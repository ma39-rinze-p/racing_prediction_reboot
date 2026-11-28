"""predict_model

学習済みモデルで各馬の「1 着になる確率」を予測し, ワイド (2 頭とも 3 着以内) の確率を計算する.

モデル (条件付きロジット / レース内ソフトマックス):
    win_score : モデルの出力 (各馬の強さの対数). レース内の差だけに意味がある
    p_win     : レース内でソフトマックス (exp(win_score) / Σ exp(win_score)) した 1 着になる確率.
                作り方そのものから, レースごとの合計は必ず 1 になる
    オッズとの差を学習したモデル (FeatureBuilder.market_offset=True):
                win_score = モデルの出力 + log(単勝オッズから逆算した確率). モデルは「オッズがどれだけ外れているか」
                だけを学習している (オッズは特徴量に入れず, 出発点として足す). 予測には発走前の単勝オッズが必要
    3 着以内のモデル (任意. model_top3.txt):
                p_top3 = 各馬が 3 着以内に入る確率 (二値分類のロジットをシグモイドで確率にする).
                オッズとの差を学習したモデルでは, 単勝オッズから Harville モデルで計算した 3 着以内の確率の
                ロジットを出発点にする
    着順のシミュレーション (simulate_orders):
                1 着は p_win の確率で選び, 2・3 着は残りの馬から「2・3 着用の強さ」に比例した確率で選ぶ.
                3 着以内のモデルがあれば, 各馬の 3 着以内の確率が p_top3 に合うように 2・3 着用の強さを調整する
                (1 着の確率は p_win のまま). なければ 2・3 着用の強さも p_win (Harville モデル).
                ここから単勝・複勝・ワイド・3 連複・3 連単の確率が全部出せる
    ワイドの確率: p_win を各馬の強さとする Plackett-Luce モデル (Harville モデル) でレース結果を
                  シミュレーションし, 「2 頭とも 3 着以内」になった割合を使う
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from racing_prediction.data.make_dataset import TARGET_COLUMN, TOP3_COLUMN
from racing_prediction.features.feature_labels import feature_label_ja
from racing_prediction.features.model_features import FeatureBuilder

TOP_N = 3  # ワイドは 3 着以内
MODEL_TARGET = "is_win"  # このコードが扱うモデルの目的変数 (metadata.json の "target")
TOP3_MODEL_TARGET = "is_top3"  # 3 着以内のモデルの目的変数 (metadata_top3.json の "target")
TOP3_MODEL_FILE = "model_top3.txt"
TOP3_METADATA_FILE = "metadata_top3.json"
PROB_FLOOR = 1e-6  # logloss の計算で log(0) にならないための下限
# 単勝オッズがない馬 (取消直後で表示がないなど) は, このオッズの馬として市場の確率を計算する
MISSING_ODDS = 300.0


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------
def load_artifacts(model_dir: str | Path) -> tuple[lgb.Booster, FeatureBuilder, dict]:
    """save_artifacts で保存したモデル・前処理・メタデータを読み込む.

    1 着の確率を予測するモデル (metadata.json の target が "is_win") 以外は読み込まない
    (以前の 3 着以内を予測するモデルの出力を 1 着の確率として扱うと, 誤った予想を送ってしまうため).
    """
    model_dir = Path(model_dir)
    if not (model_dir / "metadata.json").exists() or not (model_dir / "model.txt").exists():
        raise FileNotFoundError(
            f"{model_dir} にモデル (model.txt, metadata.json) がありません. "
            "python scripts/train_on_kaggle.py --update-dataset で学習し, models/production を commit & push してください."
        )
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    target = metadata.get("target", "is_top3")  # target がない = 1 着モデル導入前の 3 着以内モデル
    if target != MODEL_TARGET:
        raise ValueError(
            f"{model_dir} のモデルは目的変数が {target} のため使えません (1 着の確率を予測するモデル ({MODEL_TARGET}) が必要). "
            "python scripts/train_on_kaggle.py --update-dataset で学習し直してください."
        )
    booster = _read_booster(model_dir / "model.txt")
    builder = FeatureBuilder.from_dict(metadata["feature_builder"])
    return booster, builder, metadata


def _read_booster(path: Path) -> lgb.Booster:
    # model.txt は各木の長さをバイト数で持つため, Windows の git (core.autocrlf=true) などで改行が CRLF に
    # 変換されると "Model format error" で読めなくなる. 改行を LF に揃えてから読み込む
    return lgb.Booster(model_str=Path(path).read_bytes().replace(b"\r\n", b"\n").decode("utf-8"))


def load_top3_artifacts(model_dir: str | Path) -> tuple[lgb.Booster, FeatureBuilder, dict] | None:
    """3 着以内のモデル (model_top3.txt, metadata_top3.json) を読み込む. なければ None (1 着のモデルだけで予測する)."""
    model_dir = Path(model_dir)
    if not (model_dir / TOP3_MODEL_FILE).exists() or not (model_dir / TOP3_METADATA_FILE).exists():
        return None
    metadata = json.loads((model_dir / TOP3_METADATA_FILE).read_text(encoding="utf-8"))
    if metadata.get("target") != TOP3_MODEL_TARGET:
        raise ValueError(f"{model_dir / TOP3_METADATA_FILE} は 3 着以内のモデルではありません (target: {metadata.get('target')})")
    return _read_booster(model_dir / TOP3_MODEL_FILE), FeatureBuilder.from_dict(metadata["feature_builder"]), metadata


# ---------------------------------------------------------------------------
# 予測
# ---------------------------------------------------------------------------
def softmax_by_race(scores: pd.Series | np.ndarray, race_ids: pd.Series | np.ndarray) -> np.ndarray:
    """レースごとにソフトマックスをとる (レース内の合計が 1 になる)."""
    s = pd.Series(np.asarray(scores, dtype=float))
    groups = pd.Series(np.asarray(race_ids))
    e = np.exp(s - s.groupby(groups).transform("max"))  # 大きな値で exp が溢れないよう, レース内の最大を引く
    return (e / e.groupby(groups).transform("sum")).to_numpy()


def market_log_prob(df: pd.DataFrame) -> np.ndarray:
    """単勝オッズから逆算した 1 着の確率 (レース内で合計 1) の対数.

    オッズとの差を学習するモデルの出発点 (LightGBM の init_score) と, オッズとの比較に使う.
    オッズがない馬は MISSING_ODDS 倍として扱う. レースの全馬のオッズがない場合は均等.
    """
    odds = pd.to_numeric(df["win_odds"], errors="coerce") if "win_odds" in df else pd.Series(np.nan, index=df.index)
    odds = odds.where(odds > 0)
    race = df["race_id"].to_numpy()
    has_any = odds.notna().groupby(race).transform("any").to_numpy()
    inv = pd.Series(np.where(has_any, 1 / odds.fillna(MISSING_ODDS).to_numpy(dtype=float), 1.0), index=df.index)
    return np.log((inv / inv.groupby(race).transform("sum")).to_numpy(dtype=float))


def predict_win(df: pd.DataFrame, booster: lgb.Booster, builder: FeatureBuilder) -> pd.DataFrame:
    """df (make_dataset の出力と同じ列構成) に win_score, p_win, pred_rank を付けて返す.

    p_win はレース内のソフトマックスなので, レースごとの合計は必ず 1 になる.
    オッズとの差を学習したモデル (builder.market_offset) では, モデルの出力に単勝オッズの確率の対数を足す
    (df の win_odds 列が必要. 推論では発走前のオッズを入れておく).
    """
    out = df.copy()
    out["win_score"] = booster.predict(builder.transform(df), raw_score=True)
    if getattr(builder, "market_offset", False):
        if "win_odds" not in df or pd.to_numeric(df["win_odds"], errors="coerce").notna().sum() == 0:
            raise ValueError("このモデルはオッズとの差を学習したモデルのため, 単勝オッズ (win_odds 列) が必要です")
        out["win_score"] = out["win_score"].to_numpy() + market_log_prob(df)
    out["p_win"] = softmax_by_race(out["win_score"], out["race_id"])
    out["pred_rank"] = out.groupby("race_id")["p_win"].rank(ascending=False, method="first").astype(int)
    return out


def market_top3_prob(df: pd.DataFrame) -> np.ndarray:
    """単勝オッズから逆算した 1 着の確率 (market_log_prob) を各馬の強さとする Harville モデルで計算した,
    各馬が 3 着以内に入る確率 (レース内の合計は min(3, 頭数)). 3 着以内のモデルの出発点と比較に使う.

    P(i が 3 着以内) = P(1 着) + P(2 着) + P(3 着)
        P(2 着) = Σ_j p_j × p_i / (1 - p_j)
        P(3 着) = Σ_{j, k} p_j × p_k / (1 - p_j) × p_i / (1 - p_j - p_k)   (j, k, i はすべて別の馬)
    """
    q = np.exp(market_log_prob(df))
    out = np.ones(len(df))
    groups = pd.DataFrame({"race": df["race_id"].to_numpy()}).groupby("race", sort=False).indices
    for idx in groups.values():
        p = q[idx]
        if len(p) <= 3:
            continue  # 3 頭以下なら全馬 3 着以内
        a = p / np.clip(1 - p, 1e-12, None)
        p2 = p * (a.sum() - a)
        m = np.outer(a, p)  # m[j, k] = j が 1 着で k が 2 着の確率
        np.fill_diagonal(m, 0.0)
        t = m / np.clip(1 - p[:, None] - p[None, :], 1e-12, None)
        np.fill_diagonal(t, 0.0)
        p3 = p * (t.sum() - t.sum(axis=1) - t.sum(axis=0))
        out[idx] = np.clip(p + p2 + p3, 0.0, 1.0)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), PROB_FLOOR, 1 - PROB_FLOOR)
    return np.log(p / (1 - p))


def market_top3_logit(df: pd.DataFrame) -> np.ndarray:
    """market_top3_prob のロジット (3 着以内のモデルの出発点. LightGBM の init_score)."""
    return _logit(market_top3_prob(df))


def predict_top3(df: pd.DataFrame, booster: lgb.Booster, builder: FeatureBuilder) -> pd.DataFrame:
    """3 着以内のモデルで, df に top3_score (ロジット), p_top3 (3 着以内に入る確率), top3_rank を付けて返す.

    オッズとの差を学習したモデル (builder.market_offset) では, 単勝オッズから計算した 3 着以内の確率の
    ロジットを足す (df の win_odds 列が必要).
    """
    out = df.copy()
    score = booster.predict(builder.transform(df), raw_score=True)
    if getattr(builder, "market_offset", False):
        if "win_odds" not in df or pd.to_numeric(df["win_odds"], errors="coerce").notna().sum() == 0:
            raise ValueError("このモデルはオッズとの差を学習したモデルのため, 単勝オッズ (win_odds 列) が必要です")
        score = score + market_top3_logit(df)
    out["top3_score"] = score
    out["p_top3"] = 1 / (1 + np.exp(-score))
    out["top3_rank"] = out.groupby("race_id")["p_top3"].rank(ascending=False, method="first").astype(int)
    return out


def odds_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """確定単勝オッズから逆算した 1 着の確率 (比較用. モデルの特徴量には使わない).

    p_win = (1 / 単勝オッズ) をレース内の合計が 1 になるよう揃えた値 (控除率の分は揃える処理で取り除かれる).
    オッズがない馬 (取消など) は 0. レースの全馬のオッズがない場合は均等. pred_rank は p_win の高い順 (≒ 人気順).
    """
    out = df.copy()
    inv = (1 / pd.to_numeric(out["win_odds"], errors="coerce")).fillna(0.0)
    total = inv.groupby(out["race_id"]).transform("sum")
    size = out.groupby("race_id")["race_id"].transform("size")
    out["p_win"] = np.where(total > 0, inv / total.where(total > 0, 1), 1 / size)
    out["pred_rank"] = out.groupby("race_id")["p_win"].rank(ascending=False, method="first").astype(int)
    return out


def _top3_target(p_win: np.ndarray, p_top3: np.ndarray) -> np.ndarray:
    """3 着以内のモデルの確率を, シミュレーションで再現できる形に整える.

    各馬で p_win 以上 1 以下にし, レース内の合計が 3 (頭数が 3 以下なら頭数) になるよう,
    「3 着以内の確率 - 1 着の確率」(2・3 着に入る分) を同じ比率で伸縮する.
    """
    n = len(p_win)
    total = min(TOP_N, n)
    target = np.clip(np.nan_to_num(p_top3, nan=0.0), p_win, 1.0)
    for _ in range(20):
        extra = target - p_win
        free = target < 1.0 - 1e-12
        need = total - target.sum()
        if abs(need) < 1e-9 or extra[free].sum() <= 0:
            break
        target = np.where(free, np.clip(p_win + extra * (1 + need / extra[free].sum()), p_win, 1.0), target)
    return target


def simulate_orders(
    race: pd.DataFrame,
    n_sims: int = 20000,
    seed: int = 0,
    p_top3: np.ndarray | None = None,
    n_iter: int = 50,
    tol: float = 0.002,
) -> np.ndarray:
    """1 レース分の予測から, 1〜3 着の着順を n_sims 回シミュレーションする.

    1 着: p_win の確率で選ぶ (Plackett-Luce の 1 段目)
    2・3 着: 残りの馬から, 2・3 着用の強さ w に比例した確率で順に選ぶ.
        p_top3 (3 着以内のモデルの確率. race と同じ行の順) を渡せば, 各馬の 3 着以内の割合が p_top3 に合うよう
        w を調整する (同じ乱数のまま, ずれの比で w を掛け直すのを繰り返す). 渡さなければ w = p_win (Harville モデル).
    log(強さ) に Gumbel ノイズを加えて大きい順に選ぶと, 強さに比例した確率で選んだことになる.

    Returns:
        (n_sims, min(3, 頭数)) の int 配列. [s, r] = s 回目のシミュレーションで (r+1) 着になった馬の race の行番号
    """
    race = race.reset_index(drop=True)
    p = np.clip(race["p_win"].to_numpy(dtype=float), PROB_FLOOR, None)
    p = p / p.sum()
    n = len(p)
    k = min(TOP_N, n)
    rng = np.random.default_rng(seed)
    g1 = rng.gumbel(size=(n_sims, n))
    g2 = rng.gumbel(size=(n_sims, n))
    first = np.argmax(np.log(p) + g1, axis=1)
    rows = np.arange(n_sims)

    def rest(log_w: np.ndarray) -> np.ndarray:
        s = log_w + g2
        s[rows, first] = -np.inf
        picks = []
        for _ in range(k - 1):
            pick = np.argmax(s, axis=1)
            picks.append(pick)
            s[rows, pick] = -np.inf
        return np.column_stack(picks) if picks else np.empty((n_sims, 0), dtype=int)

    log_w = np.log(p)
    if p_top3 is not None and n > TOP_N:
        target = _top3_target(p, np.asarray(p_top3, dtype=float))
        p_first = np.bincount(first, minlength=n) / n_sims
        want = np.clip(target - p_first, 1e-6, None)  # 2・3 着に入ってほしい割合
        for _ in range(n_iter):
            got = np.bincount(rest(log_w).ravel(), minlength=n) / n_sims
            if np.abs(got - want).max() < tol:
                break
            log_w = log_w + np.log(want / np.clip(got, 1 / n_sims, None))
            log_w = log_w - log_w.max()
    return np.column_stack([first, rest(log_w)]).astype(int)


def orders_to_in_top(orders: np.ndarray, n: int) -> np.ndarray:
    """simulate_orders の結果を, (n_sims, 頭数) の「3 着以内か」の bool 配列にする."""
    in_top = np.zeros((len(orders), n), dtype=bool)
    np.put_along_axis(in_top, orders, True, axis=1)
    return in_top


def simulate_top3(
    race: pd.DataFrame, n_sims: int = 20000, seed: int = 0, p_top3: np.ndarray | None = None
) -> np.ndarray:
    """1 レース分の予測から, レース結果を n_sims 回シミュレーションする (simulate_orders).

    p_top3 (3 着以内のモデルの確率) を渡せば, 各馬の 3 着以内の割合がそれに合うようにする.

    Returns:
        (n_sims, 頭数) の bool 配列. [s, i] = s 回目のシミュレーションで race の i 行目の馬が 3 着以内か
    """
    orders = simulate_orders(race, n_sims=n_sims, seed=seed, p_top3=p_top3)
    return orders_to_in_top(orders, len(race))


def wide_probabilities(
    race: pd.DataFrame,
    n_sims: int = 20000,
    seed: int = 0,
    in_top: np.ndarray | None = None,
) -> pd.DataFrame:
    """1 レース分の予測 (predict_win の出力) から, 全組み合わせのワイド的中確率を計算する.

    simulate_top3 のシミュレーションで「2 頭とも 3 着以内」になった割合を使う.
    in_top (simulate_top3 の結果) を渡せば, シミュレーションをやり直さずに使う.

    Returns:
        umaban_1, umaban_2, horse_name_1, horse_name_2, p_wide を p_wide の降順で並べた DataFrame
    """
    race = race.reset_index(drop=True)
    n = len(race)
    if n < 2:
        return pd.DataFrame(columns=["umaban_1", "umaban_2", "horse_name_1", "horse_name_2", "p_wide"])
    if in_top is None:
        in_top = simulate_top3(race, n_sims=n_sims, seed=seed)

    in_top = in_top.astype(float)
    pair = in_top.T @ in_top / len(in_top)

    rows = [
        {
            "umaban_1": int(race.loc[i, "umaban"]),
            "umaban_2": int(race.loc[j, "umaban"]),
            "horse_name_1": race.loc[i, "horse_name"],
            "horse_name_2": race.loc[j, "horse_name"],
            "p_wide": pair[i, j],
        }
        for i, j in combinations(range(n), 2)
    ]
    return pd.DataFrame(rows).sort_values("p_wide", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------
def _races_with_winner(pred: pd.DataFrame) -> pd.Index:
    has = pred[TARGET_COLUMN].astype(int).groupby(pred["race_id"]).sum() > 0
    return has[has].index


def race_logloss(pred: pd.DataFrame, prob_col: str = "p_win") -> float:
    """レース単位の logloss: 1 着の馬に付けた確率の -log の平均 (小さいほど良い. 学習で最小化する指標と同じ).

    同着で 1 着が複数いるレースは, 1 着の馬の確率の合計を使う. 1 着がいないレースは除く.
    """
    pred = pred[pred[TARGET_COLUMN].notna()]
    is_win = pred[TARGET_COLUMN].astype(int) == 1
    p_winner = pred[prob_col].where(is_win, 0).groupby(pred["race_id"]).sum()
    p_winner = p_winner[p_winner.index.isin(_races_with_winner(pred))]
    return float(-np.log(p_winner.clip(lower=PROB_FLOOR)).mean())


def evaluate_predictions(pred: pd.DataFrame, prob_col: str = "p_win", rank_col: str = "pred_rank") -> dict:
    """予測結果 (predict_win / odds_implied_probabilities の出力, 結果確定済み) の評価指標.

    roc_auc           : ROC-AUC. 1 着の馬をどれだけ上位に並べられたか (全レースの馬をまとめて計算)
    roc_auc_by_race   : レースごとの ROC-AUC の平均 (同じレース内での順番付けの良さ)
    logloss           : レース単位の logloss (1 着の馬に付けた確率の -log の平均. 小さいほど良い)
    brier             : 各馬の 1 着の確率の Brier スコア (小さいほど良い)
    win_hit           : 予測 1 位の馬が 1 着だった割合 (単勝の的中率)
    winner_in_top3    : 1 着の馬が予測上位 3 頭に入っていた割合
    top3_precision    : 予測上位 3 頭のうち実際に 3 着以内だった割合 (レース平均)
    wide_top_pair_hit : 予測上位 2 頭のワイドが的中した割合 (レース単位)
    """
    pred = pred[pred[TARGET_COLUMN].notna()]
    y = pred[TARGET_COLUMN].astype(int)
    rank = pred[rank_col]
    race = pred["race_id"]
    metrics = {
        "n_races": int(race.nunique()),
        "n_rows": int(len(pred)),
        **roc_auc_summary(pred, prob_col),
        "logloss": race_logloss(pred, prob_col),
        "brier": float(brier_score_loss(y, pred[prob_col].clip(0, 1))),
        "win_hit": float((y[rank == 1].groupby(race[rank == 1]).sum() > 0).mean()),
        "winner_in_top3": float((y[rank <= 3].groupby(race[rank <= 3]).sum() > 0).mean()),
    }
    if TOP3_COLUMN in pred:
        top3 = pred[TOP3_COLUMN].astype(int)
        metrics["top3_precision"] = float((top3[rank <= TOP_N].groupby(race[rank <= TOP_N]).sum() / TOP_N).mean())
        metrics["wide_top_pair_hit"] = float((top3[rank <= 2].groupby(race[rank <= 2]).sum() == 2).mean())
    return metrics


def roc_auc_summary(pred: pd.DataFrame, score_col: str, target_col: str = TARGET_COLUMN) -> dict:
    """score_col (大きいほど 1 着になりやすい) の ROC-AUC.

    roc_auc         : 全レースの馬をまとめて計算した ROC-AUC
    roc_auc_by_race : レースごとに計算した ROC-AUC の平均 (1 着とそれ以外が両方いるレースのみ)
    """
    pred = pred[pred[target_col].notna() & pred[score_col].notna()]
    y = pred[target_col].astype(int)
    overall = float(roc_auc_score(y, pred[score_col])) if y.nunique() == 2 else float("nan")
    by_race = [
        roc_auc_score(g[target_col].astype(int), g[score_col])
        for _, g in pred.groupby("race_id")
        if g[target_col].nunique() == 2
    ]
    return {"roc_auc": overall, "roc_auc_by_race": float(np.mean(by_race)) if by_race else float("nan")}


def odds_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """確定単勝オッズから逆算した確率で予想した場合の予測結果 (evaluate_predictions にそのまま渡せる)."""
    return odds_implied_probabilities(df.drop(columns=["p_win", "pred_rank", "win_score"], errors="ignore"))


def top_pick_table(pred: pd.DataFrame) -> pd.DataFrame:
    """レースごとに, モデルの予測 1 位 (スコアがトップの馬) と 1 番人気 (単勝オッズが一番低い馬) を並べる.

    pred: predict_win の出力 (結果確定済み. win_odds 列が必要)
    列: race_id, race_venue, model_umaban / model_odds / model_win (予測 1 位の馬番 / 単勝オッズ / その馬が 1 着か),
        fav_umaban / fav_odds / fav_win (1 番人気の馬番 / 単勝オッズ / その馬が 1 着か), same (同じ馬か)
    1 番人気が複数 (同じオッズ) のレースは, そのうちモデルのスコアが高い方を 1 番人気とする
    (モデルに有利にも不利にもならないよう, 同じ馬を選んでいれば「同じ」扱いにする).
    単勝オッズがないレース (全馬取消など) は除く.
    """
    pred = pred[pred[TARGET_COLUMN].notna()].copy()
    pred["_odds"] = pd.to_numeric(pred["win_odds"], errors="coerce").where(lambda o: o > 0)
    pred = pred[pred.groupby("race_id")["_odds"].transform("count") > 0]
    model = pred[pred["pred_rank"] == 1].set_index("race_id")
    fav = (
        pred.sort_values(["race_id", "_odds", "p_win"], ascending=[True, True, False])
        .groupby("race_id").head(1).set_index("race_id")
    )
    out = pd.DataFrame({
        "race_venue": model["race_venue"],
        "model_umaban": model["umaban"],
        "model_odds": model["_odds"],
        "model_win": model[TARGET_COLUMN].astype(int),
    })
    out["fav_umaban"] = fav["umaban"].reindex(out.index)
    out["fav_odds"] = fav["_odds"].reindex(out.index)
    out["fav_win"] = fav[TARGET_COLUMN].astype(int).reindex(out.index)
    out["same"] = out["model_umaban"] == out["fav_umaban"]
    return out.reset_index()


def compare_top_pick(pred: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 0) -> dict:
    """「スコアがトップの馬が 1 着になった割合」と「単勝オッズが一番低い馬 (1 番人気) が 1 着になった割合」を比べる.

    n_races        : 比べたレース数
    model_hit      : 予測 1 位の馬が 1 着だった割合
    favorite_hit   : 1 番人気の馬が 1 着だった割合
    diff           : model_hit - favorite_hit (プラスならモデルの方が当たる)
    diff_ci_low / diff_ci_high: diff の 95% 信頼区間 (レースを復元抽出するブートストラップ). 0 をまたげば「差があるとは言えない」
    p_value        : 差の検定 (マクネマー検定. 片方だけが当たったレースの数を比べる) の p 値
    same_rate      : 予測 1 位と 1 番人気が同じ馬だった割合
    n_model_only / n_favorite_only: 予測 1 位だけが当たった / 1 番人気だけが当たったレース数
    hit_when_same  : 同じ馬だったレースでの的中率
    model_hit_when_diff / favorite_hit_when_diff: 違う馬だったレースでの, それぞれの的中率
    単勝 100 円ずつ買った場合 (払戻 = 単勝オッズ x 100 円. 予測 1 位の単勝オッズがない場合は払戻 0 円):
    bet_per_race   : 1 レースあたりの賭け金 (100 円)
    model_profit / favorite_profit: 予測 1 位 / 1 番人気 に賭けた場合の総利益 (払戻の合計 - 賭け金の合計. 円)
    model_return / favorite_return: 回収率 (払戻の合計 / 賭け金の合計. 1.0 で元が取れる)
    return_diff    : model_return - favorite_return
    return_diff_ci_low / return_diff_ci_high: return_diff の 95% 信頼区間 (ブートストラップ)
    """
    from scipy.stats import binomtest

    t = top_pick_table(pred)
    if t.empty:
        return {"n_races": 0}
    m, f = t["model_win"].to_numpy(), t["fav_win"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), size=(n_bootstrap, len(t)))
    diffs = m[idx].mean(axis=1) - f[idx].mean(axis=1)
    bet = 100
    # 払戻 (円). 単勝オッズ (例 3.5) x 100 円
    m_pay = np.where(m == 1, t["model_odds"].fillna(0).to_numpy() * bet, 0.0)
    f_pay = np.where(f == 1, t["fav_odds"].fillna(0).to_numpy() * bet, 0.0)
    return_diffs = (m_pay[idx].mean(axis=1) - f_pay[idx].mean(axis=1)) / bet
    model_only, fav_only = int(((m == 1) & (f == 0)).sum()), int(((m == 0) & (f == 1)).sum())
    discordant = model_only + fav_only
    same, diff = t["same"].to_numpy(), ~t["same"].to_numpy()
    return {
        "n_races": int(len(t)),
        "model_hit": float(m.mean()),
        "favorite_hit": float(f.mean()),
        "diff": float(m.mean() - f.mean()),
        "diff_ci_low": float(np.quantile(diffs, 0.025)),
        "diff_ci_high": float(np.quantile(diffs, 0.975)),
        "p_value": float(binomtest(model_only, discordant, 0.5).pvalue) if discordant else 1.0,
        "same_rate": float(same.mean()),
        "n_model_only": model_only,
        "n_favorite_only": fav_only,
        "hit_when_same": float(m[same].mean()) if same.any() else float("nan"),
        "model_hit_when_diff": float(m[diff].mean()) if diff.any() else float("nan"),
        "favorite_hit_when_diff": float(f[diff].mean()) if diff.any() else float("nan"),
        "bet_per_race": bet,
        "model_profit": float(m_pay.sum() - bet * len(t)),
        "favorite_profit": float(f_pay.sum() - bet * len(t)),
        "model_return": float(m_pay.mean() / bet),
        "favorite_return": float(f_pay.mean() / bet),
        "return_diff": float((m_pay.mean() - f_pay.mean()) / bet),
        "return_diff_ci_low": float(np.quantile(return_diffs, 0.025)),
        "return_diff_ci_high": float(np.quantile(return_diffs, 0.975)),
    }


# ---------------------------------------------------------------------------
# オッズとの比較: モデルを混ぜるとオッズ単体より当たるか (メインの目標)
# ---------------------------------------------------------------------------
def _race_arrays(race_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """同じレースが連続して並んだ race_ids から, 各レースの先頭行と各行のレース番号を作る."""
    starts = np.flatnonzero(np.r_[True, race_ids[1:] != race_ids[:-1]])
    gid = np.cumsum(np.r_[True, race_ids[1:] != race_ids[:-1]]) - 1
    return starts, gid


def _blend_probs(w: np.ndarray, x: np.ndarray, starts: np.ndarray, gid: np.ndarray) -> np.ndarray:
    s = x @ w
    e = np.exp(s - np.maximum.reduceat(s, starts)[gid])
    return e / np.add.reduceat(e, starts)[gid]


def _fit_blend(x: np.ndarray, y: np.ndarray, starts: np.ndarray, gid: np.ndarray) -> tuple[np.ndarray, float]:
    """条件付きロジット s = x @ w (レース内ソフトマックス) を最尤推定する. (w, レースあたりの負の対数尤度) を返す.

    y はレース内で合計 1 の正解 (同着は等分).
    """
    from scipy.optimize import minimize

    n_races = len(starts)

    def nll(w):
        p = _blend_probs(w, x, starts, gid)
        loss = -(y * np.log(np.clip(p, PROB_FLOOR, None))).sum() / n_races
        grad = -((y - p)[:, None] * x).sum(axis=0) / n_races
        return loss, grad

    res = minimize(nll, np.ones(x.shape[1]), jac=True, method="L-BFGS-B")
    return res.x, float(res.fun)


def compare_with_market(pred: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 0, value_threshold: float = 1.0) -> dict:
    """モデルの予測をオッズと混ぜると, オッズ単体より当たるか (= モデルにオッズにない情報があるか) を調べる.

    混ぜ方: 各馬のスコア = a × log(モデルの確率) + b × log(単勝オッズの確率) をレース内でソフトマックスする
    (条件付きロジット. a, b は最尤推定). a > 0 で, 混ぜた予測がオッズ単体より当たれば, モデルはオッズにない情報を持っている.
    比べる相手の「オッズ単体」は, そのままのオッズと, b × log(単勝オッズの確率) で補正したもの (b だけ推定) の
    うち logloss が良い方にする (オッズは人気薄が過大評価されやすく, 補正しただけで良くなる分をモデルの手柄にしないため.
    レースが少ないと補正が逆効果になることもあるので, 良い方を使う = モデルに厳しい比べ方).
    a, b を決めたのと同じレースで評価すると甘くなるので, テスト期間を前半・後半に分け,
    片方で a, b を決めてもう片方を予測する (交差して両方を予測する).

    pred: predict_win の出力 (結果確定済み. win_odds 列が必要)
    n_races          : 比べたレース数 (1 着がいて, 単勝オッズがあるレース)
    model_weight / market_weight: テスト期間全体で推定した a (モデルの重み) / b (オッズの重み)
    model_weight_p_value: a = 0 (モデルを混ぜても良くならない) の検定 (尤度比検定, 片側) の p 値
    market_logloss / model_logloss / blend_logloss: レース単位の logloss (オッズ単体 / モデル単体 / 混ぜた予測)
    raw_market_logloss / calibrated_market_logloss: そのままのオッズ / 補正したオッズの logloss (良い方が market_logloss)
    logloss_gain     : market_logloss - blend_logloss (プラスなら混ぜた方が当たる. メインの目標)
    logloss_gain_ci_low / logloss_gain_ci_high: logloss_gain の 95% 信頼区間 (レースを復元抽出するブートストラップ)
    model_gain / model_gain_ci_low / model_gain_ci_high: market_logloss - model_logloss とその 95% 信頼区間
        (オッズとの差を学習したモデルは, モデル単体がすでにオッズと混ざった予測)
    market_hit / model_hit / blend_hit: それぞれの予測 1 位の馬が 1 着だった割合 (オッズの 1 位 = 1 番人気)
    blend_hit_diff   : blend_hit - market_hit
    value_threshold  : 期待値 (混ぜた予測の確率 × 単勝オッズ) がこれを超える馬を「買う」
    n_value_bets / value_bet_return: その馬の単勝を 100 円ずつ買った数と回収率
    goal_met         : 目標を達成したか (混ぜた予測かモデル単体が, オッズ単体より当たる = 改善の 95% 信頼区間が 0 より上)
    """
    from scipy.stats import chi2

    d = pred[pred[TARGET_COLUMN].notna() & pred["p_win"].notna()].copy()
    d = d[d["race_id"].isin(_races_with_winner(d))]
    odds = pd.to_numeric(d.get("win_odds"), errors="coerce").where(lambda o: o > 0)
    d = d[odds.notna().groupby(d["race_id"]).transform("any")]
    if d.empty:
        return {"n_races": 0}
    d = d.sort_values(["race_date", "race_id", "umaban"], kind="stable").reset_index(drop=True)
    race_ids = d["race_id"].to_numpy()
    starts, gid = _race_arrays(race_ids)
    win = d[TARGET_COLUMN].astype(int).to_numpy()
    y = win / np.add.reduceat(win, starts)[gid]
    lp = np.log(np.clip(d["p_win"].to_numpy(dtype=float), PROB_FLOOR, None))
    lq = market_log_prob(d)
    x = np.column_stack([lp, lq])

    # テスト期間全体での重みと, a = 0 の尤度比検定
    w_full, nll_full = _fit_blend(x, y, starts, gid)
    _, nll_market = _fit_blend(x[:, [1]], y, starts, gid)
    stat = max(0.0, 2 * (nll_market - nll_full) * len(starts))
    p_two = float(chi2.sf(stat, 1))
    p_value = p_two / 2 if w_full[0] > 0 else 1 - p_two / 2

    # 前半・後半で交差して, 決めた重みを使っていない方のレースを予測する
    race_dates = d["race_date"].to_numpy()[starts]
    first = race_dates < race_dates[len(starts) // 2]  # 同じ開催日のレースは同じ側に入れる
    if first.all() or not first.any():
        first = np.arange(len(starts)) < len(starts) // 2
    row_first = first[gid]
    pb, pm = np.zeros(len(d)), np.zeros(len(d))  # 混ぜた予測 / 補正したオッズ
    for fit_rows, pred_rows in ((row_first, ~row_first), (~row_first, row_first)):
        fs, fg = _race_arrays(race_ids[fit_rows])
        ps, pg = _race_arrays(race_ids[pred_rows])
        w, _ = _fit_blend(x[fit_rows], y[fit_rows], fs, fg)
        pb[pred_rows] = _blend_probs(w, x[pred_rows], ps, pg)
        w, _ = _fit_blend(x[fit_rows][:, [1]], y[fit_rows], fs, fg)
        pm[pred_rows] = _blend_probs(w, x[pred_rows][:, [1]], ps, pg)

    q = np.exp(lq)
    p_model = np.exp(lp) / np.add.reduceat(np.exp(lp), starts)[gid]

    def race_loss(p):
        return -np.log(np.clip(np.add.reduceat(p * (win == 1), starts), PROB_FLOOR, None))

    loss_raw, loss_cal = race_loss(q), race_loss(pm)
    loss_market = loss_raw if loss_raw.mean() <= loss_cal.mean() else loss_cal
    loss_model, loss_blend = race_loss(p_model), race_loss(pb)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(starts), size=(n_bootstrap, len(starts)))

    def gain_ci(gain):
        gains = gain[idx].mean(axis=1)
        return float(gain.mean()), float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))

    blend_gain, blend_low, blend_high = gain_ci(loss_market - loss_blend)
    model_gain, model_low, model_high = gain_ci(loss_market - loss_model)

    def top_hit(score):
        # レース内で score が最大の馬 (同じならモデルの確率が高い方) が 1 着か
        order = np.lexsort((-p_model, -score, gid))
        top_rows = order[np.r_[True, gid[order][1:] != gid[order][:-1]]]
        return float(win[top_rows].mean())

    odds_v = pd.to_numeric(d["win_odds"], errors="coerce").where(lambda o: o > 0).to_numpy()
    value = (pb * np.nan_to_num(odds_v, nan=0.0)) > value_threshold
    n_value = int(value.sum())
    market_hit, blend_hit = top_hit(q), top_hit(pb)
    return {
        "n_races": int(len(starts)),
        "model_weight": float(w_full[0]),
        "market_weight": float(w_full[1]),
        "model_weight_p_value": float(p_value),
        "market_logloss": float(loss_market.mean()),
        "raw_market_logloss": float(loss_raw.mean()),
        "calibrated_market_logloss": float(loss_cal.mean()),
        "model_logloss": float(loss_model.mean()),
        "blend_logloss": float(loss_blend.mean()),
        "logloss_gain": blend_gain,
        "logloss_gain_ci_low": blend_low,
        "logloss_gain_ci_high": blend_high,
        "model_gain": model_gain,
        "model_gain_ci_low": model_low,
        "model_gain_ci_high": model_high,
        "market_hit": market_hit,
        "model_hit": top_hit(p_model),
        "blend_hit": blend_hit,
        "blend_hit_diff": blend_hit - market_hit,
        "value_threshold": value_threshold,
        "n_value_bets": n_value,
        "value_bet_return": float((win[value] * odds_v[value]).sum() / n_value) if n_value else float("nan"),
        "goal_met": bool(blend_low > 0 or model_low > 0),
    }


def market_blend_verdict(r: dict) -> str:
    """compare_with_market の結果を一言で表す."""
    if not r.get("n_races"):
        return "比べられるレースがありません"
    if r.get("goal_met"):
        return "目標達成: モデルを混ぜるとオッズ単体より当たる (モデルにオッズにない情報がある)"
    if max(r.get("logloss_gain") or 0, r.get("model_gain") or 0) > 0:
        return "目標未達: オッズよりわずかに良いが, 偶然の範囲を出ない"
    return "目標未達: オッズ単体の方が当たる (モデルにオッズにない情報はまだない)"


def calibration_table(pred: pd.DataFrame, prob_col: str = "p_win") -> pd.DataFrame:
    """予測確率の帯ごとに, 平均予測確率と実際の 1 着率を比べる (ずれが小さいほど良い).

    1 着の確率は小さい値が多いので, 帯は低い方を細かく区切る.
    """
    pred = pred[pred[TARGET_COLUMN].notna()]
    edges = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    bins = pd.cut(pred[prob_col], bins=edges, include_lowest=True)
    return (
        pred.assign(_y=pred[TARGET_COLUMN].astype(float))
        .groupby(bins, observed=True)
        .agg(n=(prob_col, "size"), mean_pred=(prob_col, "mean"), actual_rate=("_y", "mean"))
        .reset_index(names="p_win_bin")
    )


def calibration_bins(
    pred: pd.DataFrame,
    prob_col: str = "p_win",
    n_bins: int = 10,
    method: str = "quantile",
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """スコア (1 着の確率) を n_bins 個の帯に分け, 帯ごとにスコアの平均と実際の 1 着率を比べる.

    method:
        "quantile": スコアの低い順に並べ, 頭数が同じになるように n_bins 等分する (既定. 1 着の確率は低い馬が
                    ほとんどなので, 低い確率のところも細かく見られる. どの帯も頭数が同じなのでぶれの大きさも揃う)
        "width"   : スコアの値で 0-10%, 10-20%, ... と等間隔に分ける (高い帯は馬が少ない)
    列: bin ("quantile" なら 1〜n_bins. 1 がスコアの最も低い帯 / "width" なら "10-20%" など),
        range (帯のスコアの範囲. 例 "1.2%〜2.5%"), n (馬の数), mean_score (スコアの平均), win_rate (実際の 1 着率),
        diff (win_rate - mean_score. プラスならスコアが低すぎ = 過小評価, マイナスなら過大評価),
        ci_low / ci_high (実際の 1 着率の 95% 信頼区間. Wilson 法. 馬が少ない帯ほど広い)
    馬がいない帯は n = 0 で, ほかの列は欠損.
    """
    if method not in ("quantile", "width"):
        raise ValueError(f"method は 'quantile' か 'width' にしてください: {method}")
    pred = pred[pred[target_col].notna() & pred[prob_col].notna()]
    score = pred[prob_col].clip(0, 1).to_numpy(dtype=float)
    y = pred[target_col].astype(float).to_numpy()
    if method == "quantile":
        # 同じスコアの馬が帯の境目をまたいでも頭数が揃うよう, 順位 (同じ値は出てきた順) で分ける
        order = np.argsort(score, kind="stable")
        idx = np.empty(len(score), dtype=int)
        idx[order] = np.arange(len(score)) * n_bins // max(len(score), 1)
    else:
        idx = np.minimum((score * n_bins + 1e-9).astype(int), n_bins - 1)
    rows = []
    for k in range(n_bins):
        mask = idx == k
        n = int(mask.sum())
        label = str(k + 1) if method == "quantile" else f"{round(k * 100 / n_bins)}-{round((k + 1) * 100 / n_bins)}%"
        row = {"bin": label, "range": f"{score[mask].min():.1%}〜{score[mask].max():.1%}" if n else "-", "n": n,
               "mean_score": np.nan, "win_rate": np.nan, "diff": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        if n:
            rate = float(y[mask].mean())
            z = 1.96  # Wilson の 95% 信頼区間
            center = (rate + z * z / (2 * n)) / (1 + z * z / n)
            half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            row.update(mean_score=float(score[mask].mean()), win_rate=rate, ci_low=center - half, ci_high=center + half)
            row["diff"] = rate - row["mean_score"]
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_summary(bins: pd.DataFrame) -> dict:
    """calibration_bins の結果のまとめ.

    ece          : 帯ごとの |実際の 1 着率 - スコアの平均| を馬の数で重み付けした平均 (Expected Calibration Error. 小さいほど良い)
    max_abs_diff : |差| の最大 (馬が 30 頭以上の帯だけ. 少ない帯は偶然のぶれが大きいため)
    n_outside_ci : スコアの平均が実際の 1 着率の 95% 信頼区間の外にある帯の数 (偶然では説明しにくいずれ)
    bins         : 帯ごとの表 (calibration_bins の行)
    """
    b = bins[bins["n"] > 0]
    enough = b[b["n"] >= 30]
    outside = (b["mean_score"] < b["ci_low"]) | (b["mean_score"] > b["ci_high"])
    return {
        "ece": float((b["diff"].abs() * b["n"]).sum() / b["n"].sum()) if len(b) else float("nan"),
        "max_abs_diff": float(enough["diff"].abs().max()) if len(enough) else float("nan"),
        "n_outside_ci": int(outside.sum()),
        "bins": json.loads(bins.to_json(orient="records")),
    }


def compare_calibration(pred: pd.DataFrame, n_bins: int = 10, method: str = "quantile", task: str = "win") -> dict:
    """モデルのスコアと, 単勝オッズから計算した確率のそれぞれで, 帯ごとのスコアの平均と実際の割合を比べる.

    task="win" : p_win と実際の 1 着率 (オッズ側は単勝オッズから逆算した確率)
    task="top3": p_top3 (predict_top3 の出力) と実際の 3 着以内率 (オッズ側は Harville モデルで計算した 3 着以内の確率)
    既定は頭数が同じになるように 10 等分する (calibration_bins の method). 帯はそれぞれのスコアで別々に分ける.
    {"method": ..., "n_bins": ..., "model": calibration_summary(...), "odds": calibration_summary(...)}
    """
    prob_col, target_col, baseline = (
        ("p_top3", TOP3_COLUMN, odds_top3_baseline) if task == "top3" else ("p_win", TARGET_COLUMN, odds_baseline)
    )
    out = {"method": method, "n_bins": n_bins, "task": task,
           "model": calibration_summary(calibration_bins(pred, prob_col, n_bins, method, target_col))}
    if "win_odds" in pred:
        out["odds"] = calibration_summary(calibration_bins(baseline(pred), prob_col, n_bins, method, target_col))
    return out


# ---------------------------------------------------------------------------
# 3 着以内のモデルの評価
# ---------------------------------------------------------------------------
def odds_top3_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """単勝オッズから Harville モデルで計算した 3 着以内の確率で予想した場合 (evaluate_top3 にそのまま渡せる)."""
    out = df.drop(columns=["p_top3", "top3_rank", "top3_score"], errors="ignore").copy()
    out["p_top3"] = market_top3_prob(out)
    out["top3_rank"] = out.groupby("race_id")["p_top3"].rank(ascending=False, method="first").astype(int)
    return out


def evaluate_top3(pred: pd.DataFrame, prob_col: str = "p_top3", rank_col: str = "top3_rank") -> dict:
    """3 着以内の確率の予測 (predict_top3 / odds_top3_baseline の出力, 結果確定済み) の評価指標.

    roc_auc / roc_auc_by_race: 3 着以内か否かの ROC-AUC (全馬まとめて / レースごとの平均)
    logloss        : 各馬の 3 着以内か否かの logloss (二値. 小さいほど良い)
    brier          : Brier スコア (小さいほど良い)
    place_hit      : 予測 1 位の馬が 3 着以内だった割合 (複勝の的中率)
    top3_precision : 予測上位 3 頭のうち実際に 3 着以内だった割合 (レース平均)
    sum_mean       : レースごとの確率の合計の平均 (3 に近いほど, レースの中で辻褄が合っている)
    """
    pred = pred[pred[TOP3_COLUMN].notna() & pred[prob_col].notna()]
    y = pred[TOP3_COLUMN].astype(int)
    p = pred[prob_col].clip(PROB_FLOOR, 1 - PROB_FLOOR)
    rank, race = pred[rank_col], pred["race_id"]
    return {
        "n_races": int(race.nunique()),
        "n_rows": int(len(pred)),
        **roc_auc_summary(pred, prob_col, TOP3_COLUMN),
        "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(brier_score_loss(y, p)),
        "place_hit": float(y[rank == 1].mean()),
        "top3_precision": float((y[rank <= TOP_N].groupby(race[rank <= TOP_N]).sum() / TOP_N).mean()),
        "sum_mean": float(pred[prob_col].groupby(race).sum().mean()),
    }


def compare_top3_with_market(pred: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 0) -> dict:
    """3 着以内のモデルが, 単勝オッズから計算した 3 着以内の確率 (Harville モデル) より当たるか.

    model_logloss / market_logloss: 各馬の二値の logloss (レースごとに合計し, レース数で割った値)
    model_gain     : market_logloss - model_logloss (プラスならモデルの方が当たる)
    model_gain_ci_low / model_gain_ci_high: model_gain の 95% 信頼区間 (レースを復元抽出するブートストラップ)
    model_place_hit / market_place_hit: 予測 1 位の馬が 3 着以内だった割合 (オッズ側は Harville の 3 着以内の確率が最大の馬)
    goal_met       : model_gain の 95% 信頼区間が 0 より上か
    注意: Harville モデルは人気馬の 3 着以内の確率を高く見積もりがちなので, オッズ側にやや不利な比べ方になる.
    """
    d = pred[pred[TOP3_COLUMN].notna() & pred["p_top3"].notna()].copy()
    odds = pd.to_numeric(d.get("win_odds"), errors="coerce").where(lambda o: o > 0)
    d = d[odds.notna().groupby(d["race_id"]).transform("any")]
    if d.empty:
        return {"n_races": 0}
    m = odds_top3_baseline(d)
    y = d[TOP3_COLUMN].astype(int).to_numpy()

    def race_loss(p):
        p = np.clip(np.asarray(p, dtype=float), PROB_FLOOR, 1 - PROB_FLOOR)
        return pd.Series(-(y * np.log(p) + (1 - y) * np.log(1 - p))).groupby(d["race_id"].to_numpy()).sum().to_numpy()

    loss_model, loss_market = race_loss(d["p_top3"]), race_loss(m["p_top3"])
    gain = loss_market - loss_model
    idx = np.random.default_rng(seed).integers(0, len(gain), size=(n_bootstrap, len(gain)))
    gains = gain[idx].mean(axis=1)
    low = float(np.quantile(gains, 0.025))
    return {
        "n_races": int(len(gain)),
        "model_logloss": float(loss_model.mean()),
        "market_logloss": float(loss_market.mean()),
        "model_gain": float(gain.mean()),
        "model_gain_ci_low": low,
        "model_gain_ci_high": float(np.quantile(gains, 0.975)),
        "model_place_hit": float(y[(d["top3_rank"] == 1).to_numpy()].mean()),
        "market_place_hit": float(y[(m["top3_rank"] == 1).to_numpy()].mean()),
        "goal_met": bool(low > 0),
    }


def evaluate_joint_top3(
    pred_win: pd.DataFrame, pred_top3: pd.DataFrame | None = None, n_sims: int = 2000, seed: int = 0
) -> dict:
    """着順のシミュレーション (simulate_orders) から出した確率の当たり具合 (1 着のモデル + 3 着以内のモデル の併用の確認).

    pred_win: predict_win の出力 / pred_top3: predict_top3 の出力 (同じ馬. なければ 1 着のモデルだけで Harville)
    top3_logloss : シミュレーションした 3 着以内の割合の, 各馬の二値の logloss (レースごとに合計した平均)
    wide_hit     : シミュレーションでワイドの確率が最も高い組が的中した割合
    trio_logloss : 実際の 3 着以内の 3 頭の組 (3 連複) に付けた確率の -log の平均
    trifecta_logloss: 実際の 1〜3 着の順番 (3 連単) に付けた確率の -log の平均
    (確率が 0 になった組は 1 / n_sims で数える)
    """
    keys = ["race_id", "umaban"]
    d = pred_win.sort_values(keys).reset_index(drop=True)
    if pred_top3 is not None:
        d = d.merge(pred_top3[keys + ["p_top3"]].rename(columns={"p_top3": "_p_top3"}), on=keys, how="left")
    d = d[d[TOP3_COLUMN].notna() & d["rank"].notna()]
    top3_losses, wide_hits, trio, trifecta = [], [], [], []
    for _, race in d.groupby("race_id", sort=False):
        race = race.reset_index(drop=True)
        if len(race) <= TOP_N:
            continue
        orders = simulate_orders(race, n_sims=n_sims, seed=seed,
                                 p_top3=race["_p_top3"].to_numpy() if "_p_top3" in race else None)
        in_top = orders_to_in_top(orders, len(race))
        p3 = np.clip(in_top.mean(axis=0), 1 / n_sims, 1 - 1 / n_sims)
        y = race[TOP3_COLUMN].astype(int).to_numpy()
        top3_losses.append(float(-(y * np.log(p3) + (1 - y) * np.log(1 - p3)).sum()))
        pair = in_top.T.astype(float) @ in_top.astype(float) / n_sims
        np.fill_diagonal(pair, -1)
        i, j = np.unravel_index(np.argmax(pair), pair.shape)
        wide_hits.append(bool(y[i] and y[j]))
        rank = pd.to_numeric(race["rank"], errors="coerce").to_numpy()
        actual = [int(np.flatnonzero(rank == r)[0]) if (rank == r).sum() == 1 else -1 for r in (1, 2, 3)]
        if -1 in actual:
            continue  # 同着などで 1〜3 着が 1 頭ずつ決まらないレースは 3 連複・3 連単の評価から除く
        hit_set = np.sort(orders, axis=1) == np.sort(actual)
        trio.append(-np.log(max(hit_set.all(axis=1).mean(), 1 / n_sims)))
        trifecta.append(-np.log(max((orders == np.array(actual)).all(axis=1).mean(), 1 / n_sims)))
    return {
        "n_races": len(top3_losses),
        "top3_logloss": float(np.mean(top3_losses)) if top3_losses else float("nan"),
        "wide_hit": float(np.mean(wide_hits)) if wide_hits else float("nan"),
        "n_races_order": len(trio),
        "trio_logloss": float(np.mean(trio)) if trio else float("nan"),
        "trifecta_logloss": float(np.mean(trifecta)) if trifecta else float("nan"),
    }


def feature_importance(
    booster: lgb.Booster,
    builder: FeatureBuilder | None = None,
    df: pd.DataFrame | None = None,
    sort_by: str = "gain",
) -> pd.DataFrame:
    """特徴量の重要度を重要度の高い順に並べた DataFrame を返す.

    feature    : 特徴量の名前 (英語)
    feature_ja : 特徴量の日本語名
    gain       : その特徴量で分岐したときの損失の減少量の合計 (LightGBM 標準の重要度)
    gain_share : gain の全体に占める割合
    split      : 分岐に使われた回数
    shap       : df (と builder) を渡した場合のみ. SHAP 値の絶対値の平均
                 = その特徴量が予測 (各馬の強さの対数 win_score) を平均してどれだけ動かしたか
    """
    names = booster.feature_name()
    gain = booster.feature_importance("gain")
    imp = pd.DataFrame(
        {
            "feature": names,
            "feature_ja": [feature_label_ja(n) for n in names],
            "gain": gain,
            "gain_share": gain / gain.sum() if gain.sum() > 0 else 0.0,
            "split": booster.feature_importance("split"),
        }
    )
    if df is not None and builder is not None:
        contrib = booster.predict(builder.transform(df), pred_contrib=True)  # 最後の列はバイアス
        imp["shap"] = np.abs(contrib[:, :-1]).mean(axis=0)
    imp = imp.sort_values(sort_by, ascending=False, ignore_index=True)
    imp.index = imp.index + 1  # 順位 (1 始まり)
    imp.index.name = "rank"
    return imp
