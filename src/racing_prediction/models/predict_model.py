"""predict_model

学習済みモデルで各馬の「1 着になる確率」を予測し, ワイド (2 頭とも 3 着以内) の確率を計算する.

モデル (条件付きロジット / レース内ソフトマックス):
    win_score : モデルの出力 (各馬の強さの対数). レース内の差だけに意味がある
    p_win     : レース内でソフトマックス (exp(win_score) / Σ exp(win_score)) した 1 着になる確率.
                作り方そのものから, レースごとの合計は必ず 1 になる
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
PROB_FLOOR = 1e-6  # logloss の計算で log(0) にならないための下限


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
    # model.txt は各木の長さをバイト数で持つため, Windows の git (core.autocrlf=true) などで改行が CRLF に
    # 変換されると "Model format error" で読めなくなる. 改行を LF に揃えてから読み込む
    model_str = (model_dir / "model.txt").read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    booster = lgb.Booster(model_str=model_str)
    builder = FeatureBuilder.from_dict(metadata["feature_builder"])
    return booster, builder, metadata


# ---------------------------------------------------------------------------
# 予測
# ---------------------------------------------------------------------------
def softmax_by_race(scores: pd.Series | np.ndarray, race_ids: pd.Series | np.ndarray) -> np.ndarray:
    """レースごとにソフトマックスをとる (レース内の合計が 1 になる)."""
    s = pd.Series(np.asarray(scores, dtype=float))
    groups = pd.Series(np.asarray(race_ids))
    e = np.exp(s - s.groupby(groups).transform("max"))  # 大きな値で exp が溢れないよう, レース内の最大を引く
    return (e / e.groupby(groups).transform("sum")).to_numpy()


def predict_win(df: pd.DataFrame, booster: lgb.Booster, builder: FeatureBuilder) -> pd.DataFrame:
    """df (make_dataset の出力と同じ列構成) に win_score, p_win, pred_rank を付けて返す.

    p_win はレース内のソフトマックスなので, レースごとの合計は必ず 1 になる.
    """
    out = df.copy()
    out["win_score"] = booster.predict(builder.transform(df), raw_score=True)
    out["p_win"] = softmax_by_race(out["win_score"], out["race_id"])
    out["pred_rank"] = out.groupby("race_id")["p_win"].rank(ascending=False, method="first").astype(int)
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


def simulate_top3(race: pd.DataFrame, n_sims: int = 20000, seed: int = 0) -> np.ndarray:
    """1 レース分の予測 (predict_win の出力) から, レース結果を n_sims 回シミュレーションする.

    p_win を各馬の強さとする Plackett-Luce モデル (Harville モデル) に従う:
    1 着は p_win の確率で選ばれ, 2 着以降は残った馬の中で強さに比例した確率で選ばれる.
    (log(強さ) に Gumbel ノイズを加えて並べ替えると Plackett-Luce の着順サンプルになる)

    Returns:
        (n_sims, 頭数) の bool 配列. [s, i] = s 回目のシミュレーションで race の i 行目の馬が 3 着以内か
    """
    race = race.reset_index(drop=True)
    p = np.clip(race["p_win"].to_numpy(dtype=float), PROB_FLOOR, None)
    n = len(p)
    k = min(TOP_N, n)
    rng = np.random.default_rng(seed)
    noisy = np.log(p) + rng.gumbel(size=(n_sims, n))
    top = np.argpartition(-noisy, k - 1, axis=1)[:, :k]
    in_top = np.zeros((n_sims, n), dtype=bool)
    np.put_along_axis(in_top, top, True, axis=1)
    return in_top


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
