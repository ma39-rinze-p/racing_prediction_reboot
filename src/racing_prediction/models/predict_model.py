"""predict_model

学習済みモデルで各馬の「3 着以内に入る確率」を予測し, ワイド (2 頭とも 3 着以内) の確率を計算する.

    p_top3_raw : モデルの出力そのまま
    p_top3     : レース内の合計が「3 着以内に入る頭数 (= min(3, 出走頭数))」になるよう補正した確率
    ワイドの確率: 各馬の確率の積ではなく, p_top3 に合うように強さを調整した
                  Plackett-Luce モデルでレース結果をシミュレーションして求める
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from racing_prediction.data.make_dataset import TARGET_COLUMN
from racing_prediction.features.feature_labels import feature_label_ja
from racing_prediction.features.model_features import FeatureBuilder

TOP_N = 3  # 3 着以内


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------
def load_artifacts(model_dir: str | Path) -> tuple[lgb.Booster, FeatureBuilder, dict]:
    """save_artifacts で保存したモデル・前処理・メタデータを読み込む."""
    model_dir = Path(model_dir)
    # model.txt は各木の長さをバイト数で持つため, Windows の git (core.autocrlf=true) などで改行が CRLF に
    # 変換されると "Model format error" で読めなくなる. 改行を LF に揃えてから読み込む
    model_str = (model_dir / "model.txt").read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    booster = lgb.Booster(model_str=model_str)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    builder = FeatureBuilder.from_dict(metadata["feature_builder"])
    return booster, builder, metadata


# ---------------------------------------------------------------------------
# 予測
# ---------------------------------------------------------------------------
def normalize_to_total(p: np.ndarray, total: float, max_p: float = 0.995, n_iter: int = 50) -> np.ndarray:
    """確率の合計が total になるようにスケールする (1 を超えないよう上限で切って残りを再配分)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, None)
    for _ in range(n_iter):
        p = np.minimum(p * total / p.sum(), max_p)
        if abs(p.sum() - total) < 1e-9:
            break
    return p


def predict_top3(df: pd.DataFrame, booster: lgb.Booster, builder: FeatureBuilder) -> pd.DataFrame:
    """df (make_dataset の出力と同じ列構成) に p_top3_raw, p_top3, pred_rank を付けて返す."""
    out = df.copy()
    out["p_top3_raw"] = booster.predict(builder.transform(df))
    out["p_top3"] = out.groupby("race_id")["p_top3_raw"].transform(
        lambda s: normalize_to_total(s.to_numpy(), min(TOP_N, len(s)))
    )
    out["pred_rank"] = out.groupby("race_id")["p_top3"].rank(ascending=False, method="first").astype(int)
    return out


def simulate_top3(race: pd.DataFrame, n_sims: int = 20000, n_iter: int = 30, seed: int = 0) -> np.ndarray:
    """1 レース分の予測 (predict_top3 の出力) から, レース結果を n_sims 回シミュレーションする.

    各馬の「強さ」を, シミュレーションした 3 着以内率が p_top3 に一致するよう反復調整し,
    その強さで Plackett-Luce モデルに従ってレースを n_sims 回シミュレーションする.
    (Gumbel ノイズを加えて並べ替えると Plackett-Luce の着順サンプルになる)

    Returns:
        (n_sims, 頭数) の bool 配列. [s, i] = s 回目のシミュレーションで race の i 行目の馬が 3 着以内か
    """
    race = race.reset_index(drop=True)
    p = race["p_top3"].to_numpy(dtype=float)
    n = len(p)
    k = min(TOP_N, n)

    rng = np.random.default_rng(seed)
    gumbel = rng.gumbel(size=(n_sims, n))  # 反復中は同じノイズを使う (調整を安定させるため)
    p_target = np.clip(p, 1e-4, 1 - 1e-4)
    log_strength = np.log(p_target / (1 - p_target))

    for _ in range(n_iter):
        top = np.argpartition(-(log_strength + gumbel), k - 1, axis=1)[:, :k]
        in_top = np.zeros((n_sims, n), dtype=bool)
        np.put_along_axis(in_top, top, True, axis=1)
        p_sim = np.clip(in_top.mean(axis=0), 1e-4, None)
        log_strength += np.log(p_target / p_sim)
    return in_top


def wide_probabilities(
    race: pd.DataFrame,
    n_sims: int = 20000,
    n_iter: int = 30,
    seed: int = 0,
    in_top: np.ndarray | None = None,
) -> pd.DataFrame:
    """1 レース分の予測 (predict_top3 の出力) から, 全組み合わせのワイド的中確率を計算する.

    各馬の確率の積ではなく, simulate_top3 のシミュレーションで「2 頭とも 3 着以内」になった割合を使う.
    in_top (simulate_top3 の結果) を渡せば, シミュレーションをやり直さずに使う.

    Returns:
        umaban_1, umaban_2, horse_name_1, horse_name_2, p_wide を p_wide の降順で並べた DataFrame
    """
    race = race.reset_index(drop=True)
    n = len(race)
    if n < 2:
        return pd.DataFrame(columns=["umaban_1", "umaban_2", "horse_name_1", "horse_name_2", "p_wide"])
    if in_top is None:
        in_top = simulate_top3(race, n_sims=n_sims, n_iter=n_iter, seed=seed)

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
def evaluate_predictions(pred: pd.DataFrame) -> dict:
    """予測結果 (predict_top3 の出力, 結果確定済み) の評価指標.

    logloss / brier       : 確率としての正確さ (p_top3 = レース内補正後. *_raw は補正前)
    roc_auc               : ROC-AUC. 3 着以内の馬をどれだけ上位に並べられたか (全レースまとめて)
    roc_auc_by_race       : レースごとの ROC-AUC の平均 (同じレース内での順番付けの良さ. ワイド向き)
    top3_precision        : 予測上位 3 頭のうち実際に 3 着以内だった割合 (レース平均)
    wide_top_pair_hit     : 予測上位 2 頭のワイドが的中した割合 (レース単位)
    """
    pred = pred[pred[TARGET_COLUMN].notna()]
    y = pred[TARGET_COLUMN].astype(int)
    hit = y.rename("hit")
    top3_hits = hit[pred["pred_rank"] <= TOP_N].groupby(pred["race_id"]).sum()
    top2_both = hit[pred["pred_rank"] <= 2].groupby(pred["race_id"]).sum() == 2
    metrics = {
        "n_races": int(pred["race_id"].nunique()),
        "n_rows": int(len(pred)),
        "logloss": float(log_loss(y, pred["p_top3"], labels=[0, 1])),
        "logloss_raw": float(log_loss(y, pred["p_top3_raw"], labels=[0, 1])),
        "brier": float(brier_score_loss(y, pred["p_top3"])),
        **roc_auc_summary(pred, "p_top3"),
        "roc_auc_raw": roc_auc_summary(pred, "p_top3_raw")["roc_auc"],
        "top3_precision": float((top3_hits / TOP_N).mean()),
        "wide_top_pair_hit": float(top2_both.mean()),
    }
    return metrics


def roc_auc_summary(pred: pd.DataFrame, score_col: str) -> dict:
    """score_col (大きいほど 3 着以内になりやすい) の ROC-AUC.

    roc_auc         : 全レースの馬をまとめて計算した ROC-AUC
    roc_auc_by_race : レースごとに計算した ROC-AUC の平均 (3 着以内とそれ以外が両方いるレースのみ)
    """
    pred = pred[pred[TARGET_COLUMN].notna() & pred[score_col].notna()]
    y = pred[TARGET_COLUMN].astype(int)
    overall = float(roc_auc_score(y, pred[score_col])) if y.nunique() == 2 else float("nan")
    by_race = [
        roc_auc_score(g[TARGET_COLUMN].astype(int), g[score_col])
        for _, g in pred.groupby("race_id")
        if g[TARGET_COLUMN].nunique() == 2
    ]
    return {"roc_auc": overall, "roc_auc_by_race": float(np.mean(by_race)) if by_race else float("nan")}


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
                 = その特徴量が予測 (対数オッズ) を平均してどれだけ動かしたか
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


def calibration_table(pred: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """予測確率の帯ごとに, 平均予測確率と実際の 3 着以内率を比べる (ずれが小さいほど良い)."""
    pred = pred[pred[TARGET_COLUMN].notna()]
    bins = pd.cut(pred["p_top3"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    return (
        pred.groupby(bins, observed=True)
        .agg(n=("p_top3", "size"), mean_pred=("p_top3", "mean"), actual_rate=(TARGET_COLUMN, "mean"))
        .reset_index(names="p_top3_bin")
    )
