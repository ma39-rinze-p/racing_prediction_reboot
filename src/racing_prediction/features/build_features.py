"""build_features

過去戦績から条件別の平均着順を計算し, 出馬表 (馬ごとのデータ) に追加する.
"""

from __future__ import annotations

import warnings

import pandas as pd
import requests

from racing_prediction.data.scrape_horse_results import fetch_horse_results

# 距離区分 (両端を含む). 区分の隙間 (例: 1401〜1499m, 1000m 未満) のレースはどこにも入らない
DISTANCE_BINS = {
    "short": (1000, 1400),
    "mid": (1500, 2400),
    "long": (2500, float("inf")),
}

TRACK_CONDITIONS = ["good", "yielding", "soft", "heavy"]  # 良, 稍重, 重, 不良

VENUES = {
    "sapporo": "札幌",
    "hakodate": "函館",
    "fukushima": "福島",
    "niigata": "新潟",
    "tokyo": "東京",
    "nakayama": "中山",
    "chukyo": "中京",
    "kyoto": "京都",
    "hanshin": "阪神",
    "kokura": "小倉",
}

SURFACES = ["turf", "dirt"]


def distance_category(distance: float | None) -> str | None:
    """距離 (m) を "short" / "mid" / "long" に区分する. どの区分にも入らなければ None."""
    if distance is None or pd.isna(distance):
        return None
    return next((name for name, (lo, hi) in DISTANCE_BINS.items() if lo <= distance <= hi), None)


def avg_rank_feature_names() -> list[str]:
    """compute_avg_rank_features が返す列名の一覧."""
    names = []
    for surface in SURFACES:
        names.append(f"avg_rank_{surface}")  # 芝 / ダート全体
        names += [f"avg_rank_{surface}_{b}" for b in DISTANCE_BINS]
        names += [f"avg_rank_{surface}_{c}" for c in TRACK_CONDITIONS]
        names += [f"avg_rank_{surface}_{v}" for v in VENUES]
        names.append(f"avg_rank_{surface}_overseas")  # 海外レース
    return names


def compute_avg_rank_features(results: pd.DataFrame, before: pd.Timestamp | None = None) -> dict:
    """1 頭分の過去戦績から条件別の平均着順を計算する.

    Args:
        results: fetch_horse_results の戻り値
        before: この日付より前のレースだけを集計する (対象レース当日以降の結果を含めないため)

    Returns:
        {列名: 平均着順} の dict. 該当レースがない条件は NaN.
        着順が数字でないレース (中止・除外・取消など) は集計から除く.
    """
    df = results.dropna(subset=["rank"])
    if before is not None:
        df = df[df["date"] < before]
    rank = df["rank"].astype(float)

    features = {}
    for surface in SURFACES:
        on_surface = df["surface"] == surface
        features[f"avg_rank_{surface}"] = rank[on_surface].mean()

        for name, (lo, hi) in DISTANCE_BINS.items():
            mask = on_surface & df["distance"].between(lo, hi).fillna(False).astype(bool)
            features[f"avg_rank_{surface}_{name}"] = rank[mask].mean()

        for cond in TRACK_CONDITIONS:
            mask = on_surface & (df["track_condition"] == cond)
            features[f"avg_rank_{surface}_{cond}"] = rank[mask].mean()

        for key, venue in VENUES.items():
            mask = on_surface & (df["venue"] == venue)
            features[f"avg_rank_{surface}_{key}"] = rank[mask].mean()

        mask = on_surface & df["is_overseas"].astype(bool)
        features[f"avg_rank_{surface}_overseas"] = rank[mask].mean()

    return features


def add_avg_rank_features(df_shutuba: pd.DataFrame) -> pd.DataFrame:
    """出馬表の各馬について戦績ページにアクセスし, 条件別の平均着順の列を追加する.

    df_shutuba に race_date 列があれば, その日より前のレースだけを集計する.
    1 頭ごとに 1 回アクセスする (アクセスごとに 1〜3 秒待機するため, 16 頭で 30 秒前後かかる).
    """
    rows = []
    for _, horse in df_shutuba.iterrows():
        before = horse.get("race_date")
        try:
            results = fetch_horse_results(horse["horse_id"])
            features = compute_avg_rank_features(results, before=before)
        except requests.RequestException as e:
            warnings.warn(f"戦績の取得に失敗しました (horse_id={horse['horse_id']}): {e}")
            features = {}
        rows.append(features)

    features_df = pd.DataFrame(rows, index=df_shutuba.index, columns=avg_rank_feature_names())
    return pd.concat([df_shutuba, features_df.astype(float)], axis=1)
