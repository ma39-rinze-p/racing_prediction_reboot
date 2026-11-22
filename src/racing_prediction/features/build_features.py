"""build_features

過去戦績から条件別の平均着順・出走回数を計算し, 出馬表 (馬ごとのデータ) に追加する.
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


def _condition_suffixes() -> list[str]:
    """集計条件の名前 (芝/ダート × [全体, 距離, 馬場状態, 競馬場, 海外])."""
    suffixes = []
    for surface in SURFACES:
        suffixes.append(surface)  # 芝 / ダート全体
        suffixes += [f"{surface}_{b}" for b in DISTANCE_BINS]
        suffixes += [f"{surface}_{c}" for c in TRACK_CONDITIONS]
        suffixes += [f"{surface}_{v}" for v in VENUES]
        suffixes.append(f"{surface}_overseas")  # 海外レース
    return suffixes


def avg_rank_feature_names() -> list[str]:
    """条件別の平均着順の列名一覧 (avg_rank_turf, avg_rank_turf_short, ...)."""
    return [f"avg_rank_{s}" for s in _condition_suffixes()]


def n_starts_feature_names() -> list[str]:
    """条件別の出走回数の列名一覧 (n_starts_turf, n_starts_turf_short, ...)."""
    return [f"n_starts_{s}" for s in _condition_suffixes()]


def _condition_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """{条件名: その条件に該当するレースの bool マスク}."""
    masks = {}
    for surface in SURFACES:
        on_surface = df["surface"] == surface
        masks[surface] = on_surface
        for name, (lo, hi) in DISTANCE_BINS.items():
            masks[f"{surface}_{name}"] = on_surface & df["distance"].between(lo, hi).fillna(False).astype(bool)
        for cond in TRACK_CONDITIONS:
            masks[f"{surface}_{cond}"] = on_surface & (df["track_condition"] == cond)
        for key, venue in VENUES.items():
            masks[f"{surface}_{key}"] = on_surface & (df["venue"] == venue)
        masks[f"{surface}_overseas"] = on_surface & df["is_overseas"].astype(bool)
    return masks


def compute_avg_rank_features(results: pd.DataFrame, before: pd.Timestamp | None = None) -> dict:
    """1 頭分の過去戦績から条件別の平均着順と出走回数を計算する.

    Args:
        results: fetch_horse_results の戻り値
        before: この日付より前のレースだけを集計する (対象レース当日以降の結果を含めないため)

    Returns:
        {列名: 値} の dict.
          avg_rank_*: 平均着順. 該当レースがない条件は NaN
          n_starts_*: 平均着順の計算に使ったレース数 (該当レースがなければ 0)
        着順が数字でないレース (中止・除外・取消など) は平均・回数のどちらにも含めない.
    """
    df = results.dropna(subset=["rank"])
    if before is not None:
        df = df[df["date"] < before]
    rank = df["rank"].astype(float)

    features = {}
    for suffix, mask in _condition_masks(df).items():
        features[f"avg_rank_{suffix}"] = rank[mask].mean()
        features[f"n_starts_{suffix}"] = int(mask.sum())
    return features


def add_avg_rank_features(df_shutuba: pd.DataFrame) -> pd.DataFrame:
    """出馬表の各馬について戦績ページにアクセスし, 条件別の平均着順・出走回数の列を追加する.

    戦績の取得に失敗した馬は, 平均着順・出走回数ともに欠損 (不明) になる.

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

    avg_df = pd.DataFrame(rows, index=df_shutuba.index, columns=avg_rank_feature_names()).astype(float)
    n_df = pd.DataFrame(rows, index=df_shutuba.index, columns=n_starts_feature_names()).astype("Int64")
    return pd.concat([df_shutuba, avg_df, n_df], axis=1)
