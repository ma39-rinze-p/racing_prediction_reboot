"""build_features

過去戦績から 条件別の平均着順・出走回数 と 近走・能力の特徴量 (hist_*) を計算し,
出馬表 (馬ごとのデータ) に追加する.
"""

from __future__ import annotations

import warnings

import numpy as np
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


# ---------------------------------------------------------------------------
# 近走・能力の特徴量 (hist_*)
# ---------------------------------------------------------------------------
# モデルの特徴量にする列 (数値)
HISTORY_FEATURE_NAMES = [
    "hist_n_starts",              # 通算の出走数 (着順が付いたレース)
    "hist_n_starts_365",          # 直近 1 年の出走数
    "hist_days_since_last",       # 前走からの日数
    "hist_win_rate",              # 通算の勝率
    "hist_top3_rate",             # 通算の 3 着以内率
    "hist_prize_total",           # 通算の獲得賞金 (万円)
    "hist_prize_365",             # 直近 1 年の獲得賞金 (万円)
    "hist_last1_rank",            # 前走の着順
    "hist_last1_rank_ratio",      # 前走の着順 / 頭数 (0 = 1 着, 1 = 最下位)
    "hist_avg3_rank_ratio",       # 直近 3 走の 着順 / 頭数 の平均
    "hist_avg5_rank_ratio",       # 直近 5 走の 着順 / 頭数 の平均
    "hist_best5_rank_ratio",      # 直近 5 走の 着順 / 頭数 の最小
    "hist_last1_margin",          # 前走の着差 (秒. 1 着はマイナス)
    "hist_avg3_margin",           # 直近 3 走の着差の平均
    "hist_best5_margin",          # 直近 5 走の着差の最小
    "hist_last1_popularity",      # 前走の人気
    "hist_avg3_popularity",       # 直近 3 走の人気の平均
    "hist_avg5_rank_minus_pop",   # 直近 5 走の (着順 - 人気) の平均. マイナスなら人気以上に走っている
    "hist_last1_log_odds",        # 前走の単勝オッズの対数
    "hist_avg3_log_odds",         # 直近 3 走の単勝オッズの対数の平均
    "hist_last1_agari",           # 前走の上がり 3F (秒)
    "hist_best3_agari",           # 直近 3 走の上がり 3F の最速
    "hist_avg3_first_corner",     # 直近 3 走の最初のコーナーの位置 / 頭数 の平均 (0 = 先頭. 脚質)
    "hist_avg3_last_corner",      # 直近 3 走の最後のコーナーの位置 / 頭数 の平均
    "hist_avg3_gain",             # 直近 3 走の (最後のコーナーの位置 - 着順) / 頭数 の平均. プラスなら直線で追い上げる
    "hist_last1_class",           # 前走のクラス (0: 新馬・未勝利 〜 7: G1)
    "hist_max5_class",            # 直近 5 走の最高クラス
    "hist_class_diff",            # 今回のクラス - 前走のクラス (プラスなら昇級)
    "hist_distance_diff",         # 今回の距離 - 前走の距離 (m)
    "hist_surface_change",        # 前走と芝・ダートが違えば 1
    "hist_jockey_change",         # 前走と騎手が違えば 1
    "hist_weight_carried_diff",   # 今回の斤量 - 前走の斤量
]

# スピード指数の計算用 (直近 3 走の 走破タイム・距離・芝/ダート・競馬場・馬場状態).
# 基準タイムは学習データから決めるため (model_features.FeatureBuilder), 生の値のまま CSV に保存する
HISTORY_RAW_FIELDS = ["time", "distance", "surface", "venue", "track_condition"]
N_SPEED_RACES = 3
HISTORY_RAW_NAMES = [f"hist_last{k}_{f}" for k in range(1, N_SPEED_RACES + 1) for f in HISTORY_RAW_FIELDS]


def history_feature_names() -> list[str]:
    return list(HISTORY_FEATURE_NAMES)


def history_raw_names() -> list[str]:
    return list(HISTORY_RAW_NAMES)


def _mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def _min(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.min()) if len(s) else np.nan


def _ratio(pos: pd.Series, n_field: pd.Series) -> pd.Series:
    """着順・通過順位を 0 (先頭) 〜 1 (最後方) にする. 頭数が 1 以下なら欠損."""
    pos = pd.to_numeric(pos, errors="coerce")
    n = pd.to_numeric(n_field, errors="coerce")
    return ((pos - 1) / (n - 1)).where(n > 1).clip(0, 1)


def _diff(a, b) -> float:
    a, b = pd.to_numeric(pd.Series([a, b]), errors="coerce")
    return float(a - b) if pd.notna(a) and pd.notna(b) else np.nan


def _changed(a, b) -> float:
    """a と b が違えば 1. 片方が欠損なら NaN.

    騎手名は出馬表が略称 ("丹内"), 戦績ページが正式名 ("丹内祐次") のことがあるため, 前方一致なら同じとみなす.
    """
    if a is None or b is None or pd.isna(a) or pd.isna(b) or a == "" or b == "":
        return np.nan
    a, b = str(a), str(b)
    return float(not (a.startswith(b) or b.startswith(a)))


def compute_history_features(results: pd.DataFrame, before: pd.Timestamp | None = None, race: dict | None = None) -> dict:
    """1 頭分の過去戦績から, 近走・能力の特徴量 (HISTORY_FEATURE_NAMES) とスピード指数用の生の値を計算する.

    Args:
        results: fetch_horse_results の戻り値
        before: この日付より前のレースだけを使う (対象レース当日以降の結果を含めないため)
        race: 今回のレースの情報 (race_class, race_distance, race_surface, jockey_name, weight_carried).
              前走との差 (昇級・距離変更・騎手の乗り替わりなど) の計算に使う
    """
    race = race or {}
    df = results
    if before is not None and pd.notna(before):
        df = df[df["date"] < pd.Timestamp(before)]
    df = df.sort_values("date", ascending=False)
    started = df[df["rank"].notna()]  # 着順が付いたレース (中止・除外などを除く)
    rank = pd.to_numeric(started["rank"], errors="coerce")
    rank_ratio = _ratio(started["rank"], started["n_field"])
    last = started.iloc[0] if len(started) else pd.Series(dtype=object)
    last3, last5 = started.head(3), started.head(5)
    ref = pd.Timestamp(before) if before is not None and pd.notna(before) else pd.Timestamp.now().normalize()
    recent = df["date"] >= ref - pd.Timedelta(days=365)

    first_corner = _ratio(last3["first_corner"], last3["n_field"])
    last_corner = _ratio(last3["last_corner"], last3["n_field"])
    margin = pd.to_numeric(started["margin"], errors="coerce").clip(-3, 10)
    log_odds = np.log(pd.to_numeric(started["odds"], errors="coerce").clip(lower=1.0))

    f = {
        "hist_n_starts": float(len(started)),
        "hist_n_starts_365": float(recent[started.index].sum()) if len(started) else 0.0,
        "hist_days_since_last": float((ref - df["date"].max()).days) if len(df) and pd.notna(df["date"].max()) else np.nan,
        "hist_win_rate": float((rank == 1).mean()) if len(rank) else np.nan,
        "hist_top3_rate": float((rank <= 3).mean()) if len(rank) else np.nan,
        "hist_prize_total": float(pd.to_numeric(df["prize"], errors="coerce").fillna(0).sum()),
        "hist_prize_365": float(pd.to_numeric(df.loc[recent, "prize"], errors="coerce").fillna(0).sum()),
        "hist_last1_rank": float(rank.iloc[0]) if len(rank) else np.nan,
        "hist_last1_rank_ratio": float(rank_ratio.iloc[0]) if len(rank_ratio) and pd.notna(rank_ratio.iloc[0]) else np.nan,
        "hist_avg3_rank_ratio": _mean(rank_ratio.head(3)),
        "hist_avg5_rank_ratio": _mean(rank_ratio.head(5)),
        "hist_best5_rank_ratio": _min(rank_ratio.head(5)),
        "hist_last1_margin": _mean(margin.head(1)),
        "hist_avg3_margin": _mean(margin.head(3)),
        "hist_best5_margin": _min(margin.head(5)),
        "hist_last1_popularity": _mean(last3["popularity"].head(1)),
        "hist_avg3_popularity": _mean(last3["popularity"]),
        "hist_avg5_rank_minus_pop": _mean(rank.head(5) - pd.to_numeric(last5["popularity"], errors="coerce")),
        "hist_last1_log_odds": _mean(log_odds.head(1)),
        "hist_avg3_log_odds": _mean(log_odds.head(3)),
        "hist_last1_agari": _mean(last3["agari"].head(1)),
        "hist_best3_agari": _min(last3["agari"]),
        "hist_avg3_first_corner": _mean(first_corner),
        "hist_avg3_last_corner": _mean(last_corner),
        "hist_avg3_gain": _mean(last_corner - rank_ratio.head(3)),
        "hist_last1_class": _mean(last3["race_class"].head(1)),
        "hist_max5_class": float(pd.to_numeric(last5["race_class"], errors="coerce").max()) if len(last5) else np.nan,
        "hist_class_diff": _diff(race.get("race_class"), last.get("race_class")),
        "hist_distance_diff": _diff(race.get("race_distance"), last.get("distance")),
        "hist_surface_change": _changed(race.get("race_surface"), last.get("surface")),
        "hist_jockey_change": _changed(race.get("jockey_name"), last.get("jockey")),
        "hist_weight_carried_diff": _diff(race.get("weight_carried"), last.get("weight_carried")),
    }

    # スピード指数用: 走破タイムがある直近 3 走
    timed = started[pd.to_numeric(started["time_sec"], errors="coerce").notna()].head(N_SPEED_RACES)
    for k in range(1, N_SPEED_RACES + 1):
        row = timed.iloc[k - 1] if len(timed) >= k else None
        for field, src in zip(HISTORY_RAW_FIELDS, ["time_sec", "distance", "surface", "venue", "track_condition"]):
            value = row[src] if row is not None else None
            f[f"hist_last{k}_{field}"] = None if value is None or pd.isna(value) else value
    return f


def add_history_features(df_shutuba: pd.DataFrame) -> pd.DataFrame:
    """出馬表の各馬について戦績ページにアクセスし, 次の列を追加する.

    - 条件別の平均着順・出走回数 (avg_rank_*, n_starts_*)
    - 近走・能力の特徴量 (hist_*) と, スピード指数用の直近 3 走の生の値 (hist_last{k}_*)

    戦績の取得に失敗した馬は, これらの列が欠損 (不明) になる.
    df_shutuba に race_date 列があれば, その日より前のレースだけを集計する.
    1 頭ごとに 1 回アクセスする (アクセスごとに 1〜3 秒待機するため, 16 頭で 30 秒前後かかる).
    保存済みの戦績 (scrape_horse_results のディスクキャッシュ) が使える馬はアクセスしない.
    """
    avg_rows, hist_rows = [], []
    for _, horse in df_shutuba.iterrows():
        before = horse.get("race_date")
        try:
            results = fetch_horse_results(horse["horse_id"], needed_until=before)
            avg_rows.append(compute_avg_rank_features(results, before=before))
            hist_rows.append(compute_history_features(results, before=before, race=horse.to_dict()))
        except requests.RequestException as e:
            warnings.warn(f"戦績の取得に失敗しました (horse_id={horse['horse_id']}): {e}")
            avg_rows.append({})
            hist_rows.append({})

    idx = df_shutuba.index
    avg_df = pd.DataFrame(avg_rows, index=idx, columns=avg_rank_feature_names()).astype(float)
    n_df = pd.DataFrame(avg_rows, index=idx, columns=n_starts_feature_names()).astype("Int64")
    hist_df = pd.DataFrame(hist_rows, index=idx, columns=HISTORY_FEATURE_NAMES).astype(float)
    raw_df = pd.DataFrame(hist_rows, index=idx, columns=HISTORY_RAW_NAMES)
    return pd.concat([df_shutuba, avg_df, n_df, hist_df, raw_df], axis=1)
