"""特徴量の日本語名.

    >>> feature_label_ja("avg_rank_dirt_tokyo")
    'ダート・東京の平均着順'
    >>> feature_label_ja("shr_avg_rank_distance_race_rank")
    '今回の距離での平均着順 (補正後) のレース内順位'

対応表にない名前は, そのまま (英語名) を返す.
"""

from __future__ import annotations

import re

import pandas as pd

from racing_prediction.features.build_features import VENUES

SURFACE_JA = {"turf": "芝", "dirt": "ダート"}
DISTANCE_JA = {"short": "短距離", "mid": "中距離", "long": "長距離"}
CONDITION_JA = {"good": "良", "yielding": "稍重", "soft": "重", "heavy": "不良"}
CONDITION_KIND_JA = {"surface": "芝・ダート", "distance": "距離", "condition": "馬場状態", "venue": "競馬場"}

BASE_LABELS = {
    # レース
    "race_venue": "競馬場",
    "race_distance_category": "距離区分",
    "race_surface": "芝・ダート",
    "race_track_condition": "当日の馬場状態",
    "n_horses": "出走頭数",
    "race_distance": "距離",
    "race_class": "クラス",
    "race_prize": "1着の本賞金",
    "race_n_front_runners": "レース内の逃げ・先行馬の頭数",
    # 近走・能力 (build_features.HISTORY_FEATURE_NAMES)
    "hist_n_starts": "通算の出走数",
    "hist_n_starts_365": "直近1年の出走数",
    "hist_days_since_last": "前走からの日数",
    "hist_win_rate": "通算の勝率",
    "hist_top3_rate": "通算の3着以内率",
    "hist_prize_total": "通算の獲得賞金",
    "hist_prize_365": "直近1年の獲得賞金",
    "hist_last1_rank": "前走の着順",
    "hist_last1_rank_ratio": "前走の着順/頭数",
    "hist_avg3_rank_ratio": "直近3走の着順/頭数の平均",
    "hist_avg5_rank_ratio": "直近5走の着順/頭数の平均",
    "hist_best5_rank_ratio": "直近5走の着順/頭数の最小",
    "hist_last1_margin": "前走の着差",
    "hist_avg3_margin": "直近3走の着差の平均",
    "hist_best5_margin": "直近5走の着差の最小",
    "hist_last1_popularity": "前走の人気",
    "hist_avg3_popularity": "直近3走の人気の平均",
    "hist_avg5_rank_minus_pop": "直近5走の(着順-人気)の平均",
    "hist_last1_log_odds": "前走の単勝オッズ(対数)",
    "hist_avg3_log_odds": "直近3走の単勝オッズ(対数)の平均",
    "hist_last1_agari": "前走の上がり3F",
    "hist_best3_agari": "直近3走の上がり3Fの最速",
    "hist_avg3_first_corner": "直近3走の最初のコーナーの位置(脚質)",
    "hist_avg3_last_corner": "直近3走の最後のコーナーの位置",
    "hist_avg3_gain": "直近3走の直線での追い上げ",
    "hist_last1_class": "前走のクラス",
    "hist_max5_class": "直近5走の最高クラス",
    "hist_class_diff": "前走からのクラスの変化(昇級)",
    "hist_distance_diff": "前走からの距離の変化",
    "hist_surface_change": "前走から芝・ダートが変わったか",
    "hist_jockey_change": "前走から騎手が変わったか",
    "hist_weight_carried_diff": "前走からの斤量の変化",
    # 単勝オッズ (model_features)
    "odds_log": "単勝オッズ(対数)",
    "odds_prob": "単勝オッズから見た1着の確率",
    "odds_rank": "単勝人気(オッズの順位)",
    "odds_log_minus_hist": "単勝オッズ(対数)-直近3走の平均(人気の変化)",
    # スピード指数 (model_features)
    "speed_last1": "前走のスピード指数",
    "speed_best3": "直近3走のスピード指数の最高",
    "speed_avg3": "直近3走のスピード指数の平均",
    # 馬
    "waku": "枠番",
    "sex": "性別",
    "age": "馬齢",
    "weight_carried": "斤量",
    "jockey_name": "騎手",
    "trainer_name": "調教師",
    "horse_weight": "馬体重",
    "weight_diff": "馬体重の増減",
    # 血統
    "sire": "父",
    "sire_line": "父の系統",
    "dam": "母",
    "sire_sire": "父父",
    "sire_dam": "父母",
    "dam_sire": "母父",
    "dam_dam": "母母",
}


def _condition_ja(suffix: str) -> str | None:
    """"dirt_tokyo" -> "ダート・東京", "turf" -> "芝"."""
    surface, _, rest = suffix.partition("_")
    if surface not in SURFACE_JA:
        return None
    if not rest:
        return SURFACE_JA[surface]
    detail = (
        DISTANCE_JA.get(rest)
        or CONDITION_JA.get(rest)
        or VENUES.get(rest)
        or ("海外" if rest == "overseas" else None)
    )
    return f"{SURFACE_JA[surface]}・{detail}" if detail else None


def feature_label_ja(name: str) -> str:
    """特徴量の英語名を日本語名にする. 対応がなければ英語名のまま返す."""
    if name in BASE_LABELS:
        return BASE_LABELS[name]

    # レース内での比較 (model_features で追加): *_race_rank, *_race_diff
    m = re.fullmatch(r"(.+)_race_(rank|diff)", name)
    if m:
        base = feature_label_ja(m.group(1))
        return f"{base}のレース内順位" if m.group(2) == "rank" else f"{base}のレース内平均との差"

    # 騎手・調教師・父・母父の成績 (model_features で追加): {列名}_win_rate / _top3_rate / _n_runs
    m = re.fullmatch(r"(jockey_name|trainer_name|sire|dam_sire)_(win_rate|top3_rate|n_runs)", name)
    if m:
        what = {"win_rate": "勝率", "top3_rate": "3着以内率", "n_runs": "出走数"}[m.group(2)]
        return f"{BASE_LABELS[m.group(1)]}の{what}"

    # 今回のレース条件に対応する値 (model_features で追加): cur_*, shr_*
    m = re.fullmatch(r"(cur|shr)_(avg_rank|n_starts)_(surface|distance|condition|venue)", name)
    if m:
        kind = CONDITION_KIND_JA[m.group(3)]
        what = "平均着順" if m.group(2) == "avg_rank" else "出走回数"
        label = f"今回の{kind}での{what}"
        return label + (" (補正後)" if m.group(1) == "shr" else "")

    # 過去戦績の条件別集計: avg_rank_*, n_starts_*
    for prefix, what in (("avg_rank_", "平均着順"), ("n_starts_", "出走回数")):
        if name.startswith(prefix):
            condition = _condition_ja(name[len(prefix):])
            if condition:
                return f"{condition}の{what}"
    return name


def add_feature_label_ja(df: pd.DataFrame, column: str = "feature", new_column: str = "feature_ja") -> pd.DataFrame:
    """特徴量名の列の右隣に日本語名の列を追加した DataFrame を返す."""
    out = df.copy()
    out.insert(out.columns.get_loc(column) + 1, new_column, out[column].map(feature_label_ja))
    return out
