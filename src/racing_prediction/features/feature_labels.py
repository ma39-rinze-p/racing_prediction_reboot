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
