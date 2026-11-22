"""netkeiba からワイドのオッズを取得する.

出馬表の HTML にはオッズが入っておらず, ページを開いた後に JavaScript で次の API から読み込まれる.
    https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=...&type=5&action=init
    (type=5 がワイド)

レスポンスの例:
    {"status": "middle", "data": {"official_datetime": "2026-11-23 01:26:05",
     "odds": {"5": {"0105": ["2.8", "3.2", "1"], ...}}}}
    "0105" = 1 番と 5 番の組み合わせ, ["最低オッズ", "最高オッズ", "人気"].
    status: "middle" = 発売中 (オッズは変動する), "result" = 確定

アクセスは scrape_shutuba.polite_get を経由するため,
User-Agent の付与とアクセスごとの 1〜3 秒ランダム待機が必ず行われる.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from racing_prediction.data.scrape_shutuba import polite_get

ODDS_API_URL = "https://race.netkeiba.com/api/api_get_jra_odds.html"
WIDE_TYPE = "5"

ODDS_COLUMNS = ["umaban_1", "umaban_2", "odds_min", "odds_max", "popularity"]


@dataclass
class WideOdds:
    odds: pd.DataFrame  # umaban_1, umaban_2, odds_min, odds_max, popularity
    status: str  # "middle" (発売中) / "result" (確定) など
    official_datetime: str | None  # オッズの更新時刻


def _to_float(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None  # 取消・発売前などは 0 や "---" になる


def parse_wide_odds_json(text: str) -> WideOdds:
    """オッズ API のレスポンスからワイドのオッズを取り出す. オッズがなければ空の DataFrame."""
    payload = json.loads(text)
    data = payload.get("data") or {}
    raw = ((data.get("odds") or {}).get(WIDE_TYPE)) or {}

    rows = []
    for key, values in raw.items():
        if len(key) != 4 or not key.isdigit() or not isinstance(values, list) or len(values) < 2:
            continue
        odds_min, odds_max = _to_float(values[0]), _to_float(values[1])
        if odds_min is None:
            continue
        popularity = values[2] if len(values) > 2 else None
        rows.append(
            {
                "umaban_1": int(key[:2]),
                "umaban_2": int(key[2:]),
                "odds_min": odds_min,
                "odds_max": odds_max if odds_max is not None else odds_min,
                "popularity": int(popularity) if str(popularity).isdigit() else None,
            }
        )
    odds = pd.DataFrame(rows, columns=ODDS_COLUMNS)
    return WideOdds(odds=odds, status=str(payload.get("status", "")), official_datetime=data.get("official_datetime"))


def fetch_wide_odds(race_id: str) -> WideOdds:
    """race_id のワイドのオッズを取得する (推論の時点で発売中のオッズ)."""
    response = polite_get(ODDS_API_URL, params={"race_id": race_id, "type": WIDE_TYPE, "action": "init"})
    return parse_wide_odds_json(response.text)
