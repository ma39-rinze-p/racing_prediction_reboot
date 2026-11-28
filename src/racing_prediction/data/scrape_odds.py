"""netkeiba から単勝・複勝・ワイド・3 連複・3 連単のオッズを取得する.

出馬表の HTML にはオッズが入っておらず, ページを開いた後に JavaScript で次の API から読み込まれる.
    https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=...&type=5&action=init
    type: 1 = 単勝と複勝 (レスポンスの "1" が単勝, "2" が複勝) / 5 = ワイド / 7 = 3 連複 / 8 = 3 連単
    組み合わせのキーは馬番 2 桁の連結 ("0105" = 1 番と 5 番, "010203" = 1-2-3. 3 連単は着順の順).
    値は ["オッズ (最低)", "最高オッズ (複勝・ワイドのみ. それ以外は 0.0)", "人気"]. 1,000 倍以上は "1,779.2" のようにカンマが入る.

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
        v = float(str(value).replace(",", ""))  # 1,000 倍以上はカンマ区切り
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


# ---------------------------------------------------------------------------
# 単勝オッズ (モデルの特徴量に使う)
# ---------------------------------------------------------------------------
WIN_TYPE = "1"
WIN_ODDS_COLUMNS = ["umaban", "win_odds", "popularity"]
# 推論に使えるオッズの状態. "yoso" は発売前の予想オッズ (実際の馬券のオッズではない)
LIVE_ODDS_STATUSES = ("middle", "result")


@dataclass
class WinOdds:
    odds: pd.DataFrame  # umaban, win_odds, popularity
    status: str  # "middle" (発売中) / "result" (確定) / "yoso" (発売前の予想オッズ) など
    official_datetime: str | None  # オッズの更新時刻


def parse_win_odds_json(text: str) -> WinOdds:
    """オッズ API のレスポンスから単勝オッズを取り出す. オッズがなければ空の DataFrame.

    レスポンスの例: {"status": "middle", "data": {"official_datetime": "...",
                     "odds": {"1": {"12": ["2.4", "", "1"], ...}}}}   "12" = 馬番, ["オッズ", "", "人気"]
    取消・除外の馬はオッズが 0 や "---" になるので除く.
    """
    payload = json.loads(text)
    data = payload.get("data") or {}
    raw = ((data.get("odds") or {}).get(WIN_TYPE)) or {}
    rows = []
    for key, values in raw.items():
        if not str(key).isdigit() or not isinstance(values, list) or not values:
            continue
        odds = _to_float(values[0])
        if odds is None:
            continue
        popularity = values[2] if len(values) > 2 else None
        rows.append({"umaban": int(key), "win_odds": odds, "popularity": int(popularity) if str(popularity).isdigit() else None})
    df = pd.DataFrame(rows, columns=WIN_ODDS_COLUMNS).sort_values("umaban", ignore_index=True)
    return WinOdds(odds=df, status=str(payload.get("status", "")), official_datetime=data.get("official_datetime"))


def fetch_win_odds(race_id: str) -> WinOdds:
    """race_id の単勝オッズを取得する (推論の時点で発売中のオッズ)."""
    response = polite_get(ODDS_API_URL, params={"race_id": race_id, "type": WIN_TYPE, "action": "init"})
    return parse_win_odds_json(response.text)


# ---------------------------------------------------------------------------
# 券種ごとのオッズ (買い方の期待値の計算に使う)
# ---------------------------------------------------------------------------
# 券種 -> (API の type, レスポンスの odds のキー, 組み合わせの頭数)
BET_ODDS_TYPES = {
    "win": ("1", "1", 1),
    "place": ("1", "2", 1),
    "wide": ("5", "5", 2),
    "trio": ("7", "7", 3),
    "trifecta": ("8", "8", 3),
}
BET_ODDS_COLUMNS = ["combo", "odds_min", "odds_max", "popularity"]


@dataclass
class BetOdds:
    bet_type: str  # "win" / "place" / "wide" / "trio" / "trifecta"
    odds: pd.DataFrame  # combo (馬番のタプル. 3 連単は着順の順), odds_min, odds_max (幅がない券種は odds_min と同じ), popularity
    status: str  # "middle" (発売中) / "result" (確定) など
    official_datetime: str | None


def parse_bet_odds(payload: dict, bet_type: str) -> BetOdds:
    """オッズ API のレスポンス (json を読み込んだ dict) から, bet_type のオッズを取り出す. なければ空の DataFrame."""
    _, key, size = BET_ODDS_TYPES[bet_type]
    data = payload.get("data") or {}
    raw = ((data.get("odds") or {}).get(key)) or {}
    rows = []
    for combo, values in raw.items():
        combo = str(combo)
        if len(combo) != 2 * size or not combo.isdigit() or not isinstance(values, list) or not values:
            continue
        odds_min = _to_float(values[0])
        if odds_min is None:
            continue
        odds_max = _to_float(values[1]) if len(values) > 1 else None
        popularity = values[2] if len(values) > 2 else None
        rows.append({
            "combo": tuple(int(combo[i:i + 2]) for i in range(0, 2 * size, 2)),
            "odds_min": odds_min,
            "odds_max": odds_max if odds_max is not None and odds_max >= odds_min else odds_min,
            "popularity": int(popularity) if str(popularity).isdigit() else None,
        })
    return BetOdds(bet_type, pd.DataFrame(rows, columns=BET_ODDS_COLUMNS), str(payload.get("status", "")),
                   data.get("official_datetime"))


def fetch_bet_odds(race_id: str, bet_types: tuple[str, ...] | list[str] = tuple(BET_ODDS_TYPES)) -> dict[str, BetOdds]:
    """race_id の券種ごとのオッズを取得する. 単勝と複勝は 1 回のアクセスで両方取れる (API の type=1).

    Returns:
        {券種: BetOdds}
    """
    payloads: dict[str, dict] = {}
    out = {}
    for bet_type in bet_types:
        api_type = BET_ODDS_TYPES[bet_type][0]
        if api_type not in payloads:
            response = polite_get(ODDS_API_URL, params={"race_id": race_id, "type": api_type, "action": "init"})
            payloads[api_type] = json.loads(response.text)
        out[bet_type] = parse_bet_odds(payloads[api_type], bet_type)
    return out
