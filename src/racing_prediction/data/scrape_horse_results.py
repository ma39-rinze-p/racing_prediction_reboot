"""netkeiba の競走馬データベースから各馬の過去戦績を取得する.

使用例:
    >>> from racing_prediction.data.scrape_horse_results import fetch_horse_results
    >>> df = fetch_horse_results("2023103731")

アクセスは scrape_shutuba.polite_get を経由するため,
User-Agent の付与とアクセスごとの 1〜3 秒ランダム待機が必ず行われる.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd
from bs4 import BeautifulSoup

from racing_prediction.data.scrape_shutuba import polite_get

HORSE_RESULT_URL = "https://db.netkeiba.com/horse/result/{horse_id}/"

# JRA の 10 競馬場 (開催欄 "4東京1" などから抽出する)
JRA_VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]

SURFACE_MAP = {"芝": "turf", "ダ": "dirt", "障": "steeple"}
TRACK_CONDITION_MAP = {"良": "good", "稍": "yielding", "重": "soft", "不": "heavy"}

RESULT_COLUMNS = [
    "date", "race_id", "venue", "is_overseas", "surface", "distance", "track_condition", "rank", "rank_text",
]


def _is_overseas(race_id: str | None) -> bool:
    """race_id の 5〜6 文字目 (競馬場コード) に英字が入っていれば海外レース.

    JRA は 01〜10, 地方は 30 番台以降の数字, 海外は "J0"(メイダン) のように英字を含む.
    """
    return race_id is not None and len(race_id) >= 6 and not race_id[4:6].isdigit()


def parse_horse_results_html(html: str) -> pd.DataFrame:
    """競走成績ページの HTML から 1 レース 1 行の DataFrame を作る.

    列:
        date: 開催日 (Timestamp)
        race_id: netkeiba のレース ID
        venue: 競馬場名. JRA 以外 (地方・海外) の場合は開催欄の文字列そのまま
        is_overseas: 海外レースなら True
        surface: "turf" / "dirt" / "steeple"(障害) / None
        distance: 距離 (m)
        track_condition: "good"(良) / "yielding"(稍重) / "soft"(重) / "heavy"(不良) / None
        rank: 着順 (中止・除外・取消などは欠損)
        rank_text: 着順欄の文字列そのまま
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.db_h_race_results")
    if table is None:
        # 出走歴がない馬はテーブル自体が存在しない
        return pd.DataFrame(columns=RESULT_COLUMNS)

    headers = [th.get_text(strip=True) for th in table.select("thead th")]
    col = {name: headers.index(name) for name in ("日付", "開催", "着順", "距離", "馬場")}

    rows = []
    for tr in table.select("tbody tr"):
        td_tags = tr.find_all("td", recursive=False)
        tds = [td.get_text(strip=True) for td in td_tags]
        if len(tds) < len(headers):
            continue

        race_a = td_tags[headers.index("レース名")].find("a")
        rid = re.search(r"/race/(\w+)/", race_a.get("href", "")) if race_a else None
        race_id = rid.group(1) if rid else None

        kaisai = tds[col["開催"]]
        venue = next((v for v in JRA_VENUES if v in kaisai), re.sub(r"\d", "", kaisai) or None)

        dm = re.match(r"(芝|ダ|障)(\d+)", tds[col["距離"]])
        surface = SURFACE_MAP[dm.group(1)] if dm else None
        distance = int(dm.group(2)) if dm else None

        rank_text = tds[col["着順"]]
        rm = re.match(r"\d+", rank_text)

        rows.append(
            {
                "date": pd.to_datetime(tds[col["日付"]], format="%Y/%m/%d", errors="coerce"),
                "race_id": race_id,
                "venue": venue,
                "is_overseas": _is_overseas(race_id),
                "surface": surface,
                "distance": distance,
                "track_condition": TRACK_CONDITION_MAP.get(tds[col["馬場"]][:1]),
                "rank": int(rm.group()) if rm else None,
                "rank_text": rank_text,
            }
        )

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df["rank"] = df["rank"].astype("Int64")
    df["distance"] = df["distance"].astype("Int64")
    return df


def fetch_horse_results(horse_id: str) -> pd.DataFrame:
    """horse_id を指定して競走成績ページを取得し, 過去戦績の DataFrame を返す.

    同じ馬を複数レースで使う場合 (期間内に複数回出走する馬など) は,
    実行中 (カーネル起動中) は 1 回だけアクセスし, 2 回目以降は取得済みの結果を返す.
    戦績ページには取得時点までの全レースが載っており, 集計時に対象レース日より前に
    絞り込むため, 使い回しても結果は変わらない.
    """
    return _fetch_horse_results_cached(horse_id).copy()


@lru_cache(maxsize=None)
def _fetch_horse_results_cached(horse_id: str) -> pd.DataFrame:
    response = polite_get(HORSE_RESULT_URL.format(horse_id=horse_id))
    return parse_horse_results_html(response.text)
