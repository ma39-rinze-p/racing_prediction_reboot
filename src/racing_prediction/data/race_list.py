"""指定した競馬場・期間・レース番号に該当する race_id を netkeiba から集める.

race_id の形式: YYYY(年) + VV(競馬場コード) + KK(回) + DD(日目) + RR(レース番号)
    例: 202605050507 = 2026年 東京(05) 5回 5日目 7R

使用例:
    >>> from racing_prediction.data.race_list import find_race_ids
    >>> find_race_ids("東京", "2026-11-01", "2026-11-22", race_numbers=range(7, 13))

アクセスは scrape_shutuba.polite_get を経由するため,
User-Agent の付与とアクセスごとの 1〜3 秒ランダム待機が必ず行われる.
"""

from __future__ import annotations

import re

import pandas as pd
from bs4 import BeautifulSoup

from racing_prediction.data.scrape_shutuba import polite_get

CALENDAR_URL = "https://race.netkeiba.com/top/calendar.html"
RACE_LIST_URL = "https://race.netkeiba.com/top/race_list_sub.html"

# JRA 競馬場コード
VENUE_CODES = {
    "札幌": "01",
    "函館": "02",
    "福島": "03",
    "新潟": "04",
    "東京": "05",
    "中山": "06",
    "中京": "07",
    "京都": "08",
    "阪神": "09",
    "小倉": "10",
}


def venue_code(venue: str) -> str:
    """競馬場名 ("東京") またはコード ("05") から競馬場コードを返す."""
    if venue in VENUE_CODES:
        return VENUE_CODES[venue]
    if venue in VENUE_CODES.values():
        return venue
    raise ValueError(f"未対応の競馬場です: {venue} (指定できるのは {list(VENUE_CODES)})")


def parse_kaisai_calendar_html(html: str) -> dict[pd.Timestamp, list[str]]:
    """開催カレンダーの HTML から {開催日: [その日に開催する競馬場名, ...]} を作る.

    カレンダーには実際の JRA 開催日だけがリンク付きで載っている
    (土日でも開催がない日・平日開催・祝日開催も正しく反映される).
    """
    schedule = {}
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.select("td.RaceCellBox a[href*='kaisai_date=']"):
        m = re.search(r"kaisai_date=(\d{8})", a["href"])
        venues = [span.get_text(strip=True) for span in a.select("span.JyoName")]
        if m and venues:
            schedule[pd.Timestamp(m.group(1))] = venues
    return schedule


def fetch_kaisai_calendar(year: int, month: int) -> dict[pd.Timestamp, list[str]]:
    """開催カレンダーを取得し, その月の {開催日: [競馬場名, ...]} を返す."""
    response = polite_get(CALENDAR_URL, params={"year": year, "month": month})
    schedule = parse_kaisai_calendar_html(response.text)
    return {d: v for d, v in schedule.items() if d.year == year and d.month == month}


def fetch_race_ids_on(date: pd.Timestamp) -> list[str]:
    """開催日のレース一覧ページから, その日に行われる全レースの race_id を返す."""
    response = polite_get(RACE_LIST_URL, params={"kaisai_date": date.strftime("%Y%m%d")})
    return sorted(set(re.findall(r"race_id=(\d{12})", response.text)))


def find_race_ids(
    venue: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    race_numbers: range | list[int] = range(7, 13),
) -> list[tuple[pd.Timestamp, str]]:
    """競馬場・期間 (両端を含む)・レース番号に該当するレースを (開催日, race_id) のリストで返す.

    アクセス回数は 期間の月数 (カレンダー) + 指定競馬場の開催日数 (レース一覧).
    開催カレンダーで「指定した競馬場が開催している日」だけに絞ってからレース一覧を取得するため,
    非開催日や他場だけが開催している日にはアクセスしない.
    """
    code = venue_code(venue)
    venue_name = next(name for name, c in VENUE_CODES.items() if c == code)
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start > end:
        raise ValueError(f"開始日 {start.date()} が終了日 {end.date()} より後になっています")
    numbers = {int(n) for n in race_numbers}

    schedule = {}
    for month_start in pd.date_range(start.replace(day=1), end, freq="MS"):
        schedule.update(fetch_kaisai_calendar(month_start.year, month_start.month))
    kaisai_dates = sorted(d for d, venues in schedule.items() if start <= d <= end and venue_name in venues)
    print(f"{venue_name} の開催日 ({len(kaisai_dates)} 日): {[d.strftime('%m/%d') for d in kaisai_dates]}")

    races = []
    for date in kaisai_dates:
        for race_id in fetch_race_ids_on(date):
            if race_id[4:6] == code and int(race_id[10:12]) in numbers:
                races.append((date, race_id))
    return races
