"""指定した競馬場・期間・レース番号に該当する race_id を netkeiba から集める.

race_id の形式: YYYY(年) + VV(競馬場コード) + KK(回) + DD(日目) + RR(レース番号)
    例: 202605050507 = 2026年 東京(05) 5回 5日目 7R

使用例:
    >>> from racing_prediction.data.race_list import find_race_ids
    >>> find_race_ids("東京", "2026-11-01", "2026-11-22", race_numbers=range(7, 13))
    >>> find_race_ids(["東京", "京都"], "2026-11-01", "2026-11-22")   # 複数の競馬場

アクセスは scrape_shutuba.polite_get を経由するため,
User-Agent の付与とアクセスごとの 1〜3 秒ランダム待機が必ず行われる.

開催カレンダー (月ごと) とレース ID の一覧 (開催日ごと) は, 終わった月・終わった開催日の分だけ
data/raw/race_calendar/, data/raw/race_list/ に保存し, 次からはアクセスせずに使う (racing_prediction.data.cache).
今月・今日以降は, 開催の変更 (中止・代替開催など) があり得るため毎回取得する.
"""

from __future__ import annotations

import re

import pandas as pd
from bs4 import BeautifulSoup

from racing_prediction.data.cache import cache_dir, read_json, today_jst, write_json
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
    """開催カレンダーを取得し, その月の {開催日: [競馬場名, ...]} を返す.

    終わった月 (先月以前) は保存済みならアクセスせずに使い, 取得したら保存する.
    """
    path = cache_dir("race_calendar") / f"{year:04d}{month:02d}.json"
    month_end = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    finished = month_end < today_jst()
    if finished:
        saved = read_json(path)
        if isinstance(saved, dict):
            return {pd.Timestamp(d): list(v) for d, v in saved.items()}

    response = polite_get(CALENDAR_URL, params={"year": year, "month": month})
    schedule = parse_kaisai_calendar_html(response.text)
    schedule = {d: v for d, v in schedule.items() if d.year == year and d.month == month}
    if finished and schedule:
        write_json(path, {d.strftime("%Y-%m-%d"): v for d, v in schedule.items()})
    return schedule


def fetch_race_ids_on(date: pd.Timestamp) -> list[str]:
    """開催日のレース一覧ページから, その日に行われる全レースの race_id を返す.

    終わった開催日 (昨日以前) は保存済みならアクセスせずに使い, 取得したら保存する.
    """
    date = pd.Timestamp(date).normalize()
    path = cache_dir("race_list") / f"{date:%Y%m%d}.json"
    finished = date < today_jst()
    if finished:
        saved = read_json(path)
        if isinstance(saved, list) and saved:
            return [str(r) for r in saved]

    response = polite_get(RACE_LIST_URL, params={"kaisai_date": date.strftime("%Y%m%d")})
    race_ids = sorted(set(re.findall(r"race_id=(\d{12})", response.text)))
    if finished and race_ids:
        write_json(path, race_ids)
    return race_ids


def parse_race_schedule_html(html: str, date: pd.Timestamp) -> pd.DataFrame:
    """開催日のレース一覧ページから, その日の全レースの時刻表を作る.

    Returns:
        race_id, venue (競馬場名), race_number, race_name, course (例: "ダ1400m"),
        start_time (発走時刻, 日本時間) の DataFrame
    """
    code_to_venue = {code: name for name, code in VENUE_CODES.items()}
    rows = []
    soup = BeautifulSoup(html, "html.parser")
    for item in soup.select("li.RaceList_DataItem"):
        a = item.find("a", href=re.compile(r"race_id=\d{12}"))
        time_span = item.select_one("span.RaceList_Itemtime")
        if a is None or time_span is None:
            continue
        race_id = re.search(r"race_id=(\d{12})", a["href"]).group(1)
        tm = re.search(r"(\d{1,2}):(\d{2})", time_span.get_text())
        if tm is None:
            continue
        title = item.select_one("span.ItemTitle")
        course = item.select_one("span.RaceList_ItemLong")
        rows.append(
            {
                "race_id": race_id,
                "venue": code_to_venue.get(race_id[4:6], race_id[4:6]),
                "race_number": int(race_id[10:12]),
                "race_name": title.get_text(strip=True) if title else "",
                "course": course.get_text(strip=True) if course else "",
                "start_time": date.normalize() + pd.Timedelta(hours=int(tm.group(1)), minutes=int(tm.group(2))),
            }
        )
    columns = ["race_id", "venue", "race_number", "race_name", "course", "start_time"]
    df = pd.DataFrame(rows, columns=columns).drop_duplicates("race_id")
    return df.sort_values(["start_time", "race_id"], ignore_index=True)


def fetch_race_schedule(date: str | pd.Timestamp) -> pd.DataFrame:
    """開催日の全レースの時刻表 (race_id, 競馬場, レース番号, レース名, 発走時刻) を返す. 非開催日は空."""
    date = pd.Timestamp(date)
    response = polite_get(RACE_LIST_URL, params={"kaisai_date": date.strftime("%Y%m%d")})
    return parse_race_schedule_html(response.text, date)


def venue_name_of(venue: str) -> str:
    """競馬場名 ("東京") またはコード ("05") から競馬場名を返す."""
    code = venue_code(venue)
    return next(name for name, c in VENUE_CODES.items() if c == code)


def venue_name_from_race_id(race_id: str) -> str:
    """race_id の 5〜6 文字目 (競馬場コード) から競馬場名を返す. 例: "202605050507" -> "東京"."""
    return venue_name_of(race_id[4:6])


def _as_venue_list(venues: str | list[str] | tuple[str, ...]) -> list[str]:
    """"東京" / "東京,京都" / ["東京", "京都"] / コード ("05") -> 競馬場名のリスト (重複なし, 指定順)."""
    if isinstance(venues, str):
        venues = [v for v in venues.split(",")]
    names = []
    for v in venues:
        v = str(v).strip()
        if v and venue_name_of(v) not in names:
            names.append(venue_name_of(v))
    if not names:
        raise ValueError("競馬場を 1 つ以上指定してください")
    return names


def find_race_ids(
    venues: str | list[str] | tuple[str, ...],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    race_numbers: range | list[int] = range(7, 13),
) -> list[tuple[pd.Timestamp, str]]:
    """競馬場 (複数可)・期間 (両端を含む)・レース番号に該当するレースを (開催日, race_id) のリストで返す.

    Args:
        venues: 競馬場名 ("東京") / コード ("05") / そのリスト (["東京", "京都"]) / カンマ区切り ("東京,京都")

    アクセス回数は 期間の月数 (カレンダー) + 指定競馬場のどれかが開催している日数 (レース一覧).
    開催カレンダーで「指定した競馬場のどれかが開催している日」だけに絞ってからレース一覧を取得するため,
    非開催日や他場だけが開催している日にはアクセスしない. レース一覧は 1 日 1 ページに全場が載っているので,
    同じ日に開催する競馬場を複数指定してもアクセスは増えない.
    """
    names = _as_venue_list(venues)
    codes = {VENUE_CODES[n] for n in names}
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start > end:
        raise ValueError(f"開始日 {start.date()} が終了日 {end.date()} より後になっています")
    numbers = {int(n) for n in race_numbers}

    schedule = {}
    for month_start in pd.date_range(start.replace(day=1), end, freq="MS"):
        schedule.update(fetch_kaisai_calendar(month_start.year, month_start.month))
    in_period = {d: v for d, v in schedule.items() if start <= d <= end}
    for name in names:
        dates = sorted(d for d, held in in_period.items() if name in held)
        print(f"{name} の開催日 ({len(dates)} 日): {[d.strftime('%m/%d') for d in dates]}")
    kaisai_dates = sorted(d for d, held in in_period.items() if any(n in held for n in names))

    races = []
    for date in kaisai_dates:
        for race_id in fetch_race_ids_on(date):
            if race_id[4:6] in codes and int(race_id[10:12]) in numbers:
                races.append((date, race_id))
    return sorted(races)
