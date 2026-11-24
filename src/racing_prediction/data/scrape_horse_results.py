"""netkeiba の競走馬データベースから各馬の過去戦績を取得する.

使用例:
    >>> from racing_prediction.data.scrape_horse_results import fetch_horse_results
    >>> df = fetch_horse_results("2023103731")

アクセスは scrape_shutuba.polite_get を経由するため,
User-Agent の付与とアクセスごとの 1〜3 秒ランダム待機が必ず行われる.

取得した戦績はディスク (data/raw/horse_results/{horse_id}.csv. racing_prediction.data.cache)
にも保存し, 次からは「取得した日 >= 必要な日 (対象レースの開催日)」なら取り直さずに使う.
戦績ページには取得時点までの全レースが載っているため, 対象レースより前のレースはすべて含まれている.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from racing_prediction.data.cache import cache_dir, today_jst
from racing_prediction.data.scrape_shutuba import polite_get, race_class_level

HORSE_RESULT_URL = "https://db.netkeiba.com/horse/result/{horse_id}/"

# JRA の 10 競馬場 (開催欄 "4東京1" などから抽出する)
JRA_VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]

SURFACE_MAP = {"芝": "turf", "ダ": "dirt", "障": "steeple"}
TRACK_CONDITION_MAP = {"良": "good", "稍": "yielding", "重": "soft", "不": "heavy"}

RESULT_COLUMNS = [
    "date", "race_id", "venue", "is_overseas", "surface", "distance", "track_condition", "rank", "rank_text",
    "race_name", "race_class", "n_field", "odds", "popularity", "jockey", "weight_carried",
    "time_sec", "margin", "first_corner", "last_corner", "agari", "prize",
]

CACHE_DIR = cache_dir("horse_results")


def _is_overseas(race_id: str | None) -> bool:
    """race_id の 5〜6 文字目 (競馬場コード) に英字が入っていれば海外レース.

    JRA は 01〜10, 地方は 30 番台以降の数字, 海外は "J0"(メイダン) のように英字を含む.
    """
    return race_id is not None and len(race_id) >= 6 and not race_id[4:6].isdigit()


def _num(text: str | None) -> float | None:
    """ "1,600.3" -> 1600.3. 数字でなければ None."""
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def _time_sec(text: str | None) -> float | None:
    """ "1:47.7" -> 107.7, "59.8" -> 59.8. 形式が違えば None."""
    m = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", text or "")
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2))


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
        race_name: レース名, race_class: クラス (scrape_shutuba.CLASS_LEVELS の数値. 地方・海外などは欠損)
        n_field: 出走頭数, odds: 単勝オッズ, popularity: 人気, jockey: 騎手, weight_carried: 斤量
        time_sec: 走破タイム (秒), margin: 着差 (秒. 1 着は 2 着との差をマイナスで表す)
        first_corner / last_corner: 最初 / 最後のコーナーの通過順位 ("9-9-7-7" -> 9 / 7)
        agari: 上がり 3F (秒), prize: 獲得賞金 (万円. 賞金がなければ 0)
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.db_h_race_results")
    if table is None:
        # 出走歴がない馬はテーブル自体が存在しない
        return pd.DataFrame(columns=RESULT_COLUMNS)

    headers = [th.get_text(strip=True) for th in table.select("thead th")]
    col = {name: headers.index(name) for name in ("日付", "開催", "着順", "距離", "馬場")}
    # 追加の列 (ページの構成が変わって列が無くなっても, その値が欠損になるだけにする)
    optional = ("レース名", "頭数", "オッズ", "人気", "騎手", "斤量", "タイム", "着差", "通過", "上り", "賞金")
    opt = {name: headers.index(name) for name in optional if name in headers}

    rows = []
    for tr in table.select("tbody tr"):
        td_tags = tr.find_all("td", recursive=False)
        tds = [td.get_text(strip=True) for td in td_tags]
        if len(tds) < len(headers):
            continue

        def get(name: str) -> str:
            return tds[opt[name]] if name in opt else ""

        race_a = td_tags[opt["レース名"]].find("a") if "レース名" in opt else None
        rid = re.search(r"/race/(\w+)/", race_a.get("href", "")) if race_a else None
        race_id = rid.group(1) if rid else None

        kaisai = tds[col["開催"]]
        venue = next((v for v in JRA_VENUES if v in kaisai), re.sub(r"\d", "", kaisai) or None)

        dm = re.match(r"(芝|ダ|障)(\d+)", tds[col["距離"]])
        surface = SURFACE_MAP[dm.group(1)] if dm else None
        distance = int(dm.group(2)) if dm else None

        rank_text = tds[col["着順"]]
        rm = re.match(r"\d+", rank_text)
        corners = [int(c) for c in re.findall(r"\d+", get("通過"))]
        race_name = get("レース名") or None

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
                "race_name": race_name,
                "race_class": race_class_level(race_name),
                "n_field": _num(get("頭数")),
                "odds": _num(get("オッズ")),
                "popularity": _num(get("人気")),
                "jockey": get("騎手") or None,
                "weight_carried": _num(get("斤量")),
                "time_sec": _time_sec(get("タイム")),
                "margin": _num(get("着差")),
                "first_corner": corners[0] if corners else None,
                "last_corner": corners[-1] if corners else None,
                "agari": _num(get("上り")),
                "prize": _num(get("賞金")) or 0.0,
            }
        )

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df["rank"] = df["rank"].astype("Int64")
    df["distance"] = df["distance"].astype("Int64")
    return df


# 実行中に取得・読み込みした戦績 {horse_id: (取得した日, DataFrame)}
_memory: dict[str, tuple[pd.Timestamp, pd.DataFrame]] = {}


def _cache_path(horse_id: str) -> Path:
    return CACHE_DIR / f"{horse_id}.csv"


def _load_disk_cache(horse_id: str) -> tuple[pd.Timestamp, pd.DataFrame] | None:
    """保存済みの戦績 (取得した日, DataFrame). 無い・読めない・古い形式なら None."""
    path = _cache_path(horse_id)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str, "fetched_at": str})
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if "fetched_at" not in df.columns or any(c not in df.columns for c in RESULT_COLUMNS):
        return None  # 列を追加する前の古いキャッシュは使わない
    if df.empty:  # 出走歴のない馬はファイル名と同じ場所に取得日を書けないので, 更新日時を使う
        fetched = pd.Timestamp(path.stat().st_mtime, unit="s").normalize()
    else:
        fetched = pd.Timestamp(df["fetched_at"].iloc[0]).normalize()
    df = df[RESULT_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rank"] = df["rank"].astype("Int64")
    df["distance"] = df["distance"].astype("Int64")
    return fetched, df


def _save_disk_cache(horse_id: str, fetched: pd.Timestamp, df: pd.DataFrame) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        out = df.assign(fetched_at=fetched.strftime("%Y-%m-%d"))
        out.to_csv(_cache_path(horse_id), index=False, encoding="utf-8-sig")
    except OSError:
        pass  # キャッシュが書けなくても戦績の取得自体は成功させる


def fetch_horse_results(horse_id: str, needed_until: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """horse_id を指定して競走成績ページを取得し, 過去戦績の DataFrame を返す.

    Args:
        needed_until: この日 (対象レースの開催日) より前のレースが載っていれば十分, という日付.
            実行中に取得済み, またはディスクに保存済みの戦績が, この日以降に取得したものなら取り直さない.
            None なら今日 (= 今日取得したものだけを使う).

    戦績ページには取得時点までの全レースが載っており, 集計時に対象レース日より前に
    絞り込むため, 使い回しても結果は変わらない.
    """
    needed = today_jst() if needed_until is None or pd.isna(needed_until) else pd.Timestamp(needed_until).normalize()
    for load in (lambda: _memory.get(horse_id), lambda: _load_disk_cache(horse_id)):
        hit = load()
        if hit is not None and hit[0] >= needed:
            _memory[horse_id] = hit
            return hit[1].copy()

    response = polite_get(HORSE_RESULT_URL.format(horse_id=horse_id))
    df = parse_horse_results_html(response.text)
    fetched = today_jst()
    _memory[horse_id] = (fetched, df)
    _save_disk_cache(horse_id, fetched, df)
    return df.copy()
