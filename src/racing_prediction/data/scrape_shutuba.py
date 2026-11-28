"""netkeiba の出馬表ページから出走馬情報を取得する.

使用例:
    >>> from racing_prediction.data.scrape_shutuba import fetch_shutuba
    >>> df = fetch_shutuba("202605050507")  # 確定済みレースなら着順 (rank) も付く

注意:
    - リクエストヘッダには必ず User-Agent を付与する。
    - サーバー負荷軽減のため、アクセスのたびに 1〜3 秒のランダムな待機を入れる。
    - オッズ・人気は JavaScript で後から読み込まれるため、出馬表 HTML からは取得できない。
"""

from __future__ import annotations

import random
import re
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html"
RESULT_URL = "https://race.netkeiba.com/race/result.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    )
}

SLEEP_MIN_SEC = 1.0
SLEEP_MAX_SEC = 3.0


_HEADER_CHARSET = re.compile(r"charset=[\"']?([\w-]+)", re.IGNORECASE)
_META_CHARSET = re.compile(rb"<meta[^>]+charset=[\"']?([\w-]+)", re.IGNORECASE)


def detect_encoding(response: requests.Response) -> str:
    """ページの文字コード. ページが宣言している文字コード (HTTP ヘッダ → HTML の meta タグ) を優先し,
    宣言がないときだけ中身から推定する (apparent_encoding).

    netkeiba は UTF-8 のページと EUC-JP のページが混在する. 中身からの推定は, 日本語が少ないページなどで
    外れることがある (例: UTF-8 の競走成績ページを mac_latin2 と推定し, 見出しの「日付」が文字化けして読めなくなった).
    """
    m = _HEADER_CHARSET.search(response.headers.get("Content-Type", ""))
    if m:
        return m.group(1)
    m = _META_CHARSET.search(response.content[:10000])
    if m:
        return m.group(1).decode("ascii")
    return response.apparent_encoding or "utf-8"


def polite_get(url: str, params: dict | None = None, timeout: int = 10) -> requests.Response:
    """User-Agent 付きで GET し, アクセスのたびに 1〜3 秒ランダムに待機する."""
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        # netkeiba は UTF-8 と EUC-JP のページが混在するため, ページが宣言している文字コードで読む
        response.encoding = detect_encoding(response)
        return response
    finally:
        # 成功・失敗にかかわらず必ず待機する (サーバー負荷軽減)
        time.sleep(random.uniform(SLEEP_MIN_SEC, SLEEP_MAX_SEC))


def _text(tag: Tag | None) -> str:
    return tag.get_text(strip=True) if tag else ""


def _id_from_href(a: Tag | None, pattern: str) -> str | None:
    if a is None or not a.get("href"):
        return None
    m = re.search(pattern, a["href"])
    return m.group(1) if m else None


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def _parse_weight(text: str) -> tuple[int | None, int | None]:
    """ "480(+2)" -> (480, 2). "480(計不)" -> (480, None). 発表前の空欄や "計不" -> (None, None)."""
    m = re.match(r"(\d+)(?:\(([^)]*)\))?", text)
    if not m:
        return None, None
    diff = m.group(2)
    return int(m.group(1)), int(diff) if diff and re.fullmatch(r"[+-]?\d+", diff) else None


def _parse_horse_row(tr: Tag) -> dict:
    waku = _text(tr.find("td", class_=re.compile(r"^Waku")))
    umaban = _text(tr.find("td", class_=re.compile(r"^Umaban")))

    horse_a = tr.select_one("td.HorseInfo span.HorseName a")
    horse_name = (horse_a.get("title") or _text(horse_a)) if horse_a else ""

    sex_age = _text(tr.find("td", class_="Barei"))
    m = re.match(r"(\D+)(\d+)", sex_age)
    sex, age = (m.group(1), int(m.group(2))) if m else (None, None)

    # 斤量は性齢セルの直後の td
    barei_td = tr.find("td", class_="Barei")
    weight_carried = None
    if barei_td is not None:
        nxt = barei_td.find_next_sibling("td")
        weight_carried = _to_float(_text(nxt))

    jockey_a = tr.select_one("td.Jockey a")
    jockey_name = (jockey_a.get("title") or _text(jockey_a)).strip() if jockey_a else ""

    trainer_td = tr.find("td", class_="Trainer")
    trainer_a = trainer_td.find("a") if trainer_td else None
    trainer_name = (trainer_a.get("title") or _text(trainer_a)) if trainer_a else ""
    trainer_area = _text(trainer_td.find("span")) if trainer_td else ""  # 美浦 / 栗東 など

    # 馬体重: 発表後は "480(+2)" の形式, 発表前は空
    weight_text = _text(tr.find("td", class_="Weight"))
    horse_weight, weight_diff = _parse_weight(weight_text)

    # 取消・除外の馬は tr に Cancel クラスが付き, 馬体重欄に "取消" / "除外" と表示される
    is_canceled = "Cancel" in tr.get("class", []) or weight_text in ("取消", "除外")

    return {
        "waku": int(waku) if waku.isdigit() else None,
        "umaban": int(umaban) if umaban.isdigit() else None,
        "horse_id": _id_from_href(horse_a, r"/horse/(\w+)"),
        "horse_name": horse_name,
        "sex": sex,
        "age": age,
        "weight_carried": weight_carried,
        "jockey_id": _id_from_href(jockey_a, r"/jockey/(?:result/recent/)?(\w+)"),
        "jockey_name": jockey_name,
        "trainer_id": _id_from_href(trainer_a, r"/trainer/(?:result/recent/)?(\w+)"),
        "trainer_name": trainer_name,
        "trainer_area": trainer_area,
        "horse_weight": horse_weight,
        "weight_diff": weight_diff,
        "is_canceled": is_canceled,
    }


def parse_shutuba_html(html: str) -> pd.DataFrame:
    """出馬表ページの HTML を出走馬ごとの DataFrame に変換する."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.Shutuba_Table")
    if table is None:
        raise ValueError("出馬表テーブル (table.Shutuba_Table) が見つかりませんでした")

    rows = [_parse_horse_row(tr) for tr in table.select("tr.HorseList")]
    df = pd.DataFrame(rows)
    for i, (key, value) in enumerate({"race_date": _parse_race_date(soup), **_parse_race_info(soup)}.items()):
        df.insert(i, key, value)
    return df.sort_values("umaban", ignore_index=True)


_SURFACE_MAP = {"芝": "turf", "ダ": "dirt", "障": "steeple"}

# レースのクラス (数値が大きいほど上のクラス). 過去戦績のレース名と出馬表のクラス欄の両方に使う
CLASS_LEVELS = {"maiden": 0, "1win": 1, "2win": 2, "3win": 3, "open": 4, "g3": 5, "g2": 6, "g1": 7}
_CLASS_PATTERNS = [  # 上から順に調べる (G1 を OP より先に判定するため)
    (r"\((?:G|Jpn)(?:1|I)\)", "g1"),
    (r"\((?:G|Jpn)(?:2|II)\)", "g2"),
    (r"\((?:G|Jpn)(?:3|III)\)", "g3"),
    (r"\((?:L|OP)\)|オープン", "open"),
    (r"3勝クラス|1600万", "3win"),
    (r"2勝クラス|1000万", "2win"),
    (r"1勝クラス|500万", "1win"),
    (r"新馬|未勝利", "maiden"),
]
# 出馬表のレース名に付くグレードのアイコン (Icon_GradeType1 = G1 など)
_GRADE_ICON_CLASSES = {"Icon_GradeType1": "g1", "Icon_GradeType2": "g2", "Icon_GradeType3": "g3", "Icon_GradeType15": "open"}


def race_class_level(text: str | None) -> int | None:
    """レース名・クラス欄の文字列からクラスを数値にする (CLASS_LEVELS). 判定できなければ None (地方・海外など).

    >>> race_class_level("ドンカスターC(3勝クラス)")
    3
    >>> race_class_level("天皇賞(秋)(G1)")
    7
    """
    if not text:
        return None
    text = unicodedata.normalize("NFKC", text)  # 全角数字 (３勝クラス) を半角に
    for pattern, name in _CLASS_PATTERNS:
        if re.search(pattern, text):
            return CLASS_LEVELS[name]
    return None
# 馬場状態 (過去戦績の track_condition と同じ表記に揃える)
_TRACK_CONDITION_MAP = {"良": "good", "稍": "yielding", "重": "soft", "不": "heavy"}


def _parse_race_info(soup: BeautifulSoup) -> dict:
    """レース名の下の情報欄から 競馬場・芝/ダート・距離・馬場状態・クラス・本賞金 を取り出す.

    RaceData01: "13:10発走 / ダ2100m 左 / 晴 / 良 ..."
        馬場状態は 芝なら span.Item03, ダートなら span.Item04 (当日の発表前は存在しない)
    RaceData02: "5回 東京 5日目 サラ系３歳以上 ３勝クラス ... 本賞金:1870,750,... 万円" (2 番目の span が競馬場)
    レース名 (h1.RaceName) のアイコン: グレード (G1〜G3, L)
    """
    data01 = _text(soup.select_one("div.RaceData01"))
    m = re.search(r"(芝|ダ|障)(\d+)m", data01)
    spans = soup.select("div.RaceData02 span")
    surface = _SURFACE_MAP[m.group(1)] if m else None
    item = {"turf": "Item03", "dirt": "Item04"}.get(surface)
    condition_span = soup.select_one(f"div.RaceData01 span.{item}") if item else None
    if condition_span is None:  # 障害レースなどはどちらか載っている方を使う
        condition_span = soup.select_one("div.RaceData01 span.Item03, div.RaceData01 span.Item04")
    condition = _text(condition_span)

    # クラス: グレードのアイコン (G1〜G3, L) があればそれ, なければクラス欄 ("３勝クラス", "オープン" など)
    race_class = None
    for icon in soup.select("h1.RaceName span"):
        name = next((v for c, v in _GRADE_ICON_CLASSES.items() if c in icon.get("class", [])), None)
        if name and (race_class is None or CLASS_LEVELS[name] > race_class):
            race_class = CLASS_LEVELS[name]
    data02 = " ".join(_text(s) for s in spans)
    if race_class is None:
        race_class = race_class_level(data02)
    # 1 着の本賞金 (万円). "本賞金:1870,750,470,280,187万円" -> 1870
    prize = re.search(r"本賞金:?([\d.]+)", unicodedata.normalize("NFKC", data02))
    return {
        "race_venue": _text(spans[1]) if len(spans) > 1 else None,
        "race_surface": surface,
        "race_distance": int(m.group(2)) if m else None,
        # 当日の馬場状態: "good"(良) / "yielding"(稍重) / "soft"(重) / "heavy"(不良) / None(発表前)
        "race_track_condition": _TRACK_CONDITION_MAP.get(condition[:1]),
        "race_class": race_class,  # CLASS_LEVELS の数値 (0: 新馬・未勝利 〜 7: G1). 判定できなければ None
        "race_prize": float(prize.group(1)) if prize else None,  # 1 着の本賞金 (万円)
    }


def _parse_race_date(soup: BeautifulSoup) -> pd.Timestamp | None:
    """ページタイトル "... | 2026年11月22日 東京7R ..." から開催日を取り出す."""
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", _text(soup.title))
    return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def parse_result_html(html: str) -> pd.DataFrame:
    """レース結果ページの HTML から 馬番・着順・馬体重・単勝オッズ・人気 を取り出す.

    単勝オッズ (win_odds) と人気 (popularity) は確定 (最終) オッズ.
    結果が未確定 (結果テーブルが無い) の場合は空の DataFrame を返す.
    """
    columns = ["umaban", "rank", "rank_text", "horse_weight", "weight_diff", "win_odds", "popularity"]
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("div.ResultTableWrap table")
    if table is None:
        return pd.DataFrame(columns=columns)

    rows = []
    for tr in table.select("tr.HorseList"):
        # 着順は数字, 取消・除外・中止・失格などは文字列で入る
        rank_text = _text(tr.select_one("td.Result_Num"))
        tds = tr.find_all("td", recursive=False)
        umaban = _text(tds[2]) if len(tds) > 2 else ""  # 着順, 枠, 馬番 の順
        horse_weight, weight_diff = _parse_weight(_text(tr.find("td", class_="Weight")))
        popularity = _text(tr.select_one("span.OddsPeople"))
        rows.append(
            {
                "umaban": int(umaban) if umaban.isdigit() else None,
                "rank": int(rank_text) if rank_text.isdigit() else None,
                "rank_text": rank_text,
                "horse_weight": horse_weight,
                "weight_diff": weight_diff,
                # 単勝オッズのセルは td.Odds.Txt_R (1 番人気だけ span に Odds_Ninki クラスが付く)
                "win_odds": _to_float(_text(tr.select_one("td.Odds.Txt_R"))),
                "popularity": int(popularity) if popularity.isdigit() else None,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def fetch_result(race_id: str) -> pd.DataFrame:
    """race_id を指定してレース結果ページを取得し, 馬番・着順・馬体重の DataFrame を返す."""
    response = polite_get(RESULT_URL, params={"race_id": race_id})
    return parse_result_html(response.text)


def fetch_shutuba(race_id: str, with_result: bool = True) -> pd.DataFrame:
    """race_id を指定して出馬表を取得し, 出走馬情報の DataFrame を返す.

    with_result=True の場合は結果ページにもアクセスし, 着順 (rank, rank_text) を付与する.
    馬体重が出馬表で空のときは結果ページの値で補完する. 未確定レースでは rank は欠損になる.
    """
    response = polite_get(SHUTUBA_URL, params={"race_id": race_id})
    df = parse_shutuba_html(response.text)
    df.insert(0, "race_id", race_id)

    if with_result:
        result = fetch_result(race_id)
        df = df.merge(result, on="umaban", how="left", suffixes=("", "_result"))
        for col in ("horse_weight", "weight_diff"):
            df[col] = df[col].combine_first(df.pop(f"{col}_result"))

    for col in ("horse_weight", "weight_diff", "rank", "popularity"):
        if col in df:
            df[col] = df[col].astype("Int64")
    return df


if __name__ == "__main__":
    print(fetch_shutuba("202605050507").to_string())
