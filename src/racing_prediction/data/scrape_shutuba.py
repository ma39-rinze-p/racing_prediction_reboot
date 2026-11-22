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


def polite_get(url: str, params: dict | None = None, timeout: int = 10) -> requests.Response:
    """User-Agent 付きで GET し, アクセスのたびに 1〜3 秒ランダムに待機する."""
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        # netkeiba は EUC-JP のページもあるため, 実際の文字コードを推定させる
        response.encoding = response.apparent_encoding
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


def _parse_race_info(soup: BeautifulSoup) -> dict:
    """レース名の下の情報欄から 競馬場・芝/ダート・距離 を取り出す.

    RaceData01: "13:10発走 / ダ2100m 左 / 晴 / 良 ..."
    RaceData02: "5回 東京 5日目 ..." (2 番目の span が競馬場)
    """
    data01 = _text(soup.select_one("div.RaceData01"))
    m = re.search(r"(芝|ダ|障)(\d+)m", data01)
    spans = soup.select("div.RaceData02 span")
    return {
        "race_venue": _text(spans[1]) if len(spans) > 1 else None,
        "race_surface": _SURFACE_MAP[m.group(1)] if m else None,
        "race_distance": int(m.group(2)) if m else None,
    }


def _parse_race_date(soup: BeautifulSoup) -> pd.Timestamp | None:
    """ページタイトル "... | 2026年11月22日 東京7R ..." から開催日を取り出す."""
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", _text(soup.title))
    return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def parse_result_html(html: str) -> pd.DataFrame:
    """レース結果ページの HTML から 馬番・着順・馬体重 を取り出す.

    結果が未確定 (結果テーブルが無い) の場合は空の DataFrame を返す.
    """
    columns = ["umaban", "rank", "rank_text", "horse_weight", "weight_diff"]
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
        rows.append(
            {
                "umaban": int(umaban) if umaban.isdigit() else None,
                "rank": int(rank_text) if rank_text.isdigit() else None,
                "rank_text": rank_text,
                "horse_weight": horse_weight,
                "weight_diff": weight_diff,
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

    for col in ("horse_weight", "weight_diff", "rank"):
        if col in df:
            df[col] = df[col].astype("Int64")
    return df


if __name__ == "__main__":
    print(fetch_shutuba("202605050507").to_string())
