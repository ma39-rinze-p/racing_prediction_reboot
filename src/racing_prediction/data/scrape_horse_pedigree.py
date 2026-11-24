"""netkeiba の競走馬データベースから各馬の血統 (2 代まで) を取得する.

使用例:
    >>> from racing_prediction.data.scrape_horse_pedigree import fetch_horse_pedigree
    >>> fetch_horse_pedigree("2022105585")
    {'sire': 'ナダル', 'sire_id': '000a016e99', 'sire_line': 'Roberto系', 'dam': ..., ...}

アクセスは scrape_shutuba.polite_get を経由するため,
User-Agent の付与とアクセスごとの 1〜3 秒ランダム待機が必ず行われる.

血統は変わらないため, 取得した血統は data/raw/horse_pedigree/{horse_id}.json に保存し,
次からは (別の実行・別のレースでも) アクセスせずに使う (racing_prediction.data.cache).
"""

from __future__ import annotations

import re
import warnings

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from racing_prediction.data.cache import cache_dir, read_json, write_json
from racing_prediction.data.scrape_shutuba import polite_get

HORSE_PED_URL = "https://db.netkeiba.com/horse/ped/{horse_id}/"

# 5 代血統表のセルは rowspan で世代がわかる (1 代目=16, 2 代目=8). 文書順に並んでいる
ANCESTORS_BY_ROWSPAN = {
    "16": ["sire", "dam"],  # 父, 母
    "8": ["sire_sire", "sire_dam", "dam_sire", "dam_dam"],  # 父父, 父母, 母父, 母母
}

PEDIGREE_COLUMNS = ["sire", "sire_id", "sire_line"] + [
    f"{name}{suffix}"
    for names in ANCESTORS_BY_ROWSPAN.values()
    for name in names
    if name != "sire"
    for suffix in ("", "_id")
]


def _parse_ancestor(td: Tag) -> tuple[str | None, str | None, str | None]:
    """血統表の 1 セルから (馬名, horse_id, 系統) を取り出す."""
    a = td.find("a")
    if a is None:
        return None, None, None
    # "Blame (米)" のような生産国表記は除く
    name = re.sub(r"\s*\(.+?\)$", "", a.get_text(strip=True).replace("\xa0", " ")).strip()
    m = re.search(r"/horse/(\w+)/", a.get("href", ""))
    horse_id = m.group(1) if m else None

    # 種牡馬のセルには "Roberto系" のような系統が最後の行に書かれている
    info = td.select_one("div.parent > div")
    lines = info.get_text("|", strip=True).split("|") if info else []
    line = next((s for s in reversed(lines) if s.endswith("系")), None)
    return name, horse_id, line


def parse_horse_pedigree_html(html: str) -> dict:
    """血統ページの HTML から 父・母・父父・父母・母父・母母 の馬名と ID, 父の系統を取り出す."""
    pedigree = dict.fromkeys(PEDIGREE_COLUMNS)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.blood_table")
    if table is None:
        return pedigree

    for rowspan, names in ANCESTORS_BY_ROWSPAN.items():
        for name, td in zip(names, table.find_all("td", rowspan=rowspan)):
            horse_name, horse_id, line = _parse_ancestor(td)
            pedigree[name] = horse_name
            pedigree[f"{name}_id"] = horse_id
            if name == "sire":
                pedigree["sire_line"] = line
    return pedigree


_memory: dict[str, dict] = {}  # 実行中に取得・読み込みした血統


def fetch_horse_pedigree(horse_id: str) -> dict:
    """horse_id を指定して血統ページを取得し, 2 代までの血統を dict で返す.

    血統は変わらないため, 実行中に取得済み, またはディスクに保存済みならアクセスしない.
    血統表が読めなかった (父が取れなかった) 場合は保存せず, 次回また取得する.
    """
    if horse_id in _memory:
        return dict(_memory[horse_id])
    path = cache_dir("horse_pedigree") / f"{horse_id}.json"
    saved = read_json(path)
    if isinstance(saved, dict) and all(c in saved for c in PEDIGREE_COLUMNS):
        _memory[horse_id] = saved
        return dict(saved)

    response = polite_get(HORSE_PED_URL.format(horse_id=horse_id))
    pedigree = parse_horse_pedigree_html(response.text)
    _memory[horse_id] = pedigree
    if pedigree.get("sire"):
        write_json(path, pedigree)
    return dict(pedigree)


def add_pedigree(df_horses: pd.DataFrame) -> pd.DataFrame:
    """出馬表の各馬について血統ページにアクセスし, 血統の列を追加する.

    1 頭ごとに 1 回アクセスする (アクセスごとに 1〜3 秒待機するため, 16 頭で 30 秒前後かかる).
    取得に失敗した馬は警告を出して欠損にする.
    """
    rows = []
    for horse_id in df_horses["horse_id"]:
        try:
            rows.append(fetch_horse_pedigree(horse_id))
        except requests.RequestException as e:
            warnings.warn(f"血統の取得に失敗しました (horse_id={horse_id}): {e}")
            rows.append({})

    pedigree_df = pd.DataFrame(rows, index=df_horses.index, columns=PEDIGREE_COLUMNS)
    return pd.concat([df_horses, pedigree_df], axis=1)
