"""netkeiba から取得したデータのディスクキャッシュ (同じページに何度もアクセスしないため).

保存先はリポジトリの data/raw/ (環境変数 RACING_CACHE_DIR で変更可). 実行したフォルダ (notebooks/ など) に
関係なく同じ場所を使う. git 管理外.

    data/raw/horse_results/{馬のID}.csv     各馬の戦績 (取得した日より後のレースに使うときは取り直す)
    data/raw/horse_pedigree/{馬のID}.json   各馬の血統 (変わらないので取り直さない)
    data/raw/race_calendar/{年月}.json      開催カレンダー (終わった月だけ保存する)
    data/raw/race_list/{開催日}.json        開催日のレース ID の一覧 (終わった開催日だけ保存する)

キャッシュの読み書きに失敗しても (Kaggle の読み取り専用のフォルダなど), 取得自体は普通に行う.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

# src/racing_prediction/data/cache.py -> リポジトリのルート
_REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = Path(os.environ.get("RACING_CACHE_DIR", _REPO_ROOT / "data" / "raw"))


def cache_dir(name: str) -> Path:
    return CACHE_ROOT / name


def today_jst() -> pd.Timestamp:
    """今日の日付 (日本時間, 時刻なし)."""
    return pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize()


def read_json(path: Path):
    """保存済みの JSON を読む. なければ・読めなければ None."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, data) -> None:
    """JSON を保存する. 書けなくてもエラーにしない (途中で止まっても壊れたファイルを残さないよう, 一時ファイル経由)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass
