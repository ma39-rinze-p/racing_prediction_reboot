"""make_dataset

netkeiba から取得した出馬表・結果・過去戦績・血統をもとに,
1 行 = 1 頭 (レース × 馬) の学習用テーブルを作り CSV に保存する.

使用例:
    >>> from racing_prediction.data.make_dataset import make_dataset, collect_races
    >>> df = make_dataset(["202605050507"], "data/processed/dataset.csv")
    >>> # 競馬場・期間・レース番号を指定して連続取得 (レースごとに CSV を保存)
    >>> df = collect_races("東京", "2026-11-01", "2026-11-22", race_numbers=range(7, 13))

テーブルの列:
    識別用   : race_id, race_date, umaban
    特徴量   : レースの場所・距離区分・芝/ダート, 馬ごとの情報, 条件別平均着順, 血統 (ID なし)
    目的変数 : is_top3 (3 着以内なら 1, それ以外は 0. 結果未確定のレースは欠損)
    参考     : rank (着順)
"""

from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd

from racing_prediction.data.race_list import VENUE_CODES, find_race_ids
from racing_prediction.data.scrape_horse_pedigree import add_pedigree
from racing_prediction.data.scrape_shutuba import fetch_shutuba
from racing_prediction.features.build_features import (
    VENUES,
    add_avg_rank_features,
    avg_rank_feature_names,
    distance_category,
)

ID_COLUMNS = ["race_id", "race_date", "umaban"]

RACE_FEATURE_COLUMNS = ["race_venue", "race_distance_category", "race_surface"]

HORSE_FEATURE_COLUMNS = [
    "horse_name",
    "sex",
    "age",
    "weight_carried",
    "jockey_name",
    "trainer_name",
    "horse_weight",
    "weight_diff",
]

PEDIGREE_FEATURE_COLUMNS = ["sire", "sire_line", "dam", "sire_sire", "sire_dam", "dam_sire", "dam_dam"]

TARGET_COLUMN = "is_top3"

# 出走しなかった馬 (着順欄がこれらの場合はテーブルから除く)
NOT_STARTED = ("取消", "除外")


def feature_columns() -> list[str]:
    """モデルに入れる特徴量の列名一覧."""
    return RACE_FEATURE_COLUMNS + HORSE_FEATURE_COLUMNS + avg_rank_feature_names() + PEDIGREE_FEATURE_COLUMNS


def build_race_table(race_id: str) -> pd.DataFrame:
    """1 レース分のテーブル (1 行 = 1 頭) を作る.

    アクセス回数は 出馬表 1 + 結果 1 + 頭数 × 2 (戦績・血統).
    すべて polite_get 経由なので, User-Agent 付与とアクセスごとの 1〜3 秒待機が入る.
    """
    df = fetch_shutuba(race_id, with_result=True)
    df = add_avg_rank_features(df)
    df = add_pedigree(df)

    df["race_distance_category"] = df["race_distance"].map(distance_category)

    # 取消・除外の馬は出走していないので除く (中止・失格は出走扱いで is_top3 = 0)
    not_started = df["is_canceled"] | df["rank_text"].isin(NOT_STARTED)
    df = df[~not_started].copy()

    result_fixed = df["rank_text"].notna().any()
    if result_fixed:
        df[TARGET_COLUMN] = (df["rank"] <= 3).fillna(False).astype("Int64")
    else:
        df[TARGET_COLUMN] = pd.array([pd.NA] * len(df), dtype="Int64")

    return df[ID_COLUMNS + feature_columns() + [TARGET_COLUMN, "rank"]].reset_index(drop=True)


def make_dataset(race_ids: list[str], output_path: str | Path) -> pd.DataFrame:
    """複数レースのテーブルを縦に結合し, CSV に保存して返す.

    CSV は Excel でも文字化けしないよう UTF-8 (BOM 付き) で保存する.
    """
    tables = []
    for i, race_id in enumerate(race_ids, start=1):
        print(f"[{i}/{len(race_ids)}] race_id={race_id} を取得中...")
        tables.append(build_race_table(race_id))

    dataset = pd.concat(tables, ignore_index=True)
    save_csv(dataset, output_path)
    print(f"{len(dataset)} 行を {output_path} に保存しました")
    return dataset


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Excel でも文字化けしないよう UTF-8 (BOM 付き) で保存する. 親フォルダがなければ作る."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# 競馬場名 -> フォルダ名 (ローマ字)
_VENUE_DIRNAMES = {name: key for key, name in VENUES.items()}


def race_csv_path(output_dir: str | Path, venue: str, date: pd.Timestamp, race_id: str) -> Path:
    """保存先: {output_dir}/{競馬場}/{開催日}/{レース番号}R.csv  例: .../tokyo/20261122/07R.csv"""
    return Path(output_dir) / _VENUE_DIRNAMES.get(venue, venue) / date.strftime("%Y%m%d") / f"{race_id[10:12]}R.csv"


def collect_races(
    venue: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    race_numbers: range | list[int] = range(7, 13),
    output_dir: str | Path = "data/processed/races",
    overwrite: bool = False,
) -> pd.DataFrame:
    """指定した競馬場・期間 (両端を含む)・レース番号のレースを連続で取得し, レースごとに CSV 保存する.

    Args:
        venue: 競馬場名 ("東京" など) または競馬場コード ("05")
        start_date, end_date: 集計期間 ("2026-11-01" など)
        race_numbers: 取得するレース番号 (デフォルトは 7R〜12R)
        output_dir: 保存先のルートフォルダ. {競馬場}/{開催日}/{レース番号}R.csv の形で保存する
        overwrite: False なら保存済みのレースは取得せずに CSV を読み込む (中断後の再開用)

    Returns:
        取得した全レースを縦に結合した DataFrame

    1 レースの取得に失敗しても, エラーを表示してそのレースを飛ばし, 次のレースへ進む.
    """
    races = find_race_ids(venue, start_date, end_date, race_numbers)
    print(f"{venue} {pd.Timestamp(start_date).date()}〜{pd.Timestamp(end_date).date()}: 対象 {len(races)} レース")

    # "05" のようにコードで指定された場合も競馬場名に揃える
    venue_name = next((name for name, code in VENUE_CODES.items() if code == venue), venue)
    tables, failed = [], []
    for i, (date, race_id) in enumerate(races, start=1):
        path = race_csv_path(output_dir, venue_name, date, race_id)
        label = f"[{i}/{len(races)}] {date.date()} {int(race_id[10:12])}R (race_id={race_id})"

        if path.exists() and not overwrite:
            print(f"{label}: 保存済みのため読み込みのみ -> {path}")
            tables.append(pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str}, parse_dates=["race_date"]))
            continue

        print(f"{label}: 取得中...")
        try:
            table = build_race_table(race_id)
        except Exception:
            print(f"{label}: 取得に失敗したためスキップします")
            traceback.print_exc()
            failed.append(race_id)
            continue

        save_csv(table, path)
        print(f"{label}: {len(table)} 頭分を保存 -> {path}")
        tables.append(table)

    if failed:
        print(f"取得に失敗したレース ({len(failed)} 件): {failed}")
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
