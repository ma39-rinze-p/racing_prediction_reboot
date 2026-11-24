"""make_dataset

netkeiba から取得した出馬表・結果・過去戦績・血統をもとに,
1 行 = 1 頭 (レース × 馬) の学習用テーブルを作り CSV に保存する.

使用例:
    >>> from racing_prediction.data.make_dataset import make_dataset, collect_races
    >>> df = make_dataset(["202605050507"], "data/processed/dataset.csv")
    >>> # 競馬場 (複数可)・期間・レース番号を指定して連続取得 (レースごとに CSV を保存)
    >>> df = collect_races("東京", "2026-11-01", "2026-11-22", race_numbers=range(7, 13))
    >>> df = collect_races(["東京", "京都", "福島"], "2026-11-01", "2026-11-22")

テーブルの列:
    識別用   : race_id, race_date, umaban, horse_name (馬名は過学習を防ぐため特徴量にしない)
    特徴量   : レースの場所・距離・芝/ダート・馬場状態・出走頭数・クラス・本賞金, 馬ごとの情報 (枠番含む),
               条件別の平均着順と出走回数, 近走・能力 (hist_*), 血統 (ID なし)
    計算用   : hist_last{1〜3}_* (直近 3 走のタイムなど. スピード指数の計算に使う)
    目的変数 : is_win (1 着なら 1, それ以外は 0. 結果未確定のレースは欠損)
               CSV には保存せず, 読み込み時に着順 (rank) から作る (add_win_label)
    参考     : rank (着順), is_top3 (3 着以内なら 1. ワイドの評価に使う. CSV に保存する)
    オッズ   : win_odds (単勝オッズ), popularity (人気). CSV には確定オッズを保存する
               win_odds はモデルの特徴量に使う (model_features の odds_*). 学習では確定オッズ,
               推論では推論時点 (発走 12 分前) に取得したオッズで置き換える (inference/run.py)
"""

from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd

from racing_prediction.data.race_list import find_race_ids, venue_name_from_race_id
from racing_prediction.data.scrape_horse_pedigree import add_pedigree
from racing_prediction.data.scrape_shutuba import fetch_shutuba
from racing_prediction.features.build_features import (
    VENUES,
    add_history_features,
    avg_rank_feature_names,
    distance_category,
    history_feature_names,
    history_raw_names,
    n_starts_feature_names,
)

# 馬名は馬ごとに一意で, 特徴量にするとモデルが丸暗記して新馬に効かないため識別用にする
ID_COLUMNS = ["race_id", "race_date", "umaban", "horse_id", "horse_name"]

RACE_FEATURE_COLUMNS = [
    "race_venue",
    "race_distance_category",
    "race_surface",
    "race_track_condition",
    "n_horses",  # 出走頭数 (取消・除外を除く)
    "race_distance",  # 距離 (m)
    "race_class",  # クラス (0: 新馬・未勝利, 1〜3: 1〜3 勝クラス, 4: オープン・L, 5〜7: G3〜G1)
    "race_prize",  # 1 着の本賞金 (万円)
]

HORSE_FEATURE_COLUMNS = [
    "waku",  # 枠番
    "sex",
    "age",
    "weight_carried",
    "jockey_name",
    "trainer_name",
    "horse_weight",
    "weight_diff",
]

PEDIGREE_FEATURE_COLUMNS = ["sire", "sire_line", "dam", "sire_sire", "sire_dam", "dam_sire", "dam_dam"]

# モデルの目的変数 (1 着か否か). CSV には保存せず, 着順から作る (保存済みの CSV をそのまま使えるようにするため)
TARGET_COLUMN = "is_win"
# 3 着以内か否か (CSV に保存する. ワイドの評価に使う)
TOP3_COLUMN = "is_top3"

# 結果の列 (着順と確定オッズ・人気. win_odds は model_features でオッズの特徴量にする. 推論時は推論時点のオッズ)
RESULT_COLUMNS = ["rank", "win_odds", "popularity"]

# 出走しなかった馬 (着順欄がこれらの場合はテーブルから除く)
NOT_STARTED = ("取消", "除外")


def feature_columns() -> list[str]:
    """モデルに入れる特徴量の列名一覧."""
    return (
        RACE_FEATURE_COLUMNS
        + HORSE_FEATURE_COLUMNS
        + avg_rank_feature_names()
        + n_starts_feature_names()
        + history_feature_names()
        + PEDIGREE_FEATURE_COLUMNS
    )


def build_race_table(race_id: str) -> pd.DataFrame:
    """1 レース分のテーブル (1 行 = 1 頭) を作る.

    アクセス回数は 出馬表 1 + 結果 1 + 頭数 × 2 (戦績・血統).
    すべて polite_get 経由なので, User-Agent 付与とアクセスごとの 1〜3 秒待機が入る.
    """
    df = fetch_shutuba(race_id, with_result=True)
    df = add_history_features(df)
    df = add_pedigree(df)

    df["race_distance_category"] = df["race_distance"].map(distance_category)

    # 取消・除外の馬は出走していないので除く (中止・失格は出走扱いで is_top3 = is_win = 0)
    not_started = df["is_canceled"] | df["rank_text"].isin(NOT_STARTED)
    df = df[~not_started].copy()
    df["n_horses"] = len(df)

    result_fixed = df["rank_text"].notna().any()
    if result_fixed:
        df[TOP3_COLUMN] = (df["rank"] <= 3).fillna(False).astype("Int64")
    else:
        df[TOP3_COLUMN] = pd.array([pd.NA] * len(df), dtype="Int64")

    return add_win_label(df[output_columns()].reset_index(drop=True))


def add_win_label(df: pd.DataFrame) -> pd.DataFrame:
    """目的変数 is_win (1 着なら 1) を着順から作って追加する. 結果未確定 (is_top3 が欠損) の行は欠損.

    同着の場合は 1 着の馬が複数になる (どちらも 1).
    """
    df = df.copy()
    rank = pd.to_numeric(df["rank"], errors="coerce")
    win = (rank == 1).astype("Int64")
    df[TARGET_COLUMN] = win.where(df[TOP3_COLUMN].notna(), pd.NA).astype("Int64")
    return df


def output_columns() -> list[str]:
    """CSV に保存する列の順番."""
    return ID_COLUMNS + feature_columns() + history_raw_names() + [TOP3_COLUMN] + RESULT_COLUMNS


# 出馬表から取り直す列 (horse_id がないと戦績を取れないため. 列追加前の CSV を作り直すときに使う)
_SHUTUBA_UPGRADE_COLUMNS = ["horse_id", "race_distance", "race_class", "race_prize"]


def missing_columns(df: pd.DataFrame) -> list[str]:
    """CSV に保存する列 (output_columns) のうち, df にない列."""
    return [c for c in output_columns() if c not in df.columns]


def upgrade_race_table(df: pd.DataFrame, refetch_history: bool = False) -> pd.DataFrame:
    """列を追加する前に保存した 1 レース分のテーブルに, 足りない列を追加して返す.

    すべて取り直す (build_race_table) より少ないアクセスで済むよう, 保存済みの列 (血統・着順・オッズなど) はそのまま使い,
    - horse_id・距離・クラス・本賞金 がなければ出馬表を 1 回取得して補い,
    - 戦績から計算する列 (avg_rank_*, n_starts_*, hist_*) は計算し直す
      (戦績は保存済みのもの (data/raw/horse_results) があればアクセスしない).
    refetch_history=False で戦績から計算する列がすべて揃っていれば, そのまま返す.
    """
    df = df.copy()
    if any(c not in df.columns for c in _SHUTUBA_UPGRADE_COLUMNS):
        race_id = str(df["race_id"].iloc[0])
        shutuba = fetch_shutuba(race_id, with_result=False)
        extra = shutuba[["umaban"] + _SHUTUBA_UPGRADE_COLUMNS]
        df = df.drop(columns=[c for c in _SHUTUBA_UPGRADE_COLUMNS if c in df.columns])
        df = df.merge(extra, on="umaban", how="left")
        if df["horse_id"].isna().any():
            raise ValueError(f"race_id={race_id}: 出馬表に無い馬番があります {df.loc[df['horse_id'].isna(), 'umaban'].tolist()}")

    history_cols = avg_rank_feature_names() + n_starts_feature_names() + history_feature_names() + history_raw_names()
    if refetch_history or any(c not in df.columns for c in history_cols):
        df = add_history_features(df.drop(columns=[c for c in history_cols if c in df.columns]))
    return df[output_columns()]


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
    venues: str | list[str] | tuple[str, ...],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    race_numbers: range | list[int] = range(7, 13),
    output_dir: str | Path = "data/processed/races",
    overwrite: bool = False,
) -> pd.DataFrame:
    """指定した競馬場 (複数可)・期間 (両端を含む)・レース番号のレースを連続で取得し, レースごとに CSV 保存する.

    Args:
        venues: 競馬場名 ("東京" など) / 競馬場コード ("05") / そのリスト (["東京", "京都"]) / カンマ区切り ("東京,京都")
            複数指定した場合は, 開催日順 (同じ日は競馬場コード・レース番号順) に取得する
        start_date, end_date: 集計期間 ("2026-11-01" など)
        race_numbers: 取得するレース番号 (デフォルトは 7R〜12R)
        output_dir: 保存先のルートフォルダ. {競馬場}/{開催日}/{レース番号}R.csv の形で保存する
        overwrite: False なら保存済みのレースは取得せずに CSV を読み込む (中断後の再開用)

    Returns:
        取得した全レースを縦に結合した DataFrame

    1 レースの取得に失敗しても, エラーを表示してそのレースを飛ばし, 次のレースへ進む.
    """
    races = find_race_ids(venues, start_date, end_date, race_numbers)
    print(f"{pd.Timestamp(start_date).date()}〜{pd.Timestamp(end_date).date()}: 対象 {len(races)} レース")

    tables, failed = [], []
    for i, (date, race_id) in enumerate(races, start=1):
        venue_name = venue_name_from_race_id(race_id)  # 保存先のフォルダ (競馬場ごと)
        path = race_csv_path(output_dir, venue_name, date, race_id)
        label = f"[{i}/{len(races)}] {date.date()} {venue_name}{int(race_id[10:12])}R (race_id={race_id})"

        if path.exists() and not overwrite:
            saved = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str, "horse_id": str}, parse_dates=["race_date"])
            missing = missing_columns(saved)
            if not missing:
                print(f"{label}: 保存済みのため読み込みのみ -> {path}")
                tables.append(saved)
                continue
            # 列を追加する前に保存した古い CSV は, 足りない列だけ補う (補えなければ取り直す)
            print(f"{label}: 保存済みの CSV に列がないため補います ({len(missing)} 列. 例: {missing[:3]})")
            try:
                table = upgrade_race_table(saved)
                save_csv(table, path)
                print(f"{label}: 列を補って保存 -> {path}")
                tables.append(table)
                continue
            except Exception:
                print(f"{label}: 列を補えなかったため取り直します")
                traceback.print_exc()

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
