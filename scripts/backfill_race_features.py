"""保存済みのレースの CSV (data/processed/races) に, 後から追加した列を補う.

列を追加する前に保存した CSV は学習で読み飛ばされる (model_features.load_race_csvs) ため, このスクリプトで補う.
すべて取り直すより少ないアクセスで済む:
    - horse_id・距離・クラス・本賞金: 出馬表を 1 レース 1 回取得
    - 戦績から計算する列 (avg_rank_*, n_starts_*, hist_*): 各馬の戦績ページを 1 頭 1 回取得
      (取得した戦績は data/raw/horse_results/ に保存し, 2 回目以降・別のレースでは使い回す)
    - 血統・着順・オッズなど保存済みの列はそのまま使う
アクセスごとに 1〜3 秒待機するため, 800 レース (約 4,600 頭) で 3 時間前後かかる.
途中で止めても, 補い終わったレースは保存済みなので, もう一度実行すれば続きから再開する.

使い方 (リポジトリのルートで実行):
    python scripts/backfill_race_features.py               # 列が足りない CSV だけを補う
    python scripts/backfill_race_features.py --recompute   # 全 CSV の戦績から計算する列を計算し直す
                                                           # (特徴量の定義を変えたとき. 保存済みの戦績を使うのでアクセスは少ない)
    python scripts/backfill_race_features.py --limit 5     # 試しに 5 レースだけ
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from racing_prediction.data.make_dataset import missing_columns, save_csv, upgrade_race_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--races-dir", default=str(ROOT / "data" / "processed" / "races"))
    parser.add_argument("--recompute", action="store_true", help="列が揃っている CSV も, 戦績から計算する列を計算し直す")
    parser.add_argument("--limit", type=int, default=None, help="処理するレース数の上限 (試し用)")
    args = parser.parse_args()

    paths = sorted(Path(args.races_dir).rglob("*.csv"))
    targets = []
    for path in paths:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str, "horse_id": str}, parse_dates=["race_date"])
        if args.recompute or missing_columns(df):
            targets.append((path, df))
    if args.limit is not None:
        targets = targets[: args.limit]
    print(f"CSV {len(paths)} 件のうち, 処理するもの {len(targets)} 件")

    failed = []
    start = time.time()
    for i, (path, df) in enumerate(targets, start=1):
        label = f"[{i}/{len(targets)}] {path.relative_to(args.races_dir)}"
        try:
            table = upgrade_race_table(df, refetch_history=args.recompute)
        except Exception:
            print(f"{label}: 失敗したためスキップします")
            traceback.print_exc()
            failed.append(path)
            continue
        save_csv(table, path)
        elapsed = time.time() - start
        remaining = elapsed / i * (len(targets) - i)
        print(f"{label}: {len(table)} 頭分を保存 (経過 {elapsed / 60:.0f} 分 / 残り約 {remaining / 60:.0f} 分)", flush=True)

    if failed:
        print(f"失敗した CSV ({len(failed)} 件). もう一度実行すると再試行します:")
        for path in failed:
            print(f"  {path}")
        return 1
    print("完了しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
