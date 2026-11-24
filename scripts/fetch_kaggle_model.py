"""Kaggle で学習したモデル (predict_jra_racing.ipynb の出力) を取り込む.

取り込んだモデルは新しいバージョンとして models/registry/ に保存し (scripts/model_registry.py),
自動モード (既定) なら, 登録済みの全バージョンのうちテストの ROC-AUC が最良のものを models/production/ に適用する.
GitHub Actions の推論は models/production/ のモデルを使うので, 取り込んだら commit & push する.

--kernel で取り込む場合は, Kaggle の実行結果一式もローカルに保存する:
    reports/kaggle_runs/<日時>/
        artifacts/   モデル (model.txt, metadata.json, feature_importance.csv)
        results/     評価結果 (summary.json, test_predictions.csv, グラフなど)
        run_log.txt  実行ログ (print の出力とエラー. 元のログ *.log も残す)

使い方 (リポジトリのルートで実行):
    # Kaggle のノートブックの出力を Kaggle CLI でダウンロードして取り込む
    #   (ノートブックを「Save Version」(Save & Run All) で保存した最新バージョンの出力が対象)
    python scripts/fetch_kaggle_model.py --kernel <ユーザー名>/<ノートブックのスラッグ>

    # 手動でダウンロードしたフォルダ (model.txt と metadata.json が入ったもの) から取り込む
    python scripts/fetch_kaggle_model.py --from-dir <フォルダ>

    # --force: 取り込んだモデルを適用する (手動モードになる) / --no-promote: 登録だけして適用しない

ノートブックのスラッグは, Kaggle のノートブックの URL kaggle.com/code/<ユーザー名>/<ここ> の部分.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from model_registry import load_registry, register

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "reports" / "kaggle_runs"  # Kaggle の実行結果の保存先 (git 管理外)


def find_artifacts(root: Path) -> Path:
    """root 以下で model.txt と metadata.json が両方あるフォルダを探す."""
    for model in root.rglob("model.txt"):
        if (model.parent / "metadata.json").exists():
            return model.parent
    raise SystemExit(f"{root} 以下に model.txt と metadata.json の組が見つかりません")


def import_artifacts(root: Path, promote_mode: str = "auto", source: str = "") -> str:
    """root 以下にあるモデルを新しいバージョンとして登録し, 精度が下がっていなければ適用する.

    promote_mode: "auto" (ROC-AUC が最良のモデルを適用) / "force" (取り込んだモデルを適用) / "none" (登録だけ)
    登録したバージョン番号を返す.
    """
    src = find_artifacts(Path(root))
    return register(src, source=source or str(src), promote_mode=promote_mode)


def convert_log(log_path: Path) -> Path:
    """Kaggle の実行ログ (JSON) を読みやすいテキスト (run_log.txt) にする. エラー (stderr) の行には印を付ける."""
    out = log_path.parent / "run_log.txt"
    try:
        entries = json.loads(log_path.read_text(encoding="utf-8"))
        lines = []
        for e in entries:
            prefix = "[stderr] " if e.get("stream_name") == "stderr" else ""
            lines += [prefix + line for line in e.get("data", "").rstrip("\n").splitlines()]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except (json.JSONDecodeError, AttributeError, TypeError):
        shutil.copy2(log_path, out)  # JSON でなければそのまま
    return out


def download_kernel_output(kernel: str, runs_dir: Path = RUNS_DIR, suffix: str = "") -> Path:
    """Kaggle のノートブックの最新バージョンの出力一式 (モデル・評価結果・ログ) を
    runs_dir/<日時><suffix>/ にダウンロードし, そのフォルダを返す.
    """
    run_dir = Path(runs_dir) / (datetime.now().strftime("%Y%m%d_%H%M%S") + suffix)
    run_dir.mkdir(parents=True, exist_ok=False)
    subprocess.run(["kaggle", "kernels", "output", kernel, "-p", str(run_dir), "--force", "-q"], check=True)
    for log in run_dir.glob("*.log"):
        convert_log(log)
    return run_dir


def fetch_from_kernel(kernel: str, runs_dir: Path = RUNS_DIR, promote_mode: str = "auto") -> Path:
    """Kaggle の実行結果一式をローカルに保存し, モデルを新しいバージョンとして登録する. 保存先フォルダを返す."""
    run_dir = download_kernel_output(kernel, runs_dir)
    print(f"Kaggle の実行結果を保存しました: {run_dir}")
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(run_dir).as_posix()} ({p.stat().st_size:,} bytes)")
    import_artifacts(run_dir, promote_mode=promote_mode, source=f"Kaggle {kernel} ({run_dir.name})")
    return run_dir


def print_summary() -> None:
    registry = load_registry()
    active = registry["active"]
    info = registry["versions"].get(active, {})
    print(f"適用中のモデル: {active} (学習期間 {info.get('train_period')} / {info.get('n_train_races')} レース)")
    test = info.get("test_metrics", {})
    if test:
        print(f"  テスト: ROC-AUC={test.get('roc_auc')} レース内AUC={test.get('roc_auc_by_race')} logloss={test.get('logloss')} 単勝的中率={test.get('win_hit')}")
    print("一覧: python scripts/model_registry.py list / 1 つ前に戻す: rollback / ROC-AUC 最良に戻す: best")
    print("GitHub Actions で使うには commit & push してください:")
    print("  git add models && git commit -m \"update production model\" && git push")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--kernel", help="Kaggle のノートブック (<ユーザー名>/<スラッグ>)")
    source.add_argument("--from-dir", help="model.txt と metadata.json があるフォルダ")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="取り込んだモデルを適用する (手動モードになる)")
    mode.add_argument("--no-promote", action="store_true", help="登録だけして適用しない")
    args = parser.parse_args()
    promote_mode = "force" if args.force else "none" if args.no_promote else "auto"

    if args.kernel:
        fetch_from_kernel(args.kernel, promote_mode=promote_mode)
    else:
        import_artifacts(Path(args.from_dir), promote_mode=promote_mode)
    print_summary()


if __name__ == "__main__":
    main()
