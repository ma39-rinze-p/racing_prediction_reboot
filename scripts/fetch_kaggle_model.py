"""Kaggle で学習したモデル (predict_jra_racing.ipynb の出力) を取り込む.

取り込んだモデルは新しいバージョンとして models/registry/ に保存し (scripts/model_registry.py),
自動モード (既定) なら, 登録済みの全バージョンのうちテストの ROC-AUC が最良のものを models/production/ に適用する.
GitHub Actions の推論は models/production/ のモデルを使うので, 取り込んだら commit & push する.

--kernel で取り込む場合は, Kaggle の実行結果一式もローカルに保存する (学習した日付ごとのフォルダの下に, バージョンごと):
    reports/kaggle_runs/<学習した日付 YYYYMMDD>/<バージョン名 v{日付_時刻}>/
        artifacts/   モデル (model.txt, metadata.json, feature_importance.csv)
        results/     評価結果 (summary.json, test_predictions.csv, roc_curve.png など)
        run_log.txt  実行ログ (print の出力とエラー. 元のログ *.log も残す)
    失敗した実行のログは reports/kaggle_runs/<日付>/error_<時刻>/ に保存する.

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

from model_registry import JST, load_registry, market_blend_verdict, register, version_name

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "reports" / "kaggle_runs"  # Kaggle の実行結果の保存先 (git 管理外)


def find_artifacts(root: Path) -> Path:
    """root 以下で model.txt と metadata.json が両方あるフォルダを探す."""
    for model in root.rglob("model.txt"):
        if (model.parent / "metadata.json").exists():
            return model.parent
    raise SystemExit(f"{root} 以下に model.txt と metadata.json の組が見つかりません")


def import_artifacts(root: Path, promote_mode: str = "auto", source: str = "") -> str:
    """root 以下にあるモデル (と評価結果) を新しいバージョンとして登録し, 自動モードなら ROC-AUC が最良のモデルを適用する.

    promote_mode: "auto" (ROC-AUC が最良のモデルを適用) / "force" (取り込んだモデルを適用) / "none" (登録だけ)
    登録したバージョン名を返す.
    """
    src = find_artifacts(Path(root))
    results = next((d for d in (src / "results", src.parent / "results") if (d / "summary.json").exists()), None)
    return register(src, source=source or str(src), promote_mode=promote_mode, results_dir=results)


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


def download_kernel_output(kernel: str, runs_dir: Path = RUNS_DIR, error: bool = False) -> Path:
    """Kaggle のノートブックの最新バージョンの出力一式 (モデル・評価結果・ログ) をダウンロードし, そのフォルダを返す.

    保存先は runs_dir/<学習した日付>/<バージョン名>/ (バージョン名はモデルの metadata.json の学習日時から決める).
    error=True (失敗した実行のログを取るとき) やモデルがない場合は runs_dir/<今日の日付>/error_<時刻>/.
    同じモデルの出力がすでに保存されていれば, 新しくダウンロードしたもので置き換える.
    """
    now = datetime.now(JST)
    runs_dir = Path(runs_dir)
    tmp = runs_dir / f"_downloading_{now:%Y%m%d_%H%M%S}"
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        subprocess.run(["kaggle", "kernels", "output", kernel, "-p", str(tmp), "--force", "-q"], check=True)
        for log in tmp.glob("*.log"):
            convert_log(log)
        return _move_to_run_dir(tmp, runs_dir, now, error)
    except BaseException:
        if tmp.exists():  # ダウンロードの途中で失敗した場合も, 取れた分は error_<時刻> として残す
            _move_to_run_dir(tmp, runs_dir, now, error=True)
        raise


def _move_to_run_dir(tmp: Path, runs_dir: Path, now: datetime, error: bool) -> Path:
    model = next((m.parent for m in tmp.rglob("model.txt") if (m.parent / "metadata.json").exists()), None)
    if error or model is None:
        dest = runs_dir / f"{now:%Y%m%d}" / f"error_{now:%H%M%S}"
    else:
        version = version_name(json.loads((model / "metadata.json").read_text(encoding="utf-8")))
        dest = runs_dir / version[1:9] / version  # v20261124_153012 → 20261124/v20261124_153012
        if dest.exists():
            print(f"{dest} はすでにあるため, 新しくダウンロードしたもので置き換えます")
            shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(dest))
    return dest


def fetch_from_kernel(kernel: str, runs_dir: Path = RUNS_DIR, promote_mode: str = "auto") -> Path:
    """Kaggle の実行結果一式をローカルに保存し, モデルを新しいバージョンとして登録する. 保存先フォルダを返す."""
    run_dir = download_kernel_output(kernel, runs_dir)
    print(f"Kaggle の実行結果を保存しました: {run_dir}")
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(run_dir).as_posix()} ({p.stat().st_size:,} bytes)")
    import_artifacts(run_dir, promote_mode=promote_mode, source=f"Kaggle {kernel} (reports/kaggle_runs/{run_dir.parent.name}/{run_dir.name})")
    return run_dir


def print_summary() -> None:
    registry = load_registry()
    active = registry["active"]
    info = registry["versions"].get(active, {})
    print(f"適用中のモデル: {active} (学習期間 {info.get('train_period')} / {info.get('n_train_races')} レース)")
    test = info.get("test_metrics", {})
    if test:
        print(f"  テスト: ROC-AUC={test.get('roc_auc')} レース内AUC={test.get('roc_auc_by_race')} logloss={test.get('logloss')} 単勝的中率={test.get('win_hit')}")
    mb = info.get("test_market_blend") or {}
    if mb.get("n_races"):
        print(f"  メインの目標 (混ぜるとオッズより当たるか): {market_blend_verdict(mb)}")
        print(f"    logloss オッズ単体 {mb['market_logloss']:.4f} → 混ぜた予測 {mb['blend_logloss']:.4f} "
              f"(改善 {mb['logloss_gain']:+.4f}, 95% 信頼区間 {mb['logloss_gain_ci_low']:+.4f} 〜 {mb['logloss_gain_ci_high']:+.4f})")
    cal = (info.get("test_calibration") or {}).get("model") or {}
    if cal.get("bins"):
        odds_ece = ((info.get("test_calibration") or {}).get("odds") or {}).get("ece")
        print(f"  スコアと実際の 1 着率のずれ (頭数が同じになるように 10 等分): ECE {cal['ece']:.4f}"
              + (f" (単勝オッズ {odds_ece:.4f})" if odds_ece is not None else ""))
    top = info.get("test_top_pick") or {}
    if top.get("n_races"):
        print(f"  1 着になった割合: 予測 1 位 {top['model_hit']:.1%} / 1 番人気 {top['favorite_hit']:.1%} "
              f"(差 {top['diff']:+.1%}, 95% 信頼区間 {top['diff_ci_low']:+.1%} 〜 {top['diff_ci_high']:+.1%})")
        if top.get("model_return") is not None:
            print(f"  単勝 {top['bet_per_race']} 円ずつ: 予測 1 位 総利益 {top['model_profit']:+,.0f} 円 (回収率 {top['model_return']:.1%}) / "
                  f"1 番人気 総利益 {top['favorite_profit']:+,.0f} 円 (回収率 {top['favorite_return']:.1%})")
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
