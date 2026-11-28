"""Kaggle で学習し, できたモデルを models/production に取り込むまでを 1 回で行う.

手順:
    1. (--update-dataset を付けたとき) data/processed/races, src/, configs/ を Kaggle Dataset にアップロードし,
       Kaggle 側の処理が終わるまで待つ (データセットがなければ作成, あれば新しいバージョンを追加).
       学習コード・設定と, レースの CSV (半年ごと) を別々のデータセットにしてあり, 中身が変わったものだけを
       アップロードする (scripts/prepare_kaggle_dataset.py)
    2. 作業用フォルダ (OS の一時フォルダ) に, ノートブック (出力は消す) と kernel-metadata.json を用意する
    3. kaggle kernels push でノートブックを Kaggle に送り, 最初から最後まで実行する (= Save & Run All)
    4. kaggle kernels status で実行が終わるまで待つ
    5. kaggle kernels output で出力一式をダウンロードして reports/kaggle_runs/<学習した日付>/<バージョン名>/ に保存し,
       モデルを新しいバージョン (v{学習した日付_時刻}) として models/registry/ に登録する. 自動モード (既定) なら,
       登録済みの全バージョンのうちテストの ROC-AUC が最良のものを models/production/ に適用する
       (評価結果 summary.json・roc_curve.png と精度のまとめ METRICS.md も models/production/ に置く. scripts/model_registry.py).
       (失敗した場合もログを reports/kaggle_runs/<日付>/error_<時刻>/ に保存する)

使い方 (リポジトリのルートで実行):
    python scripts/train_on_kaggle.py --update-dataset   # データ・コード・設定を更新してから学習 (通常はこれ)
    python scripts/train_on_kaggle.py                    # データ・コード・設定 (configs/train.yaml) が前回のアップロードから
                                                         # 変わっていないとき (設定だけ変えた場合も --update-dataset が必要)
    python scripts/train_on_kaggle.py --fetch-only       # 実行済みのノートブックの出力を取り込むだけ
    python scripts/train_on_kaggle.py --dry-run          # 作業用フォルダを作って, 実行するコマンドを表示するだけ

事前準備: pip install kaggle → kaggle auth login (Kaggle CLI にログイン). ユーザー名はログイン情報から自動で読む.
取り込んだモデルを GitHub Actions で使うには, models/production を commit して main に push する.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_kaggle_model import download_kernel_output, fetch_from_kernel, print_summary  # noqa: E402
from prepare_kaggle_dataset import (  # noqa: E402
    DEFAULT_SLUG as DEFAULT_DATASET_SLUG,
    build_dataset_folder,
    compute_fingerprint,
    dataset_parts,
    fingerprint_file,
    remove_tree,
    resolve_kaggle_user,
)

# ノートブックの setup セルにある, この行を期待する識別値に書き換えて送る
FINGERPRINT_PLACEHOLDER = "EXPECTED_DATASET_FINGERPRINT = None"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "predict_jra_racing.ipynb"
DEFAULT_KERNEL_SLUG = "predict-jra-racing"
WORK_DIR = Path(tempfile.gettempdir()) / "racing_prediction_kaggle_kernel"


def kaggle(*args: str, dry_run: bool = False, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    """kaggle CLI を実行する. dry_run なら表示するだけ. quiet なら出力を表示しない."""
    cmd = ["kaggle", *args]
    if not quiet:
        print("$", " ".join(f'"{a}"' if " " in a else a for a in cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = (result.stdout + result.stderr).strip()
    if output and not quiet:
        print(output)
    if check and result.returncode != 0:
        raise SystemExit(f"kaggle コマンドが失敗しました (終了コード {result.returncode})")
    return result


# ---------------------------------------------------------------------------
# 1. データセット
# ---------------------------------------------------------------------------
def remote_fingerprint(user: str, slug: str) -> str | None:
    """Kaggle 上の最新版のデータセットの識別値 (<データセット名>.fingerprint.txt). 取れなければ None.

    通信エラーや一時的な失敗で待機処理全体が止まらないよう, 例外は出さずに None を返す.
    """
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            result = subprocess.run(
                ["kaggle", "datasets", "download", f"{user}/{slug}", "-f", fingerprint_file(slug), "-p", tmp, "--force", "-q"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            )
            if result.returncode != 0:
                return None
            for path in Path(tmp).rglob("*"):
                if path.suffix == ".zip":  # 圧縮されて届いた場合
                    with zipfile.ZipFile(path) as zf:
                        name = next((n for n in zf.namelist() if n.endswith(fingerprint_file(slug))), None)
                        return zf.read(name).decode("utf-8").strip() if name else None
                if path.name == fingerprint_file(slug):
                    return path.read_text(encoding="utf-8").strip()
    except Exception as e:  # noqa: BLE001 - 待機中の一時的な失敗は再試行する
        print(f"\n(識別値の取得に失敗しました: {type(e).__name__}: {e}. 再試行します)", flush=True)
    return None


def update_dataset(user: str, base_slug: str, dry_run: bool) -> dict[str, str]:
    """中身が変わったデータセットだけをアップロードし (なければ作成, あれば新しいバージョン), 最新版が使えるまで待つ.

    {データセット名: 識別値} を返す (ノートブックに埋め込む値).
    Kaggle 上の識別値がローカルと同じデータセットはアップロードしない (終わった半年分のレースなど).
    アップロードするものの識別値はアップロード用フォルダ (その時点のスナップショット) から計算するので,
    データ抽出を並行して実行していても, アップロードした中身と必ず一致する.
    """
    fingerprints: dict[str, str] = {}
    uploaded: list[str] = []
    for part in dataset_parts(base_slug):
        local = compute_fingerprint(part.files)
        remote = None if dry_run else remote_fingerprint(user, part.slug)
        if remote == local:
            print(f"{part.slug}: 変更なし (ファイル {len(part.files):,} 件, 識別値 {local}). アップロードを省略します")
            fingerprints[part.slug] = local
            continue
        folder, fingerprint = build_dataset_folder(part, user)
        fingerprints[part.slug] = fingerprint
        reason = "新規または識別値なし" if remote is None else f"変更あり (Kaggle 上の識別値 {remote})"
        print(f"{part.slug}: {reason}. ファイル {len(part.files):,} 件をアップロードします (識別値 {fingerprint})")
        exists = not dry_run and dataset_exists(user, part.slug)
        if exists:
            args = ("datasets", "version", "-p", str(folder), "--dir-mode", "zip", "-m",
                    f"update ({part.n_csv} csv, {fingerprint})")
        else:
            args = ("datasets", "create", "-p", str(folder), "--dir-mode", "zip")
        if dry_run:
            kaggle(*args, dry_run=True)
            continue
        upload_with_retry(args, user, part.slug, fingerprint)
        uploaded.append(part.slug)

    if not uploaded and not dry_run:
        print("アップロードが必要なデータセットはありませんでした")
    for slug in uploaded:  # アップロードはすべて先に行い, Kaggle 側の処理はまとめて待つ
        wait_dataset_version(user, slug, fingerprints[slug])
    return fingerprints


def dataset_exists(user: str, slug: str) -> bool:
    """Kaggle にデータセットがあるか. まだないデータセットは Kaggle が 403 (Forbidden) を返すため, 出力は表示しない."""
    return kaggle("datasets", "status", f"{user}/{slug}", check=False, quiet=True).returncode == 0


def upload_failed(result: subprocess.CompletedProcess) -> bool:
    """アップロードが失敗したか. kaggle CLI は「Dataset creation error: ...」を表示しても終了コード 0 を返すため, 出力も見る."""
    output = result.stdout + result.stderr
    return result.returncode != 0 or "creation error" in output.lower()


def local_fingerprints(base_slug: str) -> dict[str, str]:
    """ローカルのデータ・コード・設定の {データセット名: 識別値}."""
    return {part.slug: compute_fingerprint(part.files) for part in dataset_parts(base_slug)}


def upload_with_retry(args: tuple[str, ...], user: str, slug: str, fingerprint: str, waits: tuple[int, ...] = (60, 180)) -> None:
    """アップロード (kaggle datasets version / create) を, 失敗したら少し待ってやり直す.

    Kaggle は直前の版の処理中に続けて版を作ろうとすると "400 Bad Request" を返すことがある
    (短時間に何度もアップロードした場合など). 一時的な失敗なので, 待ってからやり直す.
    やり直す前に, 失敗したように見えて実は版が作られていないかを識別値で確認し, 作られていればやり直さない.
    """
    for attempt in range(len(waits) + 1):
        result = kaggle(*args, check=False)
        if not upload_failed(result):
            return
        if "already in use" in (result.stdout + result.stderr):
            # 作成直後のデータセットは, Kaggle の処理が終わるまで status が 403 になり「ない」と判定されるため,
            # 作成し直そうとしてこのエラーになる. 処理が終われば version で更新できる
            raise SystemExit(
                f"{slug} は Kaggle 上に作成済みで, まだ処理中のようです (処理中は存在の確認ができません). "
                "数分おいてから再実行してください."
            )
        if attempt == len(waits):
            break
        print(f"アップロードに失敗しました. {waits[attempt]} 秒待ってから確認します "
              f"(Kaggle が直前の版を処理中の場合などに, 一時的に失敗することがあります)", flush=True)
        time.sleep(waits[attempt])
        if remote_fingerprint(user, slug) == fingerprint:
            print("Kaggle 上には今回の版が作られていたため, やり直しません")
            return
        print(f"やり直します ({attempt + 2} 回目)")
    raise SystemExit(
        "アップロードに失敗しました (何度か待ってやり直しても失敗). 数分おいてから再実行するか, "
        f"https://www.kaggle.com/datasets/{user}/{slug} で状況を確認してください."
    )


def wait_dataset_version(user: str, slug: str, fingerprint: str, interval: int = 20, timeout_min: int = 30) -> None:
    """アップロードした版が Kaggle で使えるようになるまで待つ.

    `kaggle datasets status` はアップロード直後に前の版の "ready" を返すことがあり, それを信じて
    すぐにノートブックを実行すると前の版のデータセット (古いコード・設定) で動いてしまう.
    そのため, 最新版の識別値のファイル (<データセット名>.fingerprint.txt) を実際にダウンロードし, 今回の識別値になるまで待つ.
    """
    print(f"{slug}: アップロードした版 (識別値 {fingerprint}) が Kaggle で使えるようになるのを待っています")
    start = time.time()
    deadline = start + timeout_min * 60
    remote = None
    while time.time() < deadline:
        time.sleep(interval)
        remote = remote_fingerprint(user, slug)
        elapsed = int(time.time() - start)
        if remote == fingerprint:
            print(f"  [{elapsed:>4} 秒] 反映されました")
            time.sleep(interval)  # ノートブックの実行環境への反映に少し余裕をもたせる
            return
        print(f"  [{elapsed:>4} 秒] Kaggle 上の識別値: {remote} (まだ前の版です)", flush=True)
    raise SystemExit(
        f"{timeout_min} 分待っても新しい版が反映されませんでした (Kaggle 上の識別値: {remote}, 期待する識別値: {fingerprint}).\n"
        f"https://www.kaggle.com/datasets/{user}/{slug} でアップロードの処理状況を確認し, 反映されていれば "
        "`python scripts/train_on_kaggle.py` (--update-dataset なし) で続きを実行してください."
    )


def ensure_dataset_current(user: str, fingerprints: dict[str, str]) -> None:
    """--update-dataset なしで実行するとき, Kaggle 上のデータセットがすべてローカルと同じ内容かを確かめる."""
    differ = []
    for slug, fingerprint in fingerprints.items():
        remote = remote_fingerprint(user, slug)
        if remote != fingerprint:
            differ.append(f"  {slug}: Kaggle 上 {remote} / ローカル {fingerprint}")
    if differ:
        raise SystemExit(
            "Kaggle 上のデータセットがローカルのデータ・コード・設定と違います:\n" + "\n".join(differ) + "\n"
            "configs/train.yaml やコード, データを変更した場合は --update-dataset を付けて実行してください."
        )
    print(f"Kaggle 上のデータセット ({len(fingerprints)} 件) はローカルと同じ内容です")


# ---------------------------------------------------------------------------
# 2〜4. ノートブックの実行
# ---------------------------------------------------------------------------
def prepare_kernel_folder(
    notebook: Path, user: str, kernel_slug: str, internet: bool, fingerprints: dict[str, str]
) -> Path:
    """作業用フォルダに出力を消したノートブックと kernel-metadata.json を置く.

    ノートブックには期待するデータセットの識別値を埋め込む. Kaggle 上で古い版のデータセットが
    使われた場合, ノートブックは古いコード・設定で学習せずにすぐエラーで止まる.
    """
    if WORK_DIR.exists():
        remove_tree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    nb = json.loads(notebook.read_text(encoding="utf-8"))
    embedded = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell["outputs"] = []  # 前回の出力は不要 (Kaggle で最初から実行し直す)
        cell["execution_count"] = None
        source = "".join(cell["source"])
        if FINGERPRINT_PLACEHOLDER in source:
            source = source.replace(FINGERPRINT_PLACEHOLDER, f"EXPECTED_DATASET_FINGERPRINT = {json.dumps(fingerprints)}")
            cell["source"] = source.splitlines(True)
            embedded = True
    if not embedded:
        raise SystemExit(f"ノートブックに `{FINGERPRINT_PLACEHOLDER}` の行がありません ({notebook})")
    (WORK_DIR / notebook.name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    metadata = {
        "id": f"{user}/{kernel_slug}",
        "title": kernel_slug,
        "code_file": notebook.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": internet,
        "dataset_sources": [f"{user}/{slug}" for slug in fingerprints],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (WORK_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"作業用フォルダを作成しました: {WORK_DIR}")
    return WORK_DIR


def parse_kernel_status(text: str) -> str:
    """`... has status "KernelWorkerStatus.COMPLETE"` などから状態 (complete / error / running ...) を取り出す."""
    m = re.search(r'status\s+"(?:KernelWorkerStatus\.)?([A-Za-z_]+)"', text)
    return m.group(1).lower() if m else text.strip().lower()


def show_kernel_error(kernel: str, max_lines: int = 40) -> None:
    """失敗した実行のログを reports/kaggle_runs/<日付>/error_<時刻>/ に保存し, エラー (Traceback 以降) を表示する."""
    try:
        run_dir = download_kernel_output(kernel, error=True)
    except subprocess.CalledProcessError:
        print("(ログを取得できませんでした)")
        return
    log = run_dir / "run_log.txt"
    if not log.exists():
        print(f"(ログがありませんでした: {run_dir})")
        return
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("[stderr] ")]
    # ノートブックのセルで起きたエラー ("Exception encountered at In [n]") を優先して表示する
    # (その前の Traceback は Kaggle の実行の仕組み (papermill) 側のもので, 原因の特定には役立たない)
    start = next((i for i, line in enumerate(lines) if "Exception encountered" in line), None)
    if start is None:
        start = next((i for i, line in enumerate(lines) if "Traceback" in line), None)
    shown = lines[start:start + max_lines] if start is not None else lines[-max_lines:]
    print("---- Kaggle の実行ログ (エラー部分) ----")
    print("\n".join(shown))
    print("---------------------------------------")
    print(f"ログ全体: {log}")


def wait_kernel_complete(kernel: str, interval: int, timeout_hours: float) -> None:
    """ノートブックの実行が終わるまで待つ. エラー・キャンセルなら終了する."""
    start = time.time()
    deadline = start + timeout_hours * 3600
    time.sleep(30)  # push 直後は前のバージョンの状態が返ることがあるため少し待つ
    while time.time() < deadline:
        result = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        status = parse_kernel_status(result.stdout + result.stderr)
        elapsed = int((time.time() - start) / 60)
        print(f"[{elapsed:>3} 分経過] 状態: {status}", flush=True)
        if status == "complete":
            return
        if status in ("error", "cancel_requested", "cancel_acknowledged"):
            show_kernel_error(kernel)
            raise SystemExit(
                f"ノートブックの実行が {status} で終了しました. "
                f"詳細は Kaggle の画面 (https://www.kaggle.com/code/{kernel}) の Logs でも確認できます."
            )
        time.sleep(interval)
    raise SystemExit(f"{timeout_hours} 時間待っても実行が終わりませんでした. Kaggle の画面で確認してください.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update-dataset", action="store_true", help="学習の前にデータとコードを Kaggle にアップロードする")
    parser.add_argument("--fetch-only", action="store_true", help="実行済みのノートブックの出力を取り込むだけ")
    parser.add_argument("--dry-run", action="store_true", help="作業用フォルダを作り, 実行するコマンドを表示するだけ")
    parser.add_argument("--kaggle-user", help="Kaggle のユーザー名 (省略時はログイン情報から自動取得)")
    parser.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG, help=f"Kaggle のノートブック名 (デフォルト: {DEFAULT_KERNEL_SLUG})")
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG, help=f"データセット名 (デフォルト: {DEFAULT_DATASET_SLUG})")
    parser.add_argument("--notebook", default=str(DEFAULT_NOTEBOOK), help="実行するノートブック")
    parser.add_argument("--internet", action="store_true", help="Kaggle での実行時にインターネットを有効にする (通常は不要)")
    parser.add_argument("--poll-interval", type=int, default=60, help="実行状況を確認する間隔 (秒)")
    parser.add_argument("--timeout-hours", type=float, default=12, help="実行の完了を待つ最大時間 (Kaggle の上限は 12 時間)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force-promote", action="store_true", help="ROC-AUC に関係なく新しいモデルを適用する (手動モードになる)")
    mode.add_argument("--no-promote", action="store_true", help="新しいモデルを登録だけして適用しない")
    args = parser.parse_args()
    promote_mode = "force" if args.force_promote else "none" if args.no_promote else "auto"

    user = resolve_kaggle_user(args.kaggle_user)
    kernel = f"{user}/{args.kernel_slug}"
    print(f"Kaggle ユーザー: {user} / ノートブック: {kernel} / データセット: {user}/{args.dataset_slug} (+ レースの CSV: {args.dataset_slug}-races-*)")

    if not args.fetch_only:
        if args.update_dataset:
            print("\n== 1. データ・コード・設定をアップロード")
            fingerprints = update_dataset(user, args.dataset_slug, args.dry_run)
        else:
            fingerprints = local_fingerprints(args.dataset_slug)  # ローカルのデータ・コード・設定の識別値
            if not args.dry_run:
                print("\n== 1. Kaggle 上のデータセットがローカルと同じか確認")
                ensure_dataset_current(user, fingerprints)

        print("\n== 2. 作業用フォルダを準備")
        folder = prepare_kernel_folder(
            Path(args.notebook), user, args.kernel_slug, args.internet, fingerprints
        )

        print("\n== 3. Kaggle でノートブックを実行 (Save & Run All)")
        kaggle("kernels", "push", "-p", str(folder), dry_run=args.dry_run)
        print(f"実行状況は https://www.kaggle.com/code/{kernel} でも確認できます (このウィンドウを閉じても実行は続きます)")

        if args.dry_run:
            print("\n(dry-run のため, ここで終了します)")
            return

        print("\n== 4. 実行の完了を待つ")
        wait_kernel_complete(kernel, args.poll_interval, args.timeout_hours)

    print("\n== 5. 実行結果をローカルに保存し, モデルを新しいバージョンとして登録する")
    run_dir = fetch_from_kernel(kernel, promote_mode=promote_mode)
    print_summary()
    if (run_dir / "results" / "summary.json").exists():
        print(f"評価結果: {run_dir / 'results'} (summary.json, test_predictions.csv, グラフなど)")
    print(f"実行ログ: {run_dir / 'run_log.txt'}")


if __name__ == "__main__":
    main()
