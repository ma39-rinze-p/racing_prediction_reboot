"""Kaggle Dataset (非公開) にアップロードするフォルダを作る.

ローカルで抽出したデータ (data/processed/races), 学習コード (src/racing_prediction),
学習の設定 (configs/) を 1 つのフォルダにまとめ, dataset-metadata.json を書き出す.

出力先はデフォルトで OS の一時フォルダ (Windows なら %TEMP%\\racing_prediction_kaggle_dataset).
リポジトリが OneDrive 内にあると, 同期中のフォルダを削除・上書きできずに失敗することがあるため.

使い方 (リポジトリのルートで実行):
    python scripts/prepare_kaggle_dataset.py

    # 実行後に表示されるコマンドでアップロードする
    kaggle datasets create  -p <出力先> --dir-mode zip                 # 初回
    kaggle datasets version -p <出力先> --dir-mode zip -m "update"     # 2 回目以降

Kaggle のユーザー名は, Kaggle CLI のログイン情報 (~/.kaggle/credentials.json または kaggle.json,
環境変数 KAGGLE_USERNAME) から自動で読み取る. --kaggle-user で指定した場合も, ログイン中の
ユーザー名と違えばエラーにする (違うと Kaggle 側で "Invalid Owner Id" になるため).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path

from download_japanese_font import ensure_japanese_font

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLUG = "racing-prediction-data"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "racing_prediction_kaggle_dataset"


def logged_in_kaggle_user() -> str | None:
    """Kaggle CLI でログイン中のユーザー名 (見つからなければ None)."""
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    for name in ("credentials.json", "kaggle.json"):  # credentials.json: `kaggle auth login` (OAuth)
        path = config_dir / name
        if path.exists():
            username = json.loads(path.read_text(encoding="utf-8")).get("username")
            if username:
                return username
    return None


def remove_tree(path: Path, retries: int = 5) -> None:
    """フォルダを削除する. 読み取り専用属性は外し, 一時的なロックは少し待って再試行する."""

    def on_error(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    for attempt in range(retries):
        try:
            shutil.rmtree(path, onerror=on_error)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise SystemExit(
                    f"{path} を削除できませんでした (別のプログラムが使用中の可能性があります).\n"
                    "エクスプローラーでそのフォルダを開いていれば閉じるか, --out で別の出力先を指定してください."
                )
            time.sleep(1)


def resolve_kaggle_user(kaggle_user: str | None = None) -> str:
    """使う Kaggle のユーザー名を決める. 指定がログイン中のユーザー名と違えばエラー."""
    logged_in = logged_in_kaggle_user()
    user = kaggle_user or logged_in
    if user is None:
        raise SystemExit(
            "Kaggle のユーザー名がわかりません. `kaggle auth login` でログインするか, --kaggle-user で指定してください."
        )
    if logged_in and user != logged_in:
        raise SystemExit(
            f"--kaggle-user ({user}) が Kaggle CLI でログイン中のユーザー名 ({logged_in}) と違います.\n"
            "表示名ではなくユーザー名 (プロフィール URL の kaggle.com/<ここ>) を指定するか, --kaggle-user を省略してください."
        )
    return user


# データセットに入れるもの: (データセット内のパス, ローカルのパス)
DATASET_CONTENTS = [
    ("races", ROOT / "data" / "processed" / "races"),
    ("src/racing_prediction", ROOT / "src" / "racing_prediction"),
    ("configs", ROOT / "configs"),
    ("assets/fonts", ROOT / "assets" / "fonts"),  # グラフ用の日本語フォント (Kaggle には日本語フォントがない)
]
FINGERPRINT_FILE = "dataset_fingerprint.txt"


def _dataset_files(contents=None):
    """データセットに入るファイルを (データセット内のパス, ローカルのパス) で列挙する (順序は固定)."""
    for prefix, local in contents or DATASET_CONTENTS:
        for path in sorted(local.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                yield f"{prefix}/{path.relative_to(local).as_posix()}", path


def compute_fingerprint(folder: Path | None = None) -> str:
    """データ・コード・設定の中身から識別値 (ハッシュ) を計算する.

    folder を指定した場合は, build_dataset_folder で作ったフォルダの中身から計算する
    (指定しなければリポジトリの data/processed/races, src/racing_prediction, configs から計算する).
    どちらで計算しても, 中身が同じなら同じ値になる.

    Kaggle 上のデータセットがローカルと同じ版か (= 最新版が反映済みか) を確かめるために使う.
    改行コードの違い (Windows の CRLF) で値が変わらないよう, 改行は LF に揃えて計算する.
    """
    contents = None if folder is None else [(prefix, Path(folder) / prefix) for prefix, _ in DATASET_CONTENTS]
    h = hashlib.sha256()
    for rel, path in _dataset_files(contents):
        h.update(rel.encode("utf-8") + b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def build_dataset_folder(out: Path, user: str, slug: str = DEFAULT_SLUG) -> tuple[int, str]:
    """out に races/, src/racing_prediction/, configs/, dataset_fingerprint.txt, dataset-metadata.json を作る.

    (CSV の件数, 識別値) を返す. 識別値はコピーした後のフォルダの中身から計算するので,
    データ抽出 (collect_races) を並行して実行していて CSV が増え続けていても, フォルダの中身と必ず一致する.
    """
    races = ROOT / "data" / "processed" / "races"
    if not any(races.rglob("*.csv")):
        raise SystemExit(f"{races} に CSV がありません. 先にデータを抽出してください.")
    ensure_japanese_font()  # グラフ用の日本語フォント (なければダウンロード)

    out = Path(out)
    if out.exists():
        remove_tree(out)
    out.mkdir(parents=True)

    for prefix, local in DATASET_CONTENTS:  # データ, 学習コード, 学習の設定 (configs/train.yaml)
        shutil.copytree(local, out / prefix, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # 中身の識別値. ノートブックはこれを見て, 最新版のデータセットが使われているかを確かめる
    fingerprint = compute_fingerprint(out)
    (out / FINGERPRINT_FILE).write_text(fingerprint + "\n", encoding="utf-8")
    n_csv = len(list((out / "races").rglob("*.csv")))
    metadata = {
        "title": slug,
        "id": f"{user}/{slug}",
        "licenses": [{"name": "other"}],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return n_csv, fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kaggle-user", help="Kaggle のユーザー名 (省略時はログイン情報から自動取得)")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help=f"データセット名 (デフォルト: {DEFAULT_SLUG})")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"出力先フォルダ (デフォルト: {DEFAULT_OUT})")
    args = parser.parse_args()

    user = resolve_kaggle_user(args.kaggle_user)
    out = Path(args.out)
    n_csv, _ = build_dataset_folder(out, user, args.slug)

    print(f"{out} を作成しました (CSV {n_csv} 件 + src/racing_prediction + configs, 所有者: {user})")
    print(f'初回      : kaggle datasets create -p "{out}" --dir-mode zip')
    print(f'2 回目以降: kaggle datasets version -p "{out}" --dir-mode zip -m "update"')


if __name__ == "__main__":
    main()
