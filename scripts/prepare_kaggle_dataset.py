"""Kaggle Dataset (非公開) にアップロードするフォルダを作る.

ローカルで抽出したデータ (data/processed/races) と学習コード (src/racing_prediction) を
1 つのフォルダにまとめ, dataset-metadata.json を書き出す.

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
import json
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kaggle-user", help="Kaggle のユーザー名 (省略時はログイン情報から自動取得)")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help=f"データセット名 (デフォルト: {DEFAULT_SLUG})")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"出力先フォルダ (デフォルト: {DEFAULT_OUT})")
    args = parser.parse_args()

    logged_in = logged_in_kaggle_user()
    user = args.kaggle_user or logged_in
    if user is None:
        raise SystemExit(
            "Kaggle のユーザー名がわかりません. `kaggle auth login` でログインするか, --kaggle-user で指定してください."
        )
    if logged_in and user != logged_in:
        raise SystemExit(
            f"--kaggle-user ({user}) が Kaggle CLI でログイン中のユーザー名 ({logged_in}) と違います.\n"
            "表示名ではなくユーザー名 (プロフィール URL の kaggle.com/<ここ>) を指定するか, --kaggle-user を省略してください."
        )

    races = ROOT / "data" / "processed" / "races"
    n_csv = len(list(races.rglob("*.csv")))
    if n_csv == 0:
        raise SystemExit(f"{races} に CSV がありません. 先にデータを抽出してください.")

    out = Path(args.out)
    if out.exists():
        remove_tree(out)
    out.mkdir(parents=True)

    shutil.copytree(races, out / "races")
    shutil.copytree(
        ROOT / "src" / "racing_prediction",
        out / "src" / "racing_prediction",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    metadata = {
        "title": args.slug,
        "id": f"{user}/{args.slug}",
        "licenses": [{"name": "other"}],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"{out} を作成しました (CSV {n_csv} 件 + src/racing_prediction, 所有者: {user})")
    print(f'初回      : kaggle datasets create -p "{out}" --dir-mode zip')
    print(f'2 回目以降: kaggle datasets version -p "{out}" --dir-mode zip -m "update"')


if __name__ == "__main__":
    main()
