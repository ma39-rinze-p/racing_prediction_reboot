"""Kaggle Dataset (非公開) にアップロードするフォルダを作る.

Kaggle のデータセットは, 新しい版を作るたびに全ファイルをアップロードし直す (差分だけの追加はできない).
そのため, 中身を次の複数のデータセットに分け, 中身が変わったデータセットだけをアップロードする
(scripts/train_on_kaggle.py --update-dataset):

    racing-prediction-data                  学習コード (src/racing_prediction), 学習の設定 (configs/), 日本語フォント (assets/fonts)
    racing-prediction-data-races-2024h1 など  レースの CSV (data/processed/races) を開催日の半年ごとに分けたもの
                              (h1 = 1〜6 月, h2 = 7〜12 月). 終わった半年分は普通は変わらないので, 次からは
                              新しく集めたデータを含む半年分だけがアップロードされる

各データセットには中身の識別値 (ハッシュ) を <データセット名>.fingerprint.txt として入れる.
Kaggle 上の識別値とローカルの識別値が同じデータセットはアップロードしない.

出力先はデフォルトで OS の一時フォルダ (Windows なら %TEMP%\\racing_prediction_kaggle_dataset\\<データセット名>).
リポジトリが OneDrive 内にあると, 同期中のフォルダを削除・上書きできずに失敗することがあるため.

使い方 (リポジトリのルートで実行):
    python scripts/prepare_kaggle_dataset.py    # 全データセットのフォルダを作り, アップロードのコマンドを表示する
    (通常は python scripts/train_on_kaggle.py --update-dataset が, 変わったものだけを自動でアップロードする)

Kaggle のユーザー名は, Kaggle CLI のログイン情報 (~/.kaggle/credentials.json または kaggle.json,
環境変数 KAGGLE_USERNAME) から自動で読み取る. --kaggle-user で指定した場合も, ログイン中の
ユーザー名と違えばエラーにする (違うと Kaggle 側で "Invalid Owner Id" になるため).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from download_japanese_font import ensure_japanese_font

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLUG = "racing-prediction-data"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "racing_prediction_kaggle_dataset"
RACES_DIR = ROOT / "data" / "processed" / "races"

# 学習コード・設定のデータセットに入れるもの: (データセット内のパス, ローカルのパス)
CODE_CONTENTS = [
    ("src/racing_prediction", ROOT / "src" / "racing_prediction"),
    ("configs", ROOT / "configs"),
    ("assets/fonts", ROOT / "assets" / "fonts"),  # グラフ用の日本語フォント (Kaggle には日本語フォントがない)
]
FINGERPRINT_SUFFIX = ".fingerprint.txt"


def fingerprint_file(slug: str) -> str:
    """データセットに入れる識別値のファイル名 (データセットごとに違う名前にして, ノートブックで区別できるようにする)."""
    return slug + FINGERPRINT_SUFFIX


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


@dataclass
class DatasetPart:
    """1 つの Kaggle データセットに入れる中身."""

    slug: str
    files: list[tuple[str, Path]]  # (データセット内のパス, ローカルのパス). 順序は固定

    @property
    def n_csv(self) -> int:
        return sum(rel.endswith(".csv") for rel, _ in self.files)


def _code_files() -> list[tuple[str, Path]]:
    files = []
    for prefix, local in CODE_CONTENTS:
        for path in sorted(local.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files.append((f"{prefix}/{path.relative_to(local).as_posix()}", path))
    return files


def race_shard(rel: Path) -> str | None:
    """races/ からの相対パス ({競馬場}/{開催日 YYYYMMDD}/{R}R.csv) -> 半年の区分 ("2024h1" など)."""
    date = next((p for p in rel.parts if re.fullmatch(r"\d{8}", p)), None)
    if date is None:
        return None
    return f"{date[:4]}h{1 if int(date[4:6]) <= 6 else 2}"


def dataset_parts(slug: str = DEFAULT_SLUG) -> list[DatasetPart]:
    """アップロードするデータセットの一覧 (先頭が学習コード・設定, 残りがレースの CSV を半年ごとに分けたもの)."""
    if not any(RACES_DIR.rglob("*.csv")):
        raise SystemExit(f"{RACES_DIR} に CSV がありません. 先にデータを抽出してください.")
    ensure_japanese_font()  # グラフ用の日本語フォント (なければダウンロード)

    shards: dict[str, list[tuple[str, Path]]] = {}
    for path in sorted(RACES_DIR.rglob("*.csv")):
        rel = path.relative_to(RACES_DIR)
        shard = race_shard(rel)
        if shard is None:
            raise SystemExit(f"開催日のフォルダ (YYYYMMDD) が見つからない CSV があります: {path}")
        shards.setdefault(shard, []).append((f"races/{rel.as_posix()}", path))
    parts = [DatasetPart(slug, _code_files())]
    parts += [DatasetPart(f"{slug}-races-{shard}", files) for shard, files in sorted(shards.items())]
    return parts


def compute_fingerprint(files: list[tuple[str, Path]]) -> str:
    """ファイルの (データセット内のパス, 中身) から識別値 (ハッシュ) を計算する.

    Kaggle 上のデータセットがローカルと同じ版か (= アップロードが必要か, 最新版が反映済みか) を確かめるために使う.
    改行コードの違い (Windows の CRLF) で値が変わらないよう, 改行は LF に揃えて計算する.
    """
    h = hashlib.sha256()
    for rel, path in files:
        h.update(rel.encode("utf-8") + b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def build_dataset_folder(part: DatasetPart, user: str, out_root: Path = DEFAULT_OUT) -> tuple[Path, str]:
    """out_root/<データセット名>/ に中身, 識別値のファイル, dataset-metadata.json を作る. (フォルダ, 識別値) を返す.

    識別値はコピーした後のフォルダの中身から計算するので, データ抽出 (collect_races) を並行して実行していて
    CSV が変わっていても, アップロードする中身と必ず一致する.
    """
    out = Path(out_root) / part.slug
    if out.exists():
        remove_tree(out)
    out.mkdir(parents=True)
    copied = []
    for rel, path in part.files:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        copied.append((rel, dest))
    fingerprint = compute_fingerprint(copied)
    (out / fingerprint_file(part.slug)).write_text(fingerprint + "\n", encoding="utf-8")
    metadata = {"title": part.slug, "id": f"{user}/{part.slug}", "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out, fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kaggle-user", help="Kaggle のユーザー名 (省略時はログイン情報から自動取得)")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help=f"データセット名の元 (デフォルト: {DEFAULT_SLUG})")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"出力先フォルダ (デフォルト: {DEFAULT_OUT})")
    args = parser.parse_args()

    user = resolve_kaggle_user(args.kaggle_user)
    for part in dataset_parts(args.slug):
        out, fingerprint = build_dataset_folder(part, user, Path(args.out))
        print(f"{out} (ファイル {len(part.files)} 件, 識別値 {fingerprint})")
        print(f'  初回      : kaggle datasets create -p "{out}" --dir-mode zip')
        print(f'  2 回目以降: kaggle datasets version -p "{out}" --dir-mode zip -m "update"')


if __name__ == "__main__":
    main()
