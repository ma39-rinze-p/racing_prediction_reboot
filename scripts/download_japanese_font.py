"""グラフ用の日本語フォント (Noto Sans JP) を assets/fonts/ にダウンロードする.

Kaggle の実行環境には日本語フォントがなく, グラフの日本語が文字化け (□) になるため,
このフォントをデータセットに同梱して使う. prepare_kaggle_dataset.py / train_on_kaggle.py は,
フォントがなければ自動でこのスクリプトの処理を呼ぶので, 通常は手動で実行する必要はない.

Noto Sans JP は SIL Open Font License 1.1 (再配布可) のフォント. ライセンス文書 (OFL.txt) も一緒に保存する.
ファイルが大きい (約 10MB) ため git では管理しない (.gitignore の assets/fonts/).

使い方 (リポジトリのルートで実行):
    python scripts/download_japanese_font.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"
BASE_URL = "https://github.com/google/fonts/raw/main/ofl/notosansjp"
FILES = {
    "NotoSansJP-VariableFont_wght.ttf": f"{BASE_URL}/NotoSansJP%5Bwght%5D.ttf",
    "OFL.txt": f"{BASE_URL}/OFL.txt",
}


def ensure_japanese_font() -> Path:
    """フォントがなければダウンロードする. フォントのフォルダを返す."""
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = FONT_DIR / name
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"日本語フォントをダウンロードしています: {name}")
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as f:
            f.write(response.read())
        tmp.replace(path)
    return FONT_DIR


def main() -> None:
    folder = ensure_japanese_font()
    for p in sorted(folder.iterdir()):
        print(f"  {p.name} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
