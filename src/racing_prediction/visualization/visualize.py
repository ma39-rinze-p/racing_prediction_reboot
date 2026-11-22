"""visualize

グラフ (matplotlib) で日本語を文字化けさせずに表示するための設定.

Kaggle の実行環境には日本語フォントがないため, リポジトリの assets/fonts/ に置いた
Noto Sans JP (SIL Open Font License) をデータセットに同梱して使う
(scripts/download_japanese_font.py でダウンロードする).
ローカル (Windows) では assets/fonts/ がなくても, Yu Gothic / Meiryo などを使う.

    >>> font = setup_japanese_font([Path("assets/fonts")])
    >>> font   # 日本語を表示できるフォント名. 見つからなければ None (この場合は英語で描く)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
from matplotlib import font_manager

# 優先して使うフォント (上から順に, 見つかったものを使う)
JAPANESE_FONT_CANDIDATES = [
    "Noto Sans JP",  # assets/fonts に同梱するフォント
    "Noto Sans CJK JP",
    "IPAexGothic",
    "IPAGothic",
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "Hiragino Sans",
    "TakaoGothic",
]

_TEST_CHARS = "あ芝着順"  # このフォントで日本語が表示できるかの確認用


def _supports_japanese(font_path: str) -> bool:
    try:
        from matplotlib.ft2font import FT2Font

        font = FT2Font(font_path)
        return all(font.get_char_index(ord(c)) != 0 for c in _TEST_CHARS)
    except Exception:  # noqa: BLE001 - 読めないフォントは使わない
        return False


def setup_japanese_font(font_dirs: list[Path] | None = None) -> str | None:
    """日本語を表示できるフォントを matplotlib に設定し, そのフォント名を返す.

    font_dirs にあるフォントファイル (.ttf / .otf) を登録してから, 候補を順に探す.
    日本語を表示できるフォントが見つからなければ何もせず None を返す
    (呼び出し側は英語の表記でグラフを描けば文字化けしない).
    """
    # 同梱の Noto Sans JP は可変フォントのため "font weight normal が見つからない" という警告が出るが,
    # 表示には問題ないので出さない
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    for d in font_dirs or []:
        for path in sorted(Path(d).glob("*")) if Path(d).exists() else []:
            if path.suffix.lower() in (".ttf", ".otf"):
                font_manager.fontManager.addfont(str(path))

    available = {f.name: f.fname for f in font_manager.fontManager.ttflist}
    for name in JAPANESE_FONT_CANDIDATES:
        if name in available and _supports_japanese(available[name]):
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け対策
            return name
    return None
