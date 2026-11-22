"""予測結果を通知用のテキスト (スマホで読みやすい短い形式) にする."""

from __future__ import annotations

import pandas as pd

TRACK_CONDITION_JA = {"good": "良", "yielding": "稍重", "soft": "重", "heavy": "不良"}

# 丸数字 (馬番 1〜20)
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def circled(n: int) -> str:
    return _CIRCLED[n - 1] if 1 <= n <= len(_CIRCLED) else f"({n})"


def format_race_report(race_info: dict, pred: pd.DataFrame, wide: pd.DataFrame, n_wide: int = 5) -> tuple[str, str]:
    """1 レース分の (件名, 本文) を作る.

    Args:
        race_info: venue, race_number, race_name, course, start_time を持つ dict (時刻表の 1 行)
        pred: predict_top3 の出力 (1 レース分)
        wide: wide_probabilities の出力
    """
    first = pred.iloc[0]
    start = pd.Timestamp(race_info["start_time"]).strftime("%H:%M")
    title = f"{race_info['venue']}{race_info['race_number']}R {race_info['race_name']}"
    subject = f"【予想】{title} {start}発走"

    condition = TRACK_CONDITION_JA.get(first.get("race_track_condition"), "未発表")
    lines = [f"{title}  {start}発走", f"{race_info.get('course', '')} / {int(first['n_horses'])}頭 / 馬場: {condition}", ""]

    if pred["horse_weight"].isna().all():
        lines += ["※ 馬体重が未発表の状態で予測しています", ""]

    lines.append("■ 3着以内に入る確率")
    for _, h in pred.sort_values("pred_rank").iterrows():
        weight = "" if pd.isna(h["horse_weight"]) else f" {int(h['horse_weight'])}kg"
        if weight and pd.notna(h["weight_diff"]):
            weight += f"({int(h['weight_diff']):+d})"
        lines.append(f"{int(h['pred_rank']):>2}. {circled(int(h['umaban']))} {h['horse_name']} {h['p_top3']:.0%}{weight}")

    lines += ["", f"■ ワイド候補 (的中確率 上位{n_wide})"]
    for _, w in wide.head(n_wide).iterrows():
        lines.append(f"{circled(int(w['umaban_1']))}-{circled(int(w['umaban_2']))} {w['p_wide']:.0%}  {w['horse_name_1']} / {w['horse_name_2']}")

    lines += ["", "※ 予測は参考情報です. 馬券の的中を保証するものではありません."]
    return subject, "\n".join(lines)
