"""予測結果を通知用のテキスト (スマホで読みやすい短い形式) にする."""

from __future__ import annotations

import pandas as pd

TRACK_CONDITION_JA = {"good": "良", "yielding": "稍重", "soft": "重", "heavy": "不良"}

# 丸数字 (馬番 1〜20)
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def circled(n: int) -> str:
    return _CIRCLED[n - 1] if 1 <= n <= len(_CIRCLED) else f"({n})"


def _box_label(umabans) -> str:
    return "".join(circled(int(u)) for u in umabans)


def _box_lines(boxes: pd.DataFrame | None, odds_note: str | None) -> tuple[list[str], str | None]:
    """おすすめボックスの欄と, 件名に付ける短い文字列を作る."""
    lines = ["■ おすすめのワイド ボックス (期待値プラス)"]
    if boxes is None:
        lines += [f"  オッズを取得できなかったため計算していません ({odds_note or '不明'})"]
        return lines, None

    positive = boxes[boxes["expected_profit"] > 0]
    if positive.empty:
        lines += ["  期待値がプラスになるボックスはありません (見送り推奨)"]
        return lines, None

    best = positive.iloc[0]
    lines += [
        f"★ {_box_label(best['umabans'])} の{best['box_size']}頭ボックス ({best['n_tickets']}点 / {best['stake']:,.0f}円)",
        f"  期待払戻 {best['expected_return']:,.0f}円 (期待回収率 {best['return_rate']:.0%}) / 的中率 {best['hit_rate']:.0%}",
        "  " + " / ".join(f"{circled(int(u))}{n}" for u, n in zip(best["umabans"], best["horse_names"])),
    ]
    others = positive.iloc[1:4]
    if not others.empty:
        lines.append("  ほかの候補:")
        for _, b in others.iterrows():
            lines.append(
                f"  ・{_box_label(b['umabans'])} {b['box_size']}頭 ({b['n_tickets']}点/{b['stake']:,.0f}円) "
                f"回収率{b['return_rate']:.0%} 的中率{b['hit_rate']:.0%}"
            )
    return lines, f"★ボックス{_box_label(best['umabans'])}"


def _ev_lines(wide_ev: pd.DataFrame | None, n_ev: int) -> list[str]:
    """期待値プラスのワイド (組み合わせ単位) の欄."""
    if wide_ev is None:
        return []
    positive = wide_ev[wide_ev["ev_min"] > 0].head(n_ev)
    lines = ["■ 期待値プラスのワイド (最低オッズで計算)"]
    if positive.empty:
        return lines + ["  なし"]
    lines.append("  的中率は「計算に使った確率 (モデル/市場)」")
    for _, w in positive.iterrows():
        lines.append(
            f"{circled(int(w['umaban_1']))}-{circled(int(w['umaban_2']))} 期待値{w['ev_min']:+.0%} "
            f"的中{w['p_used']:.0%} ({w['p_wide']:.0%}/{w['p_market']:.0%}) オッズ{w['odds_min']:.1f}-{w['odds_max']:.1f}"
        )
    return lines


def format_race_report(
    race_info: dict,
    pred: pd.DataFrame,
    wide: pd.DataFrame,
    n_wide: int = 5,
    wide_ev: pd.DataFrame | None = None,
    boxes: pd.DataFrame | None = None,
    odds_note: str | None = None,
    n_ev: int = 8,
) -> tuple[str, str]:
    """1 レース分の (件名, 本文) を作る.

    Args:
        race_info: venue, race_number, race_name, course, start_time を持つ dict (時刻表の 1 行)
        pred: predict_top3 の出力 (1 レース分)
        wide: wide_probabilities の出力
        wide_ev: wide_expected_values の出力 (オッズが取れなかったときは None)
        boxes: best_wide_boxes の出力 (オッズが取れなかったときは None)
        odds_note: オッズの取得時刻・状態など (本文に表示する)
    """
    first = pred.iloc[0]
    start = pd.Timestamp(race_info["start_time"]).strftime("%H:%M")
    title = f"{race_info['venue']}{race_info['race_number']}R {race_info['race_name']}"

    condition = TRACK_CONDITION_JA.get(first.get("race_track_condition"), "未発表")
    lines = [f"{title}  {start}発走", f"{race_info.get('course', '')} / {int(first['n_horses'])}頭 / 馬場: {condition}", ""]

    if pred["horse_weight"].isna().all():
        lines += ["※ 馬体重が未発表の状態で予測しています", ""]

    box_lines, subject_box = _box_lines(boxes, odds_note)
    lines += box_lines + [""]
    ev_lines = _ev_lines(wide_ev, n_ev)
    if ev_lines:
        lines += ev_lines + [""]
    if boxes is not None and odds_note:
        lines += [f"※ オッズは {odds_note}. 締切までに変わります", ""]

    lines.append("■ 3着以内に入る確率")
    for _, h in pred.sort_values("pred_rank").iterrows():
        weight = "" if pd.isna(h["horse_weight"]) else f" {int(h['horse_weight'])}kg"
        if weight and pd.notna(h["weight_diff"]):
            weight += f"({int(h['weight_diff']):+d})"
        lines.append(f"{int(h['pred_rank']):>2}. {circled(int(h['umaban']))} {h['horse_name']} {h['p_top3']:.0%}{weight}")

    lines += ["", f"■ ワイド的中確率 (上位{n_wide})"]
    for _, w in wide.head(n_wide).iterrows():
        lines.append(f"{circled(int(w['umaban_1']))}-{circled(int(w['umaban_2']))} {w['p_wide']:.0%}  {w['horse_name_1']} / {w['horse_name_2']}")

    lines += [
        "",
        "※ 期待値はモデルの確率とオッズから逆算した市場の確率を組み合わせた見積もりです (configs/betting.yaml).",
        "※ 予測は参考情報であり, 馬券の的中・利益を保証するものではありません.",
    ]
    subject = f"【予想】{title} {start}発走" + (f" {subject_box}" if subject_box else "")
    return subject, "\n".join(lines)
