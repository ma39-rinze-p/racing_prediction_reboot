"""予測結果を通知用のテキスト (スマホで読みやすい短い形式) にする."""

from __future__ import annotations

import re

import pandas as pd

TRACK_CONDITION_JA = {"good": "良", "yielding": "稍重", "soft": "重", "heavy": "不良"}
SURFACE_JA = {"turf": "芝", "dirt": "ダート", "steeple": "障害"}
COURSE_PREFIX_JA = {"芝": "芝", "ダ": "ダート", "障": "障害"}
WEEKDAY_JA = "月火水木金土日"

# 丸数字 (馬番 1〜20)
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def circled(n: int) -> str:
    return _CIRCLED[n - 1] if 1 <= n <= len(_CIRCLED) else f"({n})"


def _race_date(race_info: dict, pred: pd.DataFrame) -> pd.Timestamp | None:
    """開催日. 出馬表の開催日 (race_date) を優先し, なければ時刻表の発走時刻の日付."""
    value = pred["race_date"].iloc[0] if "race_date" in pred and pd.notna(pred["race_date"].iloc[0]) else None
    if value is None and race_info.get("start_time") is not None:
        value = race_info["start_time"]
    return None if value is None else pd.Timestamp(value).normalize()


def _course(race_info: dict, pred: pd.DataFrame) -> tuple[str, str]:
    """(芝・ダート, 距離). 時刻表の "ダ1400m" などから取り, なければ出馬表の芝・ダートを使う."""
    m = re.match(r"(芝|ダ|障)\D*(\d+)", str(race_info.get("course") or ""))
    if m:
        return COURSE_PREFIX_JA[m.group(1)], f"{int(m.group(2)):,}m"
    surface = pred["race_surface"].iloc[0] if "race_surface" in pred else None
    return SURFACE_JA.get(surface, "不明"), "不明"


def _box_label(umabans) -> str:
    return "".join(circled(int(u)) for u in umabans)


def _box_lines(
    boxes: pd.DataFrame | None, n_total: int | None, max_display: int, odds_note: str | None
) -> tuple[list[str], int | None]:
    """期待値がプラスのボックスの欄と, その数を返す. オッズが取れなかった場合の数は None.

    boxes: all_box_expected_values の結果に add_box_hit_rates で的中率を付けたもの (的中率の降順)
    """
    lines = ["■ 期待値プラスのワイド ボックス (的中率の高い順)"]
    if boxes is None:
        return lines + [f"  オッズを取得できなかったため計算していません ({odds_note or '不明'})"], None
    if boxes.empty:
        return lines + [f"  全{n_total:,}通りを計算しましたが, 期待値がプラスのボックスはありません (見送り推奨)"], 0

    shown = boxes.head(max_display)
    header = f"  全{n_total:,}通り中 {len(boxes):,}通りが期待値プラス"
    lines.append(header + (f" (上位{len(shown)}通りを表示)" if len(boxes) > len(shown) else ""))
    for i, b in enumerate(shown.itertuples(), start=1):
        lines += [
            f"{i:>2}. {_box_label(b.umabans)} {b.box_size}頭 {b.n_tickets}点/{b.stake:,}円",
            f"    期待払戻{b.expected_return:,.0f}円 回収率{b.return_rate:.0%} 的中率{b.hit_rate:.0%}",
        ]
    return lines, len(boxes)


def format_race_report(
    race_info: dict,
    pred: pd.DataFrame,
    wide: pd.DataFrame,
    n_wide: int = 5,
    boxes: pd.DataFrame | None = None,
    n_boxes_total: int | None = None,
    max_display: int = 20,
    odds_note: str | None = None,
    label: str | None = None,
    model_version: str | None = None,
) -> tuple[str, str]:
    """1 レース分の (件名, 本文) を作る.

    label: 通知のタイミング ("発走12分前" など). 件名と本文の先頭に入れる.

    Args:
        race_info: venue, race_number, race_name, course, start_time を持つ dict (時刻表の 1 行)
        pred: predict_win の出力 (1 レース分) に, シミュレーションした 3 着以内の確率 p_top3 を加えたもの
        wide: wide_probabilities の出力
        boxes: 期待値がプラスのボックス (all_box_expected_values + add_box_hit_rates. オッズが取れなかったときは None)
        n_boxes_total: 期待値を計算したボックスの総数
        max_display: メールに載せるボックスの最大数
        odds_note: オッズの取得時刻・状態など (本文に表示する)
    """
    first = pred.iloc[0]
    start = pd.Timestamp(race_info["start_time"]).strftime("%H:%M")
    race_date = _race_date(race_info, pred)
    weekday = WEEKDAY_JA[race_date.weekday()] if race_date is not None else ""
    date_ja = f"{race_date.year}年{race_date.month}月{race_date.day}日({weekday})" if race_date is not None else "不明"
    date_short = f"{race_date.month}/{race_date.day}({weekday})" if race_date is not None else ""
    venue = race_info.get("venue") or first.get("race_venue") or ""
    race_name = race_info.get("race_name") or "(レース名不明)"
    surface, distance = _course(race_info, pred)
    title = f"{venue}{race_info['race_number']}R {race_name}"

    condition = TRACK_CONDITION_JA.get(first.get("race_track_condition"), "未発表")
    lines = [f"【{label}のオッズで計算】", ""] if label else []
    lines += [
        "■ レース情報",
        f"開催日: {date_ja}",
        f"会場: {venue} {race_info['race_number']}R",
        f"レース名: {race_name}",
        f"コース: {surface} {distance}",
        f"発走: {start}",
        f"頭数: {int(first['n_horses'])}頭 / 馬場: {condition}",
        "",
    ]

    if pred["horse_weight"].isna().all():
        lines += ["※ 馬体重が未発表の状態で予測しています", ""]
    if race_info.get("win_odds_note"):
        lines += [f"※ 単勝オッズ ({race_info['win_odds_note']}) を出発点にして予測しています", ""]

    box_lines, n_positive = _box_lines(boxes, n_boxes_total, max_display, odds_note)
    lines += box_lines + [""]
    if boxes is not None and odds_note:
        lines += [f"※ オッズは {odds_note}. 締切までに変わります", ""]

    lines.append("■ 1着になる確率 (カッコ内は3着以内に入る確率)")
    for _, h in pred.sort_values("pred_rank").iterrows():
        weight = "" if pd.isna(h["horse_weight"]) else f" {int(h['horse_weight'])}kg"
        if weight and pd.notna(h["weight_diff"]):
            weight += f"({int(h['weight_diff']):+d})"
        top3 = f" ({h['p_top3']:.0%})" if "p_top3" in h and pd.notna(h["p_top3"]) else ""
        lines.append(f"{int(h['pred_rank']):>2}. {circled(int(h['umaban']))} {h['horse_name']} {h['p_win']:.1%}{top3}{weight}")

    lines += ["", f"■ ワイド的中確率 (上位{n_wide})"]
    for _, w in wide.head(n_wide).iterrows():
        lines.append(f"{circled(int(w['umaban_1']))}-{circled(int(w['umaban_2']))} {w['p_wide']:.0%}  {w['horse_name_1']} / {w['horse_name_2']}")

    lines += [
        "",
        "※ ボックスの期待払戻 = 含まれる各組の (確率 × 最低オッズ × 100円) の合計. 確率はモデルの確率と",
        "  オッズから逆算した市場の確率を組み合わせたもの (オッズ50倍超・モデル確率5%未満の組は市場の確率のみ.",
        "  configs/betting.yaml). 的中率はボックス内の2頭以上が3着以内に入る確率 (モデル).",
        "※ モデルは各馬が1着になる確率を予測する (レース内の合計は100%). 3着以内・ワイドの確率は,",
        "  1着の確率を各馬の強さとして着順をシミュレーションして求めたもの.",
        "※ 予測は参考情報であり, 馬券の的中・利益を保証するものではありません.",
        *( [f"※ モデル: {model_version}"] if model_version else [] ),
    ]
    course_short = surface if distance == "不明" else f"{surface}{distance}"
    ev_label = "" if n_positive is None else (f" 期待値+ボックス{n_positive:,}通り" if n_positive else " 期待値+なし")
    prefix = f"【予想・{label}】" if label else "【予想】"
    subject = f"{prefix}{date_short} {title} {course_short} {start}発走{ev_label}"
    return subject, "\n".join(lines)
