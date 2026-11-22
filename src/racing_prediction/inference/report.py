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


def _ev_lines(wide_ev: pd.DataFrame | None, pred: pd.DataFrame, odds_note: str | None) -> tuple[list[str], int | None]:
    """期待値がプラスのワイドを全て (期待値の高い順に) 並べた欄と, その組数を返す.

    各組み合わせに, モデルの的中確率とオッズ (最低〜最高), 各馬の予測順位を併記する.
    オッズが取れなかった場合の組数は None.
    """
    lines = ["■ 期待値プラスのワイド (期待値の高い順)"]
    if wide_ev is None:
        return lines + [f"  オッズを取得できなかったため計算していません ({odds_note or '不明'})"], None

    positive = wide_ev[wide_ev["ev_min"] > 0]
    if positive.empty:
        return lines + ["  期待値がプラスの組み合わせはありません (見送り推奨)"], 0

    rank_of = dict(zip(pred["umaban"].astype(int), pred["pred_rank"].astype(int)))
    name_of = dict(zip(pred["umaban"].astype(int), pred["horse_name"]))
    lines.append(f"  {len(positive)}組 / 購入額 {len(positive) * 100:,}円 (各100円の場合)")
    for _, w in positive.iterrows():
        u1, u2 = int(w["umaban_1"]), int(w["umaban_2"])
        lines += [
            f"{circled(u1)}-{circled(u2)} 期待値{w['ev_min']:+.0%}",
            f"  モデル確率 {w['p_wide']:.1%} / オッズ {w['odds_min']:.1f}〜{w['odds_max']:.1f}倍",
            f"  {name_of.get(u1, '')}(予測{rank_of.get(u1, '-')}位) / {name_of.get(u2, '')}(予測{rank_of.get(u2, '-')}位)",
        ]
    return lines, len(positive)


def format_race_report(
    race_info: dict,
    pred: pd.DataFrame,
    wide: pd.DataFrame,
    n_wide: int = 5,
    wide_ev: pd.DataFrame | None = None,
    odds_note: str | None = None,
) -> tuple[str, str]:
    """1 レース分の (件名, 本文) を作る.

    Args:
        race_info: venue, race_number, race_name, course, start_time を持つ dict (時刻表の 1 行)
        pred: predict_top3 の出力 (1 レース分)
        wide: wide_probabilities の出力
        wide_ev: wide_expected_values の出力 (オッズが取れなかったときは None)
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
    lines = [
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

    ev_lines, n_positive = _ev_lines(wide_ev, pred, odds_note)
    lines += ev_lines + [""]
    if wide_ev is not None and odds_note:
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
        "※ 期待値 = 確率 × 最低オッズ − 1. 確率はモデルの確率とオッズから逆算した市場の確率を組み合わせたもの",
        "  (オッズ50倍超・モデル確率5%未満の組は市場の確率のみ. configs/betting.yaml) のため, モデル確率 × オッズとは一致しません.",
        "※ 予測は参考情報であり, 馬券の的中・利益を保証するものではありません.",
    ]
    course_short = surface if distance == "不明" else f"{surface}{distance}"
    ev_label = "" if n_positive is None else (f" 期待値+{n_positive}組" if n_positive else " 期待値+なし")
    subject = f"【予想】{date_short} {title} {course_short} {start}発走{ev_label}"
    return subject, "\n".join(lines)
