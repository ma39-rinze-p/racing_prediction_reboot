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


def _horses(umabans) -> str:
    return "".join(circled(int(u)) for u in umabans)


BET_TYPE_ORDER = [("win", "単勝"), ("place", "複勝"), ("wide", "ワイド"), ("trio", "3連複"), ("trifecta", "3連単")]


def bet_label(bet_type: str, method: str, groups) -> str:
    """買い方の表示 (例: "ボックス ⑩⑭⑧", "1着固定ながし 1着⑩ → 2・3着⑭⑧⑪")."""
    g = [tuple(x) for x in groups]
    sep = "→" if bet_type == "trifecta" else "-"
    if method == "1点":
        return f"1点 {sep.join(circled(int(x[0])) for x in g)}"
    if method == "複数頭":
        return f"{len(g[0])}頭 {_horses(g[0])} (各1点)"
    if method == "ボックス":
        return f"ボックス {_horses(g[0])}"
    if method == "ながし":
        return f"ながし 軸{_horses(g[0])} - 相手{_horses(g[1])}"
    if method == "軸1頭ながし":
        return f"軸1頭ながし 軸{_horses(g[0])} - 相手{_horses(g[1])}"
    if method == "軸2頭ながし":
        return f"軸2頭ながし 軸{_horses(g[0])}{_horses(g[1])} - 相手{_horses(g[2])}"
    if method == "1着固定ながし":
        return f"1着固定ながし 1着{_horses(g[0])} → 2・3着{_horses(g[1])}"
    if method == "1・2着固定ながし":
        return f"1・2着固定ながし 1着{_horses(g[0])} → 2着{_horses(g[1])} → 3着{_horses(g[2])}"
    if bet_type == "trifecta":
        return "フォーメーション " + " → ".join(f"{i}着{_horses(x)}" for i, x in enumerate(g, start=1))
    return "フォーメーション " + " / ".join(f"{i}列目{_horses(x)}" for i, x in enumerate(g, start=1))


def _rate(p: float) -> str:
    """的中率の表示 (1% 未満は小数 1 桁まで出す. 0% と表示すると当たらないように見えるため)."""
    return f"{p:.1%}" if p < 0.01 else f"{p:.0%}"


def _bet_detail(b) -> str:
    return f"{b.n_tickets}点/{b.stake:,}円 的中率{_rate(b.hit_rate)} 期待払戻{b.expected_return:,.0f}円 回収率{b.return_rate:.0%}"


def _bet_lines(
    recommendations: dict | None, max_display: int, min_hit_rate: float, odds_note: str | None
) -> tuple[list[str], int | None]:
    """券種ごとのおすすめの買い方の欄と, 買い方の総数を返す. オッズが取れなかった場合の数は None.

    recommendations: betting.strategies.recommend_bets の結果 ({券種: {"bets": DataFrame, ...}})
    """
    condition = "期待値プラス" + (f" かつ 的中率{min_hit_rate:.0%}以上" if min_hit_rate > 0 else "")
    lines = [f"■ おすすめの買い方 ({condition}. 券種ごとに的中率の高い順)"]
    if recommendations is None:
        return lines + [f"  オッズを取得できなかったため計算していません ({odds_note or '不明'})"], None
    total = 0
    for bet_type, name_ja in BET_TYPE_ORDER:
        if bet_type not in recommendations:
            continue
        r = recommendations[bet_type]
        bets = r["bets"]
        if r.get("error"):
            lines += [f"【{name_ja}】{r['error']}"]
            continue
        if bets.empty:
            reason = (f"期待値プラスは{r['n_positive_ev']:,}通りあるが, 的中率{min_hit_rate:.0%}以上はなし"
                      if r.get("n_positive_ev") else "期待値プラスの買い方なし")
            lines += [f"【{name_ja}】なし ({r['n_evaluated']:,}通りを計算. {reason})"]
            continue
        total += len(bets)
        shown = bets.head(max_display)
        more = f" (上位{len(shown)}通りを表示)" if len(bets) > len(shown) else ""
        lines.append(f"【{name_ja}】{len(bets):,}通り{more}")
        for i, b in enumerate(shown.itertuples(), start=1):
            lines += [f"{i:>2}. {bet_label(bet_type, b.method, b.groups)}", f"    {_bet_detail(b)}"]
    return lines, total


def _max_return_lines(recommendations: dict | None) -> list[str]:
    """券種ごとの, 期待回収率が最大の買い方の欄 (おすすめの中で回収率が最も高いもの. 同じなら的中率が高い方)."""
    if recommendations is None:
        return []
    lines = ["■ 回収率が最大の買い方 (券種ごと. 的中率は低いことが多いので注意)"]
    best_all = None
    for bet_type, name_ja in BET_TYPE_ORDER:
        r = recommendations.get(bet_type)
        if r is None or r.get("error") or r["bets"].empty:
            continue
        b = r["bets"].sort_values(["return_rate", "hit_rate"], ascending=False).iloc[0]
        lines += [f"【{name_ja}】{bet_label(bet_type, b.method, b.groups)}", f"    {_bet_detail(b)}"]
        if best_all is None or b.return_rate > best_all[1].return_rate:
            best_all = (name_ja, b)
    if best_all is None:
        return lines + ["  期待値プラスの買い方がありません", ""]
    lines.append(f"→ 全券種で回収率が最大: 【{best_all[0]}】{bet_label(best_all[1].bet_type, best_all[1].method, best_all[1].groups)}"
                 f" (回収率{best_all[1].return_rate:.0%})")
    return lines + [""]


OBJECTIVE_JA = {
    "profit": "収支プラス重視 (収支がプラスになる確率が最も高い)",
    "no_miss": "共倒れ回避重視 (当たれば必ず収支プラスの中で, 全部外れる確率が最も低い)",
    "half_back": "大負け回避重視 (購入額の半分以上が戻る確率が最も高い)",
}


def _portfolio_lines(portfolios: list[tuple[str, dict | None]], settings: tuple[int, int] | None) -> list[str]:
    """リスクを分散した組み合わせの欄 (目的ごとの betting.strategies.build_portfolio の結果). 設定で無効なら空."""
    if settings is None or not portfolios:
        return []
    max_stake, max_bets = settings
    lines = [f"■ リスクを分散した組み合わせ (予算{max_stake:,}円・最大{max_bets}つ. 全体の期待値はプラスのまま)"]
    naive_line = None
    shown: list[tuple[str, tuple]] = []  # すでに載せた組み合わせ (同じ組み合わせは繰り返さない)
    for objective, portfolio in portfolios:
        title = f"【{OBJECTIVE_JA.get(objective, objective)}】"
        sig = None if portfolio is None else tuple(zip(portfolio["bets"]["bet_type"], map(str, portfolio["bets"]["groups"])))
        same = next((o for o, s in shown if sig is not None and s == sig), None)
        if same is not None:
            label = OBJECTIVE_JA.get(same, same).split(" (")[0]
            lines += [title, f"  {label}と同じ組み合わせ"]
            continue
        body, naive_line = _one_portfolio(portfolio, naive_line)
        lines += [title] + body
        if sig is not None:
            shown.append((objective, sig))
    if naive_line:
        lines += [naive_line]
    return lines + [""]


def _one_portfolio(portfolio: dict | None, naive_line: str | None) -> tuple[list[str], str | None]:
    if portfolio is None or portfolio["bets"].empty:
        return ["  組み合わせる買い方がありません"], naive_line
    lines = []
    for i, b in enumerate(portfolio["bets"].itertuples(), start=1):
        lines += [
            f"{i:>2}. 【{b.bet_type_ja}】{bet_label(b.bet_type, b.method, b.groups)}" + (" (押さえ)" if b.return_rate <= 1 else ""),
            f"    {b.n_tickets}点/{b.stake:,}円 的中率{_rate(b.hit_rate)} 回収率{b.return_rate:.0%}",
        ]

    def summary(m: dict) -> str:
        return (f"{m['n_tickets']}点/{m['stake']:,.0f}円 期待回収率{m['return_rate']:.0%} "
                f"収支プラスの確率{m['p_profit']:.0%} 全部外れる確率{m['p_all_miss']:.0%}")

    def key(m: dict) -> str:
        if m.get("key_horse") is None:
            return ""
        return f"{circled(int(m['key_horse']))}が3着以内に来なかったときに1つ以上当たる確率{m['p_hit_without_key']:.0%}"

    m = portfolio["metrics"]
    lines += [f"  合計 {summary(m)}"]
    if key(m):
        lines += [f"  {key(m)}"]
    naive = portfolio.get("naive") or {}
    if naive_line is None and naive.get("bets") is not None and len(naive["bets"]) > 1 and naive.get("metrics"):
        n = naive["metrics"]
        naive_line = (f"(参考: 券種ごとのおすすめの1番目を全部買った場合 {summary(n)}" + (f" / {key(n)})" if key(n) else ")"))
    return lines, naive_line


def format_race_report(
    race_info: dict,
    pred: pd.DataFrame,
    wide: pd.DataFrame,
    n_wide: int = 5,
    recommendations: dict | None = None,
    min_hit_rate: float = 0.0,
    max_display: int = 5,
    odds_note: str | None = None,
    label: str | None = None,
    model_version: str | None = None,
    portfolios: list[tuple[str, dict | None]] | None = None,
    portfolio_settings: tuple[int, int] | None = None,
) -> tuple[str, str]:
    """1 レース分の (件名, 本文) を作る.

    label: 通知のタイミング ("発走12分前" など). 件名と本文の先頭に入れる.

    Args:
        race_info: venue, race_number, race_name, course, start_time を持つ dict (時刻表の 1 行)
        pred: predict_win の出力 (1 レース分) に, シミュレーションした 3 着以内の確率 p_top3 を加えたもの
        wide: wide_probabilities の出力
        recommendations: 券種ごとのおすすめの買い方 (betting.strategies.recommend_bets. オッズが取れなかったときは None)
        min_hit_rate: メールに載せる買い方の的中率の下限 (表示用)
        max_display: 券種ごとにメールに載せる買い方の最大数
        odds_note: オッズの取得時刻・状態など (本文に表示する)
        portfolios: リスクを分散した組み合わせ [(目的, betting.strategies.build_portfolio の結果)]
        portfolio_settings: (予算, 買い方の最大数). None なら組み合わせの欄を出さない
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

    bet_lines, n_positive = _bet_lines(recommendations, max_display, min_hit_rate, odds_note)
    lines += bet_lines + [""]
    lines += _max_return_lines(recommendations)
    if recommendations is not None:
        lines += _portfolio_lines(portfolios or [], portfolio_settings)
    if recommendations is not None and odds_note:
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
        "※ 期待払戻 = 含まれる各点の (確率 × オッズ × 100円) の合計 (複勝・ワイドは最低オッズ). 確率はモデルの確率と",
        "  オッズから逆算した市場の確率を組み合わせたもの (オッズが高すぎる・モデルの確率が低すぎる点は市場の確率のみ.",
        "  configs/betting.yaml). 的中率は含まれる点のどれかが当たる確率 (モデル). 回収率 = 期待払戻 / 購入額.",
        "※ リスクを分散した組み合わせ: 着順のシミュレーションの各結果で, 選んだ買い方の払戻を合計して数えた確率.",
        "  同じ馬に頼った買い方ばかりだと, その馬が来なかったときに全部外れる (共倒れ) ため, 外れ方の違う買い方を選ぶ.",
        "  (押さえ) は単独では期待値がマイナスの買い方 (モデルが評価する馬が来なかったときの保険. 全体の期待値はプラスのまま).",
        "※ 1着の確率は1着のモデル (レース内の合計は100%). 3着以内・ワイド・3連複・3連単の確率は, 1着のモデルと",
        "  3着以内のモデル (あれば) を併用して着順をシミュレーションして求めたもの.",
        "※ 予測は参考情報であり, 馬券の的中・利益を保証するものではありません.",
        *( [f"※ モデル: {model_version}"] if model_version else [] ),
    ]
    course_short = surface if distance == "不明" else f"{surface}{distance}"
    ev_label = "" if n_positive is None else (f" おすすめ{n_positive:,}通り" if n_positive else " おすすめなし")
    prefix = f"【予想・{label}】" if label else "【予想】"
    subject = f"{prefix}{date_short} {title} {course_short} {start}発走{ev_label}"
    return subject, "\n".join(lines)
