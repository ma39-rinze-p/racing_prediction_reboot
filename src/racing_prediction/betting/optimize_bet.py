"""optimize_bet

ワイドの期待値の計算と, 期待値がプラスになるボックス買いの組み合わせの選択.

期待値 (100 円あたりの期待利益) = 使う確率 × 最低オッズ − 1
    ワイドのオッズは「最低〜最高」の幅で発表され, 実際の払戻はその間になる.
    控えめに見積もるため, 最低オッズで計算する.

使う確率 (p_used):
    モデルの確率 (p_wide) だけで計算すると, モデルが過大評価している大穴 (オッズ数百倍など) の期待値が
    異常に大きくなる. そこで, オッズから逆算した市場の確率 (p_market) と組み合わせる.
        p_market: 各組み合わせの 1 / オッズ (最低と最高の中間) に比例し, 合計が 3 (= 当たりの組数) になるよう揃えた値
                  (控除率の分は, 揃える処理で取り除かれる)
        p_used  : p_wide^(1-w) × p_market^w   (w = market_weight. 対数での加重平均)
    さらに, オッズが高すぎる組み合わせ / モデルの確率が低すぎる組み合わせは, モデルを信用せず p_market だけを使う
    (この場合の期待値は控除率の分だけマイナスになる).

ボックス買い (k 頭を選び, その中の全 k(k-1)/2 組を 100 円ずつ買う):
    期待払戻 = Σ (各組の p_used × 最低オッズ) × 100 円   (期待値は足し算できるので, 組同士が独立でなくても正確)
    的中率   = k 頭のうち 2 頭以上が 3 着以内に入る確率 (モデルのシミュレーションから計算)

注意: 期待値は使う確率が正しいことを前提にしている. また推論時点 (発走 45 分前ごろ) のオッズは締切までに変わる.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from racing_prediction.config import BettingConfig

STAKE_PER_TICKET = 100  # 1 点あたりの購入額 (円)
WINNING_PAIRS = 3  # ワイドは 1 レースで 3 組が当たり (1-2 着, 1-3 着, 2-3 着)


def wide_expected_values(wide: pd.DataFrame, odds: pd.DataFrame, cfg: BettingConfig | None = None) -> pd.DataFrame:
    """ワイドの的中確率 (wide_probabilities の出力) とオッズ (fetch_wide_odds の odds) から期待値を計算する.

    Returns:
        umaban_1 < umaban_2 に揃えた組み合わせごとに,
        p_wide (モデル), p_market (市場), trust_model (モデルを使ったか), p_used (計算に使った確率),
        odds_min, odds_max, ev_min (最低オッズでの期待値), ev_max を, ev_min の降順で並べた DataFrame.
        オッズがない組み合わせ (取消など) は含めない.
    """
    cfg = cfg or BettingConfig()
    w = wide.copy()
    lo = w[["umaban_1", "umaban_2"]].min(axis=1)
    hi = w[["umaban_1", "umaban_2"]].max(axis=1)
    swap = w["umaban_1"] > w["umaban_2"]
    w.loc[swap, ["horse_name_1", "horse_name_2"]] = w.loc[swap, ["horse_name_2", "horse_name_1"]].to_numpy()
    w["umaban_1"], w["umaban_2"] = lo, hi

    merged = w.merge(odds[["umaban_1", "umaban_2", "odds_min", "odds_max"]], on=["umaban_1", "umaban_2"], how="inner")
    inv = 1 / ((merged["odds_min"] + merged["odds_max"]) / 2)
    merged["p_market"] = inv / inv.sum() * min(WINNING_PAIRS, len(merged))

    p_model = merged["p_wide"].clip(lower=1e-6)
    blended = np.exp((1 - cfg.market_weight) * np.log(p_model) + cfg.market_weight * np.log(merged["p_market"]))
    merged["trust_model"] = (merged["odds_min"] <= cfg.max_odds) & (merged["p_wide"] >= cfg.min_model_prob)
    merged["p_used"] = np.where(merged["trust_model"], blended, merged["p_market"])

    merged["ev_min"] = merged["p_used"] * merged["odds_min"] - 1
    merged["ev_max"] = merged["p_used"] * merged["odds_max"] - 1
    return merged.sort_values("ev_min", ascending=False, ignore_index=True)


def best_wide_boxes(
    race: pd.DataFrame,
    in_top: np.ndarray,
    pair_ev: pd.DataFrame,
    cfg: BettingConfig | None = None,
) -> pd.DataFrame:
    """頭数ごとに, ボックスの期待利益 (最低オッズで計算) が最大になる馬の組み合わせを探す.

    的中率が cfg.min_hit_rate 未満のボックスは選ばない.

    Args:
        race: predict_top3 の出力 (1 レース分). in_top と同じ行の順番であること
        in_top: simulate_top3(race) の結果
        pair_ev: wide_expected_values の出力

    Returns:
        1 行 = 1 つの頭数のボックス. box_size, umabans (馬番のタプル), horse_names, n_tickets (点数),
        stake (購入額), expected_return (期待払戻), expected_profit (期待利益), return_rate (期待回収率),
        hit_rate (的中率), positive_pairs (ボックス内で期待値がプラスの組数) を, 期待利益の降順で並べた DataFrame
    """
    cfg = cfg or BettingConfig()
    race = race.reset_index(drop=True)
    row_of = {int(u): i for i, u in enumerate(race["umaban"])}
    name_of = dict(zip(race["umaban"].astype(int), race["horse_name"]))
    exp_return_of = {  # 組み合わせ -> 100 円あたりの期待払戻 (= 使う確率 × 最低オッズ)
        (int(r.umaban_1), int(r.umaban_2)): r.p_used * r.odds_min for r in pair_ev.itertuples()
    }

    # オッズがある馬の中から, 3 着以内の確率が高い順に候補を選ぶ
    with_odds = {u for pair in exp_return_of for u in pair}
    candidates = [int(u) for u in race.sort_values("p_top3", ascending=False)["umaban"] if int(u) in with_odds]
    candidates = sorted(candidates[: cfg.max_candidates])

    rows = []
    for k in cfg.box_sizes:
        best = None
        for box in combinations(candidates, k):
            pairs = list(combinations(box, 2))
            if any(p not in exp_return_of for p in pairs):
                continue
            hit_rate = float((in_top[:, [row_of[u] for u in box]].sum(axis=1) >= 2).mean())
            if hit_rate < cfg.min_hit_rate:
                continue
            expected_return = sum(exp_return_of[p] for p in pairs)
            profit = expected_return - len(pairs)
            if best is None or profit > best["profit"]:
                best = {"box": box, "profit": profit, "expected_return": expected_return, "pairs": pairs, "hit_rate": hit_rate}
        if best is None:
            continue
        n_tickets = len(best["pairs"])
        rows.append(
            {
                "box_size": k,
                "umabans": best["box"],
                "horse_names": tuple(name_of[u] for u in best["box"]),
                "n_tickets": n_tickets,
                "stake": n_tickets * STAKE_PER_TICKET,
                "expected_return": best["expected_return"] * STAKE_PER_TICKET,
                "expected_profit": best["profit"] * STAKE_PER_TICKET,
                "return_rate": best["expected_return"] / n_tickets,
                "hit_rate": best["hit_rate"],
                "positive_pairs": sum(exp_return_of[p] > 1 for p in best["pairs"]),
            }
        )
    columns = ["box_size", "umabans", "horse_names", "n_tickets", "stake", "expected_return",
               "expected_profit", "return_rate", "hit_rate", "positive_pairs"]
    return pd.DataFrame(rows, columns=columns).sort_values("expected_profit", ascending=False, ignore_index=True)
