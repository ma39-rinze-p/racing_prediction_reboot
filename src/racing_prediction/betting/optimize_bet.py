"""optimize_bet

ワイドの期待値の計算と, 出走馬で作れる全てのボックスの期待値の計算.

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
    出走馬 (オッズがある馬) で作れる全てのボックス (2 頭〜全頭. 18 頭立てなら約 26 万通り) の期待値を計算する.
    期待払戻 = Σ (各組の p_used × 最低オッズ) × 100 円   (期待値は足し算できるので, 組同士が独立でなくても正確)
    的中率   = k 頭のうち 2 頭以上が 3 着以内に入る確率 (モデルの 1 着の確率から着順をシミュレーションして計算)

注意: 期待値は使う確率が正しいことを前提にしている. また推論時点 (発走 45 分前ごろ) のオッズは締切までに変わる.
"""

from __future__ import annotations

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


def all_box_expected_values(race: pd.DataFrame, pair_ev: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """出走馬 (オッズがある馬) で作れる全てのボックス (2 頭〜全頭) の期待値を計算し, 期待値がプラスのものを返す.

    ボックスの期待払戻 (100 円あたり) は, ボックスに含まれる全組み合わせの (p_used × 最低オッズ) の合計.
    全ボックスを 0/1 の行列 X (1 行 = 1 つのボックス) で表し, 組み合わせの期待利益の行列 V を使って
    期待利益 = x^T V x / 2 を一度に計算する (18 頭立てで約 26 万通りでも 1 秒かからない).

    Args:
        race: predict_win の出力 (1 レース分)
        pair_ev: wide_expected_values の出力

    Returns:
        (期待値がプラスのボックスの DataFrame, 計算したボックスの総数).
        DataFrame の列: box_size, umabans (馬番のタプル), n_tickets (点数), stake (購入額), expected_return (期待払戻),
        expected_profit (期待利益), return_rate (期待回収率). 期待回収率の降順.
    """
    columns = ["box_size", "umabans", "n_tickets", "stake", "expected_return", "expected_profit", "return_rate"]
    horses = sorted(set(pair_ev["umaban_1"].astype(int)) | set(pair_ev["umaban_2"].astype(int)))
    n = len(horses)
    if n < 2:
        return pd.DataFrame(columns=columns), 0
    pos = {u: i for i, u in enumerate(horses)}

    # 組み合わせごとの 100 円あたりの期待払戻 (= p_used × 最低オッズ). オッズがない組は買えないので NaN
    exp_return = np.full((n, n), np.nan)
    for r in pair_ev.itertuples():
        i, j = pos[int(r.umaban_1)], pos[int(r.umaban_2)]
        exp_return[i, j] = exp_return[j, i] = r.p_used * r.odds_min
    missing = np.isnan(exp_return) & ~np.eye(n, dtype=bool)
    profit = np.nan_to_num(exp_return - 1)  # 組み合わせごとの期待利益
    np.fill_diagonal(profit, 0)
    np.fill_diagonal(exp_return, 0)
    exp_return = np.nan_to_num(exp_return)

    # 全ボックス (2 頭以上) を 0/1 の行列で表す
    masks = np.arange(1 << n, dtype=np.int64)
    X = ((masks[:, None] >> np.arange(n)) & 1).astype(np.float64)
    sizes = X.sum(axis=1)
    X, sizes = X[sizes >= 2], sizes[sizes >= 2]
    if missing.any():  # オッズがない組を含むボックスは除く
        has_missing = ((X @ missing.astype(np.float64)) * X).sum(axis=1) > 0
        X, sizes = X[~has_missing], sizes[~has_missing]
    total = len(X)

    box_profit = ((X @ profit.astype(np.float64)) * X).sum(axis=1) / 2
    keep = box_profit > 1e-9
    X, sizes, box_profit = X[keep], sizes[keep], box_profit[keep]
    box_return = ((X @ exp_return.astype(np.float64)) * X).sum(axis=1) / 2
    n_tickets = sizes * (sizes - 1) / 2

    result = pd.DataFrame(
        {
            "box_size": sizes.astype(int),
            "umabans": [tuple(horses[i] for i in np.flatnonzero(x)) for x in X],
            "n_tickets": n_tickets.astype(int),
            "stake": (n_tickets * STAKE_PER_TICKET).astype(int),
            "expected_return": box_return * STAKE_PER_TICKET,
            "expected_profit": box_profit * STAKE_PER_TICKET,
            "return_rate": box_return / n_tickets,
        },
        columns=columns,
    )
    return result.sort_values(["return_rate", "box_size"], ascending=[False, True], ignore_index=True), total


def add_box_hit_rates(boxes: pd.DataFrame, race: pd.DataFrame, in_top: np.ndarray) -> pd.DataFrame:
    """ボックスごとの的中率 (ボックス内の 2 頭以上が 3 着以内に入る確率) を hit_rate 列として追加する.

    race と in_top (simulate_top3(race) の結果) は同じ行の順番であること.

    シミュレーションを 1 つずつ数えると, ボックスが数万通りあるときに遅いため, 次の式で全ボックスを一度に計算する.
    各シミュレーションでは 3 着以内の 3 頭 T が決まり, ボックス B と T の共通部分の頭数を a とすると
        [a >= 2] = (B 内の 2 頭の組で, 2 頭とも T に入る組の数) - 2 × (B 内の 3 頭の組で, T と一致する組の数)
    (a = 2 なら 1 - 0 = 1, a = 3 なら 3 - 2 = 1, a <= 1 なら 0). これを全シミュレーションで平均すると
        的中率 = Σ_{B 内の 2 頭} P(2 頭とも 3 着以内) - 2 × Σ_{B 内の 3 頭} P(その 3 頭が 1〜3 着)
    となり, シミュレーションで数えた結果と完全に一致する.
    """
    out = boxes.copy()
    if out.empty:
        out["hit_rate"] = pd.Series(dtype=float)
        return out

    race = race.reset_index(drop=True)
    n = len(race)
    pos = {int(u): i for i, u in enumerate(race["umaban"])}
    bits = np.int64(1) << np.arange(n, dtype=np.int64)
    box_masks = np.array([sum(int(bits[pos[u]]) for u in box) for box in out["umabans"]], dtype=np.int64)
    X = ((box_masks[:, None] >> np.arange(n)) & 1).astype(np.float64)

    # 2 頭とも 3 着以内に入る確率 (シミュレーションでの割合)
    sims = in_top.astype(np.float64)
    pair_prob = sims.T @ sims / len(sims)
    np.fill_diagonal(pair_prob, 0)
    pair_term = ((X @ pair_prob) * X).sum(axis=1) / 2

    # 3 頭がそのまま 1〜3 着を占める確率 (シミュレーションに出てきた 3 頭の組ごと. 18 頭立てでも最大 816 通り)
    triple_masks, counts = np.unique((in_top.astype(np.int64) * bits).sum(axis=1), return_counts=True)
    triple_term = np.zeros(len(out))
    for t, c in zip(triple_masks, counts / len(in_top)):
        triple_term[(box_masks & t) == t] += c

    out["hit_rate"] = np.clip(pair_term - 2 * triple_term, 0, 1)
    return out
