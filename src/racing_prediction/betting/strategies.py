"""strategies

単勝・複勝・ワイド・3 連複・3 連単のそれぞれで, 1 点買い・ボックス・ながし・フォーメーションの買い方を作り,
期待値と的中率を計算して「期待値がプラス」(と, 設定すれば「的中率が min_hit_rate 以上」) の買い方を選ぶ.

確率:
    着順のシミュレーション (predict_model.simulate_orders. 1 着のモデルと 3 着以内のモデルを併用) の結果から,
    各券種の組み合わせ (馬券 1 点) が当たる確率 (p_model) を数える (単勝は p_win をそのまま使う).
期待値 (100 円あたり) = 使う確率 (p_used) × オッズ − 1
    オッズに幅がある券種 (複勝・ワイド) は, 控えめに最低オッズで計算する.
    p_used: モデルの確率と「オッズから逆算した市場の確率 (p_market)」を対数で加重平均したもの
        p_used = p_model^(1-w) × p_market^w   (w = 券種ごとの market_weight)
        p_market = 1 / オッズ (幅があれば最低と最高の中間) に比例し, 合計が当たりの組数になるよう揃えた値
        オッズが max_odds より高い組 / モデルの確率が min_model_prob より低い組は, モデルを信用せず p_market だけを使う
        (モデルが過大評価している大穴の期待値が異常に大きくなるのを防ぐ. この場合の期待値は控除率の分マイナス)
買い方 (複数点をまとめて買う. 1 点 100 円):
    期待払戻 = 含まれる各点の p_used × オッズ × 100 円の合計 (期待値は足し算できる)
    的中率   = 含まれる点のどれかが当たる確率 (モデルの確率. シミュレーションで数える)
    1 点買いは全ての組み合わせ, それ以外は 3 着以内の確率が高い順の上位 candidates 頭で作る:
        単勝・複勝   : 1 点 / 複数頭 (選んだ馬を 1 点ずつ)
        ワイド       : 1 点 / ボックス / ながし (軸 1 頭 - 相手) / フォーメーション (1 列目 × 2 列目)
        3 連複       : 1 点 / ボックス / 軸 1 頭ながし / 軸 2 頭ながし / フォーメーション (1〜3 列目)
        3 連単       : 1 点 / ボックス / 1 着固定ながし / 1・2 着固定ながし / フォーメーション (1〜3 着)
    同じ点の組み合わせになる買い方は, 簡単な方 (1 点 → ボックス → ながし → フォーメーション) だけを残す.

注意: 期待値はモデルの確率が正しいことを前提にしている. また推論時点のオッズは締切までに変わる.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product

import numpy as np
import pandas as pd

from racing_prediction.config import BettingConfig, BetTypeConfig
from racing_prediction.data.scrape_odds import BetOdds

STAKE_PER_TICKET = 100  # 1 点あたりの購入額 (円)


@dataclass(frozen=True)
class BetType:
    name: str
    name_ja: str
    size: int  # 組み合わせの頭数
    ordered: bool  # 着順の順番まで当てるか (3 連単)
    exclusive: bool  # 1 レースで当たる点が 1 つだけか (単勝・3 連複・3 連単. 複勝・ワイドは複数当たる)


BET_TYPES = {
    "win": BetType("win", "単勝", 1, True, True),
    "place": BetType("place", "複勝", 1, True, False),
    "wide": BetType("wide", "ワイド", 2, False, False),
    "trio": BetType("trio", "3連複", 3, False, True),
    "trifecta": BetType("trifecta", "3連単", 3, True, True),
}

# 買い方の名前 (メールの表示にも使う). 同じ点の組み合わせになったときは, この順で前のものを残す
METHODS = ["1点", "複数頭", "ボックス", "ながし", "軸1頭ながし", "軸2頭ながし", "1着固定ながし", "1・2着固定ながし", "フォーメーション"]

RESULT_COLUMNS = ["bet_type", "bet_type_ja", "method", "groups", "n_tickets", "stake", "expected_return",
                  "expected_profit", "return_rate", "hit_rate"]


def n_place_winners(n_horses: int) -> int:
    """複勝が当たる着順 (8 頭以上なら 3 着まで, 5〜7 頭なら 2 着まで, 4 頭以下は発売なし = 0)."""
    return 3 if n_horses >= 8 else 2 if n_horses >= 5 else 0


def _winning_tickets(bet: BetType, n_horses: int) -> int:
    """1 レースで当たる点の数 (市場の確率の合計をこれに揃える)."""
    return {"win": 1, "place": n_place_winners(n_horses), "wide": 3, "trio": 1, "trifecta": 1}[bet.name]


# ---------------------------------------------------------------------------
# 1 点ごとの確率・期待値
# ---------------------------------------------------------------------------
def _canonical(combo: tuple, bet: BetType) -> tuple:
    return tuple(combo) if bet.ordered else tuple(sorted(combo))


def model_ticket_probs(bet: BetType, race: pd.DataFrame, orders: np.ndarray) -> dict[tuple, float]:
    """着順のシミュレーション (simulate_orders の結果. race の行番号) から, 各点が当たる確率を数える.

    単勝は p_win (1 着のモデルの確率) をそのまま使う. 1 回も当たらなかった点は含めない (確率 0).
    """
    umaban = race["umaban"].astype(int).to_numpy()
    n_sims = len(orders)
    if bet.name == "win":
        p = race["p_win"].to_numpy(dtype=float)
        return {(int(u),): float(v) for u, v in zip(umaban, p / p.sum())}
    if bet.name == "place":
        k = n_place_winners(len(race))
        if k == 0:
            return {}
        counts = np.bincount(orders[:, :k].ravel(), minlength=len(race)) / n_sims
        return {(int(umaban[i]),): float(c) for i, c in enumerate(counts) if c > 0}
    if orders.shape[1] < 3:
        return {}
    if bet.name == "wide":
        out: dict[tuple, float] = {}
        for a, b in ((0, 1), (0, 2), (1, 2)):
            pairs = np.sort(umaban[orders[:, [a, b]]], axis=1)
            keys, counts = np.unique(pairs, axis=0, return_counts=True)
            for key, c in zip(keys, counts):
                out[(int(key[0]), int(key[1]))] = out.get((int(key[0]), int(key[1])), 0.0) + c / n_sims
        return out
    triples = umaban[orders[:, :3]]
    if not bet.ordered:
        triples = np.sort(triples, axis=1)
    keys, counts = np.unique(triples, axis=0, return_counts=True)
    return {tuple(int(x) for x in key): c / n_sims for key, c in zip(keys, counts)}


def _capped_shares(weights: np.ndarray, total: float) -> np.ndarray:
    """weights に比例し, 合計が total になる確率. ただし 1 を超える点は 1 にし, 余った分をほかの点に比例して配り直す.

    複勝・ワイドは当たりが複数 (合計 3) なので, 単純に比例配分すると, 圧倒的な人気馬 (複勝 1.1 倍など) の確率が
    1 を超えてしまう (確率は 1 を超えないので, 期待払戻がオッズ × 100 円を超えるという矛盾が起きる).
    """
    w = np.asarray(weights, dtype=float)
    p = np.zeros_like(w)
    capped = np.zeros(len(w), dtype=bool)
    for _ in range(len(w) + 1):
        free = ~capped
        rest = total - capped.sum()
        if rest <= 0 or w[free].sum() <= 0:
            break
        p[free] = w[free] / w[free].sum() * rest
        p[capped] = 1.0
        over = free & (p > 1.0)
        if not over.any():
            break
        capped |= over
    return np.clip(p, 0.0, 1.0)


def ticket_table(bet: BetType, odds: pd.DataFrame, p_model: dict[tuple, float], cfg: BetTypeConfig, n_horses: int) -> pd.DataFrame:
    """発売中の各点 (オッズがある組み合わせ) の p_model, p_market, p_used, 期待値を計算する.

    列: combo, odds_min, odds_max, p_model, p_market, trust_model, p_used, exp_return (= p_used × 最低オッズ. 100 円あたりの期待払戻 / 100), ev
    """
    t = odds.copy()
    if t.empty:
        return t.assign(p_model=[], p_market=[], trust_model=[], p_used=[], exp_return=[], ev=[])
    t["combo"] = [_canonical(c, bet) for c in t["combo"]]
    t = t.drop_duplicates("combo").reset_index(drop=True)
    t["p_model"] = [p_model.get(c, 0.0) for c in t["combo"]]
    inv = 1 / ((t["odds_min"] + t["odds_max"]) / 2)
    # 合計を当たりの点数に揃える (1 点で 1 を超える分は, ほかの点に配り直す)
    t["p_market"] = _capped_shares(inv.to_numpy(), min(_winning_tickets(bet, n_horses), len(t)))
    blended = np.exp((1 - cfg.market_weight) * np.log(t["p_model"].clip(lower=1e-9))
                     + cfg.market_weight * np.log(t["p_market"].clip(lower=1e-12)))
    blended = np.clip(blended, 0.0, 1.0)  # 確率は 1 を超えない (= 期待払戻はオッズ × 100 円を超えない)
    t["trust_model"] = (t["odds_min"] <= cfg.max_odds) & (t["p_model"] >= cfg.min_model_prob)
    t["p_used"] = np.where(t["trust_model"], blended, t["p_market"])
    t["exp_return"] = t["p_used"] * t["odds_min"]
    t["ev"] = t["exp_return"] - 1
    return t


# ---------------------------------------------------------------------------
# 買い方 (点の集まり) の作成
# ---------------------------------------------------------------------------
def _subsets(n: int, min_size: int, within: int | None = None) -> np.ndarray:
    """0..n-1 のうち先頭 within 頭 (None なら全頭) から選んだ, min_size 頭以上の部分集合 (ビットマスク)."""
    m = n if within is None else min(within, n)
    masks = np.arange(1, 1 << m, dtype=np.int64)
    sizes = np.array([bin(int(x)).count("1") for x in masks])
    return masks[sizes >= min_size]


def _specs(bet: BetType, n: int) -> list[tuple[str, np.ndarray]]:
    """候補の馬 n 頭 (3 着以内の確率が高い順の 0..n-1) で作る買い方. [(買い方の名前, 列ごとのビットマスク (件数 × 頭数))]."""
    single = [np.int64(1) << np.int64(i) for i in range(n)]
    specs: list[tuple[str, np.ndarray]] = []

    def add(name, rows):
        rows = [r for r in rows if all(int(x) for x in r)]
        if rows:
            specs.append((name, np.array(rows, dtype=np.int64)))

    if bet.size == 1:
        add("複数頭", [(m,) for m in _subsets(n, 2)])
        return specs
    if bet.size == 2:
        add("ボックス", [(m, m) for m in _subsets(n, 3)])
        add("ながし", [(single[a], p) for a in range(n) for p in _subsets(n, 2) if not (p >> a) & 1])
        add("フォーメーション", [(m1, m2) for m1 in _subsets(n, 2, 4) for m2 in _subsets(n, 2) if m1 != m2])
        return specs
    if n < 3:
        return specs
    if bet.ordered:
        add("ボックス", [(m, m, m) for m in _subsets(n, 3)])
        add("1着固定ながし", [(single[a], p, p) for a in range(n) for p in _subsets(n, 2) if not (p >> a) & 1])
        add("1・2着固定ながし", [(single[a], single[b], p) for a in range(n) for b in range(n) if a != b
                                for p in _subsets(n, 1) if not ((p >> a) & 1 or (p >> b) & 1)])
    else:
        add("ボックス", [(m, m, m) for m in _subsets(n, 4)])
        add("軸1頭ながし", [(single[a], p, p) for a in range(n) for p in _subsets(n, 3) if not (p >> a) & 1])
        add("軸2頭ながし", [(single[a], single[b], p) for a, b in combinations(range(n), 2)
                           for p in _subsets(n, 2) if not ((p >> a) & 1 or (p >> b) & 1)])
    # フォーメーション: 1 列目 (1 着) は上位 3 頭から, 2 列目は上位 5 頭から, 3 列目は候補全体から選ぶ
    m1s, m2s, m3s = _subsets(n, 1, 3), _subsets(n, 1, 5), _subsets(n, 1)
    specs.append(("フォーメーション", np.array(np.meshgrid(m1s, m2s, m3s, indexing="ij")).reshape(3, -1).T))
    return specs


def _membership(cols: np.ndarray, tickets: np.ndarray, ordered: bool) -> np.ndarray:
    """買い方 (列ごとのビットマスク) に, 各点 (候補の馬の番号の組) が含まれるか. (買い方の数 × 点の数) の bool."""
    size = tickets.shape[1]
    perms = [tuple(range(size))] if ordered else list(permutations(range(size)))
    out = np.zeros((len(cols), len(tickets)), dtype=bool)
    for perm in perms:
        ok = np.ones_like(out)
        for col, pos in enumerate(perm):
            ok &= ((cols[:, [col]] >> tickets[None, :, pos]) & 1).astype(bool)
        out |= ok
    return out


def _groups(cols: np.ndarray, horses: list[int]) -> list[tuple[tuple[int, ...], ...]]:
    """ビットマスクの列を, 列ごとの馬番のタプルにする (メールの表示用)."""
    return [tuple(tuple(horses[i] for i in range(len(horses)) if (int(m) >> i) & 1) for m in row) for row in cols]


# ---------------------------------------------------------------------------
# 券種ごとの計算
# ---------------------------------------------------------------------------
def evaluate_bet_type(
    bet_type: str,
    race: pd.DataFrame,
    orders: np.ndarray,
    odds: BetOdds,
    cfg: BettingConfig,
) -> tuple[pd.DataFrame, dict]:
    """1 つの券種で, 全ての買い方の期待値と的中率を計算し, 条件 (期待値プラス, 的中率 min_hit_rate 以上,
    点数 max_tickets 以下) を満たすものを的中率の高い順に返す.

    race: 1 レース分の予測 (predict_win の出力に p_top3 を加えたもの. orders と同じ行の順)
    orders: simulate_orders の結果
    Returns:
        (条件を満たす買い方の DataFrame (RESULT_COLUMNS), 集計 {"n_evaluated", "n_positive_ev", "n_tickets_sold"})
    """
    bet = BET_TYPES[bet_type]
    type_cfg = cfg.bet_type(bet_type)
    race = race.reset_index(drop=True)
    tickets = ticket_table(bet, odds.odds, model_ticket_probs(bet, race, orders), type_cfg, len(race))
    stats = {"n_evaluated": 0, "n_positive_ev": 0, "n_tickets_sold": int(len(tickets))}
    if tickets.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS), stats

    rows = []
    # 1 点買い (全ての組み合わせ)
    single = tickets.assign(method="1点", groups=[tuple((u,) for u in c) for c in tickets["combo"]],
                            n_tickets=1, ret=tickets["exp_return"], hit=tickets["p_model"])
    rows.append(single[["method", "groups", "n_tickets", "ret", "hit"]])
    stats["n_evaluated"] += len(single)

    # 複数点の買い方 (3 着以内の確率が高い順の候補 candidates 頭で作る. オッズがない馬は除く)
    sold = {u for c in tickets["combo"] for u in c}
    order = race.assign(_p=race["p_top3"] if "p_top3" in race else race["p_win"]).sort_values("_p", ascending=False)
    horses = [int(u) for u in order["umaban"] if int(u) in sold][: cfg.candidates]
    pos = {u: i for i, u in enumerate(horses)}
    in_cand = tickets["combo"].map(lambda c: all(u in pos for u in c))
    cand = tickets[in_cand].reset_index(drop=True)
    if len(horses) >= max(2, bet.size) and len(cand):
        tix = np.array([[pos[u] for u in c] for c in cand["combo"]], dtype=np.int64)
        specs = _specs(bet, len(horses))
        if specs:
            names = np.concatenate([[name] * len(cols) for name, cols in specs])
            cols_all = np.concatenate([cols for _, cols in specs])
            member = _membership(cols_all, tix, bet.ordered)
            n_t = member.sum(axis=1)
            keep = (n_t >= 2) & (n_t <= cfg.max_tickets)
            # 同じ点の組み合わせは, 先に作った (簡単な) 買い方だけ残す
            _, first = np.unique(np.packbits(member, axis=1), axis=0, return_index=True)
            uniq = np.zeros(len(member), dtype=bool)
            uniq[first] = True
            keep &= uniq
            member, cols_all, names, n_t = member[keep], cols_all[keep], names[keep], n_t[keep]
            stats["n_evaluated"] += int(len(member))
            ret = member.astype(float) @ cand["exp_return"].to_numpy()
            # 的中率は, 期待回収率が「組み合わせの押さえの下限」以上の買い方だけ計算する (おすすめは期待値プラスのものだけ)
            floor = min(1.0, cfg.portfolio_min_bet_return) if cfg.portfolio_enabled else 1.0
            positive = ret / n_t >= floor
            member, cols_all, names, n_t, ret = member[positive], cols_all[positive], names[positive], n_t[positive], ret[positive]
            if len(member):
                hit = _hit_rates(bet, member, cand, race, orders, pos)
                multi = pd.DataFrame({"method": names, "groups": _groups(cols_all, horses), "n_tickets": n_t, "ret": ret, "hit": hit})
                rows.append(multi)

    allbets = pd.concat(rows, ignore_index=True)
    floor = min(1.0, cfg.portfolio_min_bet_return) if cfg.portfolio_enabled else 1.0
    allbets = allbets[allbets["ret"] / allbets["n_tickets"] >= floor]
    stats["n_positive_ev"] = int((allbets["ret"] / allbets["n_tickets"] > 1).sum())
    positive = allbets.assign(
        bet_type=bet_type, bet_type_ja=bet.name_ja,
        n_tickets=allbets["n_tickets"].astype(int),
        stake=allbets["n_tickets"].astype(int) * STAKE_PER_TICKET,
        expected_return=allbets["ret"] * STAKE_PER_TICKET,
        return_rate=allbets["ret"] / allbets["n_tickets"],
        hit_rate=allbets["hit"],
    )
    positive["expected_profit"] = positive["expected_return"] - positive["stake"]
    positive["_order"] = positive["method"].map({m: i for i, m in enumerate(METHODS)})
    positive = positive.sort_values(["hit_rate", "return_rate", "_order"], ascending=[False, False, True], ignore_index=True)
    positive = positive[RESULT_COLUMNS]
    # 期待回収率が押さえの下限以上の全ての買い方 (的中率の条件なし. リスクを分散した組み合わせの候補に使う)
    stats["positive"] = positive
    good = positive[(positive["return_rate"] > 1) & (positive["hit_rate"] >= cfg.min_hit_rate)].reset_index(drop=True)
    return good, stats


def _hit_rates(bet: BetType, member: np.ndarray, cand: pd.DataFrame, race: pd.DataFrame, orders: np.ndarray, pos: dict) -> np.ndarray:
    """買い方ごとの的中率 (含まれる点のどれかが当たる確率. シミュレーションで数える).

    1 レースで当たる点が 1 つだけの券種 (単勝・3 連複・3 連単) は, 点の確率の合計.
    複勝・ワイドは, シミュレーションの着順 (上位 3 頭) ごとに当たる点があるかを数える.
    """
    if bet.exclusive:
        return member.astype(float) @ cand["p_model"].to_numpy()
    umaban = race["umaban"].astype(int).to_numpy()
    top = orders[:, : (n_place_winners(len(race)) if bet.name == "place" else 3)]
    keys, counts = np.unique(np.sort(top, axis=1), axis=0, return_counts=True)
    prob = counts / len(orders)
    n_cand = len(pos)
    # 各着順で, 上位に入った候補の馬 (候補の番号. 候補外は -1)
    cand_idx = np.vectorize(lambda r: pos.get(int(umaban[r]), -1))(keys)
    in_top = np.zeros((len(keys), n_cand), dtype=bool)
    for j in range(cand_idx.shape[1]):
        ok = cand_idx[:, j] >= 0
        in_top[np.flatnonzero(ok), cand_idx[ok, j]] = True
    tix = np.array([[pos[u] for u in c] for c in cand["combo"]], dtype=np.int64)
    wins = in_top[:, tix].all(axis=2)  # (着順の種類 × 点) 当たりか
    hits = (member.astype(np.float32) @ wins.T.astype(np.float32)) > 0
    return hits.astype(float) @ prob


def recommend_bets(
    race: pd.DataFrame, orders: np.ndarray, odds: dict[str, BetOdds], cfg: BettingConfig
) -> dict[str, dict]:
    """有効な券種それぞれで, 条件を満たす買い方を計算する.

    Returns:
        {券種: {"bets": DataFrame, "n_evaluated": int, "n_positive_ev": int, "n_tickets_sold": int, "error": str | None}}
        オッズがない (発売前・発売なし) 券種は bets が空で error に理由が入る.
    """
    out = {}
    for name in cfg.enabled_bet_types():
        o = odds.get(name)
        if o is None or o.odds.empty:
            out[name] = {"bets": pd.DataFrame(columns=RESULT_COLUMNS), "n_evaluated": 0, "n_positive_ev": 0,
                         "n_tickets_sold": 0, "error": "オッズがありません (発売前・発売なしなど)"}
            continue
        bets, stats = evaluate_bet_type(name, race, orders, o, cfg)
        out[name] = {"bets": bets, **stats, "error": None}
    return out


# ---------------------------------------------------------------------------
# リスクを分散した組み合わせ (券種をまたいで, 外れ方の違う買い方を組み合わせる)
# ---------------------------------------------------------------------------
def bet_tickets(bet_type: str, groups) -> list[tuple]:
    """買い方 (列ごとの馬番) に含まれる点 (組み合わせ) の一覧."""
    bet = BET_TYPES[bet_type]
    out = {_canonical(c, bet) for c in product(*[tuple(g) for g in groups]) if len(set(c)) == len(c)}
    return sorted(out)


def _hit_tickets(bet_type: str, outcome: tuple, n_horses: int) -> list[tuple]:
    """着順 (1〜3 着の馬番) のときに当たる点."""
    a = tuple(int(x) for x in outcome)
    if bet_type == "win":
        return [a[:1]]
    if bet_type == "place":
        return [(x,) for x in a[: n_place_winners(n_horses)]]
    if bet_type == "wide":
        return [tuple(sorted(p)) for p in combinations(a[:3], 2)]
    if bet_type == "trio":
        return [tuple(sorted(a[:3]))]
    return [a[:3]]


def payout_matrix(
    bets: pd.DataFrame, race: pd.DataFrame, orders: np.ndarray, odds: dict[str, BetOdds]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """着順のシミュレーションに出てきた着順ごとに, 各買い方の払戻 (円. 最低オッズで計算) を計算する.

    Returns:
        (払戻 (買い方の数 × 着順の種類), 各着順の確率, 着順 (着順の種類 × 3 の馬番))
    """
    umaban = race["umaban"].astype(int).to_numpy()
    k = min(3, orders.shape[1])
    keys, counts = np.unique(umaban[orders[:, :k]], axis=0, return_counts=True)
    prob = counts / len(orders)
    pays = np.zeros((len(bets), len(keys)), dtype=np.float64)
    bets = bets.reset_index(drop=True)
    for name, rows in bets.groupby("bet_type").groups.items():
        bet = BET_TYPES[name]
        table = odds[name].odds
        combos = [_canonical(c, bet) for c in table["combo"]]
        col = {c: j for j, c in enumerate(combos)}
        n_t = len(combos)
        # 買い方ごとの, 各点の払戻 (その点が当たったときの払戻. 買い方に含まれない点は 0). 最後の列は「発売なし」用
        unit = np.zeros((len(rows), n_t + 1), dtype=np.float64)
        odds_min = table["odds_min"].to_numpy(dtype=float) * STAKE_PER_TICKET
        for r, i in enumerate(rows):
            js = [col[t] for t in bet_tickets(name, bets.at[i, "groups"]) if t in col]
            unit[r, js] = odds_min[js]
        # 着順ごとに当たる点の列番号 (着順の種類 × 当たる点の数. 発売なしの点は最後の列)
        hits = [[col.get(t, n_t) for t in _hit_tickets(name, tuple(o), len(race))] for o in keys]
        width = max(1, max(len(h) for h in hits))
        win = np.array([h + [n_t] * (width - len(h)) for h in hits], dtype=np.int64)
        pays[np.asarray(rows)] = unit[:, win].sum(axis=2)
    return pays, prob, keys


def portfolio_metrics(bets: pd.DataFrame, pays: np.ndarray, prob: np.ndarray, outcomes: np.ndarray) -> dict:
    """買い方の組み合わせの, 全体での収支の見込み.

    stake           : 購入額の合計
    expected_return : 期待払戻の合計 (各買い方の期待払戻の合計. 確率はモデルと市場の組み合わせ)
    return_rate     : 期待回収率 (expected_return / stake)
    p_profit        : 収支がプラス (払戻の合計 > 購入額の合計) になる確率 (モデルの着順シミュレーション)
    p_any_hit       : 1 つ以上当たる確率 / p_all_miss: 全部外れる確率 (= 共倒れ)
    key_horse / p_hit_without_key: 組み合わせが最も頼っている馬 (その馬が 3 着以内に来なかったときに当たる確率が
                      最も低い馬) と, その馬が 3 着以内に来なかったときに 1 つ以上当たる確率
    """
    stake = float(bets["stake"].sum())
    total = pays.sum(axis=0)
    hit = total > 0
    metrics = {
        "n_bets": int(len(bets)),
        "n_tickets": int(bets["n_tickets"].sum()),
        "stake": stake,
        "expected_return": float(bets["expected_return"].sum()),
        "return_rate": float(bets["expected_return"].sum() / stake) if stake else float("nan"),
        "p_profit": float(prob[total > stake].sum()),
        "p_any_hit": float(prob[hit].sum()),
        "p_all_miss": float(prob[~hit].sum()),
        "key_horse": None,
        "p_hit_without_key": None,
    }
    worst = None
    for h in sorted({u for g in bets["groups"] for col in g for u in col}):
        absent = ~(outcomes == h).any(axis=1)
        p_absent = prob[absent].sum()
        if p_absent <= 0:
            continue
        p = float(prob[absent & hit].sum() / p_absent)
        if worst is None or p < worst[1]:
            worst = (int(h), p)
    if worst is not None:
        metrics["key_horse"], metrics["p_hit_without_key"] = worst
    return metrics


def _horses_in(groups) -> set[int]:
    return {int(u) for col in groups for u in col}


def portfolio_pool(race: pd.DataFrame, recommendations: dict[str, dict], cfg: BettingConfig) -> pd.DataFrame:
    """リスクを分散した組み合わせの候補.

    券種ごとに, 的中率 portfolio_min_hit_rate 以上の買い方から
        - 期待値プラスの買い方を, 的中率の高い順に portfolio_candidates 通り
        - 押さえ: 1 着の確率が高い上位 3 頭のそれぞれについて, その馬を含まない買い方 (期待回収率 portfolio_min_bet_return
          以上. 期待値が少しマイナスでもよい) を, 的中率の高い順に portfolio_candidates 通り
    を集める (モデルが評価する馬が来なかったときに当たる買い方も候補に入れるため).
    """
    top_horses = race.sort_values("p_win", ascending=False)["umaban"].astype(int).head(3).tolist()
    parts = []
    for r in recommendations.values():
        pos = r.get("positive")
        if pos is None or pos.empty:
            continue
        pos = pos[(pos["hit_rate"] >= cfg.portfolio_min_hit_rate) & (pos["stake"] <= cfg.portfolio_max_stake)
                  & (pos["return_rate"] >= cfg.portfolio_min_bet_return)]
        parts.append(pos[pos["return_rate"] > 1].head(cfg.portfolio_candidates))
        has = pos["groups"].map(_horses_in)
        for h in top_horses:
            parts.append(pos[~has.map(lambda s, h=h: h in s)].head(cfg.portfolio_candidates))
    if not parts:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    pool = pd.concat(parts, ignore_index=True)
    pool = pool.loc[~pool[["bet_type", "method"]].astype(str).agg("|".join, axis=1).str.cat(pool["groups"].astype(str), sep="|").duplicated()]
    return pool.reset_index(drop=True)


SEARCH_POOL = 40  # 組み合わせを全通り調べる候補の数 (期待値プラスの上位 20 + 押さえの上位 20)
SEARCH_MAX_BETS = 3  # 全通り調べる組み合わせの大きさ (これより多い分は 1 つずつ足して探す)


def build_portfolio(
    race: pd.DataFrame,
    orders: np.ndarray,
    odds: dict[str, BetOdds],
    recommendations: dict[str, dict],
    cfg: BettingConfig,
    objective: str = "profit",
) -> dict | None:
    """買い方 (券種をまたいで) から, 共倒れ (全部外れる) になりにくい組み合わせを選ぶ.

    予算 (portfolio_max_stake), 買い方の数 (portfolio_max_bets), 組み合わせ全体の期待回収率 (portfolio_min_total_return
    以上), 同じ券種で同じ点を重ねない, の条件の中で, objective が最も良い組み合わせを選ぶ.
        "profit"   : 収支がプラスになる確率が最も高い (同じなら, 全部外れる確率が低い方)
        "no_miss"  : 全部外れる確率が最も低い = 共倒れ回避 (同じなら, 収支がプラスになる確率が高い方).
                     ただし「当たったときは必ず収支がプラス」(どの買い方が当たっても, 払戻の合計が購入額の合計を
                     上回る. いわゆるトリガミにならない) の組み合わせだけから選ぶ. このため 1 つ以上当たる確率 =
                     収支がプラスになる確率になる
        "half_back": 購入額の半分以上が戻る確率が最も高い
    同じ馬に頼った買い方を重ねても全部外れる確率は下がらないので, "no_miss" では外れ方の違う
    (別の馬が来たときに当たる) 買い方が選ばれる.
    モデルの評価がオッズより高い馬 (期待値プラスの元) が 1 頭だけのレースでは, その馬を含まない買い方は期待値が
    マイナスになるので, 期待値プラスの買い方で得た余裕の分だけ押さえ (期待回収率 portfolio_min_bet_return 以上) を加える.
    候補を 40 通り (期待値の余裕が大きい順に期待値プラスの 20 通り + 的中率の高い順に押さえの 20 通り) に絞り,
    3 つまでの組み合わせは全通り調べる (それより多い分は, 1 つずつ足して探す).
    比較用に「券種ごとのおすすめの 1 番目を全部買った場合」の見込みも返す.

    Returns:
        {"bets": 選んだ買い方, "metrics": portfolio_metrics, "naive": {"bets": ..., "metrics": ...}} / 候補がなければ None
    """
    pool = portfolio_pool(race, recommendations, cfg)
    has_naive = any(len(r["bets"]) for r in recommendations.values())
    naive = pd.concat([r["bets"].head(1) for r in recommendations.values() if len(r["bets"])], ignore_index=True) \
        if has_naive else pd.DataFrame(columns=RESULT_COLUMNS)
    if pool.empty:
        return None
    surplus = pool["expected_return"] - pool["stake"]
    core = pool[pool["return_rate"] > 1].assign(_s=surplus).sort_values(["_s", "hit_rate"], ascending=False).head(SEARCH_POOL // 2)
    hedge = pool[pool["return_rate"] <= 1].head(SEARCH_POOL - len(core))
    pool = pd.concat([core.drop(columns="_s"), hedge], ignore_index=True)
    if not (pool["return_rate"] > 1).any():
        return None
    both = pd.concat([pool, naive], ignore_index=True)
    pays_all, prob, outcomes = payout_matrix(both, race, orders, odds)
    pays, pays_naive = pays_all[: len(pool)], pays_all[len(pool):]
    stakes = pool["stake"].to_numpy(dtype=float)
    exp_ret = pool["expected_return"].to_numpy(dtype=float)
    n = len(pool)
    # 同じ券種で同じ点を含む買い方どうしは組み合わせない (同じ馬券を重ねて買うことになるため)
    tickets = [set(bet_tickets(t, g)) for t, g in zip(pool["bet_type"], pool["groups"])]
    clash = np.zeros((n, n), dtype=bool)
    for i, j in combinations(range(n), 2):
        clash[i, j] = clash[j, i] = pool["bet_type"].iat[i] == pool["bet_type"].iat[j] and bool(tickets[i] & tickets[j])

    def score(c: list[int]) -> tuple[float, float]:
        """大きいほど良い (目的の値, 収支がプラスになる確率)."""
        total, stake = pays[c].sum(axis=0), stakes[c].sum()
        p_profit = float(prob[total > stake].sum())
        if objective == "profit":
            return p_profit, -float(prob[total <= 0].sum())
        if objective == "no_miss":
            return -float(prob[total <= 0].sum()), p_profit
        # "half_back": 払戻が購入額の半分以上戻る確率 (大負けしない確率)
        return float(prob[total >= 0.5 * stake].sum()), p_profit

    def feasible(c: list[int]) -> bool:
        stake = stakes[c].sum()
        if not (stake <= cfg.portfolio_max_stake and exp_ret[c].sum() >= cfg.portfolio_min_total_return * stake
                and not clash[np.ix_(c, c)].any()):
            return False
        if objective == "no_miss":
            # 当たったときは必ず収支がプラス (払戻が 0 より大きく購入額以下になる着順がない)
            total = pays[c].sum(axis=0)
            return not ((total > 0) & (total <= stake)).any()
        return True

    best: tuple[tuple[float, float], list[int]] | None = None
    for size in range(1, min(cfg.portfolio_max_bets, SEARCH_MAX_BETS, n) + 1):
        for c in combinations(range(n), size):
            c = list(c)
            if feasible(c):
                sc = score(c)
                if best is None or sc > best[0]:
                    best = (sc, c)
    if best is None:
        return None
    chosen = best[1]
    # 3 つより多くてよい場合は, 1 つずつ足して, 目的の値が上がる限り加える
    while len(chosen) < cfg.portfolio_max_bets:
        cands = [(score(chosen + [i]), i) for i in range(n) if i not in chosen and feasible(chosen + [i])]
        if not cands or max(cands)[0] <= best[0]:
            break
        best = (max(cands)[0], chosen + [max(cands)[1]])
        chosen = best[1]
    selected = pool.iloc[chosen].reset_index(drop=True)
    return {
        "objective": objective,
        "bets": selected,
        "metrics": portfolio_metrics(selected, pays[chosen], prob, outcomes),
        "naive": {"bets": naive, "metrics": portfolio_metrics(naive, pays_naive, prob, outcomes) if len(naive) else None},
    }
