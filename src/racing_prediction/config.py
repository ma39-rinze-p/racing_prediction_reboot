"""学習の設定 (configs/train.yaml) の読み込み.

読み込み時に値をチェックし, 書き間違いは学習を始める前にエラーにする.

使用例:
    >>> from racing_prediction.config import load_train_config
    >>> cfg = load_train_config("configs/train.yaml")
    >>> cfg.start_date, cfg.test_ratio, cfg.n_trials
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml


# JRA の 10 競馬場 (configs/train.yaml の data.venues に書ける名前)
JRA_VENUES = ("札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉")


@dataclass(frozen=True)
class TrainConfig:
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    venues: tuple[str, ...] | None  # 学習・テストに使う競馬場. None なら data/processed/races にある全競馬場
    test_ratio: float
    n_trials: int
    timeout_sec: int | None
    n_folds: int
    # True: オッズとの差だけを学習する (単勝オッズの確率を出発点にする. 推論に発走前の単勝オッズが必要)
    market_offset: bool = True
    # 3 着以内に入る確率のモデルも学習するか / その Optuna の試行回数・打ち切り時間 (None なら tuning と同じ)
    top3_enabled: bool = True
    top3_n_trials: int | None = None
    top3_timeout_sec: int | None = None

    def to_dict(self) -> dict:
        """metadata.json などに保存できる形 (日付は文字列) にする."""
        d = asdict(self)
        for key in ("start_date", "end_date"):
            d[key] = None if d[key] is None else str(d[key].date())
        d["venues"] = None if self.venues is None else list(self.venues)
        return d


BET_TYPE_NAMES = ("win", "place", "wide", "trio", "trifecta")  # 単勝, 複勝, ワイド, 3 連複, 3 連単
# リスクを分散した組み合わせを選ぶ目的
PORTFOLIO_OBJECTIVES = ("profit", "no_miss", "half_back")


@dataclass(frozen=True)
class BetTypeConfig:
    """券種ごとの期待値の計算の設定 (configs/betting.yaml の bet_types.<券種>)."""

    enabled: bool = True
    market_weight: float = 0.5  # 市場の確率をどれだけ重視するか (0 = モデルだけ, 1 = 市場だけ)
    max_odds: float = 50.0  # オッズがこれより高い点は, モデルを信用せず市場の確率だけで計算する
    min_model_prob: float = 0.05  # モデルの確率がこれより低い点も, 市場の確率だけで計算する


# 券種ごとの既定値 (当たりにくい券種ほど, 1 点の確率は低く, オッズは高くなるので, 基準を緩める)
DEFAULT_BET_TYPES = {
    "win": BetTypeConfig(max_odds=30.0, min_model_prob=0.03),
    "place": BetTypeConfig(max_odds=10.0, min_model_prob=0.10),
    "wide": BetTypeConfig(max_odds=50.0, min_model_prob=0.05),
    "trio": BetTypeConfig(max_odds=300.0, min_model_prob=0.003),
    "trifecta": BetTypeConfig(max_odds=2000.0, min_model_prob=0.0005),
}


@dataclass(frozen=True)
class BettingConfig:
    """買い方の期待値の計算と, メールに載せる買い方の設定 (configs/betting.yaml)."""

    min_hit_rate: float = 0.0  # メールに載せる買い方の的中率の下限 (期待値プラスのうち, これ以上のものだけ載せる. 0 なら下限なし)
    candidates: int = 7  # ボックス・ながし・フォーメーションに使う馬 (3 着以内の確率が高い順) の頭数
    max_tickets: int = 30  # 1 つの買い方の最大点数
    max_display: int = 5  # 券種ごとにメールに載せる買い方の最大数
    bet_types: tuple[tuple[str, BetTypeConfig], ...] = tuple(DEFAULT_BET_TYPES.items())
    # リスクを分散した組み合わせ (券種をまたいで, 外れ方の違う買い方を組み合わせる)
    portfolio_enabled: bool = True
    portfolio_max_stake: int = 3000  # 組み合わせ全体の購入額の上限 (円)
    portfolio_max_bets: int = 3  # 組み合わせる買い方の最大数
    portfolio_candidates: int = 10  # 券種ごとに, 期待値プラスの買い方の上位何通りを組み合わせの候補にするか
    portfolio_min_hit_rate: float = 0.02  # 組み合わせの候補にする買い方の的中率の下限 (おすすめの条件より緩くする)
    portfolio_min_bet_return: float = 0.8  # 押さえにする買い方の期待回収率の下限 (期待値が少しマイナスでもよい)
    portfolio_min_total_return: float = 1.0  # 組み合わせ全体の期待回収率の下限 (1.0 = 全体では期待値プラスを保つ)
    # 組み合わせを選ぶ目的 (並べた順にメールに載せる): "profit" = 収支がプラスになる確率が最も高い /
    # "no_miss" = 全部外れる確率が最も低い (共倒れ回避) / "half_back" = 購入額の半分以上が戻る確率が最も高い
    portfolio_objectives: tuple[str, ...] = ("profit", "no_miss")

    def bet_type(self, name: str) -> BetTypeConfig:
        return dict(self.bet_types).get(name, DEFAULT_BET_TYPES[name])

    def enabled_bet_types(self) -> list[str]:
        return [name for name in BET_TYPE_NAMES if self.bet_type(name).enabled]

    # 以前のワイドの設定 (betting/optimize_bet.py) との互換用
    @property
    def market_weight(self) -> float:
        return self.bet_type("wide").market_weight

    @property
    def max_odds(self) -> float:
        return self.bet_type("wide").max_odds

    @property
    def min_model_prob(self) -> float:
        return self.bet_type("wide").min_model_prob

    @property
    def box_max_display(self) -> int:
        return self.max_display


def load_betting_config(path: str | Path | None) -> BettingConfig:
    """configs/betting.yaml を読み込む. ファイルがなければ既定値を使う.

    以前の書き方 (wide: と box: だけ) も読める (wide はワイドの設定, box.max_display はメールに載せる最大数になる).
    """
    if path is None or not Path(path).exists():
        return BettingConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    common = raw.get("common") or {}
    default = BettingConfig()
    sections = dict(raw.get("bet_types") or {})
    if "wide" in raw and "wide" not in sections:  # 以前の書き方
        sections["wide"] = raw["wide"]
    errors = []
    unknown = sorted(set(sections) - set(BET_TYPE_NAMES))
    if unknown:
        errors.append(f"bet_types に知らない券種があります: {unknown} (使えるのは {list(BET_TYPE_NAMES)})")
    bet_types = []
    for name in BET_TYPE_NAMES:
        base, sec = DEFAULT_BET_TYPES[name], sections.get(name) or {}
        t = BetTypeConfig(
            enabled=bool(sec.get("enabled", base.enabled)),
            market_weight=float(sec.get("market_weight", base.market_weight)),
            max_odds=float(sec.get("max_odds", base.max_odds)),
            min_model_prob=float(sec.get("min_model_prob", base.min_model_prob)),
        )
        if not 0 <= t.market_weight <= 1:
            errors.append(f"bet_types.{name}.market_weight は 0〜1 にしてください: {t.market_weight}")
        if t.max_odds <= 1:
            errors.append(f"bet_types.{name}.max_odds は 1 より大きくしてください: {t.max_odds}")
        if not 0 <= t.min_model_prob < 1:
            errors.append(f"bet_types.{name}.min_model_prob は 0 以上 1 未満にしてください: {t.min_model_prob}")
        bet_types.append((name, t))
    max_display = common.get("max_display", (raw.get("box") or {}).get("max_display", default.max_display))
    portfolio = raw.get("portfolio") or {}
    cfg = BettingConfig(
        min_hit_rate=float(common.get("min_hit_rate", default.min_hit_rate)),
        candidates=int(common.get("candidates", default.candidates)),
        max_tickets=int(common.get("max_tickets", default.max_tickets)),
        max_display=int(max_display),
        bet_types=tuple(bet_types),
        portfolio_enabled=bool(portfolio.get("enabled", default.portfolio_enabled)),
        portfolio_max_stake=int(portfolio.get("max_stake", default.portfolio_max_stake)),
        portfolio_max_bets=int(portfolio.get("max_bets", default.portfolio_max_bets)),
        portfolio_candidates=int(portfolio.get("candidates_per_type", default.portfolio_candidates)),
        portfolio_min_hit_rate=float(portfolio.get("min_hit_rate", default.portfolio_min_hit_rate)),
        portfolio_min_bet_return=float(portfolio.get("min_bet_return", default.portfolio_min_bet_return)),
        portfolio_min_total_return=float(portfolio.get("min_total_return", default.portfolio_min_total_return)),
        portfolio_objectives=tuple(str(x) for x in (portfolio.get("objectives") or default.portfolio_objectives)),
    )
    if cfg.portfolio_max_stake < 100:
        errors.append(f"portfolio.max_stake は 100 以上にしてください: {cfg.portfolio_max_stake}")
    if not 1 <= cfg.portfolio_max_bets <= 10:
        errors.append(f"portfolio.max_bets は 1〜10 にしてください: {cfg.portfolio_max_bets}")
    if not 0 < cfg.portfolio_min_bet_return <= cfg.portfolio_min_total_return:
        errors.append("portfolio.min_bet_return は 0 より大きく min_total_return 以下にしてください: "
                      f"{cfg.portfolio_min_bet_return} / {cfg.portfolio_min_total_return}")
    bad = [x for x in cfg.portfolio_objectives if x not in PORTFOLIO_OBJECTIVES]
    if bad:
        errors.append(f"portfolio.objectives に使えない値があります: {bad} (使えるのは {list(PORTFOLIO_OBJECTIVES)})")
    if not 1 <= cfg.portfolio_candidates <= 50:
        errors.append(f"portfolio.candidates_per_type は 1〜50 にしてください: {cfg.portfolio_candidates}")
    if not 0 <= cfg.min_hit_rate <= 1:
        errors.append(f"common.min_hit_rate は 0〜1 にしてください: {cfg.min_hit_rate}")
    if not 3 <= cfg.candidates <= 10:
        errors.append(f"common.candidates は 3〜10 にしてください (多いと計算に時間がかかる): {cfg.candidates}")
    if cfg.max_tickets < 1:
        errors.append(f"common.max_tickets は 1 以上にしてください: {cfg.max_tickets}")
    if cfg.max_display < 1:
        errors.append(f"common.max_display は 1 以上にしてください: {cfg.max_display}")
    if errors:
        raise ValueError(f"{path} の設定に誤りがあります:\n  " + "\n  ".join(errors))
    return cfg


def _date(value, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        return pd.Timestamp(str(value))
    except ValueError as e:
        raise ValueError(f"{name} を日付として読めません: {value!r} (例: \"2025-11-23\")") from e


def _venues(value) -> tuple[str, ...] | None:
    """data.venues: null / "東京" / "東京,京都" / [東京, 京都] -> 競馬場名のタプル (空なら None = 全競馬場)."""
    if value is None:
        return None
    items = value.split(",") if isinstance(value, str) else list(value)
    names = tuple(dict.fromkeys(str(v).strip() for v in items if str(v).strip()))
    return names or None


def load_train_config(path: str | Path) -> TrainConfig:
    """YAML を読み込み, 値をチェックして TrainConfig を返す."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルがありません: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data, split, tuning = raw.get("data") or {}, raw.get("split") or {}, raw.get("tuning") or {}
    model = raw.get("model") or {}
    top3 = raw.get("top3") or {}

    cfg = TrainConfig(
        start_date=_date(data.get("start_date"), "data.start_date"),
        end_date=_date(data.get("end_date"), "data.end_date"),
        venues=_venues(data.get("venues")),
        test_ratio=float(split.get("test_ratio", 0.2)),
        n_trials=int(tuning.get("n_trials", 100)),
        timeout_sec=None if tuning.get("timeout_sec") is None else int(tuning["timeout_sec"]),
        n_folds=int(tuning.get("n_folds", 3)),
        market_offset=bool(model.get("market_offset", True)),
        top3_enabled=bool(top3.get("enabled", True)),
        top3_n_trials=None if top3.get("n_trials") is None else int(top3["n_trials"]),
        top3_timeout_sec=None if top3.get("timeout_sec") is None else int(top3["timeout_sec"]),
    )

    errors = []
    if cfg.start_date is not None and cfg.end_date is not None and cfg.start_date > cfg.end_date:
        errors.append(f"data.start_date ({cfg.start_date.date()}) が data.end_date ({cfg.end_date.date()}) より後です")
    unknown = [v for v in (cfg.venues or ()) if v not in JRA_VENUES]
    if unknown:
        errors.append(f"data.venues に未対応の競馬場があります: {unknown} (書けるのは {list(JRA_VENUES)})")
    if not 0 < cfg.test_ratio < 1:
        errors.append(f"split.test_ratio は 0 より大きく 1 より小さい値にしてください: {cfg.test_ratio}")
    if cfg.n_trials < 1:
        errors.append(f"tuning.n_trials は 1 以上にしてください: {cfg.n_trials}")
    if cfg.timeout_sec is not None and cfg.timeout_sec < 1:
        errors.append(f"tuning.timeout_sec は 1 以上か null にしてください: {cfg.timeout_sec}")
    if cfg.n_folds < 1:
        errors.append(f"tuning.n_folds は 1 以上にしてください: {cfg.n_folds}")
    if errors:
        raise ValueError(f"{path} の設定に誤りがあります:\n  " + "\n  ".join(errors))
    return cfg
