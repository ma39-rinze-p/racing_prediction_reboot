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


@dataclass(frozen=True)
class TrainConfig:
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    test_ratio: float
    n_trials: int
    timeout_sec: int | None
    n_folds: int

    def to_dict(self) -> dict:
        """metadata.json などに保存できる形 (日付は文字列) にする."""
        d = asdict(self)
        for key in ("start_date", "end_date"):
            d[key] = None if d[key] is None else str(d[key].date())
        return d


@dataclass(frozen=True)
class BettingConfig:
    """ワイドの期待値とおすすめボックスの設定 (configs/betting.yaml)."""

    market_weight: float = 0.5
    max_odds: float = 50.0
    min_model_prob: float = 0.05
    box_sizes: tuple[int, ...] = (2, 3, 4, 5)
    max_candidates: int = 8
    min_hit_rate: float = 0.2


def load_betting_config(path: str | Path | None) -> BettingConfig:
    """configs/betting.yaml を読み込む. ファイルがなければ既定値を使う."""
    if path is None or not Path(path).exists():
        return BettingConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    wide, box = raw.get("wide") or {}, raw.get("box") or {}
    default = BettingConfig()
    cfg = BettingConfig(
        market_weight=float(wide.get("market_weight", default.market_weight)),
        max_odds=float(wide.get("max_odds", default.max_odds)),
        min_model_prob=float(wide.get("min_model_prob", default.min_model_prob)),
        box_sizes=tuple(int(k) for k in box.get("sizes", default.box_sizes)),
        max_candidates=int(box.get("max_candidates", default.max_candidates)),
        min_hit_rate=float(box.get("min_hit_rate", default.min_hit_rate)),
    )
    errors = []
    if not 0 <= cfg.market_weight <= 1:
        errors.append(f"wide.market_weight は 0〜1 にしてください: {cfg.market_weight}")
    if cfg.max_odds <= 1:
        errors.append(f"wide.max_odds は 1 より大きくしてください: {cfg.max_odds}")
    if not 0 <= cfg.min_model_prob < 1:
        errors.append(f"wide.min_model_prob は 0 以上 1 未満にしてください: {cfg.min_model_prob}")
    if not cfg.box_sizes or any(k < 2 for k in cfg.box_sizes):
        errors.append(f"box.sizes は 2 以上の頭数のリストにしてください: {list(cfg.box_sizes)}")
    if cfg.max_candidates < max(cfg.box_sizes or (2,)):
        errors.append(f"box.max_candidates ({cfg.max_candidates}) は box.sizes の最大値以上にしてください")
    if not 0 <= cfg.min_hit_rate < 1:
        errors.append(f"box.min_hit_rate は 0 以上 1 未満にしてください: {cfg.min_hit_rate}")
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


def load_train_config(path: str | Path) -> TrainConfig:
    """YAML を読み込み, 値をチェックして TrainConfig を返す."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルがありません: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data, split, tuning = raw.get("data") or {}, raw.get("split") or {}, raw.get("tuning") or {}

    cfg = TrainConfig(
        start_date=_date(data.get("start_date"), "data.start_date"),
        end_date=_date(data.get("end_date"), "data.end_date"),
        test_ratio=float(split.get("test_ratio", 0.2)),
        n_trials=int(tuning.get("n_trials", 100)),
        timeout_sec=None if tuning.get("timeout_sec") is None else int(tuning["timeout_sec"]),
        n_folds=int(tuning.get("n_folds", 3)),
    )

    errors = []
    if cfg.start_date is not None and cfg.end_date is not None and cfg.start_date > cfg.end_date:
        errors.append(f"data.start_date ({cfg.start_date.date()}) が data.end_date ({cfg.end_date.date()}) より後です")
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
