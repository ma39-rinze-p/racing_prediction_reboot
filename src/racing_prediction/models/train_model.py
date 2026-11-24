"""train_model

LightGBM で各馬が 1 着になる確率を予測するモデル (条件付きロジット / レース内ソフトマックス) の学習.

モデル:
    LightGBM が各馬の強さの対数 (win_score) を出力し, レース内でソフトマックスをとって 1 着の確率にする.
        p_win(i) = exp(win_score(i)) / Σ_{同じレースの馬 j} exp(win_score(j))
    → レースごとの確率の合計は必ず 1 になる. 学習では「1 着の馬に付けた確率の -log」(レース単位の logloss) を
      最小化する (ソフトマックス交差エントロピー. LightGBM のカスタム目的関数で実装).
    同着で 1 着が複数いるレースは, 正解を 1 着の馬で等分する (例: 2 頭なら 0.5 ずつ).

方針:
    - 検証は必ず時系列で行う (学習 = 過去, 検証 = その後). 1 レースの馬は同じ側に入れる.
    - Optuna は「テスト期間より前」のデータだけで, 期間をずらした時系列 CV のレース単位の logloss を最小化する.
    - カテゴリの種類などの前処理も fold ごとに学習側だけで fit する (リーク防止).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from racing_prediction.data.make_dataset import TARGET_COLUMN
from racing_prediction.features.model_features import FeatureBuilder
from racing_prediction.models.predict_model import MODEL_TARGET, PROB_FLOOR, evaluate_predictions, predict_win

SEED = 42
OBJECTIVE_NAME = "race_softmax"  # レース内ソフトマックス交差エントロピー (カスタム目的関数. 保存する params での名前)
METRIC_NAME = "race_logloss"

BASE_PARAMS = {
    "objective": OBJECTIVE_NAME,  # 学習時に race_softmax_objective に置き換える
    "metric": "None",  # 評価指標はカスタム (race_logloss_metric)
    "learning_rate": 0.05,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": SEED,
    "deterministic": True,
    "force_col_wise": True,
}

EARLY_STOPPING_ROUNDS = 100
MAX_BOOST_ROUND = 5000


# ---------------------------------------------------------------------------
# データ分割
# ---------------------------------------------------------------------------
def filter_period(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """開催日が start_date 〜 end_date (両端を含む) のレースだけに絞る. None なら制限しない."""
    mask = pd.Series(True, index=df.index)
    if start_date is not None:
        mask &= df["race_date"] >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= df["race_date"] <= pd.Timestamp(end_date)
    out = df[mask].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{start_date} 〜 {end_date} に該当するレースがありません")
    return out


def filter_venues(df: pd.DataFrame, venues: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    """競馬場 (race_venue) が venues のどれかのレースだけに絞る. None や空なら全競馬場を使う."""
    if not venues:
        return df.reset_index(drop=True)
    out = df[df["race_venue"].isin(list(venues))].reset_index(drop=True)
    if out.empty:
        found = sorted(df["race_venue"].dropna().unique())
        raise ValueError(f"競馬場 {list(venues)} のレースがありません (データにある競馬場: {found})")
    return out


def time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """開催日の新しい方から, 全開催日数の test_ratio の割合をテスト, 残りを学習にする.

    カレンダーの期間ではなく「開催日の数」で分けるので, 開催がない月があってもテストの量が安定する.
    同じ開催日のレースは必ず同じ側に入る.
    """
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio は 0 より大きく 1 より小さい値にしてください: {test_ratio}")
    unique_dates = np.sort(df["race_date"].unique())
    n_test = max(1, round(len(unique_dates) * test_ratio))
    if n_test >= len(unique_dates):
        raise ValueError(f"開催日が {len(unique_dates)} 日しかなく, 学習用の開催日が残りません")
    test_start = unique_dates[-n_test]
    train = df[df["race_date"] < test_start].reset_index(drop=True)
    test = df[df["race_date"] >= test_start].reset_index(drop=True)
    return train, test


def time_series_folds(dates: pd.Series, n_folds: int = 3, valid_ratio: float = 0.15) -> list[tuple[np.ndarray, np.ndarray]]:
    """期間をずらした時系列 CV の (学習 index, 検証 index) のリスト.

    開催日を古い順に並べ, 最後の n_folds × valid_ratio の期間を n_folds 個の検証期間に分ける.
    各 fold の学習データは「その検証期間より前」の全データ (expanding window).
    """
    unique_dates = np.sort(dates.unique())
    n_valid = max(1, int(len(unique_dates) * valid_ratio))
    if len(unique_dates) <= n_valid * n_folds:
        raise ValueError(
            f"開催日数 ({len(unique_dates)} 日) が少なすぎて {n_folds} fold の時系列 CV が組めません. "
            "データを増やすか n_folds / valid_ratio を小さくしてください."
        )

    folds = []
    for k in range(n_folds, 0, -1):
        valid_dates = unique_dates[len(unique_dates) - k * n_valid : len(unique_dates) - (k - 1) * n_valid]
        train_idx = np.where(dates < valid_dates[0])[0]
        valid_idx = np.where(dates.isin(valid_dates))[0]
        folds.append((train_idx, valid_idx))
    return folds


# ---------------------------------------------------------------------------
# レース内ソフトマックス (条件付きロジット) の目的関数・評価指標
# ---------------------------------------------------------------------------
def _group_softmax(scores: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """連続して並んだグループ (各要素はグループの行数) ごとのソフトマックスと, 各行のグループ番号を返す."""
    gid = np.repeat(np.arange(len(group)), group)
    starts = np.concatenate([[0], np.cumsum(group)[:-1]])
    e = np.exp(scores - np.maximum.reduceat(scores, starts)[gid])
    return e / np.add.reduceat(e, starts)[gid], gid


def _race_targets(label: np.ndarray, group: np.ndarray) -> np.ndarray:
    """正解 (1 着なら 1) をレース内で合計 1 にする (同着なら 1 着の馬で等分)."""
    starts = np.concatenate([[0], np.cumsum(group)[:-1]])
    gid = np.repeat(np.arange(len(group)), group)
    total = np.add.reduceat(label, starts)[gid]
    return np.divide(label, total, out=np.zeros_like(label, dtype=float), where=total > 0)


def race_softmax_objective(preds: np.ndarray, data: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """レース内ソフトマックス交差エントロピーの勾配とヘッセ行列の対角 (LightGBM のカスタム目的関数)."""
    group = data.get_group().astype(int)
    p, _ = _group_softmax(preds, group)
    y = _race_targets(data.get_label().astype(float), group)
    return p - y, np.maximum(p * (1 - p), 1e-6)


def race_logloss_metric(preds: np.ndarray, data: lgb.Dataset) -> tuple[str, float, bool]:
    """レース単位の logloss (1 着の馬に付けた確率の -log の平均) (LightGBM のカスタム評価指標)."""
    group = data.get_group().astype(int)
    p, _ = _group_softmax(preds, group)
    y = _race_targets(data.get_label().astype(float), group)
    starts = np.concatenate([[0], np.cumsum(group)[:-1]])
    p_winner = np.add.reduceat(p * (y > 0), starts)
    return METRIC_NAME, float(-np.log(np.clip(p_winner, PROB_FLOOR, None)).mean()), False


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """学習に使えるレースだけにして, 同じレースの馬が連続して並ぶよう並べ替える.

    1 着の馬がいないレース (全頭が中止など) は正解がないので除く.
    """
    df = df[df[TARGET_COLUMN].notna()]
    has_winner = df[TARGET_COLUMN].astype(int).groupby(df["race_id"]).transform("sum") > 0
    return df[has_winner].sort_values(["race_date", "race_id", "umaban"], kind="stable").reset_index(drop=True)


def _dataset(df: pd.DataFrame, builder: FeatureBuilder, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    group = df.groupby("race_id", sort=False).size().to_numpy()
    return lgb.Dataset(
        builder.transform(df), label=df[TARGET_COLUMN].astype(int).to_numpy(), group=group,
        reference=reference, free_raw_data=False,
    )


# ---------------------------------------------------------------------------
# 学習
# ---------------------------------------------------------------------------
def _lgb_params(params: dict) -> dict:
    """保存用の params を LightGBM に渡す形にする (カスタム目的関数を関数に置き換える)."""
    lgb_params = {k: v for k, v in params.items() if k != "min_category_count"}
    lgb_params["objective"] = race_softmax_objective
    lgb_params["metric"] = "None"
    return lgb_params


def _fit_one(params: dict, train: pd.DataFrame, valid: pd.DataFrame | None = None, num_boost_round: int | None = None):
    """前処理の fit + LightGBM の学習. valid があれば early stopping する."""
    train = _prepare(train)
    builder = FeatureBuilder(min_category_count=params.get("min_category_count", 5)).fit(train)
    lgb_params = _lgb_params(params)

    dtrain = _dataset(train, builder)
    if valid is None:
        booster = lgb.train(lgb_params, dtrain, num_boost_round=num_boost_round)
        return builder, booster

    dvalid = _dataset(_prepare(valid), builder, reference=dtrain)
    booster = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=MAX_BOOST_ROUND,
        valid_sets=[dvalid],
        feval=race_logloss_metric,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return builder, booster


def _suggest_params(trial: optuna.Trial) -> dict:
    """探索するハイパーパラメータ. データが少ないうちは過学習を抑える方向に効くものを中心にする."""
    return {
        **BASE_PARAMS,
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        # カテゴリ変数 (騎手・調教師・血統など種類が多い) の過学習対策
        "min_data_per_group": trial.suggest_int("min_data_per_group", 10, 200, log=True),
        "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
        "min_category_count": trial.suggest_int("min_category_count", 3, 30, log=True),
    }


@dataclass
class TuningResult:
    best_params: dict
    best_score: float
    num_boost_round: int
    study: optuna.Study = field(repr=False)


def tune(
    train: pd.DataFrame,
    n_trials: int = 50,
    timeout: int | None = None,
    n_folds: int = 3,
    valid_ratio: float = 0.15,
) -> TuningResult:
    """時系列 CV の平均 (レース単位の) logloss を最小化するハイパーパラメータを Optuna で探す."""
    folds = time_series_folds(train["race_date"], n_folds=n_folds, valid_ratio=valid_ratio)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        scores, iters = [], []
        for step, (tr_idx, va_idx) in enumerate(folds):
            _, booster = _fit_one(params, train.iloc[tr_idx], train.iloc[va_idx])
            scores.append(booster.best_score["valid_0"][METRIC_NAME])
            iters.append(booster.best_iteration)
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("best_iterations", iters)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_trial
    # 最終学習の木の本数: CV の best_iteration の平均. 学習データが増える分だけ少し多めにする
    num_boost_round = int(np.mean(best.user_attrs["best_iterations"]) * 1.1) + 1
    return TuningResult(
        best_params={**BASE_PARAMS, **best.params},
        best_score=best.value,
        num_boost_round=num_boost_round,
        study=study,
    )


def train_final(train: pd.DataFrame, params: dict, num_boost_round: int) -> tuple[FeatureBuilder, lgb.Booster]:
    """決めたパラメータ・木の本数で, 渡したデータ全体を使って学習する."""
    return _fit_one(params, train, num_boost_round=num_boost_round)


def evaluate(booster: lgb.Booster, builder: FeatureBuilder, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """テストデータでの評価指標と, 予測結果付きの DataFrame を返す."""
    pred = predict_win(test, booster, builder)
    return evaluate_predictions(pred), pred


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------
def save_artifacts(
    output_dir: str | Path,
    booster: lgb.Booster,
    builder: FeatureBuilder,
    params: dict,
    num_boost_round: int,
    train: pd.DataFrame,
    metrics: dict | None = None,
) -> Path:
    """推論に必要なもの一式を保存する.

    model.txt     : LightGBM のモデル (出力は各馬の強さの対数. レース内でソフトマックスをとると 1 着の確率)
    metadata.json : 目的変数, 前処理 (カテゴリの種類, 事前値, 特徴量の順番), パラメータ, 学習期間, 評価指標
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_dir / "model.txt"))
    metadata = {
        # 学習した日時 (日本時間). モデルのバージョン名 (v{日付_時刻}) に使う (scripts/model_registry.py)
        "trained_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "target": MODEL_TARGET,  # 推論側 (load_artifacts) はこれが "is_win" のモデルだけを使う
        "objective": OBJECTIVE_NAME,
        "feature_builder": builder.to_dict(),
        "params": params,
        "num_boost_round": num_boost_round,
        "train_period": [str(train["race_date"].min().date()), str(train["race_date"].max().date())],
        "n_train_races": int(train["race_id"].nunique()),
        "n_train_rows": int(len(train)),
        "metrics": metrics or {},
        "lightgbm_version": lgb.__version__,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir
