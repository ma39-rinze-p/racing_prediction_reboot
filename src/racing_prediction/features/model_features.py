"""model_features

data/processed/races 以下の CSV を読み込み, LightGBM に入れる特徴量行列を作る.

学習データだけから決める値 (カテゴリの種類, 平均着順の事前値, 騎手などの成績, スピード指数の基準タイム) は
FeatureBuilder.fit で学習し,
推論時も同じ FeatureBuilder (保存して読み込んだもの) で transform することで,
学習と推論の前処理を一致させる.

追加する特徴量:
    cur_*      : 今回のレース条件 (芝/ダート, 距離区分, 馬場状態, 競馬場) に対応する平均着順・出走回数
    shr_*      : cur_* の平均着順を, 出走回数が少ないほど全体平均に寄せた値 (少数レースのブレ対策)
    speed_*    : スピード指数. 直近 3 走の走破タイムが, コース・距離・馬場状態ごとの基準タイム
                 (学習データの中央値) より 1000m あたり何秒速いか
    {騎手・調教師・父・母父}_win_rate / _top3_rate / _n_runs:
                 その騎手などの 勝率・3 着以内率・出走数 (対象レースより前の学習データのレースだけで計算).
                 名前をそのままカテゴリ変数にすると, 種類が多すぎてモデルが丸暗記する (過学習) ため置き換えた
    race_n_front_runners: レース内の逃げ・先行馬の頭数 (展開)
    *_race_rank: レース内での順位 (0〜1, 小さいほど値が小さい). 他の馬との比較を表す
    *_race_diff: レース内平均との差

オッズに関する情報 (今回の単勝オッズ・人気, 過去のレースの人気・単勝オッズ) は特徴量に使わない (ODDS_COLUMNS).
モデルは市場の評価に頼らずに予測し, オッズはワイドの期待値の計算と精度の比較 (参考) にだけ使う.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from racing_prediction.data.make_dataset import (
    TARGET_COLUMN,
    TOP3_COLUMN,
    add_win_label,
    feature_columns,
    output_columns,
)
from racing_prediction.features.build_features import N_SPEED_RACES, VENUES

# LightGBM のカテゴリ変数として扱う列 (種類が少ないものだけ)
CATEGORICAL_COLUMNS = [
    "race_venue",
    "race_distance_category",
    "race_surface",
    "race_track_condition",
    "sex",
    "sire_line",
]

# 名前ごとの 勝率・3 着以内率・出走数 に置き換える列 (種類が多く, カテゴリ変数のままだと過学習するため)
RATE_COLUMNS = ["jockey_name", "trainer_name", "sire", "dam_sire"]
# 成績を出すにはデータが少なすぎる (1 頭ごとにほぼ違う) ため, 特徴量に使わない列
DROPPED_COLUMNS = ["dam", "sire_sire", "sire_dam", "dam_dam"]
# オッズに関する列 (過去のレースの人気・単勝オッズ). CSV には残すが, 特徴量には使わない.
# 今回のレースの単勝オッズ・人気 (win_odds, popularity) はもともと特徴量の列 (feature_columns) に入っていない
ODDS_COLUMNS = [
    "hist_last1_popularity",
    "hist_avg3_popularity",
    "hist_avg5_rank_minus_pop",
    "hist_last1_log_odds",
    "hist_avg3_log_odds",
]
# 勝率などを全体の平均に寄せる強さ (この出走数分だけ「全体平均の成績」を足したとみなす)
RATE_SMOOTHING = 20.0

# スピード指数の基準タイムを決めるのに必要な最少レース数
SPEED_MIN_RUNS = 10
# 直近 3 走の最初のコーナーの位置 (0 = 先頭, 1 = 最後方) がこれ未満の馬を逃げ・先行馬とみなす
FRONT_RUNNER_RATIO = 0.25

# レース内での順位・平均との差を追加する列 (shr_avg_rank_* に加えて)
RELATIVE_COLUMNS = [
    "weight_carried",
    "horse_weight",
    "age",
    "hist_avg3_rank_ratio",
    "hist_avg3_margin",
    "hist_prize_365",
    "hist_best3_agari",
    "hist_avg3_first_corner",
    "speed_best3",
    "speed_avg3",
    "jockey_name_win_rate",
    "trainer_name_win_rate",
]

# 学習データでの出現回数がこれ未満のカテゴリは "__other__" にまとめる
OTHER = "__other__"

# 平均着順を全体平均に寄せる強さ (この回数分だけ「全体平均の成績」を足したとみなす)
SHRINK_STRENGTH = 3.0

# 競馬場名 -> 列名の接尾辞 ("東京" -> "tokyo")
_VENUE_KEYS = {name: key for key, name in VENUES.items()}

# 今回のレース条件に対応させる区分
_CONDITION_KINDS = ["surface", "distance", "condition", "venue"]


def load_race_csvs(races_dir: str | Path | list[str | Path]) -> pd.DataFrame:
    """races_dir 以下の全 CSV ({競馬場}/{開催日}/{R}R.csv) を読み込んで縦に結合する.

    races_dir はフォルダのリストでもよい (Kaggle では半年ごとのデータセットに分かれているため).

    必要な列が欠けている CSV (列追加前に保存した古いもの) は警告を出して読み飛ばす.
    """
    dirs = races_dir if isinstance(races_dir, (list, tuple)) else [races_dir]
    paths = sorted(path for d in dirs for path in Path(d).rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"{races_dir} に CSV がありません")

    required = output_columns()
    tables, skipped = [], []
    for path in paths:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str, "horse_id": str}, parse_dates=["race_date"])
        if any(c not in df.columns for c in required):
            skipped.append(path)
            continue
        tables.append(df[required])

    if skipped:
        warnings.warn(
            f"列が足りない古い CSV を {len(skipped)} 件読み飛ばしました (例: {skipped[0]}). "
            "python scripts/backfill_race_features.py を実行すると列が補われます."
        )
    if not tables:
        raise ValueError("読み込める CSV がありませんでした")

    df = add_win_label(pd.concat(tables, ignore_index=True))  # 目的変数 is_win (1 着か) を着順から作る
    return df.sort_values(["race_date", "race_id", "umaban"], ignore_index=True)


def _pick_by_row(df: pd.DataFrame, colnames: pd.Series) -> np.ndarray:
    """行ごとに colnames で指定した列の値を取り出す. 列が存在しなければ NaN."""
    candidates = sorted(set(colnames.dropna()) & set(df.columns))
    values = np.full(len(df), np.nan)
    if not candidates:
        return values
    sub = df[candidates].to_numpy(dtype=float)
    col_idx = colnames.map({c: i for i, c in enumerate(candidates)})
    ok = col_idx.notna().to_numpy()
    values[ok] = sub[np.arange(len(df))[ok], col_idx[ok].astype(int).to_numpy()]
    return values


def _condition_suffixes(df: pd.DataFrame) -> dict[str, pd.Series]:
    """今回のレース条件に対応する列名の接尾辞 (例: dirt, dirt_mid, dirt_good, dirt_tokyo)."""
    surface = df["race_surface"].where(df["race_surface"].isin(["turf", "dirt"]))
    return {
        "surface": surface,
        "distance": surface + "_" + df["race_distance_category"],
        "condition": surface + "_" + df["race_track_condition"],
        "venue": surface + "_" + df["race_venue"].map(_VENUE_KEYS),
    }


class FeatureBuilder:
    """学習データで fit し, 学習・推論の両方で同じ変換 (transform) をする."""

    def __init__(self, min_category_count: int = 5):
        self.min_category_count = min_category_count
        self.categories: dict[str, list[str]] = {}
        self.rank_prior: float | None = None
        self.feature_names: list[str] = []
        # 騎手・調教師・父・母父ごとの成績 {列名: DataFrame[key, race_date, n, win, top3]} (開催日ごとの集計)
        self.rate_events: dict[str, pd.DataFrame] = {}
        self.rate_priors: dict[str, float] = {}
        # スピード指数の基準タイム {"競馬場|芝ダ|距離": 秒}, {"芝ダ|距離": 秒}, 馬場状態の補正 {"芝ダ|馬場": 倍率}
        self.speed_base: dict[str, float] = {}
        self.speed_base_coarse: dict[str, float] = {}
        self.speed_condition: dict[str, float] = {}
        # True なら「オッズとの差だけを学習したモデル」用: 予測時にモデルの出力へ単勝オッズの確率の対数を足す
        # (predict_model.predict_win. オッズは特徴量には入れない)
        self.market_offset: bool = False

    # ---- 学習 -------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        # カテゴリ: 出現回数が少ないものは "__other__" にまとめる (過学習・未知カテゴリ対策)
        self.categories = {}
        for col in CATEGORICAL_COLUMNS:
            counts = df[col].dropna().astype(str).value_counts()
            frequent = counts[counts >= self.min_category_count].index.tolist()
            self.categories[col] = sorted(frequent) + [OTHER]

        # 平均着順の事前値 = 学習データの着順の平均
        self.rank_prior = float(df["rank"].dropna().mean())

        self._fit_rates(df)
        self._fit_speed(df)
        self.feature_names = list(self._build(df.head(1)).columns)
        return self

    def _fit_rates(self, df: pd.DataFrame) -> None:
        """騎手などの名前ごとに, 開催日ごとの 出走数・1 着数・3 着以内数 を集計しておく."""
        done = df[df[TARGET_COLUMN].notna()]
        win = pd.to_numeric(done[TARGET_COLUMN], errors="coerce").fillna(0)
        top3 = pd.to_numeric(done[TOP3_COLUMN], errors="coerce").fillna(0)
        self.rate_priors = {"win": float(win.mean()), "top3": float(top3.mean())}
        self.rate_events = {}
        for col in RATE_COLUMNS:
            g = pd.DataFrame({"key": done[col].astype("string"), "race_date": done["race_date"], "win": win, "top3": top3})
            g = g.dropna(subset=["key"]).groupby(["key", "race_date"], as_index=False).agg(
                n=("win", "size"), win=("win", "sum"), top3=("top3", "sum")
            )
            self.rate_events[col] = g

    def _fit_speed(self, df: pd.DataFrame) -> None:
        """学習データの各馬の直近 3 走から, コース・距離ごとの基準タイム (中央値) と馬場状態の補正を決める."""
        runs = _past_runs(df)
        runs = runs[runs["surface"].isin(["turf", "dirt"])]
        fine = runs.groupby(["venue", "surface", "distance"])["time"].agg(["median", "size"])
        fine = fine[fine["size"] >= SPEED_MIN_RUNS]
        self.speed_base = {f"{v}|{s}|{int(d)}": float(m) for (v, s, d), m in fine["median"].items()}
        coarse = runs.groupby(["surface", "distance"])["time"].agg(["median", "size"])
        coarse = coarse[coarse["size"] >= SPEED_MIN_RUNS]
        self.speed_base_coarse = {f"{s}|{int(d)}": float(m) for (s, d), m in coarse["median"].items()}

        base = self._speed_base(runs)
        ratio = (runs["time"] / base).where(base > 0)
        cond = ratio.groupby([runs["surface"], runs["track_condition"]]).median().dropna()
        self.speed_condition = {f"{s}|{c}": float(r) for (s, c), r in cond.items()}

    # ---- 変換 -------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量行列 (LightGBM にそのまま入れられる DataFrame) を返す."""
        if self.rank_prior is None:
            raise RuntimeError("先に fit してください")
        return self._build(df)[self.feature_names]

    def _speed_base(self, runs: pd.DataFrame) -> pd.Series:
        """各レースの基準タイム (コース・距離の中央値. 少なければ芝ダ・距離の中央値)."""
        dist = runs["distance"].round().astype("Int64").astype("string")
        fine = (runs["venue"].astype("string") + "|" + runs["surface"].astype("string") + "|" + dist).map(self.speed_base)
        coarse = (runs["surface"].astype("string") + "|" + dist).map(self.speed_base_coarse)
        return pd.to_numeric(fine, errors="coerce").fillna(pd.to_numeric(coarse, errors="coerce"))

    def _speed_figures(self, df: pd.DataFrame) -> pd.DataFrame:
        """直近 3 走のスピード指数 (基準タイムより 1000m あたり何秒速いか. 大きいほど速い)."""
        runs = _past_runs(df, keep_missing=True)
        base = self._speed_base(runs)
        cond_key = runs["surface"].astype("string") + "|" + runs["track_condition"].astype("string")
        cond = pd.to_numeric(cond_key.map(self.speed_condition), errors="coerce").fillna(1.0)
        speed = (base * cond - runs["time"]) / runs["distance"] * 1000
        wide = speed.to_numpy().reshape(N_SPEED_RACES, len(df)).T  # 行 = 馬, 列 = 前走, 2 走前, 3 走前
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # 全部欠損の行の nanmax / nanmean
            return pd.DataFrame(
                {
                    "speed_last1": wide[:, 0],
                    "speed_best3": np.nanmax(wide, axis=1),
                    "speed_avg3": np.nanmean(wide, axis=1),
                },
                index=df.index,
            )

    def _rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """騎手・調教師・父・母父ごとの 勝率・3 着以内率・出走数 (対象レースの開催日より前のレースだけで計算).

        学習データの各行は「その日より前」の成績だけを使うので, 結果のリークにならない.
        推論 (学習期間より後のレース) では, 学習期間全体の成績を使う.
        少ない出走数のブレを抑えるため, 全体の平均を RATE_SMOOTHING 回分足して割る.
        """
        out = pd.DataFrame(index=df.index)
        query = pd.DataFrame({"race_date": pd.to_datetime(df["race_date"]).to_numpy(), "row": np.arange(len(df))})
        for col in RATE_COLUMNS:
            events = self.rate_events.get(col)
            n = np.zeros(len(df))
            win = np.zeros(len(df))
            top3 = np.zeros(len(df))
            if events is not None and len(events):
                ev = events.sort_values(["key", "race_date"]).copy()
                for c in ("n", "win", "top3"):
                    ev[c] = ev.groupby("key")[c].cumsum()
                # merge_asof は キー・日付の型が完全に一致している必要がある
                ev["key"] = ev["key"].astype(str).astype(object)
                ev["race_date"] = pd.to_datetime(ev["race_date"]).astype("datetime64[ns]")
                keys = df[col].astype("string").to_numpy()
                q = query.assign(key=keys).dropna(subset=["key"])
                q["key"] = q["key"].astype(str).astype(object)
                q["race_date"] = q["race_date"].astype("datetime64[ns]")
                merged = pd.merge_asof(
                    q.sort_values("race_date"), ev.sort_values("race_date"), on="race_date", by="key",
                    allow_exact_matches=False,  # 同じ日のレースの結果は使わない
                ).dropna(subset=["n"])
                rows = merged["row"].to_numpy(dtype=int)
                n[rows], win[rows], top3[rows] = merged["n"], merged["win"], merged["top3"]
            m = RATE_SMOOTHING
            out[f"{col}_n_runs"] = n
            out[f"{col}_win_rate"] = (win + m * self.rate_priors.get("win", 0.0)) / (n + m)
            out[f"{col}_top3_rate"] = (top3 + m * self.rate_priors.get("top3", 0.0)) / (n + m)
        return out

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        excluded = set(CATEGORICAL_COLUMNS) | set(RATE_COLUMNS) | set(DROPPED_COLUMNS) | set(ODDS_COLUMNS)
        base_cols = [c for c in feature_columns() if c not in excluded]
        X = df[base_cols].apply(pd.to_numeric, errors="coerce").astype(float)

        # 今回のレース条件に対応する平均着順・出走回数 (+ 全体平均に寄せた値)
        for kind, suffix in _condition_suffixes(df).items():
            avg = _pick_by_row(df, "avg_rank_" + suffix)
            n = _pick_by_row(df, "n_starts_" + suffix)
            n0 = np.nan_to_num(n, nan=0.0)
            X[f"cur_avg_rank_{kind}"] = avg
            X[f"cur_n_starts_{kind}"] = n
            X[f"shr_avg_rank_{kind}"] = (
                np.nan_to_num(avg, nan=0.0) * n0 + self.rank_prior * SHRINK_STRENGTH
            ) / (n0 + SHRINK_STRENGTH)

        # スピード指数, 騎手などの成績
        X = pd.concat([X, self._speed_figures(df), self._rates(df)], axis=1)

        # 展開: レース内の逃げ・先行馬 (直近 3 走の最初のコーナーの位置が前の方) の頭数
        front = (X["hist_avg3_first_corner"] < FRONT_RUNNER_RATIO).astype(float)
        X["race_n_front_runners"] = front.groupby(df["race_id"].to_numpy()).transform("sum").to_numpy()

        # 他の馬との比較: レース内での順位と, レース内平均との差
        relative_cols = [f"shr_avg_rank_{k}" for k in _CONDITION_KINDS] + RELATIVE_COLUMNS
        groups = X[relative_cols].groupby(df["race_id"].to_numpy())
        ranks = groups.rank(method="average", pct=True)
        diffs = X[relative_cols] - groups.transform("mean")
        X = pd.concat([X, ranks.add_suffix("_race_rank"), diffs.add_suffix("_race_diff")], axis=1)

        # カテゴリ変数 (学習時に決めた種類に揃える. 学習時に少なかった・未知のものは "__other__")
        for col in CATEGORICAL_COLUMNS:
            values = df[col].astype("string")
            known = self.categories.get(col, [OTHER])
            values = values.where(values.isna() | values.isin(known), OTHER)
            X[col] = pd.Categorical(values, categories=known)

        return X

    # ---- 保存・読み込み -------------------------------------------------------
    def to_dict(self) -> dict:
        """保存用の dict. 騎手などの成績は, 学習期間全体の合計だけを保存する
        (推論は学習期間より後のレースなので, 合計だけで同じ結果になる)."""
        rate_totals, as_of = {}, None
        for col, ev in self.rate_events.items():
            totals = ev.groupby("key")[["n", "win", "top3"]].sum()
            rate_totals[col] = {k: [int(r.n), float(r.win), float(r.top3)] for k, r in totals.iterrows()}
            if len(ev):
                last = pd.Timestamp(ev["race_date"].max())
                as_of = last if as_of is None else max(as_of, last)
        return {
            "min_category_count": self.min_category_count,
            "categories": self.categories,
            "rank_prior": self.rank_prior,
            "feature_names": self.feature_names,
            "rate_priors": self.rate_priors,
            "rate_as_of": str(as_of.date()) if as_of is not None else None,
            "rate_totals": rate_totals,
            "speed_base": self.speed_base,
            "speed_base_coarse": self.speed_base_coarse,
            "speed_condition": self.speed_condition,
            "market_offset": self.market_offset,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureBuilder":
        builder = cls(min_category_count=d["min_category_count"])
        builder.categories = d["categories"]
        builder.rank_prior = d["rank_prior"]
        builder.feature_names = d["feature_names"]
        builder.rate_priors = d.get("rate_priors", {})
        builder.market_offset = bool(d.get("market_offset", False))
        # 合計は「学習期間の最終日」の 1 回分の成績として持つ (それより後のレースにだけ使われる)
        as_of = pd.Timestamp(d["rate_as_of"]) if d.get("rate_as_of") else None
        builder.rate_events = {
            col: pd.DataFrame(
                [(k, as_of, n, w, t) for k, (n, w, t) in totals.items()],
                columns=["key", "race_date", "n", "win", "top3"],
            )
            for col, totals in d.get("rate_totals", {}).items()
            if as_of is not None
        }
        builder.speed_base = d.get("speed_base", {})
        builder.speed_base_coarse = d.get("speed_base_coarse", {})
        builder.speed_condition = d.get("speed_condition", {})
        return builder


def _past_runs(df: pd.DataFrame, keep_missing: bool = False) -> pd.DataFrame:
    """各馬の直近 3 走 (hist_last{k}_*) を縦に並べる (前走の全馬, 2 走前の全馬, 3 走前の全馬 の順).

    keep_missing=False なら, タイム・距離がない行を除く.
    """
    parts = []
    for k in range(1, N_SPEED_RACES + 1):
        part = pd.DataFrame(
            {
                "time": pd.to_numeric(df[f"hist_last{k}_time"], errors="coerce").to_numpy(),
                "distance": pd.to_numeric(df[f"hist_last{k}_distance"], errors="coerce").to_numpy(),
                "surface": df[f"hist_last{k}_surface"].astype("string").to_numpy(),
                "venue": df[f"hist_last{k}_venue"].astype("string").to_numpy(),
                "track_condition": df[f"hist_last{k}_track_condition"].astype("string").to_numpy(),
            }
        )
        parts.append(part)
    runs = pd.concat(parts, ignore_index=True)
    if not keep_missing:
        runs = runs.dropna(subset=["time", "distance"])
        runs = runs[runs["distance"] > 0]
    return runs


def labeled(df: pd.DataFrame) -> pd.DataFrame:
    """目的変数 (is_win) が入っている行 = 結果確定済みのレースだけを返す."""
    return df[df[TARGET_COLUMN].notna()].reset_index(drop=True)
