"""model_features

data/processed/races 以下の CSV を読み込み, LightGBM に入れる特徴量行列を作る.

学習データだけから決める値 (カテゴリの種類, 平均着順の事前値) は FeatureBuilder.fit で学習し,
推論時も同じ FeatureBuilder (保存して読み込んだもの) で transform することで,
学習と推論の前処理を一致させる.

追加する特徴量:
    cur_*      : 今回のレース条件 (芝/ダート, 距離区分, 馬場状態, 競馬場) に対応する平均着順・出走回数
    shr_*      : cur_* の平均着順を, 出走回数が少ないほど全体平均に寄せた値 (少数レースのブレ対策)
    *_race_rank: レース内での順位 (0〜1, 小さいほど上位). 他の馬との比較を表す
    *_race_diff: レース内平均との差
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from racing_prediction.data.make_dataset import TARGET_COLUMN, feature_columns, output_columns
from racing_prediction.features.build_features import VENUES

# LightGBM のカテゴリ変数として扱う列
CATEGORICAL_COLUMNS = [
    "race_venue",
    "race_distance_category",
    "race_surface",
    "race_track_condition",
    "sex",
    "jockey_name",
    "trainer_name",
    "sire",
    "sire_line",
    "dam",
    "sire_sire",
    "sire_dam",
    "dam_sire",
    "dam_dam",
]

# 学習データでの出現回数がこれ未満のカテゴリは "__other__" にまとめる
OTHER = "__other__"

# 平均着順を全体平均に寄せる強さ (この回数分だけ「全体平均の成績」を足したとみなす)
SHRINK_STRENGTH = 3.0

# 競馬場名 -> 列名の接尾辞 ("東京" -> "tokyo")
_VENUE_KEYS = {name: key for key, name in VENUES.items()}

# 今回のレース条件に対応させる区分
_CONDITION_KINDS = ["surface", "distance", "condition", "venue"]


def load_race_csvs(races_dir: str | Path) -> pd.DataFrame:
    """races_dir 以下の全 CSV ({競馬場}/{開催日}/{R}R.csv) を読み込んで縦に結合する.

    必要な列が欠けている CSV (列追加前に保存した古いもの) は警告を出して読み飛ばす.
    """
    paths = sorted(Path(races_dir).rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"{races_dir} に CSV がありません")

    required = output_columns()
    tables, skipped = [], []
    for path in paths:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str}, parse_dates=["race_date"])
        if any(c not in df.columns for c in required):
            skipped.append(path)
            continue
        tables.append(df[required])

    if skipped:
        warnings.warn(
            f"列が足りない古い CSV を {len(skipped)} 件読み飛ばしました (例: {skipped[0]}). "
            "collect_races を再実行すると取り直されます."
        )
    if not tables:
        raise ValueError("読み込める CSV がありませんでした")

    df = pd.concat(tables, ignore_index=True)
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

        self.feature_names = list(self._build(df.head(1)).columns)
        return self

    # ---- 変換 -------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量行列 (LightGBM にそのまま入れられる DataFrame) を返す."""
        if self.rank_prior is None:
            raise RuntimeError("先に fit してください")
        return self._build(df)[self.feature_names]

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        base_cols = [c for c in feature_columns() if c not in CATEGORICAL_COLUMNS]
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

        # 他の馬との比較: レース内での順位と, レース内平均との差
        relative_cols = [f"shr_avg_rank_{k}" for k in _CONDITION_KINDS] + [
            "weight_carried",
            "horse_weight",
            "age",
        ]
        groups = X[relative_cols].groupby(df["race_id"].to_numpy())
        ranks = groups.rank(method="average", pct=True)
        diffs = X[relative_cols] - groups.transform("mean")
        for col in relative_cols:
            X[f"{col}_race_rank"] = ranks[col]
            X[f"{col}_race_diff"] = diffs[col]

        # カテゴリ変数 (学習時に決めた種類に揃える. 学習時に少なかった・未知のものは "__other__")
        for col in CATEGORICAL_COLUMNS:
            values = df[col].astype("string")
            known = self.categories.get(col, [OTHER])
            values = values.where(values.isna() | values.isin(known), OTHER)
            X[col] = pd.Categorical(values, categories=known)

        return X

    # ---- 保存・読み込み -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "min_category_count": self.min_category_count,
            "categories": self.categories,
            "rank_prior": self.rank_prior,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureBuilder":
        builder = cls(min_category_count=d["min_category_count"])
        builder.categories = d["categories"]
        builder.rank_prior = d["rank_prior"]
        builder.feature_names = d["feature_names"]
        return builder


def labeled(df: pd.DataFrame) -> pd.DataFrame:
    """目的変数 (is_top3) が入っている行 = 結果確定済みのレースだけを返す."""
    return df[df[TARGET_COLUMN].notna()].reset_index(drop=True)
