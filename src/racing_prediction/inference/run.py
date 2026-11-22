"""発走前に予測し, 発走直前のオッズで期待値を計算して通知する. GitHub Actions から 5 分おきに実行する想定.

オッズは発走直前によく変わるため, 「発走 12 分前」(--odds-minutes-before で変更可. カンマ区切りで複数も可) の
時点のオッズで期待値プラスのワイド ボックスを計算し, メールを送る (1 レースにつき 1 通).

1 回の実行でやること:
    1. その日の時刻表を取得 (初回のみ. 以降は状態ファイルに保存したものを使う)
    2. 予測: 発走 50〜14 分前のレースの出馬表・戦績・血統を取得して予測し, 結果を状態ファイルに保存する
       (馬体重は発走 60 分前に発表される. 予測には 1 レース 1〜2 分かかるので, オッズのタイミングより前に済ませておく)
    3. 通知: 今から --lookahead 分以内に「発走 12 分前」が来るレースがあれば, その時刻まで待ち,
       その時点のワイドのオッズで全ボックスの期待値を計算してメールを送る (保存した予測を使うので数秒で送れる).
       GitHub Actions の起動が遅れて時刻を過ぎていても, --late-tolerance 分以内 (かつ発走前) なら送る.
    4. 送ったタイミングを状態ファイルに記録する (次の実行で重複して送らないため)

使い方:
    python -m racing_prediction.inference.run                      # 通常 (今の時刻で判定)
    python -m racing_prediction.inference.run --dry-run --no-wait  # 通知せず・待たずに本文を表示
    python -m racing_prediction.inference.run --race-id 202605050611 --dry-run   # 時刻に関係なく今のオッズで予測
    python -m racing_prediction.inference.run --venues 東京,京都 --race-numbers 7-12 --odds-minutes-before 12
    python -m racing_prediction.inference.run --prepare-only       # 時刻表を取得して, 対象レースがある日かだけ判定する
                                                                   # (GITHUB_OUTPUT に race_day=true/false を書く)

PYTHONPATH に src を通しておくこと. 通知先の設定は racing_prediction.inference.notify を参照.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from racing_prediction.betting.optimize_bet import add_box_hit_rates, all_box_expected_values, wide_expected_values
from racing_prediction.config import BettingConfig, load_betting_config
from racing_prediction.data.make_dataset import build_race_table
from racing_prediction.data.race_list import fetch_race_schedule, venue_code
from racing_prediction.data.scrape_odds import fetch_wide_odds
from racing_prediction.inference.notify import notify
from racing_prediction.inference.report import format_race_report
from racing_prediction.models.predict_model import load_artifacts, predict_top3, simulate_top3, wide_probabilities

JST = ZoneInfo("Asia/Tokyo")
DEFAULT_MODEL_DIR = "models/production"
DEFAULT_STATE_FILE = ".inference_state/state.json"
DEFAULT_BETTING_CONFIG = "configs/betting.yaml"

# 状態ファイルに保存する予測結果の列 (メールの作成に使う)
PRED_COLUMNS = [
    "race_id", "race_date", "race_venue", "race_surface", "race_track_condition", "n_horses",
    "umaban", "horse_name", "p_top3", "pred_rank", "horse_weight", "weight_diff",
]


def now_jst() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(JST).replace(tzinfo=None))


# ---------------------------------------------------------------------------
# 状態ファイル (その日の時刻表, 予測結果, 送った通知)
# ---------------------------------------------------------------------------
def load_state(path: Path, date: pd.Timestamp) -> dict:
    """状態ファイルを読む. 日付が変わっていれば新しい状態にする."""
    state = None
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("date") != str(date.date()):
            state = None
    state = state or {"date": str(date.date()), "schedule": None}
    state.setdefault("predictions", {})  # race_id -> {"race_info": ..., "pred": [...]}
    state.setdefault("sent", {})  # race_id -> [送った「発走何分前」のリスト]
    state.setdefault("failures", {})
    return state


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def schedule_from_state(state: dict) -> pd.DataFrame:
    df = pd.DataFrame(state["schedule"], columns=["race_id", "venue", "race_number", "race_name", "course", "start_time"])
    df["start_time"] = pd.to_datetime(df["start_time"])
    return df


def schedule_to_state(schedule: pd.DataFrame) -> list[dict]:
    return schedule.assign(start_time=schedule["start_time"].astype(str)).to_dict("records")


def pred_to_state(pred: pd.DataFrame) -> list[dict]:
    df = pred[PRED_COLUMNS].copy()
    df["race_date"] = df["race_date"].astype(str)
    return df.astype(object).where(df.notna(), None).to_dict("records")


def pred_from_state(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records, columns=PRED_COLUMNS)
    df["race_date"] = pd.to_datetime(df["race_date"])
    for col in ("horse_weight", "weight_diff"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 対象レースの選択
# ---------------------------------------------------------------------------
def parse_race_numbers(text: str) -> set[int]:
    """"7-12" や "9,10,11" を {7, ..., 12} にする."""
    numbers = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            numbers |= set(range(int(lo), int(hi) + 1))
        elif part:
            numbers.add(int(part))
    return numbers


def parse_minutes(text: str) -> list[int]:
    """"12,7" -> [12, 7] (大きい順 = 早いタイミング順)."""
    return sorted({int(x) for x in text.split(",") if x.strip()}, reverse=True)


def target_races(
    schedule: pd.DataFrame, venues: list[str] | None = None, race_numbers: set[int] | None = None
) -> pd.DataFrame:
    """時刻表のうち, 対象の競馬場・レース番号のレースだけを返す."""
    df = schedule
    if venues:
        codes = {venue_code(v) for v in venues}
        df = df[df["race_id"].str[4:6].isin(codes)]
    if race_numbers:
        df = df[df["race_number"].isin(race_numbers)]
    return df


def races_to_predict(
    targets: pd.DataFrame, now: pd.Timestamp, predicted: dict, minutes_before_min: float, minutes_before_max: float
) -> pd.DataFrame:
    """予測すべきレース (発走 max〜min 分前で, まだ予測していない) を返す."""
    df = targets[~targets["race_id"].isin(predicted)]
    minutes_to_start = (df["start_time"] - now).dt.total_seconds() / 60
    return df[minutes_to_start.between(minutes_before_min, minutes_before_max)]


def due_notifications(
    targets: pd.DataFrame,
    now: pd.Timestamp,
    sent: dict,
    odds_minutes: list[int],
    lookahead: float,
    late_tolerance: float,
) -> list[tuple[pd.Timestamp, dict, int]]:
    """今から lookahead 分以内に通知のタイミング (発走 m 分前) が来る, まだ送っていない通知を時刻順に返す.

    Returns:
        [(通知する時刻, レース情報, 発走何分前), ...]
    """
    events = []
    for race in targets.to_dict("records"):
        if race["start_time"] <= now:
            continue
        for m in odds_minutes:
            if m in sent.get(race["race_id"], []):
                continue
            at = race["start_time"] - pd.Timedelta(minutes=m)
            if now - pd.Timedelta(minutes=late_tolerance) <= at <= now + pd.Timedelta(minutes=lookahead):
                events.append((at, race, m))
    return sorted(events, key=lambda e: e[0])


# ---------------------------------------------------------------------------
# 予測と通知文の作成
# ---------------------------------------------------------------------------
def complete_race_info(race_info: dict, table: pd.DataFrame) -> dict:
    """レース名・コース・発走時刻がない場合 (手動で別の日のレース ID を指定したときなど) は,
    出馬表の開催日の時刻表を取得して補う. 取れなければそのまま返す.
    """
    if race_info.get("race_name") and race_info.get("course"):
        return race_info
    try:
        schedule = fetch_race_schedule(pd.Timestamp(table["race_date"].iloc[0]))
        match = schedule[schedule["race_id"] == race_info["race_id"]]
        if not match.empty:
            return {**race_info, **match.iloc[0].to_dict()}
    except Exception as e:  # noqa: BLE001 - レース情報が補えなくても予測は通知する
        print(f"{race_info['race_id']}: レース情報を補えませんでした: {e}")
    return race_info


def predict_race(race_info: dict, booster, builder) -> tuple[dict, pd.DataFrame]:
    """1 レースの出馬表・戦績・血統を取得して予測する. (補ったレース情報, 予測結果) を返す."""
    table = build_race_table(race_info["race_id"])
    race_info = complete_race_info(race_info, table)
    pred = predict_top3(table, booster, builder).reset_index(drop=True)
    return race_info, pred


def build_report(
    race_info: dict, pred: pd.DataFrame, betting: BettingConfig, label: str | None = None
) -> tuple[str, str]:
    """予測結果と, 今のワイドのオッズから全ボックスの期待値を計算し, 通知の (件名, 本文) を返す.

    オッズが取れなくても, 予測だけで通知文を作る.
    """
    in_top = simulate_top3(pred)  # 乱数の種が固定なので, 同じ予測からは同じ結果になる
    wide = wide_probabilities(pred, in_top=in_top)
    boxes = n_boxes_total = None
    try:
        odds = fetch_wide_odds(race_info["race_id"])
        status = {"middle": "発売中", "result": "確定"}.get(odds.status, odds.status or "状態不明")
        odds_note = f"{odds.official_datetime or '時刻不明'} 時点 ({status})"
        if odds.odds.empty:
            odds_note = f"ワイドのオッズがまだありません ({status})"
        else:
            wide_ev = wide_expected_values(wide, odds.odds, betting)
            positive_boxes, n_boxes_total = all_box_expected_values(pred, wide_ev)
            # 期待値がプラスのボックスを, 的中率の高い順に並べる (同じ的中率なら期待回収率の高い順)
            boxes = add_box_hit_rates(positive_boxes, pred, in_top).sort_values(
                ["hit_rate", "return_rate"], ascending=[False, False], ignore_index=True
            )
    except Exception as e:  # noqa: BLE001 - オッズが取れなくても予測は通知する
        odds_note = f"取得エラー: {type(e).__name__}"
        print(f"{race_info['race_id']}: ワイドのオッズを取得できませんでした: {e}")
    return format_race_report(
        race_info, pred, wide, boxes=boxes, n_boxes_total=n_boxes_total,
        max_display=betting.box_max_display, odds_note=odds_note, label=label,
    )


def send(subject: str, body: str, dry_run: bool) -> None:
    if dry_run:
        print("=" * 40, subject, body, sep="\n")
    else:
        sent = notify(subject, body)
        print(f"通知しました ({', '.join(sent)}): {subject}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="発走前に予測し, 発走直前のオッズで期待値を計算して通知する")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="model.txt と metadata.json があるフォルダ")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE, help="時刻表・予測結果・送った通知を記録するファイル")
    parser.add_argument("--betting-config", default=DEFAULT_BETTING_CONFIG, help="ワイドの期待値・ボックスの設定")
    parser.add_argument("--venues", default="", help="対象の競馬場 (カンマ区切り. 例: 東京,京都). 空なら全場")
    parser.add_argument("--race-numbers", default="7-12", help="対象のレース番号 (例: 7-12, 9,10,11)")
    parser.add_argument("--odds-minutes-before", default="12", help="オッズを取得して通知するタイミング (発走何分前. カンマ区切りで複数も可)")
    parser.add_argument("--predict-minutes-before-max", type=float, default=50, help="発走何分前から予測するか")
    # 5 分おきの起動で取りこぼさない範囲で小さくする (待機時間も GitHub Actions の実行時間に数えられるため)
    parser.add_argument("--lookahead", type=float, default=7, help="今から何分先までの通知のタイミングを, 待って処理するか")
    parser.add_argument("--late-tolerance", type=float, default=3, help="通知のタイミングを何分過ぎていても送るか")
    parser.add_argument("--race-id", help="時刻に関係なく, このレースを今のオッズで予測する (動作確認用)")
    parser.add_argument("--now", help="現在時刻を上書きする (動作確認用. 例: '2026-11-22 14:30'. 待機もしない)")
    parser.add_argument("--no-wait", action="store_true", help="通知のタイミングまで待たずにすぐ処理する (動作確認用)")
    parser.add_argument("--dry-run", action="store_true", help="通知せずに本文を表示する")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="時刻表を取得して状態ファイルに保存し, 対象レースがある日かを判定するだけ (予測はしない)",
    )
    args = parser.parse_args(argv)

    fixed_now = pd.Timestamp(args.now) if args.now else None
    no_wait = args.no_wait or fixed_now is not None

    def current_time() -> pd.Timestamp:
        return fixed_now if fixed_now is not None else now_jst()

    now = current_time()
    state_path = Path(args.state_file)
    state = load_state(state_path, now.normalize())

    if state["schedule"] is None:
        state["schedule"] = schedule_to_state(fetch_race_schedule(now.normalize()))
        save_state(state_path, state)
    schedule = schedule_from_state(state)
    venues = [v.strip() for v in args.venues.split(",") if v.strip()] or None
    targets = target_races(schedule, venues, parse_race_numbers(args.race_numbers))
    odds_minutes = parse_minutes(args.odds_minutes_before)
    sent_total = sum(len(v) for v in state["sent"].values())
    print(f"{now:%Y-%m-%d %H:%M} (JST) / 対象レース: {len(targets)} / 予測済み: {len(state['predictions'])} / 送った通知: {sent_total}")

    if args.prepare_only:
        # 時刻表の取得に失敗した場合は上の fetch_race_schedule で例外になり, ここには来ない
        # (= 取得失敗を非開催日と取り違えない)
        race_day = not targets.empty
        print(f"{'開催日' if race_day else '対象レースなし (非開催日)'}")
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
                f.write(f"race_day={'true' if race_day else 'false'}\n")
        return 0

    betting = load_betting_config(args.betting_config)
    model = {}

    def get_model():
        if not model:
            booster, builder, metadata = load_artifacts(args.model_dir)
            print(f"モデル: 学習期間 {metadata['train_period']}")
            model.update(booster=booster, builder=builder)
        return model["booster"], model["builder"]

    def ensure_prediction(race_info: dict) -> tuple[dict, pd.DataFrame]:
        """予測済みならそれを使い, なければ予測して状態ファイルに保存する."""
        saved = state["predictions"].get(race_info["race_id"])
        if saved:
            return {**race_info, **saved["race_info"]}, pred_from_state(saved["pred"])
        print(f"{race_info['race_id']}: 予測しています...")
        info, pred = predict_race(race_info, *get_model())
        state["predictions"][race_info["race_id"]] = {
            "race_info": {k: (str(v) if k == "start_time" else v) for k, v in info.items()},
            "pred": pred_to_state(pred),
        }
        save_state(state_path, state)
        return info, pred

    # 手動実行: 指定レースを今のオッズで予測して 1 通送る
    if args.race_id:
        races = schedule[schedule["race_id"] == args.race_id]
        race_info = races.iloc[0].to_dict() if not races.empty else {
            "race_id": args.race_id, "venue": "", "race_number": int(args.race_id[10:12]),
            "race_name": "", "course": "", "start_time": now,
        }
        info, pred = ensure_prediction(race_info)
        send(*build_report(info, pred, betting, label="手動実行"), args.dry_run)
        return 0

    failed = []

    # 1) 予測: 発走 50〜(最初の通知タイミング + 2) 分前のレースを, 通知より前に予測しておく
    #    (通知のタイミングが近いレースを優先する)
    events = due_notifications(targets, now, state["sent"], odds_minutes, args.lookahead, args.late_tolerance)
    need = races_to_predict(targets, now, state["predictions"], max(odds_minutes) + 2, args.predict_minutes_before_max)
    to_predict = [race for _, race, _ in events] + need.sort_values("start_time").to_dict("records")
    seen = set()
    for race in to_predict:
        if race["race_id"] in seen or race["race_id"] in state["predictions"]:
            continue
        seen.add(race["race_id"])
        try:
            ensure_prediction(race)
        except Exception:
            print(f"{race['race_id']}: 予測に失敗しました (次回の実行で再試行します)")
            traceback.print_exc()
            state["failures"][race["race_id"]] = state["failures"].get(race["race_id"], 0) + 1
            save_state(state_path, state)
            failed.append(race["race_id"])

    # 2) 通知: 発走 m 分前になるまで待ち, その時点のオッズで期待値を計算して送る
    events = due_notifications(targets, current_time(), state["sent"], odds_minutes, args.lookahead, args.late_tolerance)
    if not events:
        print("今から送る通知はありません")
    for at, race, m in events:
        race_id = race["race_id"]
        if race_id not in state["predictions"]:
            continue  # 予測に失敗したレース
        wait = (at - current_time()).total_seconds()
        if wait > 0 and not no_wait:
            print(f"{race_id}: 発走{m}分前 ({at:%H:%M}) まで {wait / 60:.1f} 分待ちます", flush=True)
            time.sleep(wait)
        if current_time() >= race["start_time"]:
            print(f"{race_id}: 発走時刻を過ぎたため, 発走{m}分前の通知を送りません")
            continue
        try:
            info, pred = ensure_prediction(race)
            send(*build_report(info, pred, betting, label=f"発走{m}分前"), args.dry_run)
            if not args.dry_run:
                state["sent"].setdefault(race_id, []).append(m)
                save_state(state_path, state)
        except Exception:
            print(f"{race_id}: 発走{m}分前の通知に失敗しました")
            traceback.print_exc()
            failed.append(race_id)

    # 失敗があれば終了コードを 1 にする (GitHub Actions の実行が失敗扱いになり, 気づける)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
