"""発走が近いレースを予測して通知する. GitHub Actions から 10 分おきに実行する想定.

1 回の実行でやること:
    1. その日の時刻表を取得 (初回のみ. 以降は状態ファイルに保存したものを使う)
    2. 「発走 minutes_before_max 分前 〜 minutes_before_min 分前」に入っていて, まだ通知していないレースを選ぶ
       (馬体重は発走 60 分前に発表されるので, デフォルトは発走 50〜10 分前)
    3. 各レースの出馬表・戦績・血統を取得 → 学習済みモデルで予測 → ワイドの確率を計算 → 通知
    4. 通知したレースを状態ファイルに記録 (次の実行で重複して通知しないため)

使い方:
    python -m racing_prediction.inference.run                      # 通常 (今の時刻で判定)
    python -m racing_prediction.inference.run --dry-run            # 通知せずに本文を表示
    python -m racing_prediction.inference.run --race-id 202605050611 --dry-run   # 時刻に関係なく指定レースを予測
    python -m racing_prediction.inference.run --venues 東京,京都 --race-numbers 7-12
    python -m racing_prediction.inference.run --prepare-only       # 時刻表を取得して, 対象レースがある日かだけ判定する
                                                                   # (GITHUB_OUTPUT に race_day=true/false を書く)

PYTHONPATH に src を通しておくこと. 通知先の設定は racing_prediction.inference.notify を参照.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from racing_prediction.betting.optimize_bet import best_wide_boxes, wide_expected_values
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


# ---------------------------------------------------------------------------
# 状態ファイル (その日の時刻表と, 通知済みのレース)
# ---------------------------------------------------------------------------
def load_state(path: Path, date: pd.Timestamp) -> dict:
    """状態ファイルを読む. 日付が変わっていれば新しい状態にする."""
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("date") == str(date.date()):
            return state
    return {"date": str(date.date()), "schedule": None, "notified": [], "failures": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def schedule_from_state(state: dict) -> pd.DataFrame:
    df = pd.DataFrame(state["schedule"], columns=["race_id", "venue", "race_number", "race_name", "course", "start_time"])
    df["start_time"] = pd.to_datetime(df["start_time"])
    return df


def schedule_to_state(schedule: pd.DataFrame) -> list[dict]:
    return schedule.assign(start_time=schedule["start_time"].astype(str)).to_dict("records")


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


def due_races(
    schedule: pd.DataFrame,
    now: pd.Timestamp,
    notified: list[str],
    minutes_before_min: int,
    minutes_before_max: int,
    venues: list[str] | None = None,
    race_numbers: set[int] | None = None,
) -> pd.DataFrame:
    """今予測すべきレース (発走 max〜min 分前の範囲に入っていて, まだ通知していない) を返す.

    GitHub Actions の定期実行は数分〜十数分遅れることがあるため, 範囲は実行間隔 (10 分) より広くとり,
    重複は notified で防ぐ.
    """
    df = target_races(schedule[~schedule["race_id"].isin(notified)], venues, race_numbers)
    minutes_to_start = (df["start_time"] - now).dt.total_seconds() / 60
    return df[minutes_to_start.between(minutes_before_min, minutes_before_max)]


# ---------------------------------------------------------------------------
# 予測と通知
# ---------------------------------------------------------------------------
def predict_race(race_info: dict, booster, builder, betting: BettingConfig | None = None) -> tuple[str, str]:
    """1 レースを取得・予測し, その時点のワイドのオッズから期待値とおすすめボックスを計算して,
    通知の (件名, 本文) を返す. オッズが取れなくても, 予測だけで通知する.
    """
    table = build_race_table(race_info["race_id"])
    pred = predict_top3(table, booster, builder).reset_index(drop=True)
    in_top = simulate_top3(pred)
    wide = wide_probabilities(pred, in_top=in_top)

    wide_ev = boxes = None
    try:
        odds = fetch_wide_odds(race_info["race_id"])
        status = {"middle": "発売中", "result": "確定"}.get(odds.status, odds.status or "状態不明")
        odds_note = f"{odds.official_datetime or '時刻不明'} 時点 ({status})"
        if odds.odds.empty:
            odds_note = f"ワイドのオッズがまだありません ({status})"
        else:
            wide_ev = wide_expected_values(wide, odds.odds, betting)
            boxes = best_wide_boxes(pred, in_top, wide_ev, betting)
    except Exception as e:  # noqa: BLE001 - オッズが取れなくても予測は通知する
        odds_note = f"取得エラー: {type(e).__name__}"
        print(f"{race_info['race_id']}: ワイドのオッズを取得できませんでした: {e}")
    return format_race_report(race_info, pred, wide, wide_ev=wide_ev, boxes=boxes, odds_note=odds_note)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="発走が近いレースを予測して通知する")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="model.txt と metadata.json があるフォルダ")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE, help="時刻表と通知済みレースを記録するファイル")
    parser.add_argument("--venues", default="", help="対象の競馬場 (カンマ区切り. 例: 東京,京都). 空なら全場")
    parser.add_argument("--race-numbers", default="7-12", help="対象のレース番号 (例: 7-12, 9,10,11)")
    parser.add_argument("--minutes-before-max", type=int, default=50, help="発走何分前から予測するか")
    parser.add_argument("--minutes-before-min", type=int, default=10, help="発走何分前まで予測するか")
    parser.add_argument("--race-id", help="時刻に関係なくこのレースを予測する (動作確認用)")
    parser.add_argument("--now", help="現在時刻を上書きする (動作確認用. 例: '2026-11-22 14:30')")
    parser.add_argument("--dry-run", action="store_true", help="通知せずに本文を表示する")
    parser.add_argument(
        "--betting-config", default=DEFAULT_BETTING_CONFIG, help="ワイドの期待値・おすすめボックスの設定 (なければ既定値)"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="時刻表を取得して状態ファイルに保存し, 対象レースがある日かを判定するだけ (予測はしない)",
    )
    args = parser.parse_args(argv)

    now = pd.Timestamp(args.now) if args.now else pd.Timestamp(datetime.now(JST).replace(tzinfo=None))
    date = now.normalize()
    state_path = Path(args.state_file)
    state = load_state(state_path, date)

    if state["schedule"] is None:
        schedule = fetch_race_schedule(date)
        state["schedule"] = schedule_to_state(schedule)
        save_state(state_path, state)
    schedule = schedule_from_state(state)
    print(f"{now:%Y-%m-%d %H:%M} (JST) / この日のレース: {len(schedule)} / 通知済み: {len(state['notified'])}")
    venues = [v.strip() for v in args.venues.split(",") if v.strip()] or None
    race_numbers = parse_race_numbers(args.race_numbers)

    if args.prepare_only:
        # 時刻表の取得に失敗した場合は上の fetch_race_schedule で例外になり, ここには来ない
        # (= 取得失敗を非開催日と取り違えない)
        targets = target_races(schedule, venues, race_numbers)
        race_day = not targets.empty
        print(f"対象レース: {len(targets)} ({'開催日' if race_day else '対象レースなし (非開催日)'})")
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
                f.write(f"race_day={'true' if race_day else 'false'}\n")
        return 0

    if args.race_id:
        races = schedule[schedule["race_id"] == args.race_id]
        if races.empty:  # 別の日のレースなど. 時刻表にない情報は最低限で埋める
            races = pd.DataFrame([{
                "race_id": args.race_id, "venue": "", "race_number": int(args.race_id[10:12]),
                "race_name": "", "course": "", "start_time": now,
            }])
    else:
        races = due_races(
            schedule, now, state["notified"], args.minutes_before_min, args.minutes_before_max,
            venues=venues, race_numbers=race_numbers,
        )

    if races.empty:
        print("今予測するレースはありません")
        return 0

    booster, builder, metadata = load_artifacts(args.model_dir)
    betting = load_betting_config(args.betting_config)
    print(f"モデル: 学習期間 {metadata['train_period']} / 対象 {len(races)} レース: {races['race_id'].tolist()}")

    failed = []
    for race_info in races.to_dict("records"):
        race_id = race_info["race_id"]
        try:
            subject, body = predict_race(race_info, booster, builder, betting)
            if args.dry_run:
                print("=" * 40, subject, body, sep="\n")
            else:
                sent = notify(subject, body)
                print(f"{race_id}: 通知しました ({', '.join(sent)})")
                state["notified"].append(race_id)
                save_state(state_path, state)
        except Exception:
            print(f"{race_id}: 失敗しました (次回の実行で再試行します)")
            traceback.print_exc()
            state["failures"][race_id] = state["failures"].get(race_id, 0) + 1
            save_state(state_path, state)
            failed.append(race_id)

    # 失敗があれば終了コードを 1 にする (GitHub Actions の実行が失敗扱いになり, 気づける)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
