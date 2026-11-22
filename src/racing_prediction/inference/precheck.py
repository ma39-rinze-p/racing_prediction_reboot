"""推論を実行する必要があるかを, 状態ファイルだけで素早く判定する (標準ライブラリのみ・ネットアクセスなし).

GitHub Actions は開催日に 5 分おきに起動するため, 「今は予測も通知もない」時間帯は
重いライブラリ (pandas, lightgbm など) をインストールする前に終わらせる.
判定の条件は run.py と揃えること.

判定:
    - 今日の状態ファイルがない (今日の初回) → 実行する (時刻表の取得が必要)
    - 今日の時刻表が空 (非開催日) → 実行しない
    - 発走 50〜(最初の通知タイミング + 2) 分前で, まだ予測していない対象レースがある → 実行する (予測)
    - 今から lookahead 分以内に, まだ送っていない通知のタイミング (発走 12 分前) がある → 実行する (通知)
    - それ以外 → 実行しない

結果は GITHUB_OUTPUT に run=true / run=false で書き出す (なければ標準出力に表示).
あわせて, 今日これから送る通知が残っているかを remaining=true / remaining=false で書き出す.
remaining=false なら, その日はもう起動する必要がない (ワークフローを無効にして無料枠を節約する).

使い方:
    python src/racing_prediction/inference/precheck.py --state-file .inference_state/state.json \\
        --venues 東京 --race-numbers 7-12 --odds-minutes-before 12
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")


def parse_race_numbers(text: str) -> set[int]:
    """"7-12" や "9,10,11" を {7, ..., 12} にする (run.parse_race_numbers と同じ)."""
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
    """"12,7" -> [12, 7] (run.parse_minutes と同じ)."""
    return sorted({int(x) for x in text.split(",") if x.strip()}, reverse=True)


def _load_today(state_file: Path, now: datetime) -> dict | None:
    """今日の時刻表を取得済みの状態ファイルなら読み込む. なければ None."""
    if not state_file.exists():
        return None
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if state.get("date") != now.date().isoformat() or state.get("schedule") is None:
        return None
    return state


def _targets(state: dict, venues: list[str], race_numbers: set[int]) -> list[dict]:
    return [
        race for race in state["schedule"]
        if (not venues or race["venue"] in venues) and (not race_numbers or int(race["race_number"]) in race_numbers)
    ]


def _unsent_notifications(state: dict, race: dict, odds_minutes: list[int]) -> list[tuple[int, datetime]]:
    """まだ送っていない通知の (発走何分前, 通知する時刻)."""
    start = datetime.fromisoformat(race["start_time"])
    sent = state.get("sent", {}).get(race["race_id"], [])
    return [(m, start - timedelta(minutes=m)) for m in odds_minutes if m not in sent]


def should_run(
    state_file: Path,
    now: datetime,
    venues: list[str],
    race_numbers: set[int],
    odds_minutes: list[int],
    predict_minutes_before_max: float,
    lookahead: float,
    late_tolerance: float,
) -> tuple[bool, str]:
    """(実行するか, 理由) を返す."""
    state = _load_today(state_file, now)
    if state is None:
        return True, "今日の時刻表をまだ取得していない"
    if not state["schedule"]:
        return False, "今日は開催がない"

    predicted = state.get("predictions", {})
    for race in _targets(state, venues, race_numbers):
        start = datetime.fromisoformat(race["start_time"])
        if start <= now:
            continue
        minutes_to_start = (start - now) / timedelta(minutes=1)
        if race["race_id"] not in predicted and max(odds_minutes) + 2 <= minutes_to_start <= predict_minutes_before_max:
            return True, f"予測するレースがある ({race['race_id']})"
        for m, at in _unsent_notifications(state, race, odds_minutes):
            if now - timedelta(minutes=late_tolerance) <= at <= now + timedelta(minutes=lookahead):
                return True, f"通知のタイミングが近い ({race['race_id']} 発走{m}分前 {at:%H:%M})"
    return False, "今は予測も通知もない"


def has_remaining(
    state_file: Path, now: datetime, venues: list[str], race_numbers: set[int], odds_minutes: list[int], late_tolerance: float
) -> bool:
    """今日これから送る通知が残っているか. 時刻表が未取得なら (わからないので) True."""
    state = _load_today(state_file, now)
    if state is None:
        return True
    for race in _targets(state, venues, race_numbers):
        if datetime.fromisoformat(race["start_time"]) <= now:
            continue
        if any(at >= now - timedelta(minutes=late_tolerance) for _, at in _unsent_notifications(state, race, odds_minutes)):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="推論を実行する必要があるかを判定する")
    parser.add_argument("--state-file", default=".inference_state/state.json")
    parser.add_argument("--venues", default="", help="対象の競馬場名 (カンマ区切り). 空なら全場")
    parser.add_argument("--race-numbers", default="7-12")
    parser.add_argument("--odds-minutes-before", default="12")
    parser.add_argument("--predict-minutes-before-max", type=float, default=50)
    parser.add_argument("--lookahead", type=float, default=7)  # run.py と揃える
    parser.add_argument("--late-tolerance", type=float, default=3)
    parser.add_argument("--now", help="現在時刻を上書きする (動作確認用. 例: '2026-11-22 14:30')")
    args = parser.parse_args()

    now = datetime.fromisoformat(args.now) if args.now else datetime.now(JST).replace(tzinfo=None)
    venues = [v.strip() for v in args.venues.split(",") if v.strip()]
    state_file, race_numbers = Path(args.state_file), parse_race_numbers(args.race_numbers)
    odds_minutes = parse_minutes(args.odds_minutes_before)
    run, reason = should_run(
        state_file, now, venues, race_numbers, odds_minutes,
        args.predict_minutes_before_max, args.lookahead, args.late_tolerance,
    )
    remaining = has_remaining(state_file, now, venues, race_numbers, odds_minutes, args.late_tolerance)
    print(f"{now:%Y-%m-%d %H:%M} (JST): {'実行する' if run else '実行しない'} - {reason}")
    print(f"今日これから送る通知: {'あり' if remaining else 'なし'}")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as f:
            f.write(f"run={'true' if run else 'false'}\n")
            f.write(f"remaining={'true' if remaining else 'false'}\n")


if __name__ == "__main__":
    main()
