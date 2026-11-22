"""推論を実行する必要があるかを, 状態ファイルだけで素早く判定する (標準ライブラリのみ・ネットアクセスなし).

GitHub Actions は毎日 10 分おきに起動するため, 非開催日や「今は予測するレースがない」時間帯は
重いライブラリ (pandas, lightgbm など) をインストールする前に終わらせる.

判定:
    - 今日の状態ファイルがない (今日の初回) → 実行する (時刻表の取得が必要)
    - 今日の時刻表が空 (非開催日) → 実行しない
    - 発走 minutes_before_max 〜 minutes_before_min 分前で, まだ通知していない対象レースがある → 実行する
    - それ以外 → 実行しない

結果は GITHUB_OUTPUT に run=true / run=false で書き出す (なければ標準出力に表示).
あわせて, 今日これから予測できる (まだ通知しておらず, 発走 minutes_before_min 分前を過ぎていない)
対象レースが残っているかを remaining=true / remaining=false で書き出す.
remaining=false なら, その日はもう起動する必要がない (ワークフローを無効にして無料枠を節約する).

使い方:
    python src/racing_prediction/inference/precheck.py --state-file .inference_state/state.json \\
        --venues 東京 --race-numbers 7-12
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


def _pending_targets(state: dict, now: datetime, venues: list[str], race_numbers: set[int]) -> list[tuple[dict, float]]:
    """まだ通知していない対象レースと, 発走までの分数のリスト."""
    notified = set(state.get("notified", []))
    pending = []
    for race in state["schedule"]:
        if race["race_id"] in notified:
            continue
        if venues and race["venue"] not in venues:
            continue
        if race_numbers and int(race["race_number"]) not in race_numbers:
            continue
        minutes_to_start = (datetime.fromisoformat(race["start_time"]) - now) / timedelta(minutes=1)
        pending.append((race, minutes_to_start))
    return pending


def has_remaining(
    state_file: Path, now: datetime, venues: list[str], race_numbers: set[int], minutes_before_min: int
) -> bool:
    """今日これから予測できる対象レースが残っているか. 時刻表が未取得なら (わからないので) True."""
    if not state_file.exists():
        return True
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if state.get("date") != now.date().isoformat() or state.get("schedule") is None:
        return True
    return any(m >= minutes_before_min for _, m in _pending_targets(state, now, venues, race_numbers))


def should_run(
    state_file: Path,
    now: datetime,
    venues: list[str],
    race_numbers: set[int],
    minutes_before_min: int,
    minutes_before_max: int,
) -> tuple[bool, str]:
    """(実行するか, 理由) を返す."""
    if not state_file.exists():
        return True, "今日の時刻表をまだ取得していない"
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if state.get("date") != now.date().isoformat() or state.get("schedule") is None:
        return True, "今日の時刻表をまだ取得していない"
    if not state["schedule"]:
        return False, "今日は開催がない"

    for race, minutes_to_start in _pending_targets(state, now, venues, race_numbers):
        if minutes_before_min <= minutes_to_start <= minutes_before_max:
            return True, f"予測するレースがある ({race['race_id']})"
    return False, "今予測するレースはない"


def main() -> None:
    parser = argparse.ArgumentParser(description="推論を実行する必要があるかを判定する")
    parser.add_argument("--state-file", default=".inference_state/state.json")
    parser.add_argument("--venues", default="", help="対象の競馬場名 (カンマ区切り). 空なら全場")
    parser.add_argument("--race-numbers", default="7-12")
    parser.add_argument("--minutes-before-max", type=int, default=50)
    parser.add_argument("--minutes-before-min", type=int, default=10)
    parser.add_argument("--now", help="現在時刻を上書きする (動作確認用. 例: '2026-11-22 14:30')")
    args = parser.parse_args()

    now = datetime.fromisoformat(args.now) if args.now else datetime.now(JST).replace(tzinfo=None)
    venues = [v.strip() for v in args.venues.split(",") if v.strip()]
    state_file, race_numbers = Path(args.state_file), parse_race_numbers(args.race_numbers)
    run, reason = should_run(
        state_file, now, venues, race_numbers, args.minutes_before_min, args.minutes_before_max
    )
    remaining = has_remaining(state_file, now, venues, race_numbers, args.minutes_before_min)
    print(f"{now:%Y-%m-%d %H:%M} (JST): {'実行する' if run else '実行しない'} - {reason}")
    print(f"今日これから予測できる対象レース: {'あり' if remaining else 'なし'}")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as f:
            f.write(f"run={'true' if run else 'false'}\n")
            f.write(f"remaining={'true' if remaining else 'false'}\n")


if __name__ == "__main__":
    main()
