"""今日これから送る通知が残っているかを, 状態ファイルだけで素早く判定する (標準ライブラリのみ・ネットアクセスなし).

race-inference ワークフローは, 通知が残っているときだけ重いライブラリ (pandas, lightgbm など) を
インストールして常駐する. 判定の条件は run.pending_notifications と揃えること.

判定:
    - 今日の状態ファイルがない (今日の初回) → 残っている (時刻表の取得が必要なので実行する)
    - 今日の時刻表が空 (非開催日) → 残っていない
    - 発走 (min_minutes_before_start 分前) より前で, まだ送っていない通知がある対象レースがある → 残っている
    - それ以外 → 残っていない

結果は GITHUB_OUTPUT に run=true / run=false と remaining=true / remaining=false で書き出す (両者は同じ値).
remaining=false なら, その日はもう起動する必要がない (ワークフローを無効にする).

使い方:
    python src/racing_prediction/inference/precheck.py --state-file .inference_state/state.json \
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


def next_notification(
    state_file: Path, now: datetime, venues: list[str], race_numbers: set[int], odds_minutes: list[int],
    min_minutes_before_start: float,
) -> tuple[bool, str]:
    """(今日これから送る通知が残っているか, 理由) を返す."""
    state = _load_today(state_file, now)
    if state is None:
        return True, "今日の時刻表をまだ取得していない"
    if not state["schedule"]:
        return False, "今日は開催がない"
    pending = []
    for race in _targets(state, venues, race_numbers):
        start = datetime.fromisoformat(race["start_time"])
        if start - timedelta(minutes=min_minutes_before_start) <= now:
            continue
        sent = state.get("sent", {}).get(race["race_id"], [])
        pending += [(start - timedelta(minutes=m), race["race_id"], m) for m in odds_minutes if m not in sent]
    if not pending:
        return False, "今日これから送る通知はない"
    at, race_id, m = min(pending)
    return True, f"次の通知: {race_id} 発走{m}分前 {at:%H:%M} (残り {len(pending)} 件)"


def main() -> None:
    parser = argparse.ArgumentParser(description="今日これから送る通知が残っているかを判定する")
    parser.add_argument("--state-file", default=".inference_state/state.json")
    parser.add_argument("--venues", default="", help="対象の競馬場名 (カンマ区切り). 空なら全場")
    parser.add_argument("--race-numbers", default="7-12")
    parser.add_argument("--odds-minutes-before", default="12")
    parser.add_argument("--min-minutes-before-start", type=float, default=1)  # run.py と揃える
    parser.add_argument("--now", help="現在時刻を上書きする (動作確認用. 例: '2026-11-22 14:30')")
    args = parser.parse_args()

    now = datetime.fromisoformat(args.now) if args.now else datetime.now(JST).replace(tzinfo=None)
    venues = [v.strip() for v in args.venues.split(",") if v.strip()]
    remaining, reason = next_notification(
        Path(args.state_file), now, venues, parse_race_numbers(args.race_numbers),
        parse_minutes(args.odds_minutes_before), args.min_minutes_before_start,
    )
    print(f"{now:%Y-%m-%d %H:%M} (JST): {'実行する' if remaining else '実行しない'} - {reason}")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as f:
            value = "true" if remaining else "false"
            f.write(f"run={value}\nremaining={value}\n")


if __name__ == "__main__":
    main()
