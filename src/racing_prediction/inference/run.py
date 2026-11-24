"""発走 12 分前に推論を始め, その時点のオッズで期待値を計算して通知する. GitHub Actions で開催日に常駐して実行する.

「発走 12 分前」(--odds-minutes-before で変更可. カンマ区切りで複数も可) になったら推論を始め
(出馬表・馬体重・戦績・血統と, その時点の単勝オッズを取得して予測. 1 レース 1〜2 分), 続けてその時点の
ワイドのオッズで期待値プラスのワイド ボックスを計算し, メールを送る (1 レースにつき 1 通).

単勝オッズはモデルの特徴量なので, 推論のたびにその時点のオッズを取得する. 発売中 (または確定) のオッズが
取れなければ推論を失敗扱いにして再試行する (--race-id の手動実行だけは, 発売前の予想オッズでも予測する).

1 回の実行でやること:
    1. その日の時刻表を取得 (初回のみ. 以降は状態ファイルに保存したものを使う)
    2. その日の通知が残っている間, 次のレースの「発走 12 分前」まで待つ → 推論 → オッズ取得 → 通知 を繰り返す.
       起動が遅れてタイミングを過ぎていても, 発走前 (--min-minutes-before-start 分前まで) ならすぐ推論して送る.
       失敗したら --retry 回まで試す.
    3. 送ったタイミングを状態ファイルに記録する (次の実行で重複して送らないため)
    4. --max-runtime-minutes を過ぎてから始まる通知が残っていれば, GITHUB_OUTPUT に continue=true を書いて終了する
       (GitHub Actions のジョブは最長 6 時間のため. ワークフローが次の実行を起動して続きを処理する)

使い方:
    python -m racing_prediction.inference.run                      # 通常 (今の時刻で判定)
    python -m racing_prediction.inference.run --dry-run --no-wait  # 通知せず・待たずに本文を表示
    python -m racing_prediction.inference.run --race-id 202605050611 --dry-run   # 時刻に関係なく今のオッズで予測
                                                                   # (発売前なら予想オッズで予測する)
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
from racing_prediction.data.scrape_odds import LIVE_ODDS_STATUSES, fetch_wide_odds, fetch_win_odds
from racing_prediction.inference.notify import notify
from racing_prediction.inference.report import format_race_report
from racing_prediction.models.predict_model import load_artifacts, predict_win, simulate_top3, wide_probabilities

JST = ZoneInfo("Asia/Tokyo")
DEFAULT_MODEL_DIR = "models/production"
DEFAULT_STATE_FILE = ".inference_state/state.json"
DEFAULT_BETTING_CONFIG = "configs/betting.yaml"

# 状態ファイルに保存する予測結果の列 (メールの作成に使う)
PRED_COLUMNS = [
    "race_id", "race_date", "race_venue", "race_surface", "race_track_condition", "n_horses",
    "umaban", "horse_name", "p_win", "pred_rank", "horse_weight", "weight_diff", "win_odds", "popularity",
]

ODDS_STATUS_JA = {"middle": "発売中", "result": "確定", "yoso": "発売前の予想オッズ"}


class OddsNotAvailable(RuntimeError):
    """推論に使える単勝オッズがまだない (発売前など)."""


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


def pending_notifications(
    targets: pd.DataFrame,
    now: pd.Timestamp,
    sent: dict,
    odds_minutes: list[int],
    min_minutes_before_start: float,
) -> list[tuple[pd.Timestamp, dict, int]]:
    """まだ送っていない通知のうち, 今から発走前 (発走 min_minutes_before_start 分前まで) に送れるものを時刻順に返す.

    通知のタイミング (発走 m 分前) を過ぎていても, 発走前なら返す (起動が遅れた場合でも送るため).
    同じレースで複数のタイミングを過ぎている場合は, 一番遅いタイミングだけ返す (古いオッズで重ねて送らない).

    Returns:
        [(推論を始める時刻, レース情報, 発走何分前), ...]
    """
    events = []
    for race in targets.to_dict("records"):
        if race["start_time"] - pd.Timedelta(minutes=min_minutes_before_start) <= now:
            continue
        unsent = [
            (race["start_time"] - pd.Timedelta(minutes=m), race, m)
            for m in odds_minutes
            if m not in sent.get(race["race_id"], [])
        ]
        passed = [e for e in unsent if e[0] <= now]
        events += ([passed[-1]] if passed else []) + [e for e in unsent if e[0] > now]
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


def apply_live_win_odds(table: pd.DataFrame, race_id: str, allow_preliminary: bool = False) -> tuple[pd.DataFrame, dict]:
    """今の単勝オッズを取得して, 出馬表の win_odds・popularity 列を置き換える. (置き換えた表, オッズの情報) を返す.

    発売中・確定のオッズが取れなければ OddsNotAvailable を出す
    (allow_preliminary=True なら, 発売前の予想オッズ (status="yoso") でも使う).
    オッズがない馬 (取消など) は欠損になる.
    """
    odds = fetch_win_odds(race_id)
    ok = odds.status in LIVE_ODDS_STATUSES or (allow_preliminary and odds.status == "yoso")
    if odds.odds.empty or not ok:
        raise OddsNotAvailable(f"{race_id}: 推論に使える単勝オッズがありません (状態: {odds.status or '不明'}, {len(odds.odds)} 頭分)")
    table = table.drop(columns=["win_odds", "popularity"], errors="ignore").merge(odds.odds, on="umaban", how="left")
    missing = table.loc[table["win_odds"].isna(), "umaban"].tolist()
    if missing:
        print(f"{race_id}: 単勝オッズがない馬番 {missing} (取消などの可能性)")
    status = ODDS_STATUS_JA.get(odds.status, odds.status or "状態不明")
    info = {"win_odds_status": odds.status, "win_odds_note": f"{odds.official_datetime or '時刻不明'} 時点 ({status})"}
    return table, info


def predict_race(
    race_info: dict, booster, builder, table: pd.DataFrame | None = None, allow_preliminary: bool = False
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """1 レースを予測する. (補ったレース情報, 予測結果, 出馬表) を返す.

    table (取得済みの出馬表・戦績・血統) がなければ取得する. 単勝オッズはモデルの特徴量なので, 毎回その時点の
    オッズを取得して使う.
    """
    if table is None:
        table = build_race_table(race_info["race_id"])
    race_info = complete_race_info(race_info, table)
    with_odds, odds_info = apply_live_win_odds(table, race_info["race_id"], allow_preliminary=allow_preliminary)
    print(f"{race_info['race_id']}: 単勝オッズ {odds_info['win_odds_note']} で予測します")
    pred = predict_win(with_odds, booster, builder).reset_index(drop=True)
    return {**race_info, **odds_info}, pred, table


def model_version_label(model_dir: str | Path) -> str:
    """モデルのバージョン (models/production/VERSION.txt. scripts/model_registry.py が書く), 学習期間,
    テストの ROC-AUC (全レース / レース内) を 1 行にまとめる."""
    model_dir = Path(model_dir)
    version_file = model_dir / "VERSION.txt"
    version = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else "バージョン不明"
    try:
        meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return version
    details = []
    if meta.get("train_period"):
        details.append(f"学習期間 {'〜'.join(meta['train_period'])}")
    test = meta.get("metrics", {}).get("test", {})
    if test.get("roc_auc") is not None:
        auc = f"テストのROC-AUC {test['roc_auc']:.3f}"
        if test.get("roc_auc_by_race") is not None:
            auc += f" (レース内 {test['roc_auc_by_race']:.3f})"
        details.append(auc)
    return f"{version} ({' / '.join(details)})" if details else version


def build_report(
    race_info: dict, pred: pd.DataFrame, betting: BettingConfig, label: str | None = None, model_version: str | None = None
) -> tuple[str, str]:
    """予測結果と, 今のワイドのオッズから全ボックスの期待値を計算し, 通知の (件名, 本文) を返す.

    オッズが取れなくても, 予測だけで通知文を作る.
    """
    pred = pred.reset_index(drop=True)
    # 1 着の確率 (p_win) から着順をシミュレーションし, 3 着以内の確率とワイドの確率を出す
    in_top = simulate_top3(pred)  # 乱数の種が固定なので, 同じ予測からは同じ結果になる
    pred = pred.assign(p_top3=in_top.mean(axis=0))
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
        max_display=betting.box_max_display, odds_note=odds_note, label=label, model_version=model_version,
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
    parser.add_argument(
        "--min-minutes-before-start", type=float, default=1,
        help="起動が遅れてタイミングを過ぎていても, 発走何分前までなら推論・通知するか",
    )
    # GitHub Actions のジョブは最長 6 時間. それより前に終了し, 続きは次の実行 (ワークフローが自分で起動する) に任せる
    parser.add_argument(
        "--max-runtime-minutes", type=float, default=0,
        help="この分数を過ぎてから始まる通知は次の実行に任せて終了する (GITHUB_OUTPUT に continue=true を書く). 0 なら無制限",
    )
    parser.add_argument("--retry", type=int, default=3, help="推論・通知に失敗したときに何回まで試すか")
    parser.add_argument("--race-id", help="時刻に関係なく, このレースを今のオッズで予測する (動作確認用)")
    parser.add_argument("--now", help="現在時刻を上書きする (動作確認用. 例: '2026-11-22 14:30'. 待機せず, 残りを全部すぐ処理する)")
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
            print(f"モデル: {model_version_label(args.model_dir)}")
            model.update(booster=booster, builder=builder)
        return model["booster"], model["builder"]

    tables: dict[str, pd.DataFrame] = {}  # 取得済みの出馬表・戦績・血統 (同じ実行の中で使い回す)

    def ensure_prediction(race_info: dict, fresh: bool = False, allow_preliminary: bool = False) -> tuple[dict, pd.DataFrame]:
        """今の単勝オッズで予測し, 状態ファイルに保存する.

        単勝オッズはモデルの特徴量なので, 予測は毎回やり直す. 出馬表・戦績・血統は, 同じ実行の中で取得済みなら
        使い回す (fresh=True なら取り直す. 馬体重・出走取消などを反映するため).
        """
        race_id = race_info["race_id"]
        table = None if fresh else tables.get(race_id)
        print(f"{race_id}: 予測しています" + ("..." if table is None else " (取得済みの出馬表を使い, オッズだけ取り直します)..."))
        info, pred, table = predict_race(race_info, *get_model(), table=table, allow_preliminary=allow_preliminary)
        tables[race_id] = table
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
        info, pred = ensure_prediction(race_info, fresh=True, allow_preliminary=True)
        send(*build_report(info, pred, betting, label="手動実行", model_version=model_version_label(args.model_dir)), args.dry_run)
        return 0

    # 常駐: その日の通知が残っている間, 「発走 12 分前」になるまで待ち, その時点で推論を始めて
    # (出馬表・馬体重・戦績を取得して予測 → その時点のワイドのオッズで期待値を計算) メールを送る.
    # GitHub Actions の定期実行 (schedule) は数十分起動しないことがあるため, 起動のタイミングに頼らず
    # 1 回の実行の中で時刻まで待つ.
    started = now_jst()
    deadline = started + pd.Timedelta(minutes=args.max_runtime_minutes) if args.max_runtime_minutes > 0 else None
    sent_local = {k: list(v) for k, v in state["sent"].items()}  # dry-run でも同じ通知を繰り返さないため
    gave_up: set[tuple[str, int]] = set()
    attempts: dict[tuple[str, int], int] = {}
    failed = []
    to_continue = False

    while True:
        events = [
            e for e in pending_notifications(targets, current_time(), sent_local, odds_minutes, args.min_minutes_before_start)
            if (e[1]["race_id"], e[2]) not in gave_up
        ]
        if not events:
            print("今日これから送る通知はありません")
            break
        at, race, m = events[0]
        race_id = race["race_id"]
        if deadline is not None and at > deadline and fixed_now is None:
            # 次の実行に引き継ぐ. 次のタイミングまで余裕があるうちに終了し, 次の実行の起動を待つ時間を確保する
            print(f"実行時間の上限が近いため終了します. 次は {race_id} 発走{m}分前 ({at:%H:%M}) から次の実行で処理します")
            to_continue = True
            break
        wait = (at - current_time()).total_seconds()
        if wait > 0 and not no_wait:
            print(f"{race_id}: 発走{m}分前 ({at:%H:%M}) まで {wait / 60:.1f} 分待ちます", flush=True)
            time.sleep(wait)
        elif wait < -30:
            print(f"{race_id}: 発走{m}分前 ({at:%H:%M}) を {-wait / 60:.1f} 分過ぎていますが, 発走前なので今から推論します")
        key = (race_id, m)
        attempts[key] = attempts.get(key, 0) + 1
        try:
            # その時点の単勝オッズで推論する. 最初のタイミング (発走 12 分前) では出馬表も取り直す (馬体重・出走取消などを反映)
            info, pred = ensure_prediction(race, fresh=(m == odds_minutes[0]))
            send(*build_report(info, pred, betting, label=f"発走{m}分前", model_version=model_version_label(args.model_dir)), args.dry_run)
            sent_local.setdefault(race_id, []).append(m)
            if not args.dry_run:
                state["sent"].setdefault(race_id, []).append(m)
                save_state(state_path, state)
        except Exception:
            print(f"{race_id}: 発走{m}分前の推論・通知に失敗しました ({attempts[key]}/{args.retry} 回目)")
            traceback.print_exc()
            state["failures"][race_id] = state["failures"].get(race_id, 0) + 1
            save_state(state_path, state)
            if attempts[key] >= args.retry:
                gave_up.add(key)
                failed.append(race_id)
            elif not no_wait:
                time.sleep(20)

    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"continue={'true' if to_continue else 'false'}\n")
    # 失敗があれば終了コードを 1 にする (GitHub Actions の実行が失敗扱いになり, 気づける)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
