"""推論 (GitHub Actions) で使うモデルのバージョン管理.

    models/registry/
        registry.json             各バージョンの情報 (学習期間・テストの精度など) と, 適用の履歴
        v20261124_153012/ ...     各バージョンのモデルと評価結果 (下の「バージョンのフォルダの中身」)
    models/production/            今適用しているバージョンのコピー (推論はここを読む)
        + VERSION.txt             適用中のバージョン名
        + METRICS.md              適用中のモデルの精度 (ROC-AUC など. 単勝オッズとの比較) と ROC 曲線

バージョン名: v{学習した日付_時刻} (日本時間. 例: v20261124_153012).
    metadata.json の trained_at (学習時に記録) から決める. 同じモデルを 2 回取り込んでも同じバージョン名になり,
    二重には登録しない.

バージョンのフォルダの中身:
    model.txt, metadata.json, feature_importance.csv   モデル (推論に使う)
    summary.json      テストの評価指標 (モデルと単勝オッズ), 学習・テスト期間, 最良パラメータ
    roc_curve.png     テスト期間の ROC 曲線 (モデルと単勝オッズ)
    calibration.csv   予測確率帯ごとの実際の 1 着率

適用するモデルの選び方 (registry.json の "policy"):
    "best"   (既定・自動モード): 1 着の確率を予測するモデル (目的変数 is_win) のうち, テストの ROC-AUC が最も高いものを適用する.
             新しいモデルを取り込むたび (scripts/train_on_kaggle.py / fetch_kaggle_model.py から自動で呼ばれる) に選び直す.
    "manual" (手動モード)     : promote / rollback で選んだバージョンに固定する (新しいモデルを取り込んでも切り替えない).
             best コマンドで自動モードに戻る.

使い方 (リポジトリのルートで実行):
    python scripts/model_registry.py list                       # バージョンの一覧 (* が適用中, ★ が ROC-AUC 最良)
    python scripts/model_registry.py best                       # 自動モードに戻し, ROC-AUC が最良のバージョンを適用する
    python scripts/model_registry.py rollback                   # 1 つ前に適用していたモデルに戻す (手動モードになる)
    python scripts/model_registry.py promote v20261124_153012   # 指定したバージョンを適用する (手動モードになる)
    python scripts/model_registry.py register <フォルダ>        # model.txt と metadata.json があるフォルダを登録する
        (評価結果は <フォルダ>/results か, <フォルダ>/../results にあれば一緒に登録する.
         --force: 登録したモデルを適用 (手動モード) / --no-promote: 登録だけ)

適用 (promote / rollback) したら, GitHub Actions に反映するため commit & push する:
    git add models && git commit -m "switch production model to v20261124_153012" && git push
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "models" / "registry"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"
PRODUCTION_DIR = ROOT / "models" / "production"
MODEL_FILES = ["model.txt", "metadata.json", "feature_importance.csv"]
REQUIRED_FILES = ["model.txt", "metadata.json"]
# 評価結果 (ノートブックの results/ の出力) のうち, バージョンごとに残すもの
RESULT_FILES = ["summary.json", "roc_curve.png", "calibration.csv"]
METRICS_FILE = "METRICS.md"
JST = ZoneInfo("Asia/Tokyo")

# 最良のモデルを選ぶ指標 (各モデルを学習したときのテスト期間の ROC-AUC. 大きいほど良い)
PRIMARY_METRIC = "roc_auc"
# 推論で使えるモデルの目的変数 (racing_prediction.models.predict_model.MODEL_TARGET と揃える).
# metadata.json に target がないモデルは, 1 着モデル導入前の 3 着以内モデル (is_top3)
MODEL_TARGET = "is_win"
LEGACY_TARGET = "is_top3"


# ---------------------------------------------------------------------------
# 管理簿 (registry.json)
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")


def _read_metadata(model_dir: Path) -> dict:
    return json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))


def version_name(metadata: dict) -> str:
    """バージョン名 v{学習した日付_時刻} (日本時間). 学習日時の記録がなければ今の日時."""
    trained_at = metadata.get("trained_at")
    when = datetime.fromisoformat(trained_at).astimezone(JST) if trained_at else datetime.now(JST)
    return "v" + when.strftime("%Y%m%d_%H%M%S")


def _summary(model_dir: Path) -> dict:
    """metadata.json から, 一覧に表示する情報を取り出す."""
    meta = _read_metadata(model_dir)
    metrics = meta.get("metrics", {})
    test = metrics.get("test", {})
    odds = metrics.get("test_odds_baseline", {})
    keys = ("roc_auc_by_race", "roc_auc", "logloss", "win_hit", "top3_precision", "wide_top_pair_hit")
    return {
        "target": meta.get("target", LEGACY_TARGET),
        "trained_at": meta.get("trained_at"),
        "train_period": meta.get("train_period"),
        "n_train_races": meta.get("n_train_races"),
        "test_period": metrics.get("test_period"),
        "test_metrics": {k: test.get(k) for k in keys if k in test},
        "test_odds_baseline": {k: odds.get(k) for k in keys if k in odds},
    }


def _target(info: dict) -> str:
    return info.get("target", LEGACY_TARGET)


def load_registry() -> dict:
    """管理簿を読み込む. まだなければ空の管理簿を作る
    (models/production に 1 着モデルがあれば, それを最初のバージョンとして登録する)."""
    if REGISTRY_FILE.exists():
        registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        registry.setdefault("policy", "best")
        return registry
    registry = {"active": None, "policy": "best", "versions": {}, "history": [], "stack": []}
    if all((PRODUCTION_DIR / f).exists() for f in REQUIRED_FILES) and _read_metadata(PRODUCTION_DIR).get("target") == MODEL_TARGET:
        version = version_name(_read_metadata(PRODUCTION_DIR))
        _copy_version_files(PRODUCTION_DIR, REGISTRY_DIR / version)
        registry["versions"][version] = {
            "registered_at": _now(), "source": "管理簿の作成前の models/production", **_summary(REGISTRY_DIR / version),
        }
        registry["active"] = version
        registry["stack"] = [version]
        registry["history"].append({"version": version, "activated_at": _now(), "reason": "初期登録"})
        save_registry(registry)
        _write_version_file(version)
        print(f"今の models/production を {version} として登録しました")
    return registry


def save_registry(registry: dict) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_version_files(src: Path, dest: Path, results_dir: Path | None = None) -> None:
    """モデル (src) と評価結果 (results_dir. なければ src) をバージョンのフォルダにコピーする."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
    for name in RESULT_FILES:
        found = next((d / name for d in (results_dir, src) if d is not None and (d / name).exists()), None)
        if found is not None:
            shutil.copy2(found, dest / name)


def _find_results_dir(model_dir: Path) -> Path | None:
    """モデルのフォルダに対応する評価結果のフォルダ (Kaggle の出力では artifacts/ と並ぶ results/)."""
    for d in (model_dir / "results", model_dir.parent / "results"):
        if (d / "summary.json").exists():
            return d
    return None


def _write_version_file(version: str) -> None:
    (PRODUCTION_DIR / "VERSION.txt").write_text(version + "\n", encoding="utf-8")


def _fmt(value, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _write_metrics_file(version: str, info: dict) -> None:
    """適用中のモデルの精度を models/production/METRICS.md にまとめる (GitHub の画面でも読める)."""
    m, o = info.get("test_metrics", {}), info.get("test_odds_baseline", {})
    rows = [
        ("ROC-AUC (1 着か否か)", "roc_auc", 4),
        ("ROC-AUC (レースごとの平均)", "roc_auc_by_race", 4),
        ("logloss (レース単位. 小さいほど良い)", "logloss", 4),
        ("予測 1 位の馬が 1 着だった割合", "win_hit", 3),
        ("予測上位 3 頭のうち 3 着以内だった割合", "top3_precision", 3),
        ("予測上位 2 頭のワイドが的中した割合", "wide_top_pair_hit", 3),
    ]
    lines = [
        f"# 適用中のモデル: {version}",
        "",
        "`scripts/model_registry.py` が, モデルを適用するたびに書き直すファイル (手で編集しない).",
        "",
        "| 項目 | 内容 |",
        "|---|---|",
        f"| 学習した日時 | {info.get('trained_at') or '-'} |",
        f"| 学習期間 | {'〜'.join(info.get('train_period') or ['-', '-'])} ({info.get('n_train_races') or '-'} レース) |",
        f"| テスト期間 | {'〜'.join(info.get('test_period') or ['-', '-'])} |",
        f"| 適用した日時 | {_now()} |",
        "",
        "## テスト期間の精度",
        "",
        "単勝オッズ = 確定単勝オッズから逆算した確率で予想した場合 (比較用. オッズはレース後に確定する値なので, モデルの特徴量には使っていない).",
        "",
        "| 指標 | モデル | 単勝オッズ |",
        "|---|---:|---:|",
        *[f"| {label} | {_fmt(m.get(key), d)} | {_fmt(o.get(key), d)} |" for label, key, d in rows],
        "",
        "## ROC 曲線 (テスト期間)",
        "",
        "![ROC 曲線](roc_curve.png)" if (PRODUCTION_DIR / "roc_curve.png").exists() else "(ROC 曲線の画像がありません)",
        "",
        "その他: `summary.json` (評価指標・最良パラメータ), `calibration.csv` (予測確率帯ごとの実際の 1 着率),",
        "`feature_importance.csv` (特徴量の重要度).",
    ]
    (PRODUCTION_DIR / METRICS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 操作
# ---------------------------------------------------------------------------
def best_version(registry: dict) -> str | None:
    """1 着モデル (is_win) のうち, テストの ROC-AUC が最も高いバージョン (同じなら新しい方).
    指標がないバージョンと, 3 着以内モデル (ROC-AUC の意味が違い, 推論にも使えない) は対象外."""
    scored = [(info["test_metrics"].get(PRIMARY_METRIC), v) for v, info in registry["versions"].items()
              if _target(info) == MODEL_TARGET and info.get("test_metrics", {}).get(PRIMARY_METRIC) is not None]
    return max(scored)[1] if scored else None


def apply_best(reason: str) -> None:
    """自動モードで, ROC-AUC が最良のバージョンを適用する (すでに適用中なら何もしない)."""
    registry = load_registry()
    best = best_version(registry)
    if best is None:
        print(f"テストの {PRIMARY_METRIC} が記録された 1 着モデル ({MODEL_TARGET}) がないため, 適用するモデルを選べません")
        return
    score = registry["versions"][best]["test_metrics"][PRIMARY_METRIC]
    if best == registry["active"]:
        print(f"ROC-AUC が最良のバージョンは {best} ({score:.4f}) で, すでに適用中です")
        return
    print(f"ROC-AUC が最良のバージョンは {best} ({score:.4f}) です")
    promote(best, reason=reason, manual=False)


def register(src_dir: Path, source: str = "", promote_mode: str = "auto", results_dir: Path | None = None) -> str:
    """モデルを新しいバージョンとして登録する. 登録したバージョン名を返す.

    バージョン名は v{学習した日付_時刻}. 同じバージョンが登録済みなら (同じモデルを取り込み直した場合),
    登録し直さずにそのバージョンを使う.
    results_dir: 評価結果 (summary.json, roc_curve.png, calibration.csv) のフォルダ. None なら src_dir の近くを探す.

    promote_mode:
        "auto"  : 自動モードなら, 全バージョンのうち ROC-AUC が最良のものを適用する (手動モードなら登録だけ)
        "force" : 登録したバージョンを適用する (手動モードになる)
        "none"  : 登録だけ
    """
    src_dir = Path(src_dir)
    missing = [f for f in REQUIRED_FILES if not (src_dir / f).exists()]
    if missing:
        raise SystemExit(f"{src_dir} に {missing} がありません")
    metadata = _read_metadata(src_dir)
    if metadata.get("target", LEGACY_TARGET) != MODEL_TARGET:
        raise SystemExit(f"{src_dir} は 1 着の確率を予測するモデル ({MODEL_TARGET}) ではないため登録しません")

    registry = load_registry()
    version = version_name(metadata)
    if version in registry["versions"]:
        print(f"{version} はすでに登録済みです (同じモデルを取り込み直しました)")
    else:
        results_dir = results_dir or _find_results_dir(src_dir)
        if results_dir is None:
            print("評価結果 (summary.json, roc_curve.png など) が見つからないため, モデルだけを登録します")
        _copy_version_files(src_dir, REGISTRY_DIR / version, results_dir)
        registry["versions"][version] = {"registered_at": _now(), "source": source or str(src_dir), **_summary(REGISTRY_DIR / version)}
        save_registry(registry)
        print(f"{version} として登録しました: {REGISTRY_DIR / version}")

    new_score = registry["versions"][version]["test_metrics"].get(PRIMARY_METRIC)
    print(f"{version} のテストの ROC-AUC: {new_score if new_score is None else f'{new_score:.4f}'}")
    if promote_mode == "none":
        return version
    if promote_mode == "force":
        promote(version, reason="新しいモデルを登録 (--force で適用)")
    elif registry.get("policy", "best") == "best":
        apply_best(reason=f"{version} を登録 (ROC-AUC が最良のモデルを自動で適用)")
    else:
        print(f"手動モードのため, 適用中の {registry['active']} のままにします. "
              "ROC-AUC が最良のモデルを自動で使うには: python scripts/model_registry.py best")
    return version


def promote(version: str, reason: str = "手動で適用", is_rollback: bool = False, manual: bool = True) -> None:
    """指定したバージョンを models/production にコピーして適用する (モデル・評価結果・METRICS.md).

    適用したバージョンは「適用の積み重ね (stack)」に積む. rollback はこれを 1 つずつさかのぼる
    (rollback で戻したときは積まない. そのため rollback を続けると, 適用してきた順に古い方へ戻っていく).
    manual=True (promote / rollback コマンド) なら手動モードにする (新しいモデルを取り込んでも自動で切り替えない).
    """
    registry = load_registry()
    if version not in registry["versions"]:
        raise SystemExit(f"{version} は登録されていません (登録済み: {', '.join(sorted(registry['versions'])) or 'なし'})")
    info = registry["versions"][version]
    target = _target(info)
    if target != MODEL_TARGET:
        raise SystemExit(
            f"{version} は目的変数が {target} のモデル (3 着以内を予測する以前のモデル) のため, 今の推論では使えません. "
            f"1 着の確率を予測するモデル ({MODEL_TARGET}) を選んでください"
        )
    # 前のバージョンの評価結果が残らないよう, 入れ替える前に消す
    for name in MODEL_FILES + RESULT_FILES + [METRICS_FILE]:
        (PRODUCTION_DIR / name).unlink(missing_ok=True)
    _copy_version_files(REGISTRY_DIR / version, PRODUCTION_DIR)
    _write_version_file(version)
    _write_metrics_file(version, info)
    previous = registry["active"]
    registry["active"] = version
    registry["policy"] = "manual" if manual else "best"
    stack = registry.setdefault("stack", [previous] if previous else [])
    if not is_rollback and (not stack or stack[-1] != version):
        stack.append(version)
    registry["history"].append({"version": version, "activated_at": _now(), "reason": reason, "previous": previous})
    save_registry(registry)
    mode = "手動モード: 新しいモデルを取り込んでも自動で切り替えません. 戻すには best コマンド" if manual else "自動モード"
    print(f"{version} を適用しました ({previous} → {version}) [{mode}]. GitHub Actions に反映するには commit & push してください:")
    print(f'  git add models && git commit -m "switch production model to {version}" && git push')


def rollback() -> None:
    """1 つ前に適用していたバージョンに戻す (続けて実行すると, さらに前のバージョンへ戻る)."""
    registry = load_registry()
    current = registry["active"]
    stack = registry.get("stack") or ([current] if current else [])
    if stack and stack[-1] == current:
        stack = stack[:-1]  # 今のバージョンを積み重ねから外す
    if not stack:
        raise SystemExit("戻せる以前のバージョンがありません")
    registry["stack"] = stack
    save_registry(registry)
    promote(stack[-1], reason=f"rollback ({current} から戻す)", is_rollback=True)


def list_versions() -> None:
    registry = load_registry()
    if not registry["versions"]:
        print("登録されたモデルはありません")
        return
    best = best_version(registry)
    policy = {"best": "自動 (ROC-AUC が最良のモデルを適用)", "manual": "手動 (選んだモデルに固定)"}.get(registry.get("policy"), "?")
    print(f"適用モード: {policy}\n")
    print(f"{'':3}{'バージョン':<17} {'学習期間':<23} {'レース数':>6}  {'ROC-AUC':>8}  {'レース内AUC':>10}  "
          f"{'logloss':>7}  {'(オッズ AUC)':>11}  テスト期間")
    for version, info in sorted(registry["versions"].items()):
        m, o = info.get("test_metrics", {}), info.get("test_odds_baseline", {})
        mark = ("*" if version == registry["active"] else " ") + ("★" if version == best else " ") + " "
        if _target(info) != MODEL_TARGET:
            mark = mark[:2] + "×"
        period = "〜".join(info.get("train_period") or ["?", "?"])
        test_period = "〜".join(info.get("test_period") or ["?", "?"])
        print(f"{mark}{version:<17} {period:<23} {info.get('n_train_races') or '-':>6}  {_fmt(m.get('roc_auc')):>8}  "
              f"{_fmt(m.get('roc_auc_by_race')):>10}  {_fmt(m.get('logloss')):>7}  {_fmt(o.get('roc_auc')):>11}  {test_period}")
    print("\n* 適用中 / ★ ROC-AUC 最良 / × 推論に使えないモデル / (オッズ AUC) = 単勝オッズで予想した場合の ROC-AUC")
    print("\n適用の履歴 (新しい順):")
    for h in reversed(registry["history"][-10:]):
        print(f"  {h['activated_at']}  {h['version']}  {h.get('reason', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="バージョンの一覧")
    sub.add_parser("best", help="自動モードに戻し, ROC-AUC が最良のバージョンを適用する")
    sub.add_parser("rollback", help="1 つ前に適用していたモデルに戻す")
    p = sub.add_parser("promote", help="指定したバージョンを適用する")
    p.add_argument("version")
    r = sub.add_parser("register", help="モデルのフォルダを新しいバージョンとして登録する")
    r.add_argument("folder")
    group = r.add_mutually_exclusive_group()
    group.add_argument("--force", action="store_true", help="登録したモデルを適用する (手動モードになる)")
    group.add_argument("--no-promote", action="store_true", help="登録だけして適用しない")
    args = parser.parse_args()

    if args.command == "list":
        list_versions()
    elif args.command == "best":
        apply_best(reason="best コマンド (自動モードに戻す)")
        registry = load_registry()
        if registry.get("policy") != "best":  # すでに最良が適用中だった場合もモードは自動に戻す
            registry["policy"] = "best"
            save_registry(registry)
            print("自動モードに戻しました")
    elif args.command == "rollback":
        rollback()
    elif args.command == "promote":
        promote(args.version)
    elif args.command == "register":
        register(Path(args.folder), promote_mode="force" if args.force else "none" if args.no_promote else "auto")


if __name__ == "__main__":
    main()
