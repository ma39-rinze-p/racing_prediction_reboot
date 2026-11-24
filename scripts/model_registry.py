"""推論 (GitHub Actions) で使うモデルのバージョン管理.

    models/registry/
        registry.json        各バージョンの情報 (学習期間・テストの精度など) と, 適用の履歴
        v001/ v002/ ...      各バージョンのモデル (model.txt, metadata.json, feature_importance.csv)
    models/production/       今適用しているバージョンのコピー (推論はここを読む) + VERSION.txt (バージョン番号)

適用するモデルの選び方 (registry.json の "policy"):
    "best"   (既定・自動モード): 1 着の確率を予測するモデル (目的変数 is_win) のうち, テストの ROC-AUC が最も高いものを適用する.
             (以前の 3 着以内を予測するモデルは, ROC-AUC の意味が違い推論にも使えないため選ばない)
             新しいモデルを取り込むたび (scripts/train_on_kaggle.py / fetch_kaggle_model.py から自動で呼ばれる) に選び直す.
    "manual" (手動モード)     : promote / rollback で選んだバージョンに固定する (新しいモデルを取り込んでも切り替えない).
             best コマンドで自動モードに戻る.

使い方 (リポジトリのルートで実行):
    python scripts/model_registry.py list                 # バージョンの一覧 (* が適用中, ★ が ROC-AUC 最良)
    python scripts/model_registry.py best                 # 自動モードに戻し, ROC-AUC が最良のバージョンを適用する
    python scripts/model_registry.py rollback             # 1 つ前に適用していたモデルに戻す (手動モードになる)
    python scripts/model_registry.py promote v002         # 指定したバージョンを適用する (手動モードになる)
    python scripts/model_registry.py register <フォルダ>  # model.txt と metadata.json があるフォルダを登録する
                                                          # (--force: 登録したモデルを適用 (手動モード) / --no-promote: 登録だけ)

適用 (promote / rollback) したら, GitHub Actions に反映するため commit & push する:
    git add models && git commit -m "switch production model to v00X" && git push
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "models" / "registry"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"
PRODUCTION_DIR = ROOT / "models" / "production"
MODEL_FILES = ["model.txt", "metadata.json", "feature_importance.csv"]
REQUIRED_FILES = ["model.txt", "metadata.json"]

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
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _summary(model_dir: Path) -> dict:
    """metadata.json から, 一覧に表示する情報を取り出す."""
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = meta.get("metrics", {})
    test = metrics.get("test", {})
    return {
        "target": meta.get("target", LEGACY_TARGET),
        "train_period": meta.get("train_period"),
        "n_train_races": meta.get("n_train_races"),
        "test_period": metrics.get("test_period"),
        "test_metrics": {
            k: test.get(k)
            for k in ("roc_auc_by_race", "roc_auc", "logloss", "win_hit", "top3_precision", "wide_top_pair_hit")
            if k in test
        },
    }


def load_registry() -> dict:
    """管理簿を読み込む. まだなければ, 今の models/production を最初のバージョン (v001) として登録して作る."""
    if REGISTRY_FILE.exists():
        registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        registry.setdefault("policy", "best")
        return registry
    registry = {"active": None, "policy": "best", "versions": {}, "history": [], "stack": []}
    if all((PRODUCTION_DIR / f).exists() for f in REQUIRED_FILES):
        version = "v001"
        _copy_model(PRODUCTION_DIR, REGISTRY_DIR / version)
        registry["versions"][version] = {
            "registered_at": _now(),
            "source": "registry 導入前の models/production",
            **_summary(REGISTRY_DIR / version),
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


def _copy_model(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)


def _write_version_file(version: str) -> None:
    (PRODUCTION_DIR / "VERSION.txt").write_text(version + "\n", encoding="utf-8")


def _next_version(registry: dict) -> str:
    numbers = [int(v[1:]) for v in registry["versions"] if v[1:].isdigit()]
    return f"v{(max(numbers) if numbers else 0) + 1:03d}"


# ---------------------------------------------------------------------------
# 操作
# ---------------------------------------------------------------------------
def _target(info: dict) -> str:
    return info.get("target", LEGACY_TARGET)


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


def register(src_dir: Path, source: str = "", promote_mode: str = "auto") -> str:
    """モデルを新しいバージョンとして登録する. 登録したバージョン番号を返す.

    promote_mode:
        "auto"  : 自動モードなら, 全バージョンのうち ROC-AUC が最良のものを適用する (手動モードなら登録だけ)
        "force" : 登録したバージョンを適用する (手動モードになる)
        "none"  : 登録だけ
    """
    src_dir = Path(src_dir)
    missing = [f for f in REQUIRED_FILES if not (src_dir / f).exists()]
    if missing:
        raise SystemExit(f"{src_dir} に {missing} がありません")

    registry = load_registry()
    version = _next_version(registry)
    _copy_model(src_dir, REGISTRY_DIR / version)
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
    """指定したバージョンを models/production にコピーして適用する.

    適用したバージョンは「適用の積み重ね (stack)」に積む. rollback はこれを 1 つずつさかのぼる
    (rollback で戻したときは積まない. そのため rollback を続けると, 適用してきた順に古い方へ戻っていく).
    manual=True (promote / rollback コマンド) なら手動モードにする (新しいモデルを取り込んでも自動で切り替えない).
    """
    registry = load_registry()
    if version not in registry["versions"]:
        raise SystemExit(f"{version} は登録されていません (登録済み: {', '.join(sorted(registry['versions']))})")
    target = _target(registry["versions"][version])
    if target != MODEL_TARGET:
        raise SystemExit(
            f"{version} は目的変数が {target} のモデル (3 着以内を予測する以前のモデル) のため, 今の推論では使えません. "
            f"1 着の確率を予測するモデル ({MODEL_TARGET}) を選んでください"
        )
    _copy_model(REGISTRY_DIR / version, PRODUCTION_DIR)
    _write_version_file(version)
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
    print(f"{'':3}{'バージョン':<8} {'登録日時':<20} {'学習期間':<25} {'レース数':>6}  {'ROC-AUC':>8}  {'レース内AUC':>10}  {'logloss':>7}  テスト期間")
    for version, info in sorted(registry["versions"].items()):
        m = info.get("test_metrics", {})
        mark = ("*" if version == registry["active"] else " ") + ("★" if version == best else " ") + " "
        period = "〜".join(info.get("train_period") or ["?", "?"])
        test_period = "〜".join(info.get("test_period") or ["?", "?"])
        if _target(info) != MODEL_TARGET:
            mark = mark[:2] + "×"  # 3 着以内モデル (推論に使えない. 指標も 1 着モデルとは比べられない)
        roc = f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "-"
        auc = f"{m['roc_auc_by_race']:.4f}" if m.get("roc_auc_by_race") is not None else "-"
        ll = f"{m['logloss']:.4f}" if m.get("logloss") is not None else "-"
        print(f"{mark}{version:<8} {info.get('registered_at', ''):<20} {period:<25} {info.get('n_train_races') or '-':>6}  {roc:>8}  {auc:>10}  {ll:>7}  {test_period}")
    print("\n* 適用中 / ★ ROC-AUC 最良 / × 3 着以内を予測する以前のモデル (推論に使えないため適用できない)")
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
