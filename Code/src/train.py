"""Train baseline / optimized classical models on EAR+MAR CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import yaml

# Allow `python -m src.train` from Code/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit_labels import audit_dataframe
from src.dataset import load_and_split, load_csv, apply_label_policy
from src.evaluate import compute_metrics, print_metrics, save_metrics, threshold_rule_scores
from src.models.classic import build_model, predict_proba, train_model


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_csv(cfg: dict, code_dir: Path) -> Path:
    p = Path(cfg["data"]["csv_path"])
    if not p.is_absolute():
        p = (code_dir / p).resolve()
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Train drowsiness classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="Override model type: logistic|rf|xgboost")
    parser.add_argument("--no-flip", action="store_true", help="Do not flip labels")
    args = parser.parse_args()

    code_dir = ROOT
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = code_dir / cfg_path
    cfg = load_config(cfg_path)

    seed = int(cfg.get("seed", 42))
    flip = bool(cfg["data"].get("flip_labels", True)) and not args.no_flip
    model_type = args.model or cfg["model"].get("type", "xgboost")
    out_dir = code_dir / cfg["model"].get("output_dir", "artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_csv(cfg, code_dir)
    print(f"CSV: {csv_path}")
    print(f"flip_labels={flip}, model={model_type}, seed={seed}")

    # Audit raw labels first
    raw = load_csv(csv_path)
    audit = audit_dataframe(raw)
    audit_path = out_dir / "label_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"Label audit: inverted={audit['likely_inverted_labels']} corr_EAR={audit['corr_ear_drowsy']:.3f}")
    print(f"  recommendation: {audit['recommendation']}")

    splits, feature_cols, df = load_and_split(
        csv_path,
        flip_labels=flip,
        test_size=float(cfg["data"].get("test_size", 0.15)),
        val_size=float(cfg["data"].get("val_size", 0.15)),
        seed=seed,
    )
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    print(f"Features: {feature_cols}")
    print(f"Split sizes train/val/test: {len(y_train)}/{len(y_val)}/{len(y_test)}")
    print(f"Train class balance: 0={(y_train==0).sum()} 1={(y_train==1).sum()}")

    # Rule baselines on TEST (after label policy)
    rules = {}
    for mode, thr in [
        ("physiology", cfg["features"].get("report_ear_threshold", 0.41)),
        ("inverted", cfg["features"].get("report_ear_threshold", 0.41)),
    ]:
        rules[mode] = threshold_rule_scores(
            X_test,
            y_test,
            ear_thr=float(thr),
            mar_thr=float(cfg["features"].get("mar_yawn_threshold", 0.9)),
            mode=mode,
        )
        print_metrics(f"Rule[{mode}] test", rules[mode])

    # Train ML
    model = build_model(
        model_type=model_type,
        class_weight=cfg["model"].get("class_weight", "balanced"),
        seed=seed,
    )
    model = train_model(model, X_train, y_train, X_val, y_val)

    results = {
        "config": {
            "flip_labels": flip,
            "model_type": model_type,
            "feature_cols": feature_cols,
            "seed": seed,
            "csv": str(csv_path),
        },
        "label_audit_raw": audit,
        "rules_test": rules,
        "metrics": {},
    }

    for split_name, (X, y) in [("train", splits["train"]), ("val", splits["val"]), ("test", splits["test"])]:
        prob = predict_proba(model, X)
        m = compute_metrics(y, prob)
        results["metrics"][split_name] = m
        print_metrics(f"{model_type} {split_name}", m)

    model_path = out_dir / "best_model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "flip_labels": flip,
            "model_type": model_type,
        },
        model_path,
    )
    metrics_path = out_dir / "metrics.json"
    save_metrics(results, metrics_path)
    print(f"\nSaved model → {model_path}")
    print(f"Saved metrics → {metrics_path}")

    # Summary line for README
    t = results["metrics"]["test"]
    print(
        f"\nTEST summary: acc={t['accuracy']:.4f} f1={t['f1']:.4f} "
        f"auc={t['auc_roc']:.4f} (flip={flip}, model={model_type})"
    )


if __name__ == "__main__":
    main()
