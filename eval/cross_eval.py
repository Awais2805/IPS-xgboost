#!/usr/bin/env python3
"""
Generic cross-evaluation for IPS models: score ANY trained model against ANY
encoded dataset. Not tied to a specific run.

The model's own feature_names are the source of truth: eval data is reindexed to
them (missing cols -> 0, extras dropped, order matched) and the alignment is
reported, so a feature-space mismatch can't silently produce a bogus score.
Leads with threshold-independent metrics (PR-AUC / ROC-AUC), which are the fair
cross-dataset numbers; the operating-threshold confusion is reported too but the
threshold was tuned on the model's own distribution.

Examples:
  # run-A model on run-B encoded data (the cross-run generalization test)
  python3 eval/cross_eval.py \
      --model models/run_20260704_142850 \
      --data preprocessing/processes_output/encoded_datasets/<runB-encoded>.parquet \
      --tag A_on_B

  # any model on any split's held-out test set
  python3 eval/cross_eval.py --model models/run_X \
      --data dataset/data-splits/8020_strat_split/split_Y
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

BENIGN_LABEL = "Benign"


def resolve_model(model_arg):
    """Accept a run dir (…/run_*/) or a direct xgboost_ids.json path."""
    p = Path(model_arg)
    if p.is_dir():
        return p / "xgboost_ids.json", p
    return p, p.parent


def load_threshold(run_dir, override):
    if override is not None:
        return float(override)
    p = run_dir / "operating_threshold.txt"
    if p.is_file():
        return float(p.read_text().strip())
    print(f"[cross-eval] no operating_threshold.txt in {run_dir}; using 0.5")
    return 0.5


def load_xy(data_arg):
    """Encoded parquet (features + Target/Label) or split dir (X_test/y_test)."""
    p = Path(data_arg)
    if p.is_dir():
        X = pd.read_parquet(p / "X_test.parquet")
        y = pd.read_parquet(p / "y_test.parquet")["Target"].to_numpy()
        return X, y, f"{p.name}/X_test"
    df = pd.read_parquet(p)
    if "Target" in df.columns:
        y = df["Target"].to_numpy()
        X = df.drop(columns=["Target"])
    elif "Label" in df.columns:
        y = (df["Label"] != BENIGN_LABEL).astype("int8").to_numpy()
        X = df.drop(columns=["Label"])
    else:
        raise SystemExit(f"{p} has neither 'Target' nor 'Label' column")
    return X, y, p.name


def align_features(X, feats):
    """Reindex eval features to the model's training features: add missing as 0,
    drop extras, match order. Returns (X_aligned, added, dropped)."""
    X = X.copy()
    X.columns = list(map(str, X.columns))
    have = set(X.columns)
    added = [c for c in feats if c not in have]
    dropped = [c for c in X.columns if c not in feats]
    X = X.reindex(columns=feats, fill_value=0)
    return X, added, dropped


def counts_at(y_true, y_prob, t):
    pred = y_prob >= t
    pos = y_true.astype(bool)
    tp = int(np.sum(pred & pos)); fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos)); tn = int(np.sum(~pred & ~pos))
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


def evaluate(model_arg, data_arg, threshold_override=None):
    model_path, run_dir = resolve_model(model_arg)
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    threshold = load_threshold(run_dir, threshold_override)

    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    feats = clf.get_booster().feature_names

    X, y, data_name = load_xy(data_arg)

    if feats is None:
        print("[cross-eval] WARNING: model has no feature names; using data columns as-is")
        added, dropped = [], []
    else:
        X, added, dropped = align_features(X, feats)

    y_prob = clf.predict_proba(X)[:, 1]

    tp, fp, fn, tn = counts_at(y, y_prob, threshold)
    p, r, f1 = prf(tp, fp, fn)
    return {
        "model": str(model_path),
        "data": str(Path(data_arg)),
        "data_name": data_name,
        "rows": int(len(y)),
        "attack_base_rate": round(float(np.mean(y)), 6),
        "threshold": round(float(threshold), 6),
        "pr_auc": round(float(average_precision_score(y, y_prob)), 6),
        "roc_auc": round(float(roc_auc_score(y, y_prob)), 6),
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "features_added_zero": added,
        "features_dropped": dropped,
        "n_features_model": (len(feats) if feats else int(X.shape[1])),
    }


def main():
    ap = argparse.ArgumentParser(description="Cross-evaluate any IPS model on any encoded dataset")
    ap.add_argument("--model", required=True,
                    help="model run dir (models/run_*) or xgboost_ids.json path")
    ap.add_argument("--data", required=True,
                    help="encoded parquet (features + Target/Label) or split dir")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override; default = model's operating_threshold.txt")
    ap.add_argument("--out_dir", default="eval/results", help="where to write the results JSON")
    ap.add_argument("--tag", default=None, help="label for this eval (e.g. A_on_B)")
    args = ap.parse_args()

    res = evaluate(args.model, args.data, args.threshold)
    tag = args.tag or f"{Path(res['model']).parent.name}__on__{res['data_name']}"

    print(f"\n=== CROSS-EVAL: {tag} ===")
    print(f"model : {res['model']}")
    print(f"data  : {res['data']}  ({res['rows']:,} rows, {100*res['attack_base_rate']:.2f}% attack)")
    if res["features_added_zero"]:
        print(f"align : +{len(res['features_added_zero'])} missing cols set to 0: {res['features_added_zero']}")
    if res["features_dropped"]:
        print(f"align : -{len(res['features_dropped'])} extra cols dropped: {res['features_dropped']}")
    if not res["features_added_zero"] and not res["features_dropped"]:
        print("align : feature spaces match exactly")
    print(f"\nPR-AUC : {res['pr_auc']:.4f}    ROC-AUC: {res['roc_auc']:.4f}    "
          "(threshold-independent — the fair cross-eval metrics)")
    print(f"@thr {res['threshold']:.4f} -> precision {res['precision']:.4f}  "
          f"recall {res['recall']:.4f}  F1 {res['f1']:.4f}")
    c = res["confusion"]
    print(f"confusion: TN {c['tn']:,}  FP {c['fp']:,}  FN {c['fn']:,}  TP {c['tp']:,}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"cross_eval_{tag}_{ts}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
