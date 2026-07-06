#!/usr/bin/env python3
"""
Train ONE XGBoost model on a combination of several datasets' train splits, then
evaluate it on EACH dataset's held-out test split. This is the multi-environment
generalisation experiment: does training across environments produce a model
that works across them?

Mirrors train_xgboost.py exactly (two-phase early-stop -> refit, same
hyperparameters, scale_pos_weight, max-F1 threshold) so the combined model is
comparable to the single-dataset models. Saves to models/run_<ts>/ (xgboost_ids
.json + operating_threshold.txt) so cross_eval.py / post_run_viz.py also work.

Example (CIC-2017 + CIC-2018, balanced so 2018 doesn't swamp 2017):
  python3 train/train_combined.py \
      --splits dataset/data-splits/8020_strat_split/split_20260706-095227 \
               dataset/data-splits/8020_strat_split/split_20260620-215643 \
      --balance --tag cic17_cic18
"""

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix)


def load_train(split_dir, cap, ref_cols):
    X = pd.read_parquet(Path(split_dir) / "X_train.parquet")
    y = pd.read_parquet(Path(split_dir) / "y_train.parquet")["Target"]
    if ref_cols is not None:
        X = X.reindex(columns=ref_cols, fill_value=0)
    if cap and len(X) > cap:
        X = X.sample(n=cap, random_state=42)
        y = y.loc[X.index]
    return X.reset_index(drop=True), y.reset_index(drop=True)


def build_params(n_estimators, spw, early_stop):
    kw = dict(n_estimators=n_estimators, learning_rate=0.05, max_depth=6,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
              tree_method="hist", random_state=42, n_jobs=-1,
              scale_pos_weight=spw)
    if early_stop:
        kw.update(early_stopping_rounds=50, eval_metric=["logloss", "aucpr"])
    return kw


def main():
    ap = argparse.ArgumentParser(description="Train one model across several datasets' train splits")
    ap.add_argument("--splits", nargs="+", required=True,
                    help="split run dirs (each with X_train/y_train/X_test/y_test)")
    ap.add_argument("--cap", type=int, default=None,
                    help="cap each split's train to N rows (random). Default: use all")
    ap.add_argument("--balance", action="store_true",
                    help="cap every split to the smallest split's train size (equal representation)")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--tag", default=None, help="label for logging/provenance")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    log(f"Combined training | tag={args.tag} | splits={len(args.splits)}")
    for s in args.splits:
        log(f"  - {s}")

    # decide per-split cap
    sizes = {s: pd.read_parquet(Path(s) / "y_train.parquet").shape[0] for s in args.splits}
    for s, n in sizes.items():
        log(f"train rows in {Path(s).name}: {n:,}")
    cap = args.cap
    if args.balance:
        cap = min(sizes.values()) if cap is None else min(cap, min(sizes.values()))
        log(f"--balance: capping each split to {cap:,} rows")

    # assemble combined train (first split defines the reference feature order)
    ref_cols = list(pd.read_parquet(Path(args.splits[0]) / "X_train.parquet").columns)
    Xs, ys = [], []
    for s in args.splits:
        X, y = load_train(s, cap, ref_cols)
        log(f"  loaded {len(X):,} rows from {Path(s).name} (attack {100*y.mean():.2f}%)")
        Xs.append(X); ys.append(y)
    X_full = pd.concat(Xs, ignore_index=True)
    y_full = pd.concat(ys, ignore_index=True)
    del Xs, ys; gc.collect()
    log(f"combined train: {len(X_full):,} rows x {X_full.shape[1]} cols "
        f"(attack {100*y_full.mean():.2f}%)")

    # ---- PHASE 1: stratified 80/20 val carve, early-stop for tree count + threshold
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    spw1 = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    log(f"\nPhase 1: train {len(X_tr):,} | val {len(X_val):,} | scale_pos_weight {spw1:.2f}")
    clf_es = xgb.XGBClassifier(**build_params(5000, spw1, early_stop=True))
    t = time.perf_counter()
    clf_es.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
    best_n = clf_es.best_iteration + 1
    log(f"Phase 1 done in {time.perf_counter()-t:.1f}s | best trees: {best_n}")

    val_prob = clf_es.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, val_prob)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1[:-1]))
    threshold = float(thr[best_idx])
    log(f"Operating threshold (max val F1): {threshold:.4f} "
        f"(P {prec[best_idx]:.4f} R {rec[best_idx]:.4f} F1 {f1[best_idx]:.4f})")
    (run_dir / "operating_threshold.txt").write_text(f"{threshold:.6f}\n")
    del X_tr, X_val, y_tr, y_val, val_prob; gc.collect()

    # ---- PHASE 2: refit best_n trees on the full combined train
    spw_full = (y_full == 0).sum() / max(1, (y_full == 1).sum())
    log(f"\nPhase 2: refit {best_n} trees on {len(X_full):,} rows | scale_pos_weight {spw_full:.2f}")
    clf = xgb.XGBClassifier(**build_params(best_n, spw_full, early_stop=False))
    t = time.perf_counter()
    clf.fit(X_full, y_full, verbose=False)
    log(f"Phase 2 done in {time.perf_counter()-t:.1f}s")
    del X_full, y_full; gc.collect()

    model_path = run_dir / "xgboost_ids.json"
    clf.save_model(model_path)
    log(f"Saved model -> {model_path}")

    # ---- EVALUATE on each input dataset's held-out test
    log("\n=== HELD-OUT TEST per dataset (combined model) ===")
    for s in args.splits:
        Xte = pd.read_parquet(Path(s) / "X_test.parquet").reindex(columns=ref_cols, fill_value=0)
        yte = pd.read_parquet(Path(s) / "y_test.parquet")["Target"].to_numpy()
        prob = clf.predict_proba(Xte)[:, 1]
        pr = average_precision_score(yte, prob)
        roc = roc_auc_score(yte, prob)
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        log(f"\n  {Path(s).name}  (rows {len(yte):,}, attack {100*yte.mean():.2f}%)")
        log(f"    PR-AUC {pr:.4f} | ROC-AUC {roc:.4f}")
        log(f"    @thr {threshold:.3f}: TN {tn:,} FP {fp:,} FN {fn:,} TP {tp:,}")
        del Xte; gc.collect()

    (run_dir / "training_output.log").write_text("\n".join(log_lines) + "\n")
    log(f"\nAll assets in: {run_dir}/")


if __name__ == "__main__":
    main()
