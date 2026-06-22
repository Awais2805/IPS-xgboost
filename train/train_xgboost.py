from datetime import datetime
import argparse
import gc
import resource
import numpy as np
import pandas as pd
import time
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import logging

MODEL_DIR = Path("models")
SPLIT_DIR_BY_STRAT = {
    "8020_stratified": Path("dataset/data-splits/8020_strat_split"),
    "time_based": Path("dataset/data-splits/time_based_split"),
}
REQUIRED_FILES = ["X_train.parquet", "X_test.parquet", "y_train.parquet", "y_test.parquet"]


def validate_split_dir(d):
    missing = [f for f in REQUIRED_FILES if not (d / f).is_file()]
    if missing:
        raise SystemExit(f"Split dir {d} is missing: {missing}")
    return d


def select_split_dir(base):
    if not base.exists():
        raise SystemExit(f"Split base dir does not exist: {base}")
    runs = sorted(
        (r for r in base.glob("split_*") if r.is_dir()),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not runs:
        raise SystemExit(f"No split_* run dirs found in {base}")
    print(f"\nFound {len(runs)} split run(s) in {base}:")
    for i, r in enumerate(runs):
        tag = " [Most Recent]" if i == 0 else ""
        print(f"  {i + 1}. {r.name}{tag}")
    while True:
        choice = input("\nEnter number to train on (Enter for [1]): ").strip()
        if choice == "":
            return runs[0]
        if choice.isdigit() and 1 <= int(choice) <= len(runs):
            return runs[int(choice) - 1]
        print(f"Invalid — enter a number 1–{len(runs)} or press Enter.")


def resolve_split_dir(split_strat, in_path, non_interactive):
    if in_path != "auto":
        d = Path(in_path)
        if not d.is_dir():
            raise SystemExit(f"--in_path is not a directory: {d}")
        return validate_split_dir(d)
    base = SPLIT_DIR_BY_STRAT[split_strat]
    if non_interactive:
        runs = sorted(
            (r for r in base.glob("split_*") if r.is_dir()),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not runs:
            raise SystemExit(f"No split_* run dirs found in {base}")
        return validate_split_dir(runs[0])
    return validate_split_dir(select_split_dir(base))


def log_peak_ram(tag):
    # macOS reports ru_maxrss in bytes (Linux uses KiB); this is darwin.
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)
    logging.info(f"[mem] peak RSS after {tag}: {peak_gb:.2f} GB")


def train_and_evaluate(splits_dir, out_base, split_strat):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_base) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "training_output.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("XGBoost Training")
    logging.info(f"Created run directory: {run_dir}/")
    logging.info(f"Using split run: {splits_dir}/")

    logging.info("\n1. Loading train matrices (test deferred until evaluation)")
    t0 = time.perf_counter()
    X_train_full = pd.read_parquet(splits_dir / "X_train.parquet")
    y_train_full = pd.read_parquet(splits_dir / "y_train.parquet")['Target']
    logging.info(f"Loaded {len(X_train_full):,} training rows in {time.perf_counter()-t0:.1f}s")

    # -----------------------------------------------------------------
    # PHASE 1 — carve a validation set, early-stop to find the tree count
    # and the operating threshold. Free X_train_full immediately so the
    # fit doesn't hold the full frame + the carved copies at once.
    # -----------------------------------------------------------------
    if split_strat == "time_based":
        # Temporal carve: X_train rows are in chronological day order, so the
        # last 20% are the latest train days — a validation slice that mimics
        # the train->test time gap (no shuffle, no stratify).
        logging.info("\n2. Carving validation set from train (temporal tail 80/20)")
        n_val = max(1, int(len(X_train_full) * 0.2))
        X_tr, X_val = X_train_full.iloc[:-n_val], X_train_full.iloc[-n_val:]
        y_tr, y_val = y_train_full.iloc[:-n_val], y_train_full.iloc[-n_val:]
    else:
        logging.info("\n2. Carving validation set from train (stratified 80/20)")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )
    del X_train_full
    gc.collect()
    logging.info(f"Phase-1 train: {len(X_tr):,} | Validation: {len(X_val):,}")
    if int((y_val == 1).sum()) == 0:
        logging.warning("Phase-1 validation has 0 attacks — early stopping (aucpr) and "
                        "threshold selection will be unreliable; adjust --test_days/boundary.")
        
    spw_phase1 = (y_tr == 0).sum() / (y_tr == 1).sum()
    logging.info(f"Phase-1 scale_pos_weight (train subset): {spw_phase1:.2f}")

    clf_es = xgb.XGBClassifier(
        n_estimators=5000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric=["logloss", "aucpr"],   # aucpr last -> drives early stopping
        scale_pos_weight=spw_phase1
    )

    logging.info("\n3. Phase 1: training with early stopping on validation PR-AUC")
    t1 = time.perf_counter()
    clf_es.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=1
    )
    best_n_trees = clf_es.best_iteration + 1   # best_iteration is 0-indexed
    logging.info(f"Phase 1 completed in {time.perf_counter()-t1:.1f}s")
    logging.info(f"Best iteration: {clf_es.best_iteration} -> using {best_n_trees} trees for the final model")

    logging.info("\n4. Selecting operating threshold (max F1 on validation)")
    val_prob = clf_es.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_prob)
    f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1s[:-1]))   # f1s has one more element than thresholds
    best_threshold = float(thresholds[best_idx])
    logging.info(
        f"Chosen threshold: {best_threshold:.4f} "
        f"(val precision {precisions[best_idx]:.4f}, recall {recalls[best_idx]:.4f}, F1 {f1s[best_idx]:.4f})"
    )
    (run_dir / "operating_threshold.txt").write_text(f"{best_threshold:.6f}\n")
    logging.info(f"Operating threshold saved to {run_dir / 'operating_threshold.txt'}")

    # capture the learning-curve data, then free the phase-1 matrices
    es_results = clf_es.evals_result()
    es_best_iter = clf_es.best_iteration
    del X_tr, X_val, y_tr, y_val, val_prob
    gc.collect()
    log_peak_ram("phase 1")

    # -----------------------------------------------------------------
    # PHASE 2 — reload the FULL train set and refit with the discovered
    # tree count (no early stopping). This is the model we keep.
    # -----------------------------------------------------------------
    logging.info("\n5. Phase 2: reloading full train set and refitting")
    t2 = time.perf_counter()
    X_train_full = pd.read_parquet(splits_dir / "X_train.parquet")
    spw_full = (y_train_full == 0).sum() / (y_train_full == 1).sum()
    logging.info(f"Reloaded {len(X_train_full):,} rows in {time.perf_counter()-t2:.1f}s")
    logging.info(f"Phase-2 scale_pos_weight (full train): {spw_full:.2f} | trees: {best_n_trees}")

    clf = xgb.XGBClassifier(
        n_estimators=best_n_trees,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=spw_full
    )

    t3 = time.perf_counter()
    clf.fit(X_train_full, y_train_full, verbose=False)
    logging.info(f"Phase 2 fit completed in {time.perf_counter()-t3:.1f}s")
    del X_train_full, y_train_full
    gc.collect()
    log_peak_ram("phase 2")

    # -----------------------------------------------------------------
    # EVALUATION — load held-out test set now, apply the phase-1 threshold
    # -----------------------------------------------------------------
    logging.info("\n6. Loading test set and generating predictions")
    t4 = time.perf_counter()
    X_test = pd.read_parquet(splits_dir / "X_test.parquet")
    y_test = pd.read_parquet(splits_dir / "y_test.parquet")['Target']
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)
    logging.info(f"Test load + inference completed in {time.perf_counter()-t4:.1f}s")
    del X_test
    gc.collect()

    logging.info("\nPRODUCTION EVALUATION")

    pr_auc = average_precision_score(y_test, y_prob)
    logging.info(f"\nPrecision-Recall AUC (PR-AUC): {pr_auc:.4f} (Closer to 1.0 is optimal)")

    logging.info(f"\nClassification Report (threshold = {best_threshold:.4f}):")
    logging.info(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Attack (1)'], digits=4))

    logging.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"True Negatives (Benign passed):  {cm[0][0]:,}")
    logging.info(f"False Positives (False Alarms):  {cm[0][1]:,}")
    logging.info(f"False Negatives (Missed Attack): {cm[1][0]:,}")
    logging.info(f"True Positives (Attack blocked): {cm[1][1]:,}")

    logging.info("\n7. Saving model")
    model_path = run_dir / "xgboost_ids.json"
    clf.save_model(model_path)
    model_mb = model_path.stat().st_size / (1024 * 1024)
    logging.info(f"Model saved to {model_path} ({model_mb:.2f} MB)")

    logging.info("Extracting top 15 most important features for attack detection")
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(clf, max_num_features=15, importance_type='gain', show_values=False)
    plt.title("Top 15 Most Important Features for Detecting Attacks")
    plt.tight_layout()
    plt.savefig(run_dir / "feature_importance.png")
    plt.close()
    logging.info("Feature importance graph saved")

    # Learning curves from the phase-1 model (the one that had an eval_set)
    logging.info("\n8. Generating learning curves (from phase-1 early-stopping run)")
    train_aucpr = es_results['validation_0']['aucpr']
    val_aucpr = es_results['validation_1']['aucpr']
    epochs = range(1, len(train_aucpr) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_aucpr, label='Train PR-AUC', color='blue', linewidth=2)
    plt.plot(epochs, val_aucpr, label='Validation PR-AUC', color='orange', linewidth=2)
    plt.axvline(x=es_best_iter, color='red', linestyle='--', label=f'Best Tree ({es_best_iter})')

    plt.title('XGBoost Learning Curve on CIC-IDS2018 (PR-AUC over Trees)')
    plt.xlabel('Number of Trees')
    plt.ylabel('PR-AUC Score (higher is optimal)')
    plt.legend()
    plt.grid(True)

    graph_path = run_dir / "learning_curves.png"
    plt.savefig(graph_path)
    plt.close()
    logging.info(f"Learning curve graph saved to {graph_path}")
    log_peak_ram("run end")
    logging.info(f"Run completely finished. All assets stored in: {run_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost IPS model on a split run.")
    parser.add_argument("--split_strat", choices=list(SPLIT_DIR_BY_STRAT), default="8020_stratified",
                        help="Which split-strategy dir to pick the run from.")
    parser.add_argument("--in_path", type=str, default="auto",
                        help="Explicit split run dir, or 'auto' for the interactive menu.")
    parser.add_argument("--out_dir", type=str, default=str(MODEL_DIR),
                        help="Base output dir; run_<ts> is nested inside (default: models/).")
    parser.add_argument("--non_interactive", action="store_true")
    args = parser.parse_args()

    splits_dir = resolve_split_dir(args.split_strat, args.in_path, args.non_interactive)
    train_and_evaluate(splits_dir, args.out_dir, args.split_strat)