"""Post-run diagnostic visualisations for an XGBoost IPS run.

Loads a saved run's model + operating threshold, scores the test split it was
trained against (chosen via the same interactive menu as the training stage),
and writes a diagnostic pack into <run_dir>/post_run_analysis/.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

# Mirrors train_xgboost.py so the menu points at the same split dirs.
SPLIT_DIR_BY_STRAT = {
    "8020_stratified": Path("dataset/data-splits/8020_strat_split"),
    "time_based": Path("dataset/data-splits/time_based_split"),
}
REQUIRED_FILES = ["X_test.parquet", "y_test.parquet"]


def setup_logging(output_dir, log_level):
    log_file = output_dir / "post_run_analysis.log"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return log_file


# --- split resolution (interactive menu), mirroring train_xgboost.py ---------
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
        choice = input("\nEnter number to analyse (Enter for [1]): ").strip()
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


# --- metric helpers ----------------------------------------------------------
def load_threshold(run_dir):
    path = run_dir / "operating_threshold.txt"
    if not path.is_file():
        logging.warning(f"No operating_threshold.txt in {run_dir}; defaulting to 0.5")
        return 0.5
    return float(path.read_text().strip())


def counts_at(y_true, y_prob, t):
    pred = y_prob >= t
    pos = y_true.astype(bool)
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    tn = int(np.sum(~pred & ~pos))
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


# --- plots -------------------------------------------------------------------
def plot_confidence_histogram(output_dir, y_true, y_prob, threshold):
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color="steelblue", label="Benign (true)")
    plt.hist(y_prob[y_true == 1], bins=50, alpha=0.6, color="crimson", label="Attack (true)")
    plt.axvline(threshold, color="black", linestyle="--",
                label=f"Operating threshold ({threshold:.4f})")
    plt.yscale("log")
    plt.title("Prediction Confidence Distribution by True Class (Leakage Audit)")
    plt.xlabel("Predicted P(Attack)")
    plt.ylabel("Number of flows (log scale)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    out = output_dir / "confidence_histogram.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    logging.info(f"Saved {out}")


def plot_pr_curve(output_dir, y_true, y_prob, threshold, pr_auc):
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    tp, fp, fn, tn = counts_at(y_true, y_prob, threshold)
    op_p, op_r, _ = prf(tp, fp, fn)
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, color="navy", linewidth=2, label=f"PR curve (AP={pr_auc:.4f})")
    plt.scatter([op_r], [op_p], color="red", zorder=5,
                label=f"Operating point @{threshold:.4f}\n(P={op_p:.4f}, R={op_r:.4f})")
    plt.title("Precision-Recall Curve (Attack class)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    out = output_dir / "pr_curve.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    logging.info(f"Saved {out}")


def plot_roc_curve(output_dir, y_true, y_prob, threshold, roc_auc):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    tp, fp, fn, tn = counts_at(y_true, y_prob, threshold)
    op_tpr = tp / (tp + fn) if (tp + fn) else 0.0
    op_fpr = fp / (fp + tn) if (fp + tn) else 0.0
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"ROC curve (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance")
    plt.scatter([op_fpr], [op_tpr], color="red", zorder=5, label=f"Operating point @{threshold:.4f}")
    plt.title("ROC Curve (Attack class)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    out = output_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    logging.info(f"Saved {out}")


def plot_calibration(output_dir, y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker="o", color="purple", label="Model")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Perfectly calibrated")
    plt.title(f"Calibration / Reliability Curve ({n_bins} bins)")
    plt.xlabel("Mean predicted P(Attack)")
    plt.ylabel("Observed attack fraction")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    out = output_dir / "calibration_curve.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    logging.info(f"Saved {out}")


def plot_confusion_at_threshold(output_dir, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Benign (0)", "Attack (1)"]
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix @ threshold {threshold:.4f}")
    mid = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > mid else "black")
    out = output_dir / "confusion_at_threshold.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    logging.info(f"Saved {out}")


def write_threshold_sweep(output_dir, y_true, y_prob, threshold):
    grid = np.round(np.linspace(0.01, 0.99, 99), 4)
    grid = np.unique(np.append(grid, round(float(threshold), 6)))
    rows = []
    for t in grid:
        tp, fp, fn, tn = counts_at(y_true, y_prob, t)
        precision, recall, f1 = prf(tp, fp, fn)
        rows.append({
            "threshold": round(float(t), 6),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "is_operating_point": bool(np.isclose(t, threshold)),
        })
    df = pd.DataFrame(rows)
    out = output_dir / "threshold_sweep.csv"
    df.to_csv(out, index=False)
    logging.info(f"Saved {out} ({len(df)} thresholds)")


def main():
    parser = argparse.ArgumentParser(description="Post-run diagnostics for an XGBoost IPS run.")
    parser.add_argument("--run_dir", required=True, help="Path to model run dir (models/run_...).")
    parser.add_argument("--split_strat", choices=list(SPLIT_DIR_BY_STRAT), default="8020_stratified",
                        help="Which split-strategy dir to pick the split run from.")
    parser.add_argument("--in_path", type=str, default="auto",
                        help="Explicit split run dir, or 'auto' for the interactive menu.")
    parser.add_argument("--non_interactive", action="store_true",
                        help="Skip the menu and use the most recent split run.")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    model_path = run_dir / "xgboost_ids.json"
    output_dir = run_dir / "post_run_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir, args.log_level)
    logging.info("Post-run analysis")
    logging.info(f"Run dir: {run_dir}")
    if not model_path.is_file():
        raise SystemExit(f"Model not found at {model_path}")

    splits_dir = resolve_split_dir(args.split_strat, args.in_path, args.non_interactive)
    logging.info(f"Using split run: {splits_dir}")

    threshold = load_threshold(run_dir)
    logging.info(f"Operating threshold: {threshold:.6f}")

    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    logging.info("Loaded model")

    logging.info("Loading test data")
    X_test = pd.read_parquet(splits_dir / "X_test.parquet")
    y_true = pd.read_parquet(splits_dir / "y_test.parquet")["Target"].to_numpy()
    logging.info(f"Test rows: {len(X_test):,} | features: {X_test.shape[1]}")

    try:
        feats = clf.get_booster().feature_names
    except Exception:
        feats = None
    if feats is not None and set(feats) != set(map(str, X_test.columns)):
        logging.warning("Model feature names differ from the selected split's columns — "
                        "check you picked the split this model was trained on.")

    logging.info("Scoring test set")
    y_prob = clf.predict_proba(X_test)[:, 1]
    del X_test

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    tp, fp, fn, tn = counts_at(y_true, y_prob, threshold)
    op_p, op_r, op_f1 = prf(tp, fp, fn)
    logging.info(f"PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")
    logging.info(f"@threshold {threshold:.4f} -> precision {op_p:.4f} recall {op_r:.4f} F1 {op_f1:.4f}")
    logging.info(f"Confusion @threshold: TN {tn:,} FP {fp:,} FN {fn:,} TP {tp:,}")

    plot_confidence_histogram(output_dir, y_true, y_prob, threshold)
    plot_pr_curve(output_dir, y_true, y_prob, threshold, pr_auc)
    plot_roc_curve(output_dir, y_true, y_prob, threshold, roc_auc)
    plot_calibration(output_dir, y_true, y_prob)
    plot_confusion_at_threshold(output_dir, y_true, y_prob, threshold)
    write_threshold_sweep(output_dir, y_true, y_prob, threshold)
    logging.info(f"Post-run analysis complete — assets in {output_dir}/")


if __name__ == "__main__":
    main()
