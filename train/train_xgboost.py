from datetime import datetime
import pandas as pd 
import time 
from pathlib import Path 
import xgboost as xgb 
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import logging

SPLITS_DIR = Path("dataset/data-split/splits")
MODEL_DIR = Path("models")

def train_and_evaluate():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODEL_DIR / f"run_{timestamp}"
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

    logging.info("\n1. Loading split matrices")
    t0 = time.perf_counter()
    X_train = pd.read_parquet(SPLITS_DIR / "X_train.parquet")
    X_test = pd.read_parquet(SPLITS_DIR/ "X_test.parquet")

    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet")['Target']
    y_test = pd.read_parquet(SPLITS_DIR / "y_test.parquet")['Target']

    logging.info(f"Loaded {len(X_train):,} training rows and {len(X_test):,} testing rows in {time.perf_counter()-t0:.1f}s")

    logging.info("\nInitialise XGBoost Classifier")

    neg_class_count = (y_train == 0).sum()
    pos_class_count = (y_train ==1).sum()
    imbalance_weight = neg_class_count / pos_class_count
    logging.info(f"Calculated scale_pos_weight: {imbalance_weight:.2f}")

    clf = xgb.XGBClassifier(
        n_estimators=5000,
        learning_rate=0.05,
        max_depth=6,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric=["logloss", "aucpr", "error"],
        scale_pos_weight=imbalance_weight
    )

    logging.info("\n3. Training the model (monitoring log loss and PR-AUC)")

    t1 = time.perf_counter()

    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=1
    )
    logging.info(f"\nTraining completed in {time.perf_counter()-t1:.1f}s")
    logging.info(f"\nBest iteration occured at tree num: {clf.best_iteration}")

    logging.info("\n4. Generating prediction on test set")

    t2 = time.perf_counter()

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]
    logging.info(f"Inference completed in {time.perf_counter()-t2:.1f}s")

    logging.info("\nPRDOCUTION EVALUATION")

    pr_auc = average_precision_score(y_test, y_prob)
    logging.info(f"\nPrecision-Recall AUC (PR-AUC): {pr_auc:.4f} (Closer to 1.0 is optimal)")

    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Attack (1)'], digits=4))

    logging.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"True Negatives (Benign passed):  {cm[0][0]:,}")
    logging.info(f"False Positives (False Alarms):  {cm[0][1]:,}")
    logging.info(f"False Negatives (Missed Attack): {cm[1][0]:,}")
    logging.info(f"True Positives (Attack blocked): {cm[1][1]:,}")

    logging.info("\n5. Saving model")

    model_path = run_dir / "xgboost_ids.json"
    clf.save_model(model_path)
    model_mb = model_path.stat().st_size / (1024 * 1024)

    logging.info(f"Model saved to {model_path} ({model_mb:.2f} MB)")

    logging.info("Extracting top 15 most important features for attack detection")

    plt.figure(figsize=(10,8))
    xgb.plot_importance(clf, max_num_features=15, importance_type='gain', show_values=False)
    plt.title("Top 15 Most Important Features for Detecting Attacks")
    plt.tight_layout()
    plt.savefig(run_dir / "feature_importance.png")
    plt.close()
    logging.info("Feature importance graph saved")

    logging.info("\n6. Generating learning curves")
    results = clf.evals_result()

    train_aucpr = results['validation_0']['aucpr']
    test_aucpr = results['validation_1']['aucpr']
    epochs = range(1, len(train_aucpr)+1)

    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_aucpr, label='Train PR-AUC', color='blue', linewidth=2)
    plt.plot(epochs, test_aucpr, label='Test PR-AUC', color='orange', linewidth=2)

    best_iteration = clf.best_iteration
    plt.axvline(x=best_iteration, color='red', linestyle='--', label=f'Best Tree ({best_iteration})')

    plt.title('XGBoost Learning Curve on CIC-IDS2018 (PR-AUC over Trees)')
    plt.xlabel('Number of Trees')
    plt.ylabel('PR-AUC Score (higher is optimal)')
    plt.legend()
    plt.grid(True)

    graph_path = run_dir / "learning_curves.png"
    plt.savefig(graph_path)
    plt.close()
    logging.info(f"Learning curve graph saved to {graph_path}")
    logging.info(f"Run completely finished. All assets stored in: {run_dir}/")

if __name__ == "__main__":
    train_and_evaluate()