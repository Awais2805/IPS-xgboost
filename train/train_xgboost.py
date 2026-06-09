import datetime
import pandas as pd 
import time 
from pathlib import Path 
import xgboost as xgb 
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
import matplot as plt

SPLITS_DIR = Path("dataset/data-split/splits")
MODEL_DIR = Path("models")

def train_and_evaluate():
    print("\nXGBoost Training")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODEL_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created run directory: {run_dir}/")

    print("\n1. Loading split matrices")
    t0 = time.perf_counter()
    X_train = pd.read_parquet(SPLITS_DIR / "X_train.parquet")
    X_test = pd.read_parquet(SPLITS_DIR/ "X_test.parquet")

    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet")['Target']
    y_test = pd.read_parquet(SPLITS_DIR / "y_test.parquet")['Target']

    print(f"Loaded {len(X_train):,} training rows and {len(X_test):,} testing rows in {time.perf_counter()-t0:.1f}s")
    print("\nInitialise XGBoost Classifier")
    clf = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=10,
        eval_metrics="aucpr"
    )

    print("\n3. Training the model (monitoring log loss and PR-AUC)")

    t1 = time.perf_counter()

    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=10
    )
    print(f"\nTraining completed in {time.perf_counter()-t1:.1f}s")
    print(f"\nBest iteration occured at tree num: {clf.best_iteration}")

    print("\n4. Generating prediction on test set")

    t2 = time.perf_counter()

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]
    print(f"Inference completed in {time.perf_counter()-t2:.1f}s")

    print("\n\n\n\n\nPRDOCUTION EVALUATION")

    pr_auc = average_precision_score(y_test, y_prob)
    print(f"\nPrecision-Recall AUC (PR-AUC): {pr_auc:.4f} (Closer to 1.0 is optimal)")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Attack (1)'], digits=4))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives (Benign passed):  {cm[0][0]:,}")
    print(f"False Positives (False Alarms):  {cm[0][1]:,}")
    print(f"False Negatives (Missed Attack): {cm[1][0]:,}")
    print(f"True Positives (Attack blocked): {cm[1][1]:,}")

    print("\n5. Saving model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "xgboost_ids_baseline.json"
    clf.save_model(model_path)
    model_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Model saved safely to {model_path} ({model_mb:.2f} MB)")

    print("\n6. Generating learning curves")
    results = clf.evals_result()

    train_aucpr = results['validation_0']['aucpr']
    test_aucpr = results['validation-1']['aucpr']
    epochs = range(1, len(train_aucpr)+1)

    plt.figure(figize=(10,6))
    plt.plot(epochs, train_aucpr, label='Train PR-AUC', color='blue', linewidth=2)
    plt.plot(epochs, test_aucpr, label='Test PR-AUC', color='orange', linewidth=2)

    best_iteration = clf.best_iteration
    plt.axvline(x=best_iteration, color='red', linestyle='--', label=f'Best Tree ({best_iteration})')

    plt.title('XGBoost Learning Curve on CIC-IDS2018 (PR-AUC over Trees)')
    plt.xlabel('Number of Trees')
    plt.ylabel('PR-AUC Score (higher is optimal)')
    plt.legend()
    plt.grid(True)

    graph_path = run_dir / "learning_crubes.png"
    plt.save(graph_path)
    plt.close()
    print(f"Learning curve graph saved to {graph_path}")

if __name__ == "__main__":
    train_and_evaluate()