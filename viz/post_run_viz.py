import pandas as pd 
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os

SPLITS_DIR = Path("dataset/data-split/splits")


def post_run_checks():

    parser = argparse.ArgumentParser(description="Data leakage/Confidence checks for XGBoost")
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to model run dir (models/run_...)"
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    model_path = os.path.join(run_dir, "xgboost_ids.json")

    output_dir = os.path.join(run_dir, "post_run_analysis")
    os.makedirs(output_dir, exist_ok=True)

    output_image = os.path.join(output_dir, "confidence_histogram.png")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model: {args} not found at {model_path}")
    if not os.path.exists(SPLITS_DIR):
        raise FileNotFoundError(f"Test data not found at {SPLITS_DIR}")

    print("\n1. Loading test data")

    X_test = pd.read_parquet(SPLITS_DIR / "X_test.parquet")
    y_test = pd.read_parquet(SPLITS_DIR / "y_test.parquet")

    print("\n2. Retreiving saved model")
    clf = xgb.XGBClassifier()

    clf.load_model(model_path)
    
    print("\n3. Generating probabilty scores")
    probabilities = clf.predict_proba(X_test)[:,1]

    print("\n4. Generating confidence histogram")

    plt.figure(figsize=(10, 6))
    plt.hist(probabilities, bins=50, color='blue', alpha=0.7)
    plt.title('Prediction Confidence Distribution (Leakage Audit)')
    plt.xlabel('Probability of being an Attack (0.0 to 1.0)')
    plt.ylabel('Number of Packets')
    plt.grid(True)
    plt.savefig(output_image)

    print(f"Post run analysis complete - confidence histogram at {output_image}")


if __name__ == "__main__":
    post_run_checks()