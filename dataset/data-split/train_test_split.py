import pandas as pd
import time 
from pathlib import Path 
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("dataset/cic-ids2018-encoded.parquet")
SPLITS_DIR = Path("dataset/data-split/splits")

def generate_splits(in_path, out_dir):
    print("\nStratified Split")
    print(f"\nLoading encoded data from {in_path}")

    t0 = time.perf_counter()
    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df):,} rows in {time.perf_counter()-t0:.1f}s")

    print("\nSeperating Features (X) and Target (y)")
    X = df.drop(columns=['Target'])
    y = df['Target']

    print(f"X Matrix shape: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"Target (y) Class Distribution before split:")
    distribution = y.value_counts().rename({0:'0 (Benign)', 1: '1 (Attack)'})
    print(distribution.to_string())


    print(f"\nExecuting Stratified 80/20 Split")
    t1 = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size = 0.2,
        random_state = 42,
        stratify=y
    )

    print(f"Split complete in {time.perf_counter()-t1:.1f}s")


    print("\nStratification Math Verification")
    full_pct = y.mean() * 100
    train_pct = y_train.mean() * 100
    test_pct = y_test.mean() * 100

    print(f"   - Full Dataset: {full_pct:6.4f}% Attacks")
    print(f"   - Training Set: {train_pct:6.4f}% Attacks ({y_train.sum():,} target rows)")
    print(f"   - Testing Set:  {test_pct:6.4f}% Attacks ({y_test.sum():,} target rows)")

    assert abs(train_pct - test_pct) < 0.01, "ERROR: stratification failed"

    print("\n Saving Matrices")

    # Save X matrices as parquet
    X_train.to_parquet(out_dir / "X_train.parquet", compression='snappy', index=False)
    X_test.to_parquet(out_dir / "X_test.parquet", compression='snappy', index=False)
    
    # Save y vectors as parquet (requires conversion to DataFrame first)
    y_train.to_frame().to_parquet(out_dir / "y_train.parquet", compression='snappy', index=False)
    y_test.to_frame().to_parquet(out_dir / "y_test.parquet", compression='snappy', index=False)
    print(f"Saved successfully to {out_dir}/")

    print("\n--- Final Pipeline Output ---")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")


if __name__ == "__main__":
    generate_splits(INPUT_PATH, SPLITS_DIR)





