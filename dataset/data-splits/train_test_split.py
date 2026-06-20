import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path 

import pandas as pd 
from sklearn.model_selection import train_test_split


STAGE = "split"
TARGET_COL = "Target"
TEST_SIZE = 0.2
RANDOM_STATE = 42

INPUT_DIR = Path("preprocessing/processes_output/encoded_datasets")
SPLIT_DIR_BY_STRAT = {
    "8020_stratified": Path("dataset/data-splits/8020_strat_split"),
    "time_based": Path("dataset/data-splits/time_based_split")
}


def set_logging(log_file, log_level="INFO"):
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format = "%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True
    )

def select_encoded_dataset(search_dir):
    if not search_dir.exists:
        sys.exit(f"Error: {search_dir} does note exist")
    
    files = sorted(
        search_dir.glob("*_encoded_*.parquet"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )

    if not files:
        sys.exit(f"Error: No encoded parquets found in {search_dir}")

    print(f"\nFound {len(files)} encoded parquets in {search_dir}")
    for i, f in enumerate(files):
        tag = " [Most Recent]" if i == 0 else ""
        print(f" {i+1}. {f.name}{tag}")

    while True:
        choice = input("\nEnter number of dataset to split (Enter for [1])").strip()
        if choice == "":
            return files[0]
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice)-1]
        print(f"Invalid - enter a number 1 - {len(files)} or press Enter")

def resolve_input(in_path_arg, non_interactive):
    if in_path_arg != "auto":
        in_path = Path(in_path_arg)
        if not in_path.is_file():
            sys.exit(f"Error: input file does not exist: {in_path}")
        return in_path

    if non_interactive:
        files = sorted(
            INPUT_DIR.glob("*_encoded_*.parquet"),
            key=lambda f: f.stat().st_mtime,
            reverse=True 
        )    
    
        if not files:
            sys.exit(f"ERror: no neocded parquets found in {INPUT_DIR}")
        return files[0]
    return select_encoded_dataset(INPUT_DIR)


def parse_split_strat(file_name):
    for token in ("8020_stratified", "time_based"):
        if token in file_name:
            return token
        
    sys.exit(
        f"Cannot parse split start from {file_name}"
        "Expected an '80280_stratified' or 'time_based' token"
    )



def generate_splits(in_path, out_dir):
    logging.info(f"STAGE :{STAGE}.upper()")
    logging.info(f"Input: {in_path}")
    logging.info(f"Output: {out_dir}")

    logging.info("1. Loading Encoded Data")
    t0 = time.perf_counter()
    df = pd.read_parquet(in_path)
    logging.info(f" Loaded {len(df):,} rows x {len(df.columns)} cols in {time.perf_counter()-t0:.1f}s")

    if TARGET_COL not in df.columns:
        sys.exit(f"ABORT: '{TARGET_COL}' column missing")

    logging.info("2. Seperating featutres (X) and target (y)")
    X = df.drop(columns={TARGET_COL})
    y = df([TARGET_COL])
    logging.info(f"X: {X.shape[0]:,} dimensions x {X.shape[1]} features")
    dist = y.value_counts().sort_index().rename({0: "o (Benign)", 1: "1 (Attack)"})
    logging.info("Class dist before split: \n" + dist.to_string())

    logging.info(f"3. Startified {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE)*100} split")
    t1 = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logging.info(f"Split complete in {time.perf_counter()-t1:.1f}s")


    full_pct = y.mean() * 100
    train_pct = y_train.mean() * 100
    test_pct = y_test.mean() * 100
    logging.info(f"   Full:  {full_pct:6.4f}% attacks")
    logging.info(f"   Train: {train_pct:6.4f}% attacks ({y_train.sum():,} rows)")
    logging.info(f"   Test:  {test_pct:6.4f}% attacks ({y_test.sum():,} rows)")
    assert abs(train_pct - test_pct) < 0.01, "ERROR: stratification failed"


    logging.info("4. Saving matrices")
    X_train.to_parquet(out_dir / "X_train.parquet", compression="snappy", index=False)
    X_test.to_parquet(out_dir / "X_test.parquet", compression="snappy", index=False)
    y_train.to_frame().to_parquet(out_dir / "y_train.parquet", compression="snappy", index=False)
    y_test.to_frame().to_parquet(out_dir / "y_test.parquet", compression="snappy", index=False)

    logging.info(f"   X_train {X_train.shape} | X_test {X_test.shape}")
    logging.info(f"   y_train {y_train.shape} | y_test {y_test.shape}")
    logging.info(f"=== {STAGE.upper()} DONE — matrices written to {out_dir}/ ===")