import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path 
import gc 
import numpy as np

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
    logging.info(f"=== {STAGE.upper()} STAGE START ===")
    logging.info(f"Input:  {in_path}")
    logging.info(f"Output: {out_dir}/")

    logging.info("1. Loading encoded data")
    t0 = time.perf_counter()
    df = pd.read_parquet(in_path)
    logging.info(f"   Loaded {len(df):,} rows x {len(df.columns)} cols in {time.perf_counter()-t0:.1f}s")

    if TARGET_COL not in df.columns:
        sys.exit(f"ABORT: '{TARGET_COL}' column missing from encoded input.")

    # Halve the in-RAM footprint: 4.9 GB -> ~3.3 GB per copy.
    # Lossless for tree splitting (XGBoost works in float32 regardless).
    float_cols = df.select_dtypes("float64").columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype("float32")
        logging.info(f"   Downcast {len(float_cols)} float64 cols -> float32")

    logging.info("2. Separating target (in-place pop) and computing stratified indices")
    y = df.pop(TARGET_COL)   # removes Target from df in place -> no full-frame copy
    dist = y.value_counts().sort_index().rename({0: "0 (Benign)", 1: "1 (Attack)"})
    logging.info("   Class distribution before split:\n" + dist.to_string())

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train_pct = y.iloc[train_idx].mean() * 100
    test_pct = y.iloc[test_idx].mean() * 100
    logging.info(f"   Attacks — full {y.mean()*100:6.4}% | test {test_pct:6.4f}%")
    assert abs(train_pct - test_pct) < 0.01, "ERROR: stratification failed"

    logging.info(f"3. Stratified {int(round((1-TEST_SIZE*100)))} split — writing one side at a time")
    for name, idx in (("train", train_idx), ("test", test_idx)):
        t1 = time.perf_counter()
        df.iloc[idx].to_parquet(out_dir / f"X_{name}.parquet", compression="snappy", index=False)
        y.iloc[idx].to_frame().to_parquet(out_dir / f"y_{name}.parquet", compression="snappy", index=False)
        logging.info(f"   {name}: {len(idx):,} rows ->parquet ({time.perf_counter()-t1:.1f}s)")
        gc.collect()

    logging.info(f"=== {STAGE.upper()} DONE — matrices saved to {out_dir}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stratified train/test split of an encoded CIC-IDS2018 parquet.")
    parser.add_argument("--in_path", type=str, default="auto",
                        help="Path to an encoded parquet, or 'auto' for the interactive menu.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Override output dir. Default: chosen from the split-strategy token.")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--non_interactive", action="store_true")
    args = parser.parse_args()

    in_path = resolve_input(args.in_path, args.non_interactive)
    split_strat = parse_split_strat(in_path.name)

    if split_strat == "time_based":
        sys.exit("ABORT: time_based split not implemented — it needs chronological Timestamp handling, "
                 "not a stratified random split. Refusing to split this artefact.")

    base_dir = Path(args.out_dir) if args.out_dir else SPLIT_DIR_BY_STRAT[split_strat]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"split_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.log_file or str(run_dir / "split.log")
    set_logging(log_file, args.log_level)
    logging.info(f"Logging to {log_file}")
    logging.info(f"Detected split strategy: {split_strat}")
    logging.info(f"Run directory: {run_dir}/")

    generate_splits(in_path, run_dir)