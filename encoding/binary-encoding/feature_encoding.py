import argparse
import logging
import sys 
import time 
from datetime import datetime
from pathlib import Path 

import numpy as np
import pandas as pd

STAGE = "encoding"
EXPECTED_PROTOCOLS = [0,6,17]
LEAKEAGE_COLS = ("Dst_Port",) 
TARGET_ENCODING = ("binary")
BENIGN_LABEL = "Benign"

INPUT_DIR = Path("preprocessing/processes_output/cleaned_datasets")
OUTPUT_DIR = Path("preprocessing/processes_output/encoded_datasets")


def set_logging(log_file, log_level="info"):
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level = level,
        format = "%(asctime)s %(levelname)s %(message)s",
        handlers = handlers,
        force=True
    )

def select_clean_dataset(search_dir):
    if not search_dir.exists():
        sys.exit(f"Error: input dior {search_dir} does not exist")

    files = sorted(
        search_dir.glob("*.parquet"),
        key = lambda f: f.stat().st_mtime,
        reverse=True  
    )

    if not files: 
        sys.exit(f"No parquet files found")

    print(f"\nFound {len(files)} datasets in {search_dir}")

    for i, f in enumerate(files): 
        tag = " [Most Recent]" if i == 0 else ""
        print(f"    {i+1}. {f.name}{tag}")

    while True:
        choice = input("\nEnter number to encode (Enter for [1])").strip()
        if choice == "":
            return files[0]
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice)-1]
        print(f"Invalid - enter a number: 1-{len(files)} or Enter")


def resolve_input(in_path_arg):
    if in_path_arg == "auto":
        return select_clean_dataset(INPUT_DIR)
    in_path = Path(in_path_arg)

    if not in_path.is_file():
        sys.exit(f"Error: input file does not exist: {in_path}")
    return in_path


def parse_split_strat(file_name):

    for token in ("8020_stratified", "time-based"):
        if token in file_name:
            return token
    
    sys.exit(f"Cannot parse split strategy from: {file_name}")

def validate_input(df):

    leaked = [c for c in LEAKEAGE_COLS if c in df.columns]
    if leaked:
        sys.exit(f"DATA LEAKAGE IN {leaked.upper()}")

    for required in ("Protocol", "Label"):
        if required not in df.columns:
            sys.exit(f"ABORTL requied column {required.upper()} missing from input")

    protocols = sorted(df["Protocol"].dropna().astype(int).unique().tolist())
    unknown = set(protocols) - set(EXPECTED_PROTOCOLS)
    if unknown:
        logging.warning(f"UNKNOWN PROTOCOL VALUE(S): {unknown.upper()}")
    logging.info(f"Input OK: Protocol values {protocols}, labels {sorted(df['Label'].unique().tolist())}")

def encode_protocol(df):
    dummies = pd.get_dummies(df["Protocol"].astype(int), prefix="Protocol")
    cols = [f"Protocol_{p}" for p in EXPECTED_PROTOCOLS]
    dummies = dummies.reindex(columns=cols, fill_value=0).astype("int8")
    df = pd.concat([df.drop(columns=["Protocol"]), dummies], axis=1)

    logging.info("Protocol one-hot encoding:")
    for c in cols:
        logging.info(f" {c}: {int(df[c].sum()):,} rows set")
    return df

def encode_target_binary(df):
    df["Target"] = (df["Label"] != BENIGN_LABEL).astype("int8")
    counts = df["Target"].value_counts().sort_index()
    n_benign = int(counts.get(0,0))
    n_attack = int(counts.get(1,0))

    logging.info(
        f"Target binary encoding - Benign (0): {n_benign:,} Attack (1): {n_attack:,} "
        f"({100*n_attack / len(df):.3f}% attack)"
    )
    return df.drop(columns=["Label"])

def verify_machine_readable(df):
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        sys.exit(f"ABORT: Non-numeric (object) column(s) remaining: {object_cols}")
    
    n_nan = int(df.isna().sum().sum())

    if n_nan:
        sys.exit(f"ABORT: {n_nan} NaN values found")

    logging.info(f"OK: All {len(df.columns)} columns numeric, 0 NaN")
    logging.info(f"dtypes: {df.dtypes.value_counts().to_dict()}")
    logging.info("First 5 rows on encoded dataset:\n" + df.head().to_string(max_cols=12))

def run_encoding(in_path, out_path, split_strat):
    logging.info(f"=== {STAGE.upper()} STAGE START ===")
    logging.info(f"Input:  {in_path}")
    logging.info(f"Output: {out_path}")
    logging.info(f"Config: split={split_strat} target variable encoding={TARGET_ENCODING}")

    t_start = time.perf_counter()

    logging.info("1. Loading cleaned parquet")
    t0 = time.perf_counter()
    df = pd.read_parquet(in_path)
    logging.info(f"Loaded {len(df):,} rows x {len(df.columns)} cols in {time.perf_counter() - t0:.1f}s")
    
    logging.info("2. Validating Input")
    validate_input(df)

    logging.info("3. Encoding Protocol (one-hot)")
    df = encode_protocol(df)

    logging.info("4. Encoding Label -> Taregt (binary)")
    df = encode_target_binary(df)

    logging.info("5. Verifying machine-readable")
    verify_machine_readable(df)

    logging.info("6. Writing encoded parquet")
    t0= time.perf_counter()
    df.to_parquet(out_path, compression="snappy", index=False)
    out_mb = out_path.stat().st_size / (1024 * 1024)
    logging.info(f" Wrote {out_path} ({out_mb} MB) in {time.perf_counter() - t0:.1f}s")

    runtime = time.perf_counter() - t_start
    logging.info(f"{STAGE.upper()} DONE -- {len(df):,} rows x {len(df.columns)} in {runtime:.1f}s")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Feature encoding a cleaned CIC-IDS2018 parquet.")
    parser.add_argument("--in_path", type=str, default="auto",
                        help="Path to a cleaned parquet, or 'auto' for menu.")
    parser.add_argument("--out_dir", type=str, default=str(OUTPUT_DIR),
                        help="Path to write the encoded parquet.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Log file path. Default: alongside the output parquet with a .log suffix.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    in_path = resolve_input(args.in_path)
    split_strat = parse_split_strat(in_path.name)

    base = in_path.name.split("-cleaned-deduped")[0]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base}_encoded_{split_strat}_{TARGET_ENCODING}_{ts}.parquet"

    log_file = args.log_file or str(out_path.with_suffix(".log"))
    set_logging(log_file, args.log_level)
    logging.info(f"Logging to {log_file}")

    run_encoding(in_path, out_path, split_strat)

