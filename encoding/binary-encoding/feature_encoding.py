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
        sys.exit(f"Error: input dir {search_dir} does not exist")

    items = sorted(
        (p for p in search_dir.iterdir()
         if (p.is_file() and p.suffix == ".parquet") or p.is_dir()),
        key=lambda f: f.stat().st_mtime, reverse=True
    )
    if not items:
        sys.exit(f"No cleaned datasets found in {search_dir}")

    print(f"\nFound {len(items)} cleaned dataset(s) in {search_dir}")
    for i, f in enumerate(items):
        tag = " [Most Recent]" if i == 0 else ""
        kind = "dir" if f.is_dir() else "file"
        print(f"    {i+1}. {f.name}  ({kind}){tag}")

    while True:
        choice = input("\nEnter number to encode (Enter for [1]): ").strip()
        if choice == "":
            return items[0]
        if choice.isdigit() and 1 <= int(choice) <= len(items):
            return items[int(choice)-1]
        print(f"Invalid - enter a number 1-{len(items)} or press Enter")


def resolve_input(in_path_arg):
    if in_path_arg == "auto":
        return select_clean_dataset(INPUT_DIR)
    in_path = Path(in_path_arg)
    if not in_path.exists():
        sys.exit(f"Error: input does not exist: {in_path}")
    return in_path

def passthrough_timestamp(df, split_strat):
    """time_based only: parse the retained Timestamp string to int64 epoch
    seconds so it is machine-readable and sortable. It is a passthrough column
    for the split stage to order on and then DROP — never a model feature."""

    if split_strat == "time_based":
        if "Timestamp" not in df.columns:
            sys.exit("ABORT: time_based encoding expects a 'Timestamp' column from cleaning.")
        ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        n_bad = int(ts.isna().sum())
        if n_bad:
            sys.exit(f"ABORT: {n_bad:,} Timestamp values failed to parse (%d/%m/%Y %H:%M:%S)")
        df["Timestamp"] = ts.values.astype("int64") // 1_000_000_000
        logging.info(f" Timestamp -> int64 epoch seconds (passthrough; "
                     f"min={int(df['Timestamp'].min())} max={int(df['Timestamp'].max())})")
    elif "Timestamp" in df.columns:
        sys.exit("ABORT: 8020_stratified input unexpectedly contains 'Timestamp'.")
    return df

def parse_split_strat(name):
    for token in ("8020_stratified", "time_based"):
        if token in name:
            return token
    sys.exit(f"Cannot parse split strategy from: {name}")

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

def encode_one(in_path, out_path, split_strat):
    in_path = Path(in_path)
    logging.info(f"--- Encoding {in_path.name} ---")
    t0 = time.perf_counter()
    df = pd.read_parquet(in_path)
    logging.info(f" Loaded {len(df):,} rows x {len(df.columns)} cols in {time.perf_counter()-t0:.1f}s")

    validate_input(df)
    df = encode_protocol(df)
    df = encode_target_binary(df)
    df = passthrough_timestamp(df, split_strat)
    verify_machine_readable(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="snappy", index=False)
    out_mb = out_path.stat().st_size / (1024 * 1024)
    logging.info(f" Wrote {out_path} ({len(df):,} rows x {len(df.columns)} cols, {out_mb:.1f} MB)")
    return len(df)


def run_encoding(in_path, out_dir, split_strat, ts):
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = in_path.name.split("-cleaned-deduped")[0]

    logging.info(f"=== {STAGE.upper()} STAGE START === split={split_strat} target={TARGET_ENCODING}")
    t_start = time.perf_counter()

    if in_path.is_dir():
        out_folder = out_dir / f"{base}_encoded_{split_strat}_{TARGET_ENCODING}_{ts}"
        out_folder.mkdir(parents=True, exist_ok=True)
        day_files = sorted(in_path.glob("*.parquet"))
        if not day_files:
            sys.exit(f"ABORT: no day parquets in {in_path}")
        logging.info(f"Per-day encode: {len(day_files)} files -> {out_folder}/")
        grand = 0
        for f in day_files:
            grand += encode_one(f, out_folder / f.name, split_strat)
        logging.info(f"{STAGE.upper()} DONE (per-day) -- {len(day_files)} files, {grand:,} rows "
                     f"in {time.perf_counter()-t_start:.1f}s -> {out_folder}/")
    else:
        out_path = out_dir / f"{base}_encoded_{split_strat}_{TARGET_ENCODING}_{ts}.parquet"
        n = encode_one(in_path, out_path, split_strat)
        logging.info(f"{STAGE.upper()} DONE -- {n:,} rows in {time.perf_counter()-t_start:.1f}s -> {out_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Feature encoding of cleaned CIC-IDS2018 parquet(s).")
    parser.add_argument("--in_path", type=str, default="auto",
                        help="Cleaned parquet (8020) or per-day folder (time_based), or 'auto' for menu.")
    parser.add_argument("--out_dir", type=str, default=str(OUTPUT_DIR),
                        help="Dir to write encoded output.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Log file path. Default: encoding_<strat>_<ts>.log in out_dir.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    in_path = resolve_input(args.in_path)
    split_strat = parse_split_strat(in_path.name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    log_file = args.log_file or str(out_dir / f"encoding_{split_strat}_{ts}.log")
    set_logging(log_file, args.log_level)
    logging.info(f"Logging to {log_file}")

    run_encoding(in_path, out_dir, split_strat, ts)