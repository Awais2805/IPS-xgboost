import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
import gc
import numpy as np

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True
    )


def select_encoded_dataset(search_dir):
    if not search_dir.exists():
        sys.exit(f"Error: {search_dir} does not exist")

    items = sorted(
        (p for p in search_dir.iterdir()
         if (p.is_file() and p.name.endswith(".parquet") and "_encoded_" in p.name)
         or (p.is_dir() and "_encoded_" in p.name)),
        key=lambda f: f.stat().st_mtime, reverse=True
    )
    if not items:
        sys.exit(f"Error: No encoded datasets found in {search_dir}")

    print(f"\nFound {len(items)} encoded dataset(s) in {search_dir}")
    for i, f in enumerate(items):
        tag = " [Most Recent]" if i == 0 else ""
        kind = "dir" if f.is_dir() else "file"
        print(f" {i+1}. {f.name}  ({kind}){tag}")

    while True:
        choice = input("\nEnter number of dataset to split (Enter for [1]): ").strip()
        if choice == "":
            return items[0]
        if choice.isdigit() and 1 <= int(choice) <= len(items):
            return items[int(choice)-1]
        print(f"Invalid - enter 1-{len(items)} or press Enter")


def resolve_input(in_path_arg, non_interactive):
    if in_path_arg != "auto":
        in_path = Path(in_path_arg)
        if not in_path.exists():
            sys.exit(f"Error: input does not exist: {in_path}")
        return in_path

    if non_interactive:
        items = sorted(
            (p for p in INPUT_DIR.iterdir()
             if (p.is_file() and p.name.endswith(".parquet") and "_encoded_" in p.name)
             or (p.is_dir() and "_encoded_" in p.name)),
            key=lambda f: f.stat().st_mtime, reverse=True
        )
        if not items:
            sys.exit(f"Error: no encoded datasets found in {INPUT_DIR}")
        return items[0]
    return select_encoded_dataset(INPUT_DIR)


def parse_split_strat(file_name):
    for token in ("8020_stratified", "time_based"):
        if token in file_name:
            return token

    sys.exit(
        f"Cannot parse split strat from {file_name}. "
        "Expected an '8020_stratified' or 'time_based' token"
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

    # Halve the in-RAM footprint. Lossless for tree splitting (XGBoost works in float32 regardless).
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

    logging.info("3. Stratified split — writing one side at a time")
    for name, idx in (("train", train_idx), ("test", test_idx)):
        t1 = time.perf_counter()
        df.iloc[idx].to_parquet(out_dir / f"X_{name}.parquet", compression="snappy", index=False)
        y.iloc[idx].to_frame().to_parquet(out_dir / f"y_{name}.parquet", compression="snappy", index=False)
        logging.info(f"   {name}: {len(idx):,} rows ->parquet ({time.perf_counter()-t1:.1f}s)")
        gc.collect()

    logging.info(f"=== {STAGE.upper()} DONE — matrices saved to {out_dir}")


def _write_side(name, metas, out_dir):
    """Concatenate the day-files for one side, dropping Timestamp (passthrough)
    and popping Target. Writes incrementally so only one day is in RAM at a time.
    Every non-onehot feature is cast to float32 — XGBoost casts to float32 anyway,
    and it gives an identical schema across days (clean infers int64 vs float64
    per file, which otherwise breaks the incremental ParquetWriter)."""
    x_path = out_dir / f"X_{name}.parquet"
    writer = None
    ref_cols = None
    y_parts = []
    total = 0
    atk = 0

    for f, _, _ in metas:
        df = pd.read_parquet(f)
        y = df.pop(TARGET_COL)
        if "Timestamp" in df.columns:
            df = df.drop(columns=["Timestamp"])

        # Unify per-file dtype drift: cast everything except the Protocol one-hots
        # to float32 so every day writes an identical schema.
        onehot = [c for c in df.columns if c.startswith("Protocol_")]
        feat = [c for c in df.columns if c not in onehot]
        df[feat] = df[feat].astype("float32")

        if ref_cols is None:
            ref_cols = list(df.columns)
        elif set(df.columns) != set(ref_cols):
            sys.exit(f"ABORT: column mismatch in {f.name} "
                     f"(missing={set(ref_cols)-set(df.columns)}, extra={set(df.columns)-set(ref_cols)})")
        df = df[ref_cols]   # enforce identical column order across days

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(x_path, table.schema, compression="snappy")
        writer.write_table(table)


        y_parts.append(y)
        total += len(df)
        atk += int(y.sum())
        del df, table
        gc.collect()

    if writer is not None:
        writer.close()

    y_full = pd.concat(y_parts, ignore_index=True)
    y_full.to_frame(name=TARGET_COL).to_parquet(out_dir / f"y_{name}.parquet", compression="snappy", index=False)
    logging.info(f"   {name}: {total:,} rows | {len(ref_cols)} features | "
                 f"attack {100*atk/max(total,1):.4f}% -> X_{name}.parquet / y_{name}.parquet")

def generate_time_based_splits(in_dir, out_dir, test_days):
    in_dir = Path(in_dir)
    day_files = list(in_dir.glob("*.parquet"))
    if not day_files:
        sys.exit(f"ABORT: no encoded day parquets in {in_dir}")

    logging.info(f"=== {STAGE.upper()} STAGE START (time_based) ===")
    logging.info(f"Input:  {in_dir}/")
    logging.info(f"Output: {out_dir}/")

    logging.info("1. Ordering day-files chronologically by min Timestamp")
    day_files = sorted(day_files, key=lambda f: int(pd.read_parquet(f, columns=["Timestamp"])["Timestamp"].min()))

    logging.info("2. Per-day class balance:")
    day_meta = []
    total_rows = 0
    for f in day_files:
        y = pd.read_parquet(f, columns=["Target"])["Target"]
        n = len(y); a = int(y.sum())
        day_meta.append((f, n, a))
        total_rows += n
        logging.info(f"   {f.name}: {n:,} rows | attack {100*a/max(n,1):.2f}%")

    logging.info("3. Choosing chronological train/test boundary (whole days)")
    if test_days is not None:
        if not (1 <= test_days < len(day_files)):
            sys.exit(f"--test_days must be between 1 and {len(day_files)-1}")
        n_test = test_days
    else:
        cum = 0; n_test = 0
        for f, n, _ in reversed(day_meta):
            cum += n; n_test += 1
            if cum >= TEST_SIZE * total_rows:
                break

    split_at = len(day_files) - n_test
    train_meta, test_meta = day_meta[:split_at], day_meta[split_at:]
    train_atk = sum(a for _, _, a in train_meta)
    test_atk = sum(a for _, _, a in test_meta)

    logging.info(f"   Train days ({len(train_meta)}): {[f.name for f, _, _ in train_meta]}")
    logging.info(f"   Test  days ({len(test_meta)}): {[f.name for f, _, _ in test_meta]}")
    if train_atk == 0:
        logging.warning("   Train set has 0 attacks — model cannot learn the positive class!")
    if test_atk == 0:
        logging.warning("   Test set has 0 attacks — cannot evaluate attack detection!")

    logging.info("4. Writing train side")
    _write_side("train", train_meta, out_dir)
    logging.info("5. Writing test side")
    _write_side("test", test_meta, out_dir)
    logging.info(f"=== {STAGE.upper()} DONE (time_based) — matrices saved to {out_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train/test split of an encoded CIC-IDS2018 parquet (8020 stratified or time_based day-split).")
    parser.add_argument("--in_path", type=str, default="auto",
                        help="Encoded parquet (8020) or per-day folder (time_based), or 'auto' for menu.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Override output dir. Default: chosen from the split-strategy token.")
    parser.add_argument("--test_days", type=int, default=None,
                        help="time_based only: number of trailing day-files for test. Default: ~TEST_SIZE by rows.")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--non_interactive", action="store_true")
    args = parser.parse_args()

    in_path = resolve_input(args.in_path, args.non_interactive)
    split_strat = parse_split_strat(in_path.name)

    base_dir = Path(args.out_dir) if args.out_dir else SPLIT_DIR_BY_STRAT[split_strat]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"split_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.log_file or str(run_dir / "split.log")
    set_logging(log_file, args.log_level)
    logging.info(f"Logging to {log_file}")
    logging.info(f"Detected split strategy: {split_strat}")
    logging.info(f"Run directory: {run_dir}/")

    if split_strat == "time_based":
        generate_time_based_splits(in_path, run_dir, args.test_days)
    else:
        generate_splits(in_path, run_dir)
