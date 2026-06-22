import argparse
import sys
import time 
from pathlib import Path 
import pandas as pd 
import pyarrow.parquet as pq 
import numpy as np
from datetime import datetime


BASE_COLUMNS_TO_DROP = [
      # Near-constant (10)
      "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "CWE Flag Count",
      "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
      "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
      # Redundant Subflow (4)
      "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts",
  "Subflow Bwd Byts",
      # Redundant Seg Size (2) — corr=1.0 linear duplicates, override verifier's "KEEP"
      "Fwd Seg Size Avg", "Bwd Seg Size Avg",
      # Leakage (1)
      "Dst Port", "Flow ID", "Src IP", "Dst IP"]

def execution_checkpoint(step_name):
    response = input(f"\n[CHECKPOINT] Ready to execute: {step_name}. \nPress [Enter] to continue or [q] to quit: ").strip().lower()
    if response == 'q':
        sys.exit("Execution aborted at checkpoint.")

def get_strat_drops(split_strat):

    drops = BASE_COLUMNS_TO_DROP.copy()

    if split_strat == "8020_stratified":
        print(f"-> Split Strategy: {split_strat}")
        drops.append("Timestamp")
        print(f"Dropping columns: {drops}")
    elif split_strat == "time_based":
        print(f"-> Split Strategy: {split_strat}")
        print(f"Dropping columns: {drops}")

    return drops

def select_input(search_dir, split_strat):
    dir_path = Path(search_dir)
    if not dir_path.exists():
        sys.exit(f"Error: Search directory {search_dir} does not exist")

    if split_strat == "time_based":
        items = sorted((p for p in dir_path.glob("*-merged-per_day-*") if p.is_dir()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        kind = "per-day folder"
    else:
        items = sorted((p for p in dir_path.glob("*-merged-*.parquet") if p.is_file()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        kind = "merged parquet"

    if not items:
        sys.exit(f"Error: No {kind} found in {search_dir} for split_strat={split_strat}")

    print(f"\nFound {len(items)} {kind}(s) in {search_dir}:")
    for i, f in enumerate(items):
        tag = " [Most Recent]" if i == 0 else ""
        print(f" {i+1}. {f.name}{tag}")

    while True:
        choice = input(f"\nEnter the number to clean (Enter for [1]): ").strip()
        if choice == "":
            return items[0]
        if choice.isdigit() and 1 <= int(choice) <= len(items):
            return items[int(choice)-1]
        print(f"Invalid choice. Enter 1-{len(items)} or press Enter for the most recent.")


def resolve_input(in_path_arg, split_strat):
    if in_path_arg == "auto":
        return select_input(Path("preprocessing/processes_output/merged_datasets"), split_strat)
    in_path = Path(in_path_arg)
    if split_strat == "time_based":
        if not in_path.is_dir():
            sys.exit(f"Error: time_based expects a per-day FOLDER, got: {in_path}")
    elif not in_path.is_file():
        sys.exit(f"Error: input file does not exist: {in_path}")
    return in_path


def print_cleaned_stats(df):
    try:
        print("\n--- First 5 rows ---")
        print(df.head(5).to_string(max_cols=10))
        print("\n--- Feature stats (mean, std, min, max) ---")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = df[numeric_cols].describe().T[['mean', 'std', 'min', 'max']]
        pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
        print(stats.to_string())
        pd.reset_option('display.float_format')
    except MemoryError:
        print("\n(skipped stats — too large for describe())")
    except Exception as e:
        print(f"\n(stats error: {e})")

def clean_one(in_path, out_path, columns_to_drop, interactive=True, verbose=True):
    """Clean + dedup one raw parquet -> cleaned parquet. Returns final row count."""
    in_path = Path(in_path)
    all_cols = pq.read_schema(in_path).names
    dropped_cols = [c for c in all_cols if c in columns_to_drop]
    keep_cols = [c for c in all_cols if c not in columns_to_drop]

    if interactive:
        execution_checkpoint(f"Load {in_path.name} and drop predefined columns")
    print(f"\n1. Loading {len(keep_cols)} cols from {in_path.name} (dropping {len(dropped_cols)})")
    t0 = time.perf_counter()
    df = pd.read_parquet(in_path, columns=keep_cols)
    print(f"   Loaded: {len(df):,} rows x {len(df.columns)} cols ({time.perf_counter()-t0:.1f}s)")
    if dropped_cols:
        print(f"   Dropped: {dropped_cols}")

    if interactive:
        execution_checkpoint("Clean column names")
    print("\n2. Cleaning column names (whitespace/slashes -> _)")
    df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]

    if interactive:
        execution_checkpoint("Remove stray headers and empty labels")
    print("\n3. Removing stray headers and empty labels")
    n_before = len(df)
    stray_vals = [v for v in df['Protocol'].unique() if str(v).strip() == 'Protocol']
    stray_mask = df['Protocol'].isin(stray_vals)
    empty_vals = [v for v in df['Label'].unique() if pd.isna(v) or str(v).strip() == '']
    null_label_mask = df['Label'].isin(empty_vals)
    df = df[~(stray_mask | null_label_mask)].copy()
    df['Label'] = df['Label'].map({v: str(v).strip() for v in df['Label'].unique()})
    print(f"   Dropped {n_before - len(df):,} stray-header/empty-label rows")

    if interactive:
        execution_checkpoint("Coerce numeric columns and drop invalid entries")
    print("\n4. Coercing numeric columns and dropping invalid entries")
    n_before = len(df)
    feature_cols = [c for c in df.columns if c not in ['Label', 'Timestamp']]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)
    print(f"   Dropped {n_before - len(df):,} non-numeric/infinite rows")

    if interactive:
        execution_checkpoint("Drop negative Flow Duration")
    print("\n5. Dropping rows with negative Flow Duration")
    n_before = len(df)
    df = df[df["Flow_Duration"] >= 0]
    print(f"   Dropped {n_before - len(df):,} (Flow_Duration < 0)")

    if interactive:
        execution_checkpoint("Drop duplicates (features+Label, excluding Timestamp)")
    print("\n6. Dropping duplicate rows (excluding Timestamp from the key)")
    n_before = len(df)
    t0 = time.perf_counter()
    dedup_subset = [c for c in df.columns if c != "Timestamp"]
    df = df.drop_duplicates(subset=dedup_subset)
    n_dup = n_before - len(df)
    print(f"   Dropped {n_dup:,} duplicates ({100*n_dup/max(n_before,1):.2f}%) ({time.perf_counter()-t0:.1f}s)")

    if interactive:
        execution_checkpoint("Fix negative IAT artifacts")
    print("\n7. Clipping negative IAT artifacts to 0")
    iat_cols = [c for c in df.columns if 'IAT' in c]
    for col in iat_cols:
        df[col] = df[col].clip(lower=0)
    print(f"   Fixed {len(iat_cols)} IAT columns")

    if interactive:
        execution_checkpoint("Write cleaned parquet")
    print(f"\n8. Writing cleaned parquet -> {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression='snappy', index=False)
    out_mb = out_path.stat().st_size / (1024 * 1024)
    n_feat = len([c for c in df.columns if c not in ('Label', 'Timestamp')])
    has_ts = "Timestamp" in df.columns
    print(f"   Wrote {len(df):,} rows x {len(df.columns)} cols "
          f"({n_feat} features + Label{' + Timestamp' if has_ts else ''}, {out_mb:.1f} MB)")

    if verbose:
        print_cleaned_stats(df)

    return len(df)


def run_clean(in_path, out_dir, split_strat, encoding_strat):
    columns_to_drop = get_strat_drops(split_strat)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = Path(in_path)

    print(f"\nClean+dedup | split={split_strat.upper()} encoding={encoding_strat.upper()}")
    print(f"Input:  {in_path}")

    if split_strat == "time_based":
        base = in_path.name.split('-merged-per_day-')[0]
        out_folder = out_dir / f"{base}-cleaned-deduped_{split_strat}_{encoding_strat}_{ts}"
        out_folder.mkdir(parents=True, exist_ok=True)
        print(f"Output: {out_folder}/ (per-day)")

        day_files = sorted(in_path.glob("*.parquet"))
        if not day_files:
            sys.exit(f"ABORT: no day parquets found in {in_path}")
        execution_checkpoint(f"Clean {len(day_files)} day-files (straight through, no per-step prompts)")

        grand = 0
        for f in day_files:
            print(f"\n========== {f.name} ==========")
            grand += clean_one(f, out_folder / f.name, columns_to_drop, interactive=False, verbose=False)
        print(f"\n=== CLEAN DONE (per-day) — {len(day_files)} files, {grand:,} total rows ===")
        print(f"Output: {out_folder}/")
    else:
        base = in_path.name.split('-merged-')[0]
        out_path = out_dir / f"{base}-cleaned-deduped_{split_strat}_{encoding_strat}_{ts}.parquet"
        print(f"Output: {out_path}")
        clean_one(in_path, out_path, columns_to_drop, interactive=True, verbose=True)
        print("\n=== CLEAN DONE ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and deduplicate merged CIC-IDS2018 parquet(s).")

    parser.add_argument("--in_path", type=str, default="auto",
                        help="Merged parquet (8020) or per-day folder (time_based). 'auto' for menu.")
    parser.add_argument("--out_dir", type=str, default="preprocessing/processes_output/cleaned_datasets",
                        help="Dir to save cleaned output")
    parser.add_argument("--split_strat", type=str, choices=["8020_stratified", "time_based"],
                        required=True, help="Splitting strategy")
    parser.add_argument("--encoding_strat", type=str, choices=["binary", "multiclass"], default="binary",
                        help="Encoding strategy")

    args = parser.parse_args()

    in_path = resolve_input(args.in_path, args.split_strat)
    run_clean(in_path, args.out_dir, args.split_strat, args.encoding_strat)
    print("END OF PROCESS")