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
      "Dst Port",]

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

def select_merged_dataset(search_dir):

    dir_path = Path(search_dir)
    if not dir_path.exists():
        sys.exit(f"Error: Search directory {search_dir} does not exist")

    files = list(dir_path.glob("*-merged-*.parquet"))
    if not files:
        sys.exit(f"Error: No merged parquet files found in {search_dir}")

    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    print(f"Found {len(files)} merged datasets in {search_dir}:")
    for i, f in enumerate(files):
        tag = "[Most Recent]" if i == 0 else ""
        print(f" {i+1}. {f.name} {tag}")

    # User input to select file

    while True:
        choice = input(f"\nEnter the number of the merged dataset to clean (enter for [1]): ").strip()
        if choice == "":
            selected_file = files[0]
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(files):
            selected_file = files[int(choice)-1]
            break
        else:
            print(f"Invalid choice. Please enter a number between 1 and {len(files)}, or press Enter for the most recent file.")

    print(f"\nLoading selected file -> {selected_file.name}")
    return selected_file

def apply_drops_and_dedup(in_path, out_dir, split_strat, encoding_strat):

    base_name = Path(in_path).name.split('-merged-')[0]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(out_dir) / f"{base_name}-cleaned-deduped_{split_strat}_{encoding_strat}_{timestamp}.parquet"

    print(f"\nCleaning + deduping on merged file")
    print(f"\nInput: {in_path}")
    print(f"\nOutput: {out_path}")
    print(f"\nSplit: {split_strat.upper()}")
    print(f"\nEncoding: {encoding_strat.upper()}")

    columns_to_drop = get_strat_drops(split_strat)
    all_cols = pq.read_schema(in_path).names

    dropped_cols = [c for c in all_cols if c in columns_to_drop]
    keep_cols = [c for c in all_cols if c not in columns_to_drop]


    # Load df with pruned columns
    execution_checkpoint("Load merged parquet and drop predefined columns")
    print(f"\n1. Loading {len(keep_cols)} cols (dropping {len(BASE_COLUMNS_TO_DROP)} at load)")
    t0 = time.perf_counter()
    df = pd.read_parquet(in_path, columns=keep_cols)
    print(f"Loaded: {len(df):,} rows x {len(df.columns)} cols ({time.perf_counter()-t0:.1f}s)")
    
    print("\nSuccessfully dropped:")
    for col in dropped_cols:
        print(f"- {col}")

    execution_checkpoint("Clean column names")
    print("\n2. Cleaning column names (replacing whitespace/slashes with _)")
    df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]
    print(f"Cleaned column names: {df.columns[:5]} ... {df.columns[-5:]}")



    # Removing stray headers and empty labels
    execution_checkpoint("Remove stray headers and empty labels")
    print("\n3. Removing stray headers and empty labels")
    n_before = len(df)
    t0 = time.perf_counter()


    # 1. Identify stray headers
    unique_protocols = df['Protocol'].unique()
    stray_vals = [val for val in unique_protocols if str(val).strip() == 'Protocol']
    stray_mask = df['Protocol'].isin(stray_vals)

    # 2. Identify empty or null labels
    unique_labels = df['Label'].unique()
    empty_vals = [val for val in unique_labels if pd.isna(val) or str(val).strip() == '']
    null_label_mask = df['Label'].isin(empty_vals)

    # 3. Filter the dataframe (using .copy() to prevent memory fragmentation)
    df = df[~(stray_mask | null_label_mask)].copy()

    # 4. Strip whitespace from the REMAINING valid labels using a fast dictionary map
    remaining_labels = df['Label'].unique()
    clean_label_map = {val: str(val).strip() for val in remaining_labels}
    df['Label'] = df['Label'].map(clean_label_map)

    print(f"\nDropped {n_before - len(df):,} rows with stray headers or empty labels")
    print(f"Time taken: {time.perf_counter() - t0:.2f}s")



    execution_checkpoint("Coercing numeric columns and dropping invalid entries")
    print("\n4. Coercing numeric columns and dropping invalid entries")
    n_before = len(df)
    feature_cols = [c for c in df.columns if c not in ['Label', 'Timestamp']]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)
    print(f"\nDropped {n_before - len(df):,} rows with non-numeric or infinite feature values")


    # Drop rows with negative value flow durations (not possible to have neg flow duration so must be an artifact)
    execution_checkpoint("Drop negative Flow Duration")
    print("\n5. Dropping rows with negative Flow Duration")
    n_before = len(df)
    df = df[df["Flow_Duration"] >= 0]
    print(f"\nDropped {n_before - len(df):,} rows with Flow Duration < 0")
    print(f"{n_before:,}->{len(df):,}")



    # Drop duplicates
    execution_checkpoint("Dropping duplicate rows")
    print(f"\n6. Dropping duplicate rows")
    n_before = len(df)
    t0 = time.perf_counter()
    df = df.drop_duplicates()

    n_dup = n_before - len(df)
    pct = 100*n_dup/n_before
    print(f" Dropped {n_dup:,} duplicates ({pct:.2f}%) ({time.perf_counter()-t0:.1f}s)")
    print(f" {n_before:,} -> {len(df):,}")


    # Fixing IAT negative artificats
    execution_checkpoint("Fixing negative IAT artifacts")
    print(f"\n7. Fixing neg IAT time artifacts (clip to 0)")
    t0 = time.perf_counter()
    iat_cols =[c for c in df.columns if 'IAT' in c]
    for col in iat_cols:
        df[col] = df[col].clip(lower=0)
    print(f"\nFixed {len(iat_cols)} IAT columns ({time.perf_counter()-t0:.1f}s)")


    # Write cleaned parquet to output file
    execution_checkpoint("Writing cleaned parquet to output file")
    print(f"\n8. Writing cleaned parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    df.to_parquet(out_path, compression='snappy', index=False)
    out_mb = out_path.stat().st_size /(1024*1024)
    print(f" Wrote {out_path} ({out_mb:.1f} MB) ({time.perf_counter()-t0:.1f}s)")



    print(f"\n  FINAL")
    print(f"    Rows:  {len(df):,}")
    print(f"    Cols:  {len(df.columns)}  (61 features + Label)")
    print(f"    Size:  {out_mb:.1f} MB")
    print(f"    Path:  {out_path}")

    print(f"\n Loading cleaned parquet stats")
    try:
        
        print("\n--- First 5 Rows of Merged Dataset ---")
        print(df.head(5).to_string(max_cols=10))
        
        
        print("\n--- Feature Stats (mean, std, min, max) ---")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_df = df[numeric_cols].describe().T[['mean', 'std', 'min', 'max']]

        # float control
        pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
        print(stats_df.to_string())
        pd.reset_option('display.float_format')
        
    except MemoryError:
        print("\nERROR: The merged dataset is too large to fit into RAM for pandas statistical analysis.")
    except Exception as e:
        print(f"\nERROR analyzing Parquet file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and deduplicate merged CIC-IDS2018 parquet file")
    
    parser.add_argument("--in_path", type=str, default = "auto",
                        help="Path to merged parquet file to clean. Use 'auto' for dataset menu")
    
    parser.add_argument("--out_dir", type=str, default="preprocessing/processes_output/cleaned_datasets",
                        help="Dir to save the cleaned parquet file")

    parser.add_argument("--split_strat", type=str, choices=["8020_stratified", "time_based"], 
                        help="Splitting strategy", required=True)
    parser.add_argument("--encoding_strat", type=str, choices=["binary", "multiclass"],
                        help="Encoding strategy", default="binary")
    
    args = parser.parse_args()

    input_file = args.in_path
    if input_file == "auto":
        input_file = select_merged_dataset("preprocessing/processes_output/merged_datasets")

    apply_drops_and_dedup(
        in_path = input_file,
        out_dir=args.out_dir,
        split_strat=args.split_strat,
        encoding_strat=args.encoding_strat
    )

    print("END OF PROCESS")
