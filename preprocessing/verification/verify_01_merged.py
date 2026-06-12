from pathlib import Path 
import numpy as np
import pandas as pd
import pyarrow.parquet as pq 
import argparse
import sys

EXPECTED_ROWS = 16_233_002
BASE_COLS = 80
EXTRA_COLS = 84

NEAR_CONSTANT = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "CWE Flag Count",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg"
]

REDUNDANT_PAIRS = [
    ("Subflow Fwd Pkts", "Tot Fwd Pkts"),
    ("Subflow Fwd Byts", "TotLen Fwd Pkts"),
    ("Subflow Bwd Pkts", "Tot Bwd Pkts"),
    ("Subflow Bwd Byts", "TotLen Bwd Pkts"),
    ("Fwd Seg Size Avg", "Fwd Pkt Len Mean"),
    ("Bwd Seg Size Avg", "Bwd Pkt Len Mean")
]


def select_merged_dataset(search_dir):
    dir_path = Path(search_dir)
    if not dir_path.exists():
        sys.exit(f"Error: Search path does not exist - {search_dir}")

    files = list(dir_path.glob("*-merged-*.parquet"))
    if not files:
        sys.exit(f"Error: No merged parquet files not found in {search_dir}")

    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    print(f"Found {len(files)} merged datasets in {search_dir}")
    for i, f in enumerate(files):
        tag = "[Most Recent]" if i == 0 else ""
        print(f" {i+1}. {f.name} {tag}")

    while True:

        choice = input(f"\nEnter the number of the merged dataset to verify (Enter for [1])").strip()
        if choice == "":
            selected_file = files[0]
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(files):
            selected_file = files[int(choice)-1]
            break
        else:
            print(f"Invalid choice - pick a dataset between 1 and {len(files)}")
    return selected_file


    

def compare_before_after(csv_dir, parquet_file):
    print("\n1. Raw CSV files vs merged Parquet")
    csv_path = Path(csv_dir)

    if csv_path.exists():
        csv_files = list(csv_path.glob("*.csv"))
        raw_size_mb = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
        print(f"Raw CSV total size: {raw_size_mb:.1f} MB across {len(csv_files)} files")
    else:
        print(f"Raw CSV file input directory {csv_dir} not found")
        raw_size_mb = 0
        n_csv_rows = 0
        for file in csv_path.glob("*.csv"):
            n_csv_rows += sum(1 for _ in open(file)) - 1  # subtract header row
        print(f"Total rows across all {len(csv_files)} CSV files: {n_csv_rows:,}")


    metadata = pq.read_metadata(parquet_file)
    n_rows = metadata.num_rows
    pq_size_mb = Path(parquet_file).stat().st_size / (1024 * 1024)

    print(f"Merged Parquet size: {pq_size_mb:.1f} MB")
    if raw_size_mb > 0:
        print(f"Compression ratio (CSV to Parquet): {raw_size_mb / pq_size_mb:.1f}x smaller")

    print(f"Total rows in Parquet: {n_rows:,} (expected ~{EXPECTED_ROWS:,})")
    assert abs(n_rows - EXPECTED_ROWS) < 50_000, f"Row count off by {abs(n_rows - EXPECTED_ROWS):,} from expected"
    print("OK: Row count matches expected range")

def verify_schema_and_mismatch(parquet_file):

    print("\n2. Schema and column mismatch checks")
    schema = pq.read_schema(parquet_file)
    cols = schema.names

    print(f"Total columns detected: {len(cols)}")

    if len(cols) == BASE_COLS:
        print("OK: Schema matches expected base columns (80 features + Label) - extra columns successfully dropped")
        print(f"Columns: {cols}")
    elif len(cols) == EXTRA_COLS:
        print("WARNING: Schema has 84 columns - expected 80 features + Label. Some extra columns may not have been dropped")
        print(f"Columns: {cols}")
    else:
        print(f"ERROR: Unexpected column count: {len(cols)} (expected {BASE_COLS} or {EXTRA_COLS})")

    
    print(f"\nChecking for non-string columns (all columns should be string post merge to avoid type inference issues)")
    types = {f.name: str(f.type) for f in schema}
    non_strings = {col: typ for col, typ in types.items() if typ not in ("string", "large_string")}
    if non_strings:
        print(f"ERROR: Non-string columns detected: {non_strings}")
    else:
        print("OK: All columns are string type as expected")

def verify_ghost_data(parquet_file):

    print("\n3. Checking for ghost data (stray headers/empty labels)")

    df = pd.read_parquet(parquet_file, columns=["Protocol", "Label"])

    stray_count = (df['Protocol'].str.strip() == 'Protocol').sum()
    print(f"Stray header rows detected: {stray_count:,} stray rows")

    null_strings = df['Label'].isin(['nan', 'NaN', 'null', 'Null', '']).sum()
    print(f"Empty/null label rows detected: {null_strings:,} rows")

    print("OK: ghost data checks complete - cleaning will take care of this")

def verify_required_feature_stats(parquet_file):

    print("\n4. Feature Stats Check & Negative Flow Rows/IAT Artifacts")
    print("Loading sample columns and converting string to numeric")
    sample_cols = ["Flow Duration", "Tot Fwd Pkts", "Flow Byts/s", "Pkt Len Mean", "Down/Up Ratio"]
    df = pd.read_parquet(parquet_file, columns=sample_cols+ ["Label"])

    for col in sample_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
    stats = df[sample_cols].describe().T[['min', 'mean', 'max', 'std']]
    print("\n4.1 Sample Feature Statistics:")
    print(stats.to_string())
    pd.reset_option('display.float_format')

def verify_data_artifacts(parquet_file):

    df = pd.read_parquet(parquet_file, columns=["Flow Duration", "Label"])
    df["Flow Duration"] = pd.to_numeric(df["Flow Duration"], errors='coerce')

    neg_mask = df["Flow Duration"] < 0
    n_neg = int(neg_mask.sum())
    print(f"\n4.2 Rows with Flow Duration < 0: {n_neg:,} ({100*n_neg/len(df):.4f}%)")

    found_neg_flow = False
    if n_neg > 0:
        found_neg_flow = True
        by_label = df.loc[neg_mask, "Label"].value_counts().head(3)
        print("Top Labels associated with negative flow duration:")
        for label, count in by_label.items():
            print(f"  - {label}: {count:,}")

    if found_neg_flow:
        print("\nNegative Flow Duration artifacts found - must be clipped to 0 during cleaning")
    else:
        print("\nNegative Flow Duration artifacts not found - clear")

    print("\n4.3 Scanning IAT columns for negative artifacts")
    schema = pq.read_schema(parquet_file).names
    iat_cols = [c for c in schema if "IAT" in c]

    df = pd.read_parquet(parquet_file, columns = iat_cols)
    total_rows = len(df)

    found_neg_iat = False
    for col in iat_cols:
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        n_neg = (numeric_col<0).sum()
        if n_neg > 0:
            found_neg_iat = True
            min_val = numeric_col.min()
            print(f"Warning: {col:20s}: {n_neg:>8,} rows < 0 | Min value: {min_val:,.0f}")

    if found_neg_iat:
        print("\nIAT artifacts found - must be clipped to 0 during cleaning")
    else:
        print("\nIAT artifacts not found - clear")

def verify_droppable_columns(parquet_file): 
    print("\n5. Proof for dropping columns")
    schema_cols = pq.read_schema(parquet_file).names 

    # Near constant columns 
    valid_constants = [c for c in NEAR_CONSTANT if c in schema_cols]
    print(f"\n1.1. Near-constant features - ({len(valid_constants)}) candidates" )
    print("Can drop column if column has <= 3 unique values AND > 99% zeros")

    print(f"\n{'Column Name':<20} | {'Unique':<6} | {'% Zero':<8} | {'Std Dev':<8} | {'Verdict'} ")

    if valid_constants:
        df_const = pd.read_parquet(parquet_file, columns=valid_constants)
        for col in valid_constants:
            numeric_cols = pd.to_numeric(df_const[col], errors='coerce')
            unique_vals = numeric_cols.nunique()
            pct_zero = 100.0 * (numeric_cols == 0).sum() / len(numeric_cols)
            std_dev = numeric_cols.std()    

            verdict = "Safe to drop" if (unique_vals <= 3 and pct_zero > 99.0) else "Keep - do not drop"
            print(f"{col:<20} | {unique_vals:<6} | {pct_zero:>7.3f}% | {std_dev:>8.4f} | {verdict}")

    del df_const


    # Redundant pairs

    print(f"\nRedunadant (duplicate) pairs (potential - {len(REDUNDANT_PAIRS)} pairs)")
    print("Threshols to drop: Pearson Correlation > 0.9999")

    print(f"{'Target (To Drop)':<20} | {'Base (To Keep)':<20} | {'Correlation':<12} | {'Verdict'}")

    cols_to_load = []
    for target, base in REDUNDANT_PAIRS:
        if target in schema_cols and base in schema_cols:
            cols_to_load.extend([target, base])

    if cols_to_load:
        df_pairs = pd.read_parquet(parquet_file, columns=list(set(cols_to_load)))
        for col in cols_to_load:
            df_pairs[col] = pd.to_numeric(df_pairs[col], errors='coerce')
    
    for target, base in REDUNDANT_PAIRS:
        if target in df_pairs.columns and base in df_pairs.columns:
            corr = df_pairs[target].corr(df_pairs[base])
            verdict = "Proven droppable (safe to drop)" if corr > 0.9999 else "Not safe to drop"
            print(f"{target:<20} | {base:<20} | {corr:>11.6f} | {verdict}")

    del df_pairs

    #Leakage drops 
    print(f"\nLeakage drops")
    print(f"{'Column Name':<20} | {'Reason for Dropping'}")

    if "Dst Port" in schema_cols:
        print(f"{'Dst Port':<20} | DROP (port number risk class memorisation)")



def verify_labels(path):
    print("\n6. Label distribution")
    labels = pd.read_parquet(path, columns=["Label"])["Label"]
    counts = labels.value_counts()
    total = int(counts.sum())
    print(f" Unique labels: {len(counts)} ")

    for label, count in counts.items():
        pct = 100 * count / total
        print(f"{label:35s} {count:>11,} ({pct:6.3f}%)")

    return counts



if __name__ =="__main__":
    
    parser = argparse.ArgumentParser(description="Verify merged dataset")
    parser.add_argument("--in_path", type=str, default="auto", 
                        help="Path to merged file. Use 'auto' for menu.")
    parser.add_argument("--raw_dir", type=str, required=True,
                         help="Path to raw csv files")
    
    args = parser.parse_args()



    try:
        merged_dataset = Path(args.in_path)
        if args.in_path == "auto":
            merged_dataset = select_merged_dataset("preprocessing/processes_output/merged_datasets")

        raw_files = Path(args.raw_dir)
        

        print(f"\n Running checks on: {merged_dataset.name}")

        compare_before_after(raw_files, merged_dataset)
        verify_schema_and_mismatch(merged_dataset)
        verify_ghost_data(merged_dataset)
        verify_required_feature_stats(merged_dataset)
        verify_data_artifacts(merged_dataset)
        verify_droppable_columns(merged_dataset)
        verify_labels(merged_dataset)
        print("\nAll checks complete")

    except Exception as e:
        print(f"\nVerificaton failed: {e}")

    print("END OF PROCESS")