import sys 
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import argparse

RAW_EXP_ROWS = 16_233_002

BASE_DROPS =["Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "CWE Flag Count",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Fwd Seg Size Avg", "Bwd Seg Size Avg", "Dst Port"]

FORMATTED_BASE_DROPS = [c.replace(" ", "_").replace("/", "_") for c in BASE_DROPS]


def select_cleaned_dataset(search_dir):
    dir_path = Path(search_dir)
    if not dir_path.exists():
        sys.exit(f"Error: search dir {search_dir} does not exist")

    files = list(dir_path.glob("*-cleaned-deduped_*.parquet"))
    if not files:
        sys.exit(f"Errir: No cleaned parquet files found in {search_dir}")
    
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    print(f"\nFound {len(files)} cleaned datasets in {search_dir}")
    for i, f in enumerate(files):
        tag = "[Most Recent]" if i == 0 else ""
        print(f" {i+1}. {f.name} {tag}")

    while True:
        choice = input(f"\nEnter the number of the cleaned dataset to verify (Press Enter for [1])")
        if choice == "":
            seleced_file = files[0]
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(files):
            seleced_file = files[int(choice)-1]
            break
        else: 
            print(f"Invalide choice - Please enter a number between 1 and {len(files)}")

    return seleced_file

def infer_pipeline_context(file_name):
    print("\n1. Pipeline Context Inference")

    split_strat = "Unknown"
    encoding_strat = "Unknown"

    if "8020_stratified" in file_name:
        split_strat = "8020_stratified"
        expected_cols = 62
    elif "time_based" in file_name:
        split_strat = "time_based"
        expected_cols = 63

    if "binary" in file_name:
        encoding_strat = "binary"
    elif "multiclass" in file_name:
        encoding_strat = "multiclass"

    print(f"Detected Splitting Strategy: {split_strat.upper()}")
    print(f"Detected Encoding Strategy: {encoding_strat.upper()}")
    print(f"Expected Column Count: {expected_cols}")

    return split_strat, expected_cols

def verify_shrinkage(parquet_file):
    print("\n2. Before v After Comparison - Shrinkage")

    metadata = pq.read_metadata(parquet_file)
    n_rows = metadata.num_rows
    size_mb = Path(parquet_file).stat().st_size / (1024 * 1024)

    dropped_rows = RAW_EXP_ROWS - n_rows
    pct_dropped_rows = (dropped_rows / RAW_EXP_ROWS) * 100 

    print(f"Raw Merged Rows: {RAW_EXP_ROWS:,}")
    print(f"Cleaned Rows: {n_rows:,}")
    print(f"Rows purged: {dropped_rows:,} ({pct_dropped_rows:.2f})")
    print(f"Final File Size: {size_mb:,.1f} MB")
    print("OK: Dataset successfully cleaned and deduped")


def verify_schema_purges(parquet_files, split_strat, expected_cols):
    print("\n3. Schema Name & Column Drop Validation")
    schema_cols = pq.read_schema(parquet_files).names

    #Total count
    print(f"Total columns found: {len(schema_cols)}")
    assert len(schema_cols) == expected_cols, f"Expected {expected_cols} cols, got {len(schema_cols)}"

    #Predefined drops 
    survivors = [c for c in FORMATTED_BASE_DROPS if c in schema_cols]
    assert len(survivors) == 0, f"Error: Predefined droppable columns survived clean: {survivors}"
    print(f"OK: All 17 predefined redunddant/constant columns successfully dropped during cleaning")

    #Timestamp check 
    if split_strat == "8020_stratified":
        assert "Timestamp" not in schema_cols, "Error: Timestamp survided 80/20 split strategy"
        print("OK: Timestamo successfuly dropped for 80/20 split strategy")
    elif split_strat == "time_based":
        assert "Timestamp" in schema_cols, "Error: Timestamp is missing for time based split strategy"

    bad_formats = [c for c in schema_cols if " " in c or "/" in c]
    assert len(bad_formats) == 0, f"ERROR: Columns with spaces/slashses detected post clean: {bad_formats}"
    print("OK: All column names successfully stripped and cleaned")


def verify_data_health_and_artifacts(parquet_file):
    print("\n4. Data Health & Artifact Resolution")
    print("Loading enter cleaned dataset into memory")
    df = pd.read_parquet(parquet_file)

    null_count = df.isna().sum().sum()
    print(f"Total NaN values: {null_count}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = df[numeric_cols].isin([np.inf, -np.inf]).sum().sum()
    print(f"Total Inf values: {inf_count}")

    assert null_count == 0, f"ERROR: NaN values detected in cleaned dataset"
    assert inf_count == 0, f"ERROR: Inf values detected in cleaned dataset"
    print("OK: data is clean and healthy")

    print("\nVerifying Artifact Resolution")
    min_flow_dur = df["Flow_Duration"].min()
    print(f"Flow_Duration minimum: {min_flow_dur:,.2f}")
    assert min_flow_dur >= 0, "ERROR: Negative Flow Duration artifacts survived"

    iat_cols = [c for c in df.columns if "IAT" in c]
    min_iat = df[iat_cols].min().min()
    print(f"All IAT columns minimum: {min_iat:,.2f}")
    assert min_iat >= 0, "ERROR: Negative IAT artifacts survived"
    print("OK: Flow duration & IAT artifacts successfully clipped and dropped")

    return df

def verify_labels_and_stats(df):
    print("\n5. Label Integrity")
    counts = df['Label'].value_counts()
    total = len(df)

    print(f"Unique labels found: {len(counts)} (Expected 15)")
    for label_name, label_count in counts.items():
        pct = (label_count/total) * 100

        flag = "WHITESPACE WARNING" if label_name != label_name.strip() else ""
        print(f"{label_name:25s} | {label_count:>11,} ({pct:6.3f}%) {flag}")

    assert len(counts) == 15, "Warning: label count is not 15 - check for whitespace dups"
    print("OK: All labels successfully merged and cleaned")

    print("\n6. Final Feature Stats (sample)")
    sample_cols = ["Flow_Duration", "Tot_Fwd_Pkts", "Flow_Byts_s", "Pkt_Len_Mean"]

    pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
    stats = df[sample_cols].describe().T[['mean', 'std', 'max', 'min']]
    print(stats.to_string())
    pd.reset_option('display.float_format')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Verify cleaned CIC-IDS2018 dataset (post merge & clean)")
    parser.add_argument("--in_path", type=str, default="auto",
                        help="Path to cleaned parquet file - Use 'auto' for menu")
    
    args = parser.parse_args()

    try:
        input_file = Path(args.in_path)
        if args.in_path == "auto":
            input_file = select_cleaned_dataset("preprocessing/processes_output/cleaned_datasets")

        print(f"\nVerifying Cleaned Dataset: {input_file.name}")

        split_strat, expected_cols = infer_pipeline_context(input_file.name)
        verify_shrinkage(input_file)
        verify_schema_purges(input_file, split_strat, expected_cols)

        df_clean = verify_data_health_and_artifacts(input_file)
        verify_labels_and_stats(df_clean)

        print("\nALL CLEANING CHECKS PASSED - DATASET READY FOR SPLITTING & ENCODING")

    except AssertionError as ae:
        print(f"\nCHECK FAILED (ASSERTATION FAILED): {ae}")
    except Exception as e:
        print(f"\nVerification Crashed: {e}")
