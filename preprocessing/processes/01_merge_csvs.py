from os import path
from pathlib import Path 
import pandas as pd
import numpy as np 
import time 
import pyarrow as pa 
import pyarrow.parquet as pq
import argparse
from datetime import datetime
from collections import Counter

CHUNK_SIZE = 500_000

def list_files(dataset_dir):
    csv_files = sorted(dataset_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {dataset_dir}")
    for f in csv_files:
        print(f" - {f.name}")
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"{f.name:25s} {size_mb:>8.1f} MB")
    print(f"OK: {len(csv_files)} files present and available.")
    return csv_files

def raw_data_metadata(csv_files):
    print("\nRaw CSV file metadata")
    headers = {}
    for f in csv_files:
        df_empty = pd.read_csv(f, nrows=0)
        
        # Extract and clean the columns
        cols = [c.strip() for c in df_empty.columns]
        headers[f.name] = cols
        
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f" - {f.name:25s} | {size_mb:>7.1f} MB | {len(cols)} columns")

    return headers

def preview_data(csv_files):
    print("\nPreviewing first 5 rows of each CSV file:")
    for f in csv_files:
        print(f"\nFile: {f.name}")
        try:
            preview_df = pd.read_csv(f, nrows=5)
            print(preview_df.to_string(max_cols=8))
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

def resolve_schema_mismatch(headers, mismatch_action):

    col_counts = [len(cols) for cols in headers.values()]
    base_count = Counter(col_counts).most_common(1)[0][0]

    base_file = next(name for name, cols in headers.items() if len(cols) == base_count)
    base_cols = headers[base_file]

    extra_columns_map = {}

    print("\nResolving column alignment")
    for name, cols in headers.items():
        if len(cols) != base_count:
            extras = set(cols) - set(base_cols)
            extra_columns_map[name] = list(extras)
            print(f"\nMISMATCH: {name} has {len(cols)} columns")
            print(f"\nExtra columns found: {list(extras)}")

    keep_extras = False
    if extra_columns_map:
        if mismatch_action == "drop":
            print("\nAction: dropping extra columns")
            keep_extras = False
        elif mismatch_action == "keep":
            print("\nAction: keeping extra columns (will be dropped later in cleaning)")
            keep_extras = True
        else:
            raise ValueError(f"Invalid mismatch_action: {mismatch_action}")
    else:
        print("\nOK: no schema mismatches detected.")

    return base_cols, extra_columns_map, keep_extras

def merge_to_parquet(csv_files, out_path, extra_columns_map, keep_extras):
    print("\nStarting file merge into parquet")
    print(f"Output path: {out_path}")

    writer = None 
    grand_total_rows = 0
    grand_start_time = time.perf_counter()
    final_schema_cols = []

    try:
        for csv_path in csv_files:
            file_start_time = time.perf_counter()
            file_rows = 0

            for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):
                chunk.columns =[c.strip() for c in chunk.columns]

                if not keep_extras and csv_path.name in extra_columns_map:
                    chunk = chunk.drop(columns=extra_columns_map[csv_path.name], errors='ignore')

                chunk = chunk.astype(str)

                file_rows += len(chunk)
                grand_total_rows += len(chunk)

                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
                    final_schema_cols = table.schema.names
                writer.write_table(table)

            elapsed = time.perf_counter() - file_start_time
            print(f"Merged {csv_path.name:25s} | {file_rows:>10,} rows ({elapsed:6.1f}s)")
        
    finally:
        if writer is not None:
            writer.close()

    total_time = time.perf_counter() - grand_start_time
    return grand_total_rows, final_schema_cols, total_time

def convert_csvs_to_parquets(csv_files, out_dir):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nConverting CSVs to per-day parquets (columns kept as-is)")
    print(f"Output dir: {out_dir}/")

    grand_total_rows = 0
    grand_start_time = time.perf_counter()

    for csv_path in csv_files:
        out_path = out_dir / f"{csv_path.stem}.parquet"
        writer = None
        file_rows = 0
        file_start_time = time.perf_counter()
        try:
            for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):
                chunk.columns = [c.strip() for c in chunk.columns]
                chunk = chunk.astype(str)

                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
                writer.write_table(table)
                file_rows += len(chunk)
        finally:
            if writer is not None:
                writer.close()

        grand_total_rows += file_rows
        elapsed = time.perf_counter() - file_start_time
        print(f"Converted {csv_path.name:25s} | {file_rows:>10,} rows -> {out_path.name} ({elapsed:6.1f}s)")

    total_time = time.perf_counter() - grand_start_time
    print(f"\n=== PER-DAY CONVERT COMPLETE ===")
    print(f"Files:       {len(csv_files)}")
    print(f"Output dir:  {out_dir}/")
    print(f"Total Rows:  {grand_total_rows:,}")
    print(f"Time Taken:  {total_time:.1f} seconds")
    return grand_total_rows



def analyse_merged_parquet(parquet_path, final_cols, total_rows, total_time):

    print("\nAnalysing merged parquet file")

    out_mb = out_file_path.stat().st_size / (1024 * 1024)
    print(f"\n=== MERGE COMPLETE ===")
    print(f"Target Name:    {args.dataset_name}")
    print(f"File Saved To:  {out_file_path}")
    print(f"Final Size:     {out_mb:,.1f} MB")
    print(f"Total Rows:     {total_rows:,}")
    print(f"Total Columns:  {len(final_cols)}")
    print(f"Time Taken:     {total_time:.1f} seconds")
    
    try:
        df = pd.read_parquet(parquet_path)
        
        print("\n--- First 5 Rows of Merged Dataset ---")
        print(df.head(5).to_string(max_cols=10))
        
        print("\n--- Summary Statistics ---")
        print(df.describe(include='all').to_string())
        
    except MemoryError:
        print("\nERROR: The merged dataset is too large to fit into RAM for pandas statistical analysis.")
    except Exception as e:
        print(f"\nERROR analyzing Parquet file: {e}")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge CSVs into one Parquet, or convert each to its own Parquet (per_day).")

    parser.add_argument("--dataset_name", type=str, default="cic-ids2018", help="Name of the dataset to process")
    parser.add_argument("--input_dir", type=str, default="dataset/ids2018_csv", help="Dir containing raw CSV files")
    parser.add_argument("--mismatch_action", type=str, choices=['drop', 'keep'], default='drop',
                        help="Action to solve extra column mismatches on file merges")
    parser.add_argument("--mode", type=str, choices=['merge', 'per_day'], default='merge',
                        help="merge: pool all CSVs into one parquet (8020 path). "
                             "per_day: one parquet per CSV, no merge (time_based path).")
    parser.add_argument("--per_day_dir", type=str, default="preprocessing/processes_output/merged_per_day",
                        help="Output dir for per_day mode.")

    args = parser.parse_args()

    input_dir_path = Path(args.input_dir)

    print(f"\n=== Dataset {args.mode.upper()} | {args.dataset_name} ===")

    csv_files = list_files(input_dir_path)
    headers = raw_data_metadata(csv_files)
    preview_data(csv_files)

    base_cols, extra_columns_map, keep_extras = resolve_schema_mismatch(headers, args.mismatch_action)

    if args.mode == "per_day":
        convert_csvs_to_parquets(csv_files, Path(args.per_day_dir), base_cols)
        print("END OF PROCESS")
    else:
        output_dir = Path("preprocessing/processes_output/merged_datasets")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file_path = output_dir / f"{args.dataset_name}-merged-{timestamp}.parquet"

        total_rows, final_cols, total_time = merge_to_parquet(csv_files, out_file_path, extra_columns_map, keep_extras)

        out_mb = out_file_path.stat().st_size / (1024 * 1024)
        print(f"\n=== MERGE COMPLETE ===")
        print(f"Target Name:    {args.dataset_name}")
        print(f"File Saved To:  {out_file_path}")
        print(f"Final Size:     {out_mb:,.1f} MB")
        print(f"Total Rows:     {total_rows:,}")
        print(f"Total Columns:  {len(final_cols)}")
        print(f"Time Taken:     {total_time:.1f} seconds")

        analyse_merged_parquet(out_file_path, final_cols, total_rows, total_time)
        print("END OF PROCESS")