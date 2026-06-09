from os import path
from pathlib import Path 
import pandas as pd
import numpy as np 
import time 
import pyarrow as pa 
import pyarrow.parquet as pq

DATASET_DIR = Path("dataset/ids2018_csv")
EXPECTED_FILE_COUNT = 10
EXPECTED_COLUMNS = 80
WIDE_FILE = "02-20-2018.csv"
EXPECTED_COLS_WIDE = 84
EXPECTED_EXTRA_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP"]
CHUNK_SIZE = 500_000
OUTPUT_PARQUET_DIR = Path("dataset/cic-ids2018.parquet")

def list_files():
    csv_files = sorted(DATASET_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {DATASET_DIR}")
    for f in csv_files:
        print(f" - {f.name}")
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"{f.name:25s} {size_mb:>8.1f} MB")
    assert len(csv_files) == EXPECTED_FILE_COUNT, (
        f"Expected {EXPECTED_FILE_COUNT} files, found {len(csv_files)}"
        )
    print(f"OK: {EXPECTED_FILE_COUNT} files present.")
    return csv_files

def read_header(path):
    """Read only the header row using the stdlib — no pandas overhead."""
    with open(path, "r", encoding="utf-8") as f:
        return next(f).rstrip("\r\n").split(",")

def check_columns(csv_files):
    print("\n Column count per file:")
    headers = {}
    for f in csv_files:
        cols = read_header(f)
        headers[f.name] = cols
        print(f"{f.name:25s} {len(cols)} cols")

    for name, cols in headers.items():
        expected = EXPECTED_COLS_WIDE if name == WIDE_FILE else EXPECTED_COLUMNS
        assert len(cols) == expected, (
            f"{name}: expected {expected} columns, got {len(cols)}"
            )
    print(f"OK: 9 files have {EXPECTED_COLUMNS} columns, 1 file has {EXPECTED_COLS_WIDE} columns.")
    return headers

def check_wide_file_extras(headers):
    print(f"\nVerify {WIDE_FILE} extras are at the front")
    wide_cols = headers[WIDE_FILE]
    first_four = wide_cols[:4]
    print(f" First 4 cols: {first_four}")
    assert first_four == EXPECTED_EXTRA_COLS, (
        f"{WIDE_FILE}: expected first 4 cols to be {EXPECTED_EXTRA_COLS}, got {first_four}"
        f"got {first_four}"
    )
    print(f"OK: extras match {EXPECTED_EXTRA_COLS}")

def check_column_alignment(headers):
    print(f"\n Verify 80 normal columns align across all 10 files")
    normalized = {}
    for name, cols in headers.items():
        cleaned = [c.strip() for c in cols]
        if name == WIDE_FILE:
            cleaned = cleaned[4:]
        normalized[name] = cleaned

    reference_name = next(iter(normalized))
    reference = normalized[reference_name]
    print(f"Reference: {reference_name}  {len(reference)} cols")

    all_match = True
    for name, cols in normalized.items():
        if cols != reference:
            all_match = False
            missing = set(reference) - set(cols)
            extra = set(cols) - set(reference)
            print(f" MISMATCH in {name}:")
            print(f" missing: {missing}")
            print(f" extra: {extra}")
    
    assert all_match, "Column names do not align across files - see mismatches above."
    print(f"OK: all 10 files share the same {len(reference)} normalised column names")

def count_rows_streaming(path, chunk_size=CHUNK_SIZE):
    print(f"\nStreaming {path.name} in chunks of {chunk_size:,}:")
    total_rows = 0
    n_chunks = 0 
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        n_chunks += 1
        total_rows += len(chunk)
        print(f" chunk {n_chunks}: {len(chunk):>9,} rows (running total: {total_rows:>10,})")
    print(f"OK: {n_chunks} chunks, {total_rows:,} total rows")
    return total_rows

def normalize_schema(chunk, file_name):
    chunk.columns = [c.strip() for c in chunk.columns]
    if file_name == WIDE_FILE:
        chunk = chunk.drop(columns=EXPECTED_EXTRA_COLS)
    return chunk

def test_normalization(path):
    print(f"\nTest normalization on {path.name}:")
    first_chunk = next(pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False))
    print(f"Before: {len(first_chunk.columns)} cols")
    normalized = normalize_schema(first_chunk, path.name)
    print(f"After: {len(normalized.columns)} cols")
    print(f" First 5 cols after normalization: {list(normalized.columns[:5])}")

def find_drop_masks(chunk):
    return {
        "stray_header": chunk['Dst Port'].astype(str).str.strip() == 'Dst Port',
        "null_label": chunk['Label'].isna() | (chunk['Label'].astype(str).str.strip() ==''),
    }


def inspect_drop(path, max_samples = 3):
    print(f"\nInspect drops on: {path.name}:")
    n_stray =0 
    n_null_label = 0
    stray_samples = []
    unique_labels = set()

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = normalize_schema(chunk, path.name)
        masks = find_drop_masks(chunk)


        # Category 1: stray header rows where 'Dst Port' column contains the string 'Dst Port'
        n_stray += int(masks["stray_header"].sum())
        if len(stray_samples) < max_samples:
            need = max_samples - len(stray_samples)
            stray_samples.extend(chunk.loc[masks["stray_header"]].head(need).to_dict('records'))

        # Category 2: null/empty labels
        n_null_label += int(masks["null_label"].sum())

        # Category 3: track unique labels
        unique_labels.update(chunk['Label'].dropna().astype(str).unique())



    print(f"\n 1. Stray header rows: {n_stray}")
    if stray_samples:
        sample = stray_samples[0]
        print(f" Sample (first 6 fields, repr shows literal values):")
        for col in list(sample)[:6]:
            print(f"{col:25s} = {sample[col]!r}")
        
    print(f" 2. Null/empty label rows: {n_null_label}")

    print(f" 3. Unique labels values (repr shows leading/trailing spaces):")
    for label in sorted(unique_labels):
        print(f" {label!r}")


def clean_rows(chunk):
    before = len(chunk)
    masks = find_drop_masks(chunk)
    keep = ~(masks["stray_header"] | masks["null_label"])
    chunk = chunk.loc[keep].copy()
    chunk['Label'] = chunk['Label'].astype(str).str.strip()
    after = len(chunk)
    return chunk, before - after

def inspect_numeric_coercion(path):
    print(f"\nInspect numeric coercion on {path.name}:")
    nan_per_column = {}
    bad_values = {}

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = normalize_schema(chunk, path.name)
        chunk, _ = clean_rows(chunk)

        for col in chunk.columns:
            if col == 'Label':
                continue
            original = chunk[col]
            coerced = pd.to_numeric(original, errors='coerce')
            new_nan_mask = coerced.isna() & ~original.isna()
            n_new = int(new_nan_mask.sum())
            if n_new == 0:
                continue 
            nan_per_column[col] = nan_per_column.get(col, 0) + n_new
            if col not in bad_values:
                bad_values[col] = set()
            samples_values = original[new_nan_mask].astype(str).unique()[:5]
            bad_values[col].update(samples_values)
    
    if not nan_per_column:
        print("OK: No coercion failures. All non-label columns are numeric")
        return
    
    print(f"Columns with non-numeric values:")
    for col, count in sorted(nan_per_column.items(), key=lambda x: -x[1]):
        samples = list(bad_values.get(col, []))[:3]
        print(f"{col:25s}{count:>10,} e.g. {samples}")

def prep_features(chunk):

    before = len(chunk)

    if 'Timestamp' in chunk.columns:
        chunk = chunk.drop(columns=['Timestamp'])
    
    feature_cols = [c for c in chunk.columns if c != 'Label']

    for col in feature_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

    chunk[feature_cols] = chunk[feature_cols].replace([np.inf, -np.inf], np.nan)
    chunk = chunk.dropna(subset=feature_cols)

    chunk[feature_cols] = chunk[feature_cols].astype(np.float64)

    after = len(chunk)
    return chunk, before - after


def test_prep_features(path):
    print(f"\nTest prep_features on {path.name}:")
    total_before = 0 
    total_garbage = 0 
    total_nan = 0 
    final_cols = None 

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = normalize_schema(chunk, path.name)
        total_before += len(chunk)
        chunk, n_garbage = clean_rows(chunk)
        total_garbage += n_garbage
        chunk, n_nan = prep_features(chunk)
        total_nan += n_nan
        if final_cols is None:
            final_cols = list(chunk.columns)
    

    print(f"  Rows before:                 {total_before:,}")
    print(f"  Rows dropped (garbage):      {total_garbage:,}")
    print(f"  Rows dropped (NaN/inf):      {total_nan:,}")
    print(f"  Rows after:                  {total_before - total_garbage - total_nan:,}")
    print(f"  Final column count:          {len(final_cols)}(was 80 — Timestamp dropped)")
    print(f"  First 3 / last 3 columns:    {final_cols[:3]} ... {final_cols[-3:]}")




def test_clean_rows(path):
    
    print(f"\nTest clean_rows on {path.name}:")
    total_before = 0
    total_dropped = 0
    label_counts = {}
  
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = normalize_schema(chunk, path.name)
        total_before += len(chunk)             
        chunk, dropped = clean_rows(chunk)
        total_dropped += dropped
        for label, count in chunk['Label'].value_counts().items():
            label_counts[label] = label_counts.get(label, 0) + int(count)

        print(f"  Rows before cleaning: {total_before:,}")
        print(f"  Rows dropped:         {total_dropped:,}")
        print(f"  Rows after cleaning:  {total_before - total_dropped:,}")
        print(f"  Label distribution after cleaning:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
          print(f"    {label:35s} {count:>10,}")


def merged_to_parquet(csv_files, output_path=OUTPUT_PARQUET_DIR):
    print(f"\nMerging {len(csv_files)} CSV files and writing to Parquet at {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    grand_before = 0 
    grand_garbage = 0 
    grand_nan = 0 
    grand_start = time.perf_counter()

    try:
        for csv_path in csv_files:
            file_start = time.perf_counter()
            file_before = 0 
            file_garbage = 0 
            file_nan = 0 

            for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory =False):
                file_before += len(chunk)
                chunk = normalize_schema(chunk, csv_path.name)
                chunk, n_g = clean_rows(chunk)
                file_garbage += n_g
                chunk, n_n = prep_features(chunk)
                file_nan += n_n
                if len(chunk) == 0:
                    continue 

                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        table.schema,
                        compression="snappy")
                writer.write_table(table)

            elapsed = time.perf_counter() - file_start
            kept = file_before - file_garbage - file_nan
            print(f"{csv_path.name:25s}" f"in{file_before:>10,} garbage:{file_garbage:>5,}" f"nan:{file_nan:>7,} kept:{kept:>10,}({elapsed:6.1f}s)")

            grand_before += file_before
            grand_garbage += file_garbage
            grand_nan += file_nan
    finally:
        if writer is not None:
            writer.close()
    
    total = time.perf_counter() - grand_start
    grand_kept = grand_before - grand_garbage - grand_nan
    out_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n TOTAL "
          f"in:{grand_before:>10,} garbage:{grand_garbage:>5,} "
          f"nan:{grand_nan:>7,} kept:{grand_kept:>10,}({total:.1f}s) "
    )
    print(f"Output: {output_path} ({out_mb:.1f} MB)")
    


if __name__ == "__main__":
    csv_files = list_files()
    headers = check_columns(csv_files)
    check_wide_file_extras(headers)
    check_column_alignment(headers)
    
    test_file = next(f for f in csv_files if f.name == "02-28-2018.csv")

    count_rows_streaming(test_file)


    wide_path = next(f for f in csv_files if f.name == WIDE_FILE)
    test_normalization(wide_path)

    # inspect_drop(test_file)
    # test_clean_rows(test_file)

    coercion_test_file = next(f for f in csv_files if f.name == "02-14-2018.csv")
    inspect_numeric_coercion(coercion_test_file)
    test_prep_features(coercion_test_file)

    merged_to_parquet(csv_files)