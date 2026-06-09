import argparse
import time 
from pathlib import Path 
import pandas as pd 
import pyarrow.parquet as pq 

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

MERGED_PARQUET = Path("dataset/cic-ids2018.parquet")
CLEANED_PARQUET = Path("dataset/cic-ids2018-cleaned.parquet")

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

def apply_drops_and_dedup(in_path, out_path, split_strat, encoding_strat):


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
    print(f"\n1. Loading {len(keep_cols)} cols (dropping {len(BASE_COLUMNS_TO_DROP)} at load)")
    t0 = time.perf_counter()
    df = pd.read_parquet(in_path, columns=keep_cols)
    print(f"Loaded: {len(df):,} rows x {len(df.columns)} cols ({time.perf_counter()-t0:.1f}s)")
    
    print("\nSuccessfully dropped:")
    for col in dropped_cols:
        print(f"- {col}")


    print("\n2. Cleaning column names (replacing whitespace/slashes with _)")
    df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]
    print(f"Cleaned column names: {df.columns[:5]} ... {df.columns[-5:]}")


    # Drop negative Flow Duration
    print("\n3. Dropping rows with negative Flow Duration")
    n_before = len(df)
    df = df[df["Flow_Duration"] >= 0]
    print(f"\nDropped {n_before - len(df):,} rows with Flow Duration < 0")
    print(f"{n_before:,}->{len(df):,}")



    # Drop duplicates

    print(f"\n4. Dropping duplicate rows")
    n_before = len(df)
    t0 = time.perf_counter()
    df = df.drop_duplicates()

    n_dup = n_before - len(df)
    pct = 100*n_dup/n_before
    print(f" Dropped {n_dup:,} duplicates ({pct:.2f}%) ({time.perf_counter()-t0:.1f}s)")
    print(f" {n_before:,} -> {len(df):,}")


    # Fixing IAT negative artificats
    print(f"\n5. Fixing neg IAT time artifacts (clip to 0)")
    t0 = time.perf_counter()
    iat_cols =[c for c in df.columns if 'IAT' in c]
    for col in iat_cols:
        df[col] = df[col].clip(lower=0)
    print(f"\nFixed {len(iat_cols)} IAT columns ({time.perf_counter()-t0:.1f}s)")


    # Write cleaned parquet to output file

    print(f"\n6. Writing cleaned parquet")
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

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and deduplicate merged CIC-IDS2018 parquet file")
    
    parser.add_argument("--in_path", type=str, default = "preprocesssing/merged_datasets/cic-ids2018-marged.parquet",
                        help="Path to raw merged parquet file to clean")
    parser.add_argument("--out_dir", type=str, default="preprocessing/cleaned_and_deduped_dataset",
                        help="Dir to save the cleaned parquet file")

    parser.add_argument("--split-strat", type=str, choices=["8020_stratified", "time_based"], 
                        help="Splitting strategy")
    parser.add_argument("--encoding-strat", type=str, choices=["binary", "multiclass"],
                        help="Encoding strategy")
    
    args = parser.parse_args()

    apply_drops_and_dedup(
        in_path = args.in_path,
        out_dir=args.out_dir,
        split_strat=args.split_strat,
        encoding_Strat=args.encoding_strat
    )
