import time 
from pathlib import Path 
import pandas as pd 
import pyarrow.parquet as pq 

COLUMNS_TO_DROP = [
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

def apply_drops_and_dedup(in_path=MERGED_PARQUET, out_path=CLEANED_PARQUET):
    print(f"\nApply drops + dedup")
    print(f"\nInput: {in_path}")
    print(f"\nOutput: {out_path}")

    # Load df with pruned columns
    all_cols = pq.read_schema(in_path).names
    keep_cols = [c for c in all_cols if c not in COLUMNS_TO_DROP]
    print(f"\n1. Loading {len(keep_cols)} cols (dropping {len(COLUMNS_TO_DROP)} at load)")
    t0 = time.perf_counter()
    df = pd.read_parquet(in_path, columns=keep_cols)
    print(f"Loaded: {len(df):,} rows x {len(df.columns)} cols ({time.perf_counter()-t0:.1f}s)")

    # Drop negative Flow Duration
    n_before = len(df)
    df = df[df["Flow Duration"] >= 0]
    print(f"\n2. Dropped {n_before - len(df):,} rows with Flow Duration < 0")
    print(f"{n_before:,}->{len(df):,}")

    # Drop duplicates

    print(f"\n3. Dropping duplicate rows")
    n_before = len(df)
    t0 = time.perf_counter()
    df = df.drop_duplicates()

    n_dup = n_before - len(df)
    pct = 100*n_dup/n_before
    print(f" Dropped {n_dup:,} duplicates ({pct:.2f}%) ({time.perf_counter()-t0:.1f}s)")
    print(f" {n_before:,} -> {len(df):,}")


    # Fixing IAT negative artificats
    print(f"\n4. Fixing neg IAT time artifacts (clip to 0)")
    t0 = time.perf_counter()
    iat_cols =[c for c in df.columns if 'IAT' in c]
    for col in iat_cols:
        df[col] = df[col].clip(lower=0)
    print(f"\nFixed {len(iat_cols)} IAT columns ({time.perf_counter()-t0:.1f}s)")


    # Write cleaned parquet to output file

    print(f"\n5. Writing cleaned parquet")
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
    apply_drops_and_dedup(MERGED_PARQUET, CLEANED_PARQUET)