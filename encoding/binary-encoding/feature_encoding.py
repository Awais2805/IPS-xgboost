import pandas as pd
import time 
from pathlib import Path 

INPUT_PATH = Path("dataset/cic-ids2018-cleaned.parquet")
OUTPUT_PATH = Path("dataset/cic-ids2018-encoded.parquet")

def encode_features(in_path, out_path):
    print(f"\nFeature Encoding")
    print(f"\nLoading cleaned data in from {in_path}")
    t0 = time.perf_counter()
    df = pd.read_parquet(in_path)

    print(f"Loaded {len(df):,} rows in {time.perf_counter()-t0:.1f}s")

    print("\n1. Cleaning Column Names")
    df.columns = df.columns.str.replace(' ', '_').str.replace('/','_')
    print(f"Sample of cleaned columns: {list(df.columns[:5])}")



    print(f"\n2. Encoding Target (Benign -> 0, Attack -> 1)")
    df['Target'] = (df['Label'] != 'Benign').astype(int)
    print("Target Mapping Verification:")
    mapping_check = df.groupby(['Label', 'Target']).size().reset_index(name='Row Count')
    print(mapping_check.to_string(index=False))
    df = df.drop(columns=['Label'])




    print(f"\n3. One-Hot Encoding Protocols")
    df = pd.get_dummies(df, columns=['Protocol'], drop_first=False)
    protocol_cols = [c for c in df.columns if 'Protocol_' in c]
    print(f"New Protocol features generated: {protocol_cols}")
    print("\nSample of encoded protocol matrix (first 5 rows):")
    # Convert booleans to integers just for a cleaner terminal print
    print(df[protocol_cols].head(5).astype(int).to_string())



    print(f"\n4. Saving Encoded Dataset")
    t1 = time.perf_counter()
    df.to_parquet(out_path, compression='snappy', index=False)
    out_mb = out_path.stat().st_size / (1024*1024)
    print(f"Saved to {out_path} ({out_mb:.1f}MB) in {time.perf_counter()-t1:.1f}s")

    print(f"\nFinal Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

if __name__ == "__main__":
    encode_features(INPUT_PATH, OUTPUT_PATH)