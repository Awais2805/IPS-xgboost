import pandas as pd
import numpy as np

# Change this to your file path
FILE_PATH = "preprocessing/processes_output/merged_datasets/cic-ids2018-merged-20260610_230848.parquet"

def inspect_columns(path):
    print(f"Loading data from {path}...")
    # Read only a small sample to save RAM
    df = pd.read_parquet(path)
    
    print(f"\n{'Column Name':<30} | {'Type':<10} | {'Unique Values Sample'}")
    print("-" * 80)
    
    for col in df.columns:
        # Get first 3 non-null values as a string
        sample = str(df[col].dropna().unique()[:3].tolist())
        dtype = str(df[col].dtype)
        print(f"{col:<30} | {dtype:<10} | {sample}")

if __name__ == "__main__":
    inspect_columns(FILE_PATH)