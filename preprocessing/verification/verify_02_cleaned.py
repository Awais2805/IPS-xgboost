import pandas as pd 
import numpy as np 

CLEANED_PARQUET = "dataset/cic-ids2018-cleaned.parquet"

def verify_final_dataset(path):
    print(f"Loading final cleaned dataset from: {path}")
    df = pd.read_parquet(path)

    print("\n1. Dataset dimensions")
    print(f"\nRows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n2. Label distribution (exact counts & percentages)")
    counts = df['Label'].value_counts()
    total = len(df)
    for label,count in counts.items():
        pct = (count/total) * 100 
        print(f"{label:30s} {count:>12,} ({pct:6.3f})%")

    
    print("\n Final Checks (null/infs)")

    null_counts = df.isna().sum().sum()
    print(f"Total NaN values: {null_counts}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum().sum()
    print(f"Total Inf values: {inf_counts}")


    assert null_counts == 0, "Warning - NaN in dataset"
    assert inf_counts == 0, "Warning - Inf values in dataset"

    print("\n Feature Stats (mean,std,min,max)")

    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', lambda x: f'{x:,.4f}')

    stats = df[numeric_cols].describe().T[['mean', 'std', 'min', 'max']]
    print(stats.to_string())

    print("\n5. Column name overview")
    cols = df.columns.tolist()
    print(f"First 5 cols: {cols[:5]}")
    print(f"\nLast 5 cols: {cols[-5:]}")

    print(f"\nVerification Complete - data ready for encoding and splitting")


if __name__ == "__main__":
    verify_final_dataset(CLEANED_PARQUET)