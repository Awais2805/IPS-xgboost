from pathlib import Path 
import numpy as np
import pandas as pd
import pyarrow.parquet as pq 
import time 

MERGED_PARQUET = Path("dataset/cic-ids2018.parquet")
EXPECTED_COLUMNS = 79 # after dropping timestamp col in preprocessing.py
EXPECTED_TOTAL_ROWS_APPROX = 16_137_183
EXPECTED_UNIQUE_LABELS = 15 # benign + 14 attack class types 
NEAR_CONSTANT_CANDIDATES = ["Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", 
                            "CWE Flag Count", "Fwd Byts/b Avg", "Fwd Pkts/b Avg",
                            "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", 
                            "Bwd Blk Rate Avg",]

REDUNDANT_PAIRS = [("Subflow Fwd Pkts", "Tot Fwd Pkts"),
                    ("Subflow Fwd Byts", "TotLen Fwd Pkts"),
                    ("Subflow Bwd Pkts", "Tot Bwd Pkts"),
                    ("Subflow Bwd Byts", "TotLen Bwd Pkts"),
                    ("Pkt Size Avg", "Pkt Len Mean"),
                    ("Fwd Seg Size Avg", "Fwd Pkt Len Mean"),
                    ("Bwd Seg Size Avg", "Bwd Pkt Len Mean"),]

LEAKAGE_DROPS = ["Dst Port"] 


def verify_schema(path):
    print(f"\n1. Schema Check")
    schema = pq.read_schema(path)
    cols = schema.names
    types = {f.name: str(f.type) for f in schema}

    print(f" Total columns: {len(cols)} (expected {EXPECTED_COLUMNS})")
    assert len(cols) == EXPECTED_COLUMNS, f"col count mismatch: got {len(cols)}"
    assert "Label" in cols, "Label column missing"
    assert "Timestamp" not in cols, "Timestamp should have been dropped after preprocessing.py"

    feature_cols = [c for c in cols if c != "Label"]
    non_double = [c for c in feature_cols if types[c] != "double"]

    assert not non_double, f"Non-double feature colunms: {non_double}"
    label_type = types["Label"]
    assert label_type in ("string", "large-string"), f"Label dtype: {label_type}"
    print(f"OK: 79 float64 features & Label ({label_type})")
    return feature_cols

def verify_row_count(path):

    print(f"\n2. Row count")
    metadata = pq.read_metadata(path)
    n_rows = metadata.num_rows
    n_row_groups = metadata.num_row_groups 
    print(f" Rows: {n_rows:,} (expected) ~{EXPECTED_TOTAL_ROWS_APPROX:,})")

    print(f" Row groups: {n_row_groups} (one per chunk written)")
    
    diff = abs(n_rows - EXPECTED_TOTAL_ROWS_APPROX)
    assert diff < 50_000, f"Row count off by {diff:,} from expected"
    print(f"OK: Row count within tolerance")
    return n_rows 


def verify_labels(path):
    print("\n3. Label distribution")
    labels = pd.read_parquet(path, columns=["Label"])["Label"]
    counts = labels.value_counts()
    total = int(counts.sum())
    print(f" Unique labels: {len(counts)} (expected {EXPECTED_UNIQUE_LABELS})")

    for label, count in counts.items():
        pct = 100 * count / total
        print(f"{label:35s} {count:>11,} ({pct:6.3f}%)")

    assert "Label" not in counts.index, "Stray 'Label' string in data - stray header rows we not dropped"
    whitespace_vars = [l for l in counts.index if l != l.strip()]
    assert not whitespace_vars, f"Whitespace-variant labels: {whitespace_vars}"
    print(f" OK: Clean label set, no stray 'labels' or whitespace variants")

    return counts

def verify_feature_stats(path, sample_cols=("Flow Duration", "Tot Fwd Pkts", "Flow Byts/s", "Pkt Len Mean", "Down/Up Ratio")):
    print("\n4. Sample feautres stats (check for residual NaN, sensible ranges)")

    df = pd.read_parquet(path, columns=list(sample_cols))
    stats = df.describe().T[["min", "max", "mean", "std"]]
    print(stats.to_string())
    for col in sample_cols:
        s = df[col]
        assert not s.isna().any(), f"{col}: has NaN (proprocessing.py failed to drop all)"
        assert not np.isinf(s).any(), f"{col}: has inf entry (preprocessing failed to drop all)"
    print(f"OK: No NaN/Inf in sampled columns, ranges look decent")

def inspect_negative_duration(path):
    print("\n5. Inspect rows with Flow Duration < 0")
    df = pd.read_parquet(path, columns=["Flow Duration", "Label"])
    mask = df["Flow Duration"] < 0 
    n_neg = int(mask.sum())
    total = len(df)
    print(f" Rows with Flow Duration < 0: {n_neg:,} of {total:,} ({100*n_neg/total:.4f}%)")

    if n_neg == 0:
        print(" (Nothing to drop)")
        return n_neg
    
    neg = df.loc[mask, "Flow Duration"]
    print(f"Value range: min={neg.min():,.0f} max={neg.max():,.0f}" 
          f"median={neg.median():,.0f}")
    
    print(f"Distribution by Label")
    by_label = df.loc[mask, "Label"].value_counts()
    overall = df["Label"].value_counts()
    for label, count in by_label.items():
        pct_within_label = 100 * count / overall[label]
        print(f"{label:35s} {count:>9,} ({pct_within_label:6.2f}% of that class)")
    return n_neg


def verify_droppable_columns(path):
    print("\n6. Verify a-priori droppable features")

    cols_needed = (
        NEAR_CONSTANT_CANDIDATES + [c for pair in REDUNDANT_PAIRS for c in pair]
    )
    df = pd.read_parquet(path, columns=cols_needed)
    n = len(df)

    print(f"\n6a. Near-constant features ({len(NEAR_CONSTANT_CANDIDATES)} candidates)")
    print(f" Threshold: unique <= 3 and %zero > 99.0 -> DROP")

    for col in NEAR_CONSTANT_CANDIDATES:
        unique = int(df[col].nunique())
        pct_zero = 100.0 * (df[col] == 0).sum() / n
        std = float(df[col].std())
        verdict = "DROP" if (unique <= 3 and pct_zero > 99.0) else "KEEP (not as constant as expected)"
        print(f" {col:22s} unique={unique:>3} %zero={pct_zero:>6.2f} std={std:>10.4f} {verdict}")

    print(f"\n6b. Redundant Pairs ({len(REDUNDANT_PAIRS)} candidates)")
    print(f" Threshold: corr>0.9999 AND mismatched_rows<1000 -> DROP the left member")
    for a,b in REDUNDANT_PAIRS:
        corr = float(df[a].corr(df[b]))
        mismatched = int((df[a]!= df[b]).sum())
        verdict = "DROP " + a if (corr>0.9999 and mismatched<1000) else f"KEEP (corr={corr:.4f})"
        print(f" {a:22s} vs {b:22s} corr={corr:.6f} mismatched={mismatched:>9,} {verdict}")

    print(f"\n6c. Leakage drop (no empirical test, methodological choice)")
    for col in LEAKAGE_DROPS:
        print(f"{col:22s} DROP (port -> class memorisation risk)")




if __name__ == "__main__":
    print(f"Verifying merged Parquet at: {MERGED_PARQUET}")
    feature_cols = verify_schema(MERGED_PARQUET)
    verify_row_count(MERGED_PARQUET)
    counts = verify_labels(MERGED_PARQUET)
    verify_feature_stats(MERGED_PARQUET)
    inspect_negative_duration(MERGED_PARQUET)
    verify_droppable_columns(MERGED_PARQUET)
    print("\nAll post-merge checks passed\n")