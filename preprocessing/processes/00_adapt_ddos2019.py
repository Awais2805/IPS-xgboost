#!/usr/bin/env python3
"""
Adapt CIC-DDoS2019 CSV(s) -> CIC-IDS2018 merged-parquet schema.

DDoS2019 uses the long-form CICFlowMeter names (like CIC-2017) and DOES carry a
real Protocol column (no imputation needed). Reads the raw CSV(s) in chunks
(they're multi-GB), renames features to the CIC-2018 convention, drops the
identifier/junk columns (Unnamed: 0, Flow ID, IPs/ports, Timestamp, the
Fwd Header Length.1 duplicate, and DDoS2019's SimillarHTTP + Inbound), normalises
the benign label (BENIGN -> Benign), and writes one merged-format parquet that
02_clean_and_dedup -> encode consume unchanged. Skips 01_merge (reads CSV direct)
to avoid a large intermediate.

NOTE: DDoS2019 attack files are almost entirely one attack class (little benign),
so the result is an attack-heavy TEST set -> read detection RECALL as the primary
metric when you cross_eval the combined model on it.

Run from IPS/. Example (TCP flood + a UDP reflection type, subsampled):
  python3 preprocessing/processes/adapt_ddos2019.py \
      ~/Downloads/03-11/Syn.csv ~/Downloads/03-11/LDAP.csv --sample_frac 0.3
Then: 02 clean --in_path <out> --split_strat 8020_stratified --encoding_strat binary
      encode --in_path auto ; cross_eval --model <combined-run> --data <ddos encoded>
"""

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK = 500_000

# long-form -> CIC-2018 abbreviated (identical to the CIC-2017 map). Names already
# matching are omitted (Protocol, Flow Duration, Flow/Fwd/Bwd IAT Mean/Std/Max/Min,
# Fwd/Bwd PSH/URG Flags, CWE Flag Count, Down/Up Ratio, Active/Idle *, Label).
RENAME = {
    "Destination Port": "Dst Port",
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts",
    "Total Length of Bwd Packets": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow Bytes/s": "Flow Byts/s",
    "Flow Packets/s": "Flow Pkts/s",
    "Fwd IAT Total": "Fwd IAT Tot",
    "Bwd IAT Total": "Bwd IAT Tot",
    "Fwd Header Length": "Fwd Header Len",
    "Bwd Header Length": "Bwd Header Len",
    "Fwd Packets/s": "Fwd Pkts/s",
    "Bwd Packets/s": "Bwd Pkts/s",
    "Min Packet Length": "Pkt Len Min",
    "Max Packet Length": "Pkt Len Max",
    "Packet Length Mean": "Pkt Len Mean",
    "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var",
    "FIN Flag Count": "FIN Flag Cnt",
    "SYN Flag Count": "SYN Flag Cnt",
    "RST Flag Count": "RST Flag Cnt",
    "PSH Flag Count": "PSH Flag Cnt",
    "ACK Flag Count": "ACK Flag Cnt",
    "URG Flag Count": "URG Flag Cnt",
    "ECE Flag Count": "ECE Flag Cnt",
    "Average Packet Size": "Pkt Size Avg",
    "Avg Fwd Segment Size": "Fwd Seg Size Avg",
    "Avg Bwd Segment Size": "Bwd Seg Size Avg",
    "Fwd Avg Bytes/Bulk": "Fwd Byts/b Avg",
    "Fwd Avg Packets/Bulk": "Fwd Pkts/b Avg",
    "Fwd Avg Bulk Rate": "Fwd Blk Rate Avg",
    "Bwd Avg Bytes/Bulk": "Bwd Byts/b Avg",
    "Bwd Avg Packets/Bulk": "Bwd Pkts/b Avg",
    "Bwd Avg Bulk Rate": "Bwd Blk Rate Avg",
    "Subflow Fwd Packets": "Subflow Fwd Pkts",
    "Subflow Fwd Bytes": "Subflow Fwd Byts",
    "Subflow Bwd Packets": "Subflow Bwd Pkts",
    "Subflow Bwd Bytes": "Subflow Bwd Byts",
    "Init_Win_bytes_forward": "Init Fwd Win Byts",
    "Init_Win_bytes_backward": "Init Bwd Win Byts",
    "act_data_pkt_fwd": "Fwd Act Data Pkts",
    "min_seg_size_forward": "Fwd Seg Size Min",
}

DROP_COLS = ["Unnamed: 0", "Flow ID", "Source IP", "Source Port",
             "Destination IP", "Timestamp", "Fwd Header Length.1",
             "SimillarHTTP", "Inbound"]


def main():
    ap = argparse.ArgumentParser(description="Adapt CIC-DDoS2019 CSV(s) to CIC-IDS2018 schema")
    ap.add_argument("csvs", nargs="+", help="DDoS2019 CSV file(s)")
    ap.add_argument("--out_dir", default="preprocessing/processes_output/merged_datasets")
    ap.add_argument("--dataset_name", default="ddos2019")
    ap.add_argument("--sample_frac", type=float, default=1.0,
                    help="row fraction to keep per chunk (e.g. 0.3) to shrink the set / save disk")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset_name}-merged-{ts}.parquet"

    writer, total, ref_cols, lab = None, 0, None, {}
    try:
        for csv in args.csvs:
            print(f"[adapt] reading {csv}")
            for chunk in pd.read_csv(csv, chunksize=CHUNK, low_memory=False, encoding="latin-1"):
                chunk.columns = [c.strip() for c in chunk.columns]
                chunk = chunk.drop(columns=[c for c in DROP_COLS if c in chunk.columns], errors="ignore")
                chunk = chunk.rename(columns={k: v for k, v in RENAME.items() if k in chunk.columns})
                if "Label" not in chunk.columns:
                    raise SystemExit(f"ABORT: no Label column in {csv}")
                if ref_cols is None:
                    ref_cols = list(chunk.columns)
                else:
                    chunk = chunk.reindex(columns=ref_cols)
                chunk["Label"] = chunk["Label"].apply(
                    lambda v: "Benign" if str(v).strip().upper() == "BENIGN" else str(v).strip())
                if args.sample_frac < 1.0:
                    chunk = chunk.sample(frac=args.sample_frac, random_state=42)
                chunk = chunk.astype(str)
                for v, n in chunk["Label"].value_counts().items():
                    lab[v] = lab.get(v, 0) + int(n)
                total += len(chunk)
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    print(f"[adapt] wrote {total:,} rows x {len(ref_cols)} cols -> {out_path}")
    print(f"[adapt] label balance: {lab}")
    benign = lab.get("Benign", 0)
    print(f"[adapt] benign {benign:,} ({100*benign/max(total,1):.2f}%) — attack-heavy: "
          f"read detection RECALL as the primary metric on cross_eval")


if __name__ == "__main__":
    main()
