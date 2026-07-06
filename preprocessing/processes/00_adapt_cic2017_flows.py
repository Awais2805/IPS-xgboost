#!/usr/bin/env python3
"""
Adapt a merged CIC-IDS2017 parquet -> CIC-IDS2018 merged-parquet schema.

CIC-2017 uses long-form column names (e.g. 'Total Fwd Packets'). The full
'TrafficLabelling'/'GeneratedLabelledFlows' variant also carries Protocol +
identifier columns; the 'MachineLearningCVE' variant strips Protocol/Timestamp.
This renames the features to the CIC-2018 abbreviated convention, drops the
duplicate 'Fwd Header Length.1' and the identifier columns, and passes Protocol/
Timestamp through when present -- producing a CIC-2018-schema merged parquet that
02_clean_and_dedup -> encode -> split -> train consume unchanged.

Run from the IPS/ project root. Next stage (use the printed --in_path explicitly):
  python3 preprocessing/processes/02_clean_and_dedup.py \
      --in_path <this output> --split_strat 8020_stratified --encoding_strat binary
"""

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

# CIC-IDS2017 (long form) -> CIC-IDS2018 (abbreviated). Verified column-by-column
# against the CIC-2018 merged schema. Names identical in both are omitted (left
# as-is): Flow Duration, Flow/Fwd/Bwd IAT Mean/Std/Max/Min, Fwd/Bwd PSH/URG Flags,
# CWE Flag Count, Down/Up Ratio, Active/Idle Mean/Std/Max/Min, Protocol,
# Timestamp, Label.
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

# Dropped: duplicate artefact + CIC-2017 identifier columns (full variant only).
DROP_COLS = ["Fwd Header Length.1", "Flow ID", "Source IP", "Source Port",
             "Destination IP"]


def resolve_in_path(in_path, merged_dir):
    if in_path != "auto":
        return Path(in_path)
    files = sorted(Path(merged_dir).glob("cic-ids2017-merged-*.parquet"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    files = [f for f in files if "adapted" not in f.name]  # don't re-pick our own output
    if not files:
        raise SystemExit(f"ABORT: no cic-ids2017-merged-*.parquet in {merged_dir} (pass --in_path)")
    return files[0]


def main():
    ap = argparse.ArgumentParser(description="Adapt merged CIC-IDS2017 -> CIC-IDS2018 schema")
    ap.add_argument("--in_path", default="auto",
                    help="merged CIC-2017 parquet, or 'auto' for newest cic-ids2017-merged-*")
    ap.add_argument("--merged_dir", default="preprocessing/processes_output/merged_datasets")
    ap.add_argument("--out_dir", default="preprocessing/processes_output/merged_datasets")
    ap.add_argument("--dataset_name", default="cic-ids2017-adapted",
                    help="artefact name token (drives the -merged- filename + downstream base)")
    ap.add_argument("--protocol_included", choices=["y", "n"], default="n",
                    help="does the source have a Protocol column? y = TrafficLabelling variant "
                         "(pass it through); n = MachineLearningCVE variant (impute Protocol=6, "
                         "the TCP mode, so the 63-feature CIC-2018 schema matches).")
    args = ap.parse_args()

    in_path = resolve_in_path(args.in_path, args.merged_dir)
    df = pd.read_parquet(in_path)
    print(f"read {len(df):,} rows x {len(df.columns)} cols from {in_path}")

    junk = [c for c in df.columns if c.startswith("__")]  # pyarrow scan artefacts
    present_drop = [c for c in (DROP_COLS + junk) if c in df.columns]
    df = df.drop(columns=present_drop)
    if present_drop:
        print(f"dropped: {present_drop}")

    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})

    if "Label" not in df.columns:
        raise SystemExit("ABORT: required 'Label' column missing after adapt")

    if args.protocol_included == "y":
        if "Protocol" not in df.columns:
            raise SystemExit("ABORT: --protocol_included y but no 'Protocol' column found "
                             "(you likely have the MachineLearningCVE variant; re-run with "
                             "--protocol_included n to impute it).")
        print("Protocol: present, passed through")
    else:  # n
        if "Protocol" not in df.columns:
            df["Protocol"] = "6"
            print("Protocol: absent -> IMPUTED = 6 (TCP mode) for all rows. NOTE: this treats "
                  "CIC-2017's genuinely-UDP flows as TCP; documented placeholder so the "
                  "63-feature schema matches CIC-2018.")
        else:
            print("Protocol: present, kept (despite --protocol_included n)")

    # CIC-2017 labels benign as "BENIGN"; the binary encoder tests == "Benign", so
    # normalise it (else encode_target_binary flips every benign flow to attack=1).
    df["Label"] = df["Label"].apply(lambda v: "Benign" if str(v).strip().upper() == "BENIGN" else v)

    df = df.astype(str)  # mirror the merge stage; 02 coerces types

    gen_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset_name}-merged-{gen_ts}.parquet"
    df.to_parquet(out_path, compression="snappy", index=False)

    print(f"wrote {len(df):,} rows x {len(df.columns)} cols -> {out_path}")
    print("columns:", list(df.columns))


if __name__ == "__main__":
    main()
