#!/usr/bin/env python3
"""
Adapt netemDocker cicflowmeter flows -> CIC-IDS2018 merged-parquet schema.

Auto-selects the newest flows_labeled_*.csv (or --in_path), renames columns to
the CIC-IDS2018 convention, drops netem-only identifier + Family columns, and
writes an all-string 'merged' parquet that 02_clean_and_dedup.py consumes
directly (glob: *-merged-*.parquet). Output is named per run so runs don't
overwrite each other.

Run from the IPS/ project root. Next stage:
  python3 preprocessing/processes/02_clean_and_dedup.py \
      --in_path preprocessing/processes_output/merged_datasets/<this output> \
      --split_strat 8020_stratified --encoding_strat binary
"""

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

# netem cicflowmeter -> CIC-IDS2018 merged column names (79 renames).
# cwr_flag_count -> "CWE Flag Count" (CIC's known typo for the CWR flag; it is
# dropped in cleaning regardless).
RENAME = {
    'dst_port': 'Dst Port',
    'protocol': 'Protocol',
    'timestamp': 'Timestamp',
    'flow_duration': 'Flow Duration',
    'flow_byts_s': 'Flow Byts/s',
    'flow_pkts_s': 'Flow Pkts/s',
    'fwd_pkts_s': 'Fwd Pkts/s',
    'bwd_pkts_s': 'Bwd Pkts/s',
    'tot_fwd_pkts': 'Tot Fwd Pkts',
    'tot_bwd_pkts': 'Tot Bwd Pkts',
    'totlen_fwd_pkts': 'TotLen Fwd Pkts',
    'totlen_bwd_pkts': 'TotLen Bwd Pkts',
    'fwd_pkt_len_max': 'Fwd Pkt Len Max',
    'fwd_pkt_len_min': 'Fwd Pkt Len Min',
    'fwd_pkt_len_mean': 'Fwd Pkt Len Mean',
    'fwd_pkt_len_std': 'Fwd Pkt Len Std',
    'bwd_pkt_len_max': 'Bwd Pkt Len Max',
    'bwd_pkt_len_min': 'Bwd Pkt Len Min',
    'bwd_pkt_len_mean': 'Bwd Pkt Len Mean',
    'bwd_pkt_len_std': 'Bwd Pkt Len Std',
    'pkt_len_max': 'Pkt Len Max',
    'pkt_len_min': 'Pkt Len Min',
    'pkt_len_mean': 'Pkt Len Mean',
    'pkt_len_std': 'Pkt Len Std',
    'pkt_len_var': 'Pkt Len Var',
    'fwd_header_len': 'Fwd Header Len',
    'bwd_header_len': 'Bwd Header Len',
    'fwd_seg_size_min': 'Fwd Seg Size Min',
    'fwd_act_data_pkts': 'Fwd Act Data Pkts',
    'flow_iat_mean': 'Flow IAT Mean',
    'flow_iat_max': 'Flow IAT Max',
    'flow_iat_min': 'Flow IAT Min',
    'flow_iat_std': 'Flow IAT Std',
    'fwd_iat_tot': 'Fwd IAT Tot',
    'fwd_iat_max': 'Fwd IAT Max',
    'fwd_iat_min': 'Fwd IAT Min',
    'fwd_iat_mean': 'Fwd IAT Mean',
    'fwd_iat_std': 'Fwd IAT Std',
    'bwd_iat_tot': 'Bwd IAT Tot',
    'bwd_iat_max': 'Bwd IAT Max',
    'bwd_iat_min': 'Bwd IAT Min',
    'bwd_iat_mean': 'Bwd IAT Mean',
    'bwd_iat_std': 'Bwd IAT Std',
    'fwd_psh_flags': 'Fwd PSH Flags',
    'bwd_psh_flags': 'Bwd PSH Flags',
    'fwd_urg_flags': 'Fwd URG Flags',
    'bwd_urg_flags': 'Bwd URG Flags',
    'fin_flag_cnt': 'FIN Flag Cnt',
    'syn_flag_cnt': 'SYN Flag Cnt',
    'rst_flag_cnt': 'RST Flag Cnt',
    'psh_flag_cnt': 'PSH Flag Cnt',
    'ack_flag_cnt': 'ACK Flag Cnt',
    'urg_flag_cnt': 'URG Flag Cnt',
    'ece_flag_cnt': 'ECE Flag Cnt',
    'down_up_ratio': 'Down/Up Ratio',
    'pkt_size_avg': 'Pkt Size Avg',
    'init_fwd_win_byts': 'Init Fwd Win Byts',
    'init_bwd_win_byts': 'Init Bwd Win Byts',
    'active_max': 'Active Max',
    'active_min': 'Active Min',
    'active_mean': 'Active Mean',
    'active_std': 'Active Std',
    'idle_max': 'Idle Max',
    'idle_min': 'Idle Min',
    'idle_mean': 'Idle Mean',
    'idle_std': 'Idle Std',
    'fwd_byts_b_avg': 'Fwd Byts/b Avg',
    'fwd_pkts_b_avg': 'Fwd Pkts/b Avg',
    'bwd_byts_b_avg': 'Bwd Byts/b Avg',
    'bwd_pkts_b_avg': 'Bwd Pkts/b Avg',
    'fwd_blk_rate_avg': 'Fwd Blk Rate Avg',
    'bwd_blk_rate_avg': 'Bwd Blk Rate Avg',
    'fwd_seg_size_avg': 'Fwd Seg Size Avg',
    'bwd_seg_size_avg': 'Bwd Seg Size Avg',
    'cwr_flag_count': 'CWE Flag Count',
    'subflow_fwd_pkts': 'Subflow Fwd Pkts',
    'subflow_bwd_pkts': 'Subflow Bwd Pkts',
    'subflow_fwd_byts': 'Subflow Fwd Byts',
    'subflow_bwd_byts': 'Subflow Bwd Byts',
}

DROP_COLS = ['src_ip', 'dst_ip', 'src_port', 'Family']


def resolve_in_path(in_path, cap_dir):
    if in_path != "auto":
        return Path(in_path)
    files = sorted(Path(cap_dir).glob("flows_labeled_*.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise SystemExit(f"ABORT: no flows_labeled_*.csv in {cap_dir} (pass --in_path)")
    return files[0]


def derive_ts(path):
    name = Path(path).name
    if name.startswith("flows_labeled_") and name.endswith(".csv"):
        return name[len("flows_labeled_"):-len(".csv")]
    return ""


def main():
    ap = argparse.ArgumentParser(description="Adapt netem flows to CIC-IDS2018 merged schema")
    ap.add_argument("--in_path", default="auto",
                    help="labelled netem CSV, or 'auto' for newest flows_labeled_*.csv")
    ap.add_argument("--cap_dir", default="../netemDocker/capture",
                    help="dir to search when --in_path auto")
    ap.add_argument("--out_dir", default="preprocessing/processes_output/merged_datasets",
                    help="dir to write the merged-format parquet")
    ap.add_argument("--dataset_name", default=None,
                    help="artefact name token (default: netem-ids2018-<run_ts>)")
    args = ap.parse_args()

    in_path = resolve_in_path(args.in_path, args.cap_dir)
    ts = derive_ts(in_path)
    name = args.dataset_name or (f"netem-ids2018-{ts}" if ts else "netem-ids2018")

    df = pd.read_csv(in_path)
    print(f"read {len(df):,} rows x {len(df.columns)} cols from {in_path}")

    present_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=present_drop)
    print(f"dropped netem-only cols: {present_drop}")

    missing = [k for k in RENAME if k not in df.columns]
    if missing:
        raise SystemExit(f"ABORT: expected netem source columns absent: {missing}")
    df = df.rename(columns=RENAME)

    if "Label" not in df.columns:
        raise SystemExit("ABORT: no Label column — run label_flows.py first")

    df = df.astype(str)   # mirror the merge stage; 02 coerces types

    gen_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}-merged-{gen_ts}.parquet"
    df.to_parquet(out_path, compression="snappy", index=False)

    print(f"wrote {len(df):,} rows x {len(df.columns)} cols -> {out_path}")
    print("columns:", list(df.columns))


if __name__ == "__main__":
    main()
