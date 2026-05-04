"""
Remove rows from metadata.parquet whose TIF files are listed in bad_files.txt.
Run after scan_tifs.py identifies problematic files.

Usage:
  python -m temporal_pretrain.remove_bad_tifs
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=os.path.join(os.getcwd(), "data", "EO1H"))
args = parser.parse_args()

DATA_DIR = args.data_dir
PARQUET   = os.path.join(DATA_DIR, "metadata.parquet")
BAD_LIST  = os.path.join(DATA_DIR, "bad_files.txt")
BACKUP    = PARQUET.replace(".parquet", "_prebad_backup.parquet")

with open(BAD_LIST) as f:
    bad_paths = set(l.strip() for l in f if l.strip())

print(f"Bad files to remove: {len(bad_paths)}")

df = pd.read_parquet(PARQUET)
print(f"Rows before: {len(df):,}")

# Match by full path
def full_path(row):
    ip = row["image_path"]
    if "{BAND}" in ip:
        ip = os.path.dirname(ip) + ".TIF"
    return os.path.join(DATA_DIR, ip)

mask = df.apply(full_path, axis=1).isin(bad_paths)
print(f"Rows to drop: {mask.sum():,}")

df.to_parquet(BACKUP, index=False)
print(f"Backup saved to: {BACKUP}")

df_clean = df[~mask].reset_index(drop=True)
df_clean.to_parquet(PARQUET, index=False)
print(f"Rows after: {len(df_clean):,}")
print("Done.")
