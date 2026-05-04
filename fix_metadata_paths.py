"""
Fix image_path in metadata.parquet: replace _{BAND}_L1T.TIF → .TIF
to point to stacked TIF files instead of per-band L1T files.

Before: AC/.../EO1H32658:134:1931_2003295/EO1H32658:134:1931_2003295_{BAND}_L1T.TIF
After:  AC/.../EO1H32658:134:1931_2003295/EO1H32658:134:1931_2003295.TIF
"""

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=os.path.join(os.getcwd(), "data", "EO1H"))
args = parser.parse_args()

DATA_DIR = args.data_dir
PARQUET  = os.path.join(DATA_DIR, "metadata.parquet")
BACKUP   = PARQUET.replace(".parquet", "_prefix_backup.parquet")

df = pd.read_parquet(PARQUET)
print(f"Total rows: {len(df):,}")

has_band = df["image_path"].str.contains("{BAND}", regex=False)
print(f"Rows with {{BAND}} in path: {has_band.sum():,}")

# Backup
df.to_parquet(BACKUP, index=False)
print(f"Backup: {BACKUP}")

# Fix: replace _{BAND}_L1T.TIF with .TIF
df["image_path"] = df["image_path"].str.replace(
    "_{BAND}_L1T.TIF", ".TIF", regex=False
)

# Verify a sample
sample = df.loc[has_band].iloc[0]["image_path"]
full = os.path.join(DATA_DIR, sample)
print(f"\nSample fixed path: {sample}")
print(f"File exists: {os.path.exists(full)}")

df.to_parquet(PARQUET, index=False)
print(f"\nSaved. Rows still with {{BAND}}: "
      f"{df['image_path'].str.contains('{BAND}', regex=False).sum():,}")
