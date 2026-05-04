"""
Remove rows from metadata.parquet whose data directory does not exist on disk.

Each image_path has format:
  {region}/EO1H{loc}/{region}/EO1H{loc}_{YYYYDDD}/{loc}_{YYYYDDD}_{BAND}_L1T.TIF

We check existence of the timestamp subdirectory (dirname of image_path).
If that directory is missing, the sample has no data and the row is dropped.
"""

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=os.path.join(os.getcwd(), "data", "EO1H"))
args = parser.parse_args()

CACHE_DIR  = args.data_dir
PARQUET    = os.path.join(CACHE_DIR, "metadata.parquet")
OUT        = PARQUET  # overwrite in place (backup first)
BACKUP     = PARQUET.replace(".parquet", "_backup.parquet")

df = pd.read_parquet(PARQUET)
print(f"Loaded {len(df):,} rows")

# Derive the directory to check: dirname of the per-band image_path
df["_check_dir"] = df["image_path"].apply(os.path.dirname)

# Vectorised existence check
print("Checking directory existence ...")
exists = df["_check_dir"].apply(lambda d: os.path.isdir(os.path.join(CACHE_DIR, d)))

n_missing = (~exists).sum()
print(f"  Missing directories : {n_missing:,}  ({n_missing/len(df)*100:.1f}%)")
print(f"  Present directories : {exists.sum():,}")

if n_missing == 0:
    print("Nothing to remove.")
else:
    # Backup original
    df.drop(columns=["_check_dir"]).to_parquet(BACKUP, index=False)
    print(f"Backup saved to {BACKUP}")

    # Filter and save
    df_clean = df[exists].drop(columns=["_check_dir"]).reset_index(drop=True)
    df_clean.to_parquet(OUT, index=False)
    print(f"Cleaned parquet saved: {len(df_clean):,} rows → {OUT}")

    # Summary by region
    print("\nRows removed per region:")
    removed = df[~exists]
    print(removed["region"].value_counts().to_string())
