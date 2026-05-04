import os
import json
import argparse
from math import sqrt

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

# ----------------------------
# Bands to compute
# ----------------------------
STABLE_BANDS = (
    list(range(10, 58))
    + list(range(81, 98))
    + list(range(101, 120))
    + list(range(134, 165))
    + list(range(182, 222))
)

# ----------------------------
# Running stats (Welford / Chan merge)
# ----------------------------
def update_running(mean, M2, n, x):
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        m = x.size
        b_mean = x.mean()
        b_var  = x.var()   # population variance within this batch
    else:
        m = x.shape[0]
        b_mean = x.mean(axis=0)
        b_var  = x.var(axis=0)

    if n == 0:
        mean = b_mean
        M2   = b_var * m
        n    = m
    else:
        delta = b_mean - mean
        tot   = n + m
        mean  = mean + delta * (m / tot)
        M2    = M2 + b_var * m + (delta * delta) * (n * m / tot)
        n     = tot
    return mean, M2, n

# ----------------------------
# Compute one band (single-thread)
# ----------------------------
def compute_one_band(band, df, data_dir):
    data_mean = [None, None]
    data_M2   = [None, None]
    data_n    = [0, 0]

    # Faster than iterrows(); still preserves column access
    for row in tqdm(df.itertuples(index=False), total=len(df), leave=False, desc=f"Band {band}"):
        base_path = os.path.join(data_dir, getattr(row, "path"))
        band_name = f"BAND{band}_FILE_NAME"
        band_path = getattr(row, band_name)
        coord_id  = getattr(row, "coord_id")
        tile_path = os.path.join(base_path, band_path)

        # Read first band and flatten
        with rasterio.open(tile_path) as src:
            x = src.read(1).ravel()

        mean, M2, n = data_mean[coord_id], data_M2[coord_id], data_n[coord_id]
        data_mean[coord_id], data_M2[coord_id], data_n[coord_id] = update_running(mean, M2, n, x)

    # Sample variance (unbiased)
    var_0 = data_M2[0] / (data_n[0] - 1) if data_n[0] > 1 else 0.0
    var_1 = data_M2[1] / (data_n[1] - 1) if data_n[1] > 1 else 0.0
    stds  = [float(np.sqrt(var_0)), float(np.sqrt(var_1))]
    means = [float(data_mean[0]) if data_mean[0] is not None else None,
             float(data_mean[1]) if data_mean[1] is not None else None]
    return band, means, stds

# ----------------------------
# Main (single-process)
# ----------------------------
def main(data_dir, band):
    metadata_path = os.path.join(data_dir, "metadata.csv")
    df = pd.read_csv(metadata_path, low_memory=False)

    # Decide which bands to compute
    if band is None:
        bands = STABLE_BANDS
    else:
        if band not in STABLE_BANDS:
            raise ValueError(f"--band {band} is not in STABLE_BANDS.")
        bands = [band]

    means, stds = {}, {}
    for b in tqdm(bands, desc="Bands", leave=True):
        band_id, m, s = compute_one_band(b, df, data_dir)
        means[band_id] = m
        stds[band_id]  = s

    # JSON-friendly scalars
    means_clean = {int(k): [None if v is None else float(v) for v in vals] for k, vals in means.items()}
    stds_clean  = {int(k): [None if v is None else float(v) for v in vals] for k, vals in stds.items()}

    out_path = os.path.join(data_dir, "band_stats.json")
    with open(out_path, "w") as f:
        json.dump({"means": means_clean, "stds": stds_clean}, f, indent=2)
    print(f"Wrote {out_path}")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing tiles (must include metadata.csv)")
    parser.add_argument("--band", type=int, default=None,
                        help="Compute a single band (by band number). Omit to compute all STABLE_BANDS.")
    args = parser.parse_args()
    main(args.data_dir, args.band)
