# eo1h_to_hub.py
# Build & publish EO1H slices to the Hugging Face Hub (no dataset scripts).
# Each slice = one (REGION × BAND_GROUP) config on the Hub.
#
# Output schema per example:
#   - image: (C, H, W) float32
#   - timestamp_days: int32   # days since reference_date (default 1990-01-01)
#   - frame_id: string        # stable unique key to join across groups
#   - geo:                    # minimal geospatial core to reconstruct georeferencing
#       - crs_epsg: int32
#       - transform: List[float64]  # GDAL affine [a, b, d, e, xoff, yoff]
#       - bounds:    List[float64]  # [xmin, ymin, xmax, ymax]
#       - res:       List[float64]  # [xres, yres]
#       - nodata:    float64        # NaN if absent
#   - raw_band_hrefs: List[string]  # optional repo-relative hrefs to raw GeoTIFFs
#
# Usage examples at bottom (__main__).

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import rasterio

from datasets import (
    Dataset,
    Features,
    Array3D,
    Sequence,
    Value,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
)

# ------------------------------ Constants -------------------------------------

BAND_GROUPS: Dict[str, List[int]] = {
    "VNIR":  list(range(10, 58)),   # 10–57
    "SWIR1": list(range(81, 98)),   # 81–97
    "SWIR2": list(range(101, 120)), # 101–119
    "SWIR3": list(range(134, 165)), # 134–164
    "SWIR4": list(range(182, 222)), # 182–221
}

ALL_REGIONS = {'SEA', 'LA', 'OC', 'SWA', 'AC', 'EU', 'EA', 'NA', 'AF'}
_BAND_COL_RE = re.compile(r"^BAND(\d{1,3})_FILE_NAME$", re.IGNORECASE)

# ------------------------------ Utilities -------------------------------------

def resolve_metadata_path(data_dir: str, filename: str = "metadata") -> str:
    """Find <data_dir>/metadata or metadata.csv."""
    p1 = os.path.join(data_dir, filename)
    p2 = p1 + ".csv"
    if os.path.isfile(p1): return p1
    if os.path.isfile(p2): return p2
    raise FileNotFoundError(f"Top-level metadata not found at '{p1}' or '{p2}'")

def days_since(date_str: str, origin: str = "1990-01-01") -> Optional[int]:
    """Days since origin (YYYY-MM-DD). Accepts 'YYYY-MM-DD' or 'YYYY/MM/DD'."""
    if not isinstance(date_str, str):
        return None
    s = date_str.strip()
    dt = None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt)
            break
        except Exception:
            pass
    if dt is None:
        return None
    return (dt.date() - datetime.strptime(origin, "%Y-%m-%d").date()).days

def build_features(groups: List[str], tile_hw: Tuple[int, int]) -> Features:
    """Features for a (region × band_group) slice."""
    C = sum(len(BAND_GROUPS[g]) for g in groups)
    H, W = tile_hw
    return Features({
        "image": Array3D(shape=(C, H, W), dtype="float32"),
        "timestamp_days": Value("int32"),
        "frame_id": Value("string"),
        "geo": {
            "crs_epsg": Value("int32"),
            "transform": Sequence(Value("float64")),  # 6 numbers
            "bounds": Sequence(Value("float64")),     # 4 numbers
            "res": Sequence(Value("float64")),        # 2 numbers
            "nodata": Value("float64"),               # NaN if none
        },
        "raw_band_hrefs": Sequence(Value("string")),  # optional repo-relative hrefs
    })

def stable_frame_id(row: pd.Series) -> str:
    """
    Build a stable ID shared across groups for the same frame.
    We use '<Region>/<path>' which already encodes location/timeframe.
    """
    region = str(row.get("Region", "")).strip()
    base   = str(row.get("path", "")).strip()
    if not base:
        # Fallback to a minimal unique combo if 'path' missing.
        # You can customize this to your metadata schema.
        loc = str(row.get("location_uid", "")).strip()
        cid = str(row.get("coord_id", "")).strip()
        date = str(row.get("ACQUISITION_DATE", "")).strip()
        return f"{region}/{loc}/{cid}/{date}"
    # Avoid double-prefixing region if already present in path:
    return base if base.upper().startswith(region.upper() + "/") else f"{region}/{base}"

# ------------------------------ Generator -------------------------------------

def iter_examples(
    data_dir: str,
    groups: List[str],
    region: str,
    coord_versions: List[int],
    tile_hw: Tuple[int, int],
    *,
    dtype: str = "float32",
    scale: Optional[float] = None,
    strict_bands: bool = True,
    origin_date: str = "1990-01-01",
    metadata_filename: str = "metadata",
    include_raw_hrefs: bool = True,
    raw_prefix: str = "raw",  # where you'll later upload raw geotiffs in the repo
) -> Iterable[dict]:
    """
    Yield training-ready examples for ONE slice (region × union(groups)).
    """
    assert region in ALL_REGIONS, f"Unknown region {region}"

    meta_path = resolve_metadata_path(data_dir, metadata_filename)
    df = pd.read_csv(meta_path, keep_default_na=False)

    # Required columns
    if "coord_id" not in df.columns:
        raise ValueError("Required column 'coord_id' not found.")
    acq_col = next((c for c in df.columns if c.lower() == "acquisition_date"), None)
    if acq_col is None:
        raise ValueError("Required column 'ACQUISITION_DATE' not found.")
    if "Region" not in df.columns:
        raise ValueError("Required column 'Region' not found.")
    if "path" not in df.columns:
        raise ValueError("Required column 'path' not found.")

    # Filters
    df = df[(df["Region"].str.upper() == region.upper()) & (df["coord_id"].isin(coord_versions))]

    # Band columns available in CSV
    band_cols: Dict[int, str] = {}
    for col in df.columns:
        m = _BAND_COL_RE.match(col)
        if m:
            band_cols[int(m.group(1))] = col

    # Requested band IDs (in group order)
    requested: List[int] = []
    for g in groups:
        if g not in BAND_GROUPS:
            raise ValueError(f"Unknown band group '{g}'. Valid: {list(BAND_GROUPS)}")
        requested += BAND_GROUPS[g]

    if strict_bands:
        missing = [b for b in requested if b not in band_cols]
        if missing:
            raise ValueError(f"CSV missing columns for bands {missing} for requested groups {groups}")

    H_exp, W_exp = tile_hw
    C_exp = len(requested)

    for _, row in df.iterrows():
        # Timestamp
        ts_days = days_since(row[acq_col], origin_date)
        if ts_days is None:
            continue

        # Resolve band files and read
        base = os.path.join(data_dir, str(row["path"]))
        planes: List[np.ndarray] = []
        raw_hrefs: List[str] = []

        ok = True
        for b in requested:
            col = band_cols.get(b)
            if not col:
                if strict_bands: ok = False; break
                else: continue
            fn = row[col]
            if not isinstance(fn, str) or not fn:
                if strict_bands: ok = False; break
                else: continue
            fp = os.path.join(base, fn)
            if not os.path.exists(fp):
                if strict_bands: ok = False; break
                else: continue
            with rasterio.open(fp) as src:
                arr = src.read(1)  # [H, W]
                if not planes:
                    if arr.shape != (H_exp, W_exp):
                        raise ValueError(f"Tile mismatch: got {arr.shape}, expected {(H_exp, W_exp)} at {fp}")
                    # Geo pulled from first band
                    crs_epsg = int(src.crs.to_epsg()) if (src.crs and src.crs.to_epsg()) else -1
                    transform = list(src.transform)  # [a,b,d,e,xoff,yoff]
                    bounds = list(src.bounds)
                    res = list(src.res)
                    nodata = float("nan") if src.nodata is None else float(src.nodata)
                    geo = {
                        "crs_epsg": crs_epsg,
                        "transform": transform,
                        "bounds": bounds,
                        "res": res,
                        "nodata": nodata,
                    }
                planes.append(arr)
            if include_raw_hrefs:
                # repo-relative planned layout for raw COGs
                raw_hrefs.append(f"{raw_prefix}/{row['path']}/{fn}")

        if not ok or not planes:
            continue
        if strict_bands and len(planes) != C_exp:
            continue

        image = np.stack(planes, axis=0).astype(dtype)
        if scale is not None:
            image *= scale

        yield {
            "image": image,
            "timestamp_days": int(ts_days),
            "frame_id": stable_frame_id(row),
            "geo": geo,
            "raw_band_hrefs": raw_hrefs,
        }

# ------------------------------ Build / Push ----------------------------------

def build_slice_dataset(
    data_dir: str,
    region: str,
    groups: List[str],
    coord_versions: List[int],
    tile_hw: Tuple[int, int],
    *,
    dtype: str = "float32",
    scale: Optional[float] = None,
    strict_bands: bool = True,
    origin_date: str = "1990-01-01",
    metadata_filename: str = "metadata",
    include_raw_hrefs: bool = True,
    raw_prefix: str = "raw",
) -> Dataset:
    """
    Build a materialized Dataset for one (region × groups) slice.
    """
    feats = build_features(groups, tile_hw)
    gen_kwargs = dict(
        data_dir=data_dir,
        groups=groups,
        region=region,
        coord_versions=coord_versions,
        tile_hw=tile_hw,
        dtype=dtype,
        scale=scale,
        strict_bands=strict_bands,
        origin_date=origin_date,
        metadata_filename=metadata_filename,
        include_raw_hrefs=include_raw_hrefs,
        raw_prefix=raw_prefix,
    )
    ds = Dataset.from_generator(
        iter_examples,
        gen_kwargs=gen_kwargs,
        features=feats,
    )
    return ds

def push_slice_to_hub(
    ds: Dataset,
    repo_id: str,
    *,
    config_name: str,
    private: bool = False,
    token: Optional[str] = None,
    max_shard_size: str = "2GB",
    commit_message: Optional[str] = None,
):
    """
    Push one slice as a Hub config (e.g., 'SEA-VNIR').
    Consumers: load_dataset(repo_id, name=config_name, split="train").
    """
    ds.push_to_hub(
        repo_id=repo_id,
        config_name=config_name,
        private=private,
        token=token,
        max_shard_size=max_shard_size,
        commit_message=commit_message or f"Add slice {config_name}",
    )

def push_all_slices(
    data_dir: str,
    repo_id: str,
    *,
    regions: List[str],
    band_groups: List[str],
    coord_versions: List[int] = [0],
    tile_hw: Tuple[int, int] = (128, 128),
    dtype: str = "float32",
    scale: Optional[float] = None,
    strict_bands: bool = True,
    origin_date: str = "1990-01-01",
    metadata_filename: str = "metadata",
    include_raw_hrefs: bool = True,
    raw_prefix: str = "raw",
    private: bool = False,
    token: Optional[str] = None,
    max_shard_size: str = "2GB",
):
    """Build & push all (region × band_group) slices as configs."""
    for r in regions:
        if r not in ALL_REGIONS:
            raise ValueError(f"Unknown region: {r}")
        for g in band_groups:
            ds = build_slice_dataset(
                data_dir,
                region=r,
                groups=[g],        # one group per config; keep C constant and small
                coord_versions=coord_versions,
                tile_hw=tile_hw,
                dtype=dtype,
                scale=scale,
                strict_bands=strict_bands,
                origin_date=origin_date,
                metadata_filename=metadata_filename,
                include_raw_hrefs=include_raw_hrefs,
                raw_prefix=raw_prefix,
            )
            cfg_name = f"{r}-{g}"
            push_slice_to_hub(
                ds,
                repo_id=repo_id,
                config_name=cfg_name,
                private=private,
                token=token,
                max_shard_size=max_shard_size,
                commit_message=f"Add {cfg_name} slice",
            )

# ------------------------------ Merge Helpers ---------------------------------

def concat_regions(repo_id: str, group: str, regions: List[str], split: str = "train"):
    """Concatenate multiple region configs for the same band group."""
    parts = [
        load_dataset(repo_id, name=f"{r}-{group}", split=split)
        for r in regions
    ]
    return concatenate_datasets(parts)

def interleave_regions(repo_id: str, group: str, regions: List[str], split: str = "train", seed: int = 0):
    """Round-robin interleave regions for the same band group."""
    parts = [
        load_dataset(repo_id, name=f"{r}-{group}", split=split)
        for r in regions
    ]
    return interleave_datasets(parts, seed=seed)

def stack_groups_by_frame_id(
    repo_id: str,
    region: str,
    groups_in_order: List[str],
    split: str = "train",
):
    """
    Join multiple group configs for a single region by 'frame_id', concatenating channels along C.
    Returns a new in-memory Dataset with:
      - image: (C_total, H, W)
      - timestamp_days: from the first group
      - frame_id
      - geo: from the first group (sanity-check as needed)
    """
    # Load all slices
    slices = [load_dataset(repo_id, name=f"{region}-{g}", split=split) for g in groups_in_order]
    # Shapes must match in H, W
    H, W = slices[0].features["image"].shape[1:]
    for ds in slices[1:]:
        h, w = ds.features["image"].shape[1:]
        assert (h, w) == (H, W), "Spatial shapes must match across groups."

    # Build index for all but the first
    idx_maps = []
    for ds in slices[1:]:
        idx_maps.append({ex["frame_id"]: i for i, ex in enumerate(ds)})

    C_total = sum(ds.features["image"].shape[0] for ds in slices)
    features = Features({
        "image": Array3D((C_total, H, W), "float32"),
        "timestamp_days": Value("int32"),
        "frame_id": Value("string"),
        "geo": {
            "crs_epsg": Value("int32"),
            "transform": Sequence(Value("float64")),
            "bounds": Sequence(Value("float64")),
            "res": Sequence(Value("float64")),
            "nodata": Value("float64"),
        },
    })

    # Map over base (first) slice
    base = slices[0]

    def _stack(ex):
        fid = ex["frame_id"]
        images = [ex["image"]]
        for ds, idx in zip(slices[1:], idx_maps):
            j = idx.get(fid)
            if j is None:
                return None  # drop unmatched frames
            images.append(ds[int(j)]["image"])
        cat = np.concatenate(images, axis=0)
        return {
            "image": cat,
            "timestamp_days": ex["timestamp_days"],
            "frame_id": fid,
            "geo": ex["geo"],  # keep geo from base; optionally validate equality across groups
        }

    merged = base.map(_stack, features=features)
    merged = merged.filter(lambda e: e is not None)
    return merged

# ------------------------------ Raw Upload (optional) -------------------------

def print_missing_raws_for_slice(data_dir: str, region: str, groups: List[str], coord_versions: List[int], metadata_filename: str = "metadata"):
    """
    Quick audit: print repo-relative HREFs (raw/<path>/<file>) that would be referenced for this slice.
    Useful to bulk-upload the raw files via 'git lfs' or huggingface_hub.
    """
    meta_path = resolve_metadata_path(data_dir, metadata_filename)
    df = pd.read_csv(meta_path, keep_default_na=False)
    band_cols = {}
    for col in df.columns:
        m = _BAND_COL_RE.match(col)
        if m:
            band_cols[int(m.group(1))] = col
    req = []
    for g in groups:
        req += BAND_GROUPS[g]
    df = df[(df["Region"].str.upper() == region.upper()) & (df["coord_id"].isin(coord_versions))]
    for _, row in df.iterrows():
        base = str(row["path"])
        for b in req:
            col = band_cols.get(b)
            if not col: continue
            fn = row[col]
            if not isinstance(fn, str) or not fn: continue
            print(f"raw/{base}/{fn}")

# ------------------------------ Examples --------------------------------------

if __name__ == "__main__":
    """
    Example workflow (edit paths and repo_id):

    1) Build & push all slices as configs (region × single group each).
    2) Load & merge later in training code.
    """

    # --- USER INPUTS ---
    DATA_DIR = "/home/jovyan/workspace/data/EO1H_v2"     # contains top-level 'metadata' or 'metadata.csv'
    REPO_ID  = "GFM-Bench/EO1H-313K"          # dataset repo on the Hub (create beforehand or let push create)
    REGIONS  = list(ALL_REGIONS)       # or list(ALL_REGIONS)
    GROUPS   = ["VNIR", "SWIR1", "SWIR2", "SWIR3", "SWIR4"]   # publish per-group configs; e.g., "SEA-VNIR", "SEA-SWIR3"
    COORDS   = [0]                 # coord_id versions included
    TILE_HW  = (128, 128)          # expected H,W for all frames
    TOKEN    = os.environ.get("HF_TOKEN")                # or your HF token string
    PRIVATE  = False

    # 1) Push all (region × group) configs
    push_all_slices(
        data_dir=DATA_DIR,
        repo_id=REPO_ID,
        regions=REGIONS,
        band_groups=GROUPS,
        coord_versions=COORDS,
        tile_hw=TILE_HW,
        dtype="float32",
        scale=None,
        strict_bands=True,
        origin_date="1990-01-01",
        metadata_filename="metadata",
        include_raw_hrefs=True,
        raw_prefix="raw",
        private=PRIVATE,
        token=TOKEN,
        max_shard_size="2GB",
    )

    # 2) (Optional) After pushing, load and merge in client code:
    #    Concatenate regions for VNIR
    # ds_vnir = concat_regions(REPO_ID, "VNIR", REGIONS, split="train")
    # print(ds_vnir)

    # 3) Stack groups for a single region by frame_id (e.g., VNIR + SWIR3 => C = C_VNIR + C_SWIR3)
    # ds_sea_all = stack_groups_by_frame_id(REPO_ID, "SEA", ["VNIR", "SWIR3"], split="train")
    # print(ds_sea_all)
