import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

from affine import Affine
from rasterio.transform import array_bounds
from pyproj import CRS, Transformer

CRS_WGS84 = "EPSG:4326"


def _coerce_value(raw: str) -> Any:
    raw = raw.strip()
    # Strip quotes
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1]
    # Booleans
    if raw.upper() in {"TRUE", "FALSE"}:
        return raw.upper() == "TRUE"
    # Integers/Floats (including scientific notation)
    try:
        if re.match(r"^[+-]?\d+$", raw):
            return int(raw)
        if re.match(r"^[+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?$", raw):
            return float(raw)
    except Exception:
        pass
    return raw


def parse_metadata(file_path: str, patch_size: int = 128) -> Dict[str, Any]:
    """Parse a plain-text metadata file into a nested dict structure.

    - Builds a nested dict using GROUP/END_GROUP blocks
    - Coerces numeric and boolean values
    - Removes BANDxxx_FILE_NAME and METADATA*_FILE_NAME keys under PRODUCT_METADATA
    - Injects PRODUCT_SAMPLES and PRODUCT_LINES based on patch_size
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    group_stack: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    root: Dict[str, Any] = current

    # Skip the last line to mimic original behavior
    for raw_line in lines[:-1]:
        line = raw_line.strip()
        if not line:
            continue

        match_group_start = re.match(r"GROUP\s*=\s*(\S+)", line)
        if match_group_start:
            group_name = match_group_start.group(1)
            new_group: Dict[str, Any] = {}
            current[group_name] = new_group
            group_stack.append(current)
            current = new_group
            continue

        match_group_end = re.match(r"END_GROUP\s*=\s*(\S+)", line)
        if match_group_end:
            if group_stack:
                current = group_stack.pop()
            continue

        match_kv = re.match(r"(\S+)\s*=\s*(.+)", line)
        if match_kv:
            key, raw_value = match_kv.group(1), match_kv.group(2).strip()
            current[key] = _coerce_value(raw_value)

    # Sanitize product metadata keys
    # Ensure expected hierarchy exists
    if "L1_METADATA_FILE" not in root or not isinstance(
        root["L1_METADATA_FILE"], dict
    ):
        root["L1_METADATA_FILE"] = {}
    if "PRODUCT_METADATA" not in root["L1_METADATA_FILE"] or not isinstance(
        root["L1_METADATA_FILE"]["PRODUCT_METADATA"], dict
    ):
        root["L1_METADATA_FILE"]["PRODUCT_METADATA"] = {}

    product_meta = root["L1_METADATA_FILE"]["PRODUCT_METADATA"]
    keys_to_delete = [
        key
        for key in list(product_meta.keys())
        if key.startswith("BAND") or key.startswith("METADATA")
    ]
    for key in keys_to_delete:
        product_meta.pop(key, None)

    product_meta["PRODUCT_SAMPLES"] = int(patch_size)
    product_meta["PRODUCT_LINES"] = int(patch_size)

    return root

def get_corner_coords(
    ds: rasterio.DatasetReader, window: rasterio.windows.Window
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """Return (latlon_corners, proj_corners) for a given window.

    - proj_corners are in source CRS (x, y)
    - latlon_corners are (lat, lon) in WGS84
    """
    left, bottom, right, top = rasterio.windows.bounds(window, ds.transform)
    proj: Dict[str, Tuple[float, float]] = {
        "UL": (left, top),
        "UR": (right, top),
        "LL": (left, bottom),
        "LR": (right, bottom),
    }

    xs, ys = zip(*proj.values())
    lons, lats = warp_transform(ds.crs, CRS_WGS84, xs, ys)
    geo = dict(zip(["UL", "UR", "LL", "LR"], zip(lats, lons)))

    return geo, proj

def patch_corners(transform: Affine,
                  width: int, height: int,
                  crs: CRS,
                  to_crs: CRS = CRS.from_epsg(4326)
                 ) -> Tuple[Dict[str, Tuple[float,float]],
                            Dict[str, Tuple[float,float]]]:
    """
    Returns: (corners_proj, corners_latlon)
      corners_proj: {'UL': (x,y), 'UR': (x,y), 'LL': (x,y), 'LR': (x,y)} in patch CRS
      corners_latlon: {'UL': (lat,lon), ...} in WGS84 (or to_crs)
    Notes:
      - Uses pixel-is-area bounds (outer edges), consistent with Rasterio/GDAL.
      - Assumes north-up transform (a>0, e<0); works for standard cases.
    """
    # 1) projected bounds (left, bottom, right, top)
    left, bottom, right, top = array_bounds(height, width, transform)

    corners_proj = {
        "UL": (left,  top),
        "UR": (right, top),
        "LL": (left,  bottom),
        "LR": (right, bottom),
    }

    # 2) transform to lat/lon (or any target CRS)
    transformer = Transformer.from_crs(crs, to_crs, always_xy=True)
    xs = [corners_proj[k][0] for k in ("UL","UR","LR","LL")]
    ys = [corners_proj[k][1] for k in ("UL","UR","LR","LL")]
    lons, lats = transformer.transform(xs, ys)

    corners_latlon = {
        "UL": (lats[0], lons[0]),
        "UR": (lats[1], lons[1]),
        "LR": (lats[2], lons[2]),
        "LL": (lats[3], lons[3]),
    }
    return corners_proj, corners_latlon


def update_metadata_json(
    meta: Dict[str, Any],
    latlon_corners: Dict[str, Tuple[float, float]],
    proj_corners: Dict[str, Tuple[float, float]],
) -> Dict[str, Any]:
    """Update corner coordinates in the provided metadata dict."""
    for key, (lat, lon) in latlon_corners.items():
        meta["L1_METADATA_FILE"]["PRODUCT_METADATA"][f"IMAGE_{key}_CORNER_LAT"] = round(lat, 6)
        meta["L1_METADATA_FILE"]["PRODUCT_METADATA"][f"IMAGE_{key}_CORNER_LON"] = round(lon, 6)
        meta["L1_METADATA_FILE"]["PRODUCT_METADATA"][f"PRODUCT_{key}_CORNER_LAT"] = round(lat, 6)
        meta["L1_METADATA_FILE"]["PRODUCT_METADATA"][f"PRODUCT_{key}_CORNER_LON"] = round(lon, 6)

    for key, (x, y) in proj_corners.items():
        meta["L1_METADATA_FILE"]["PRODUCT_METADATA"][f"PRODUCT_{key}_CORNER_MAPX"] = round(x, 6)
        meta["L1_METADATA_FILE"]["PRODUCT_METADATA"][f"PRODUCT_{key}_CORNER_MAPY"] = round(y, 6)

    return meta