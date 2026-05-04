#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, numpy as np, pandas as pd, geopandas as gpd
from pyogrio import read_info, read_dataframe, list_layers
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.ops import unary_union
from shapely.geometry import box
from rasterio.crs import CRS

HCAT3_ALIASES = ["HCAT3_code","hcat3_code","HCAT3","hcat3","EC_hcat_c"]
BDF_ALIASES = ["CODE_TFV"]
FALLBACK_NUMERIC = ["original_code","SNAR_CODE","code","CLASS_ID","class_id"]

def pick_utm_crs_from_gdf(gdf):
    """Robust UTM chooser using bbox centroid in WGS84; avoids unary_union."""
    w84 = CRS.from_epsg(4326)
    g_w84 = gdf.to_crs(w84)
    minx, miny, maxx, maxy = g_w84.total_bounds
    lon = (minx + maxx) / 2.0
    lat = (miny + maxy) / 2.0
    zone = int((lon + 180) // 6 + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def is_geographic(crs: CRS) -> bool:
    try: return CRS.from_user_input(crs).is_geographic
    except: return False

def resolve_layer(path: str, requested: str|None) -> str|None:
    """Return a layer name or None. For Shapefile/GeoJSON, None is fine.
       For GPKG/SQLite: pick the single layer or require explicit choice."""
    if requested: return requested
    try:
        layers = list_layers(path)  # [[name, geom], ...] for drivers with layers
        if not layers: return None
        if len(layers) == 1: return layers[0][0]
        names = [l[0] for l in layers]
        sys.exit(f"Multiple layers found: {names}\nPass --layer <name>.")
    except Exception:
        return None  # drivers without layer concept
       
def select_attribute_field(columns, preferred=None):
    cols_lower = {c.lower(): c for c in columns}
    if preferred:
        if preferred in columns: return preferred
        if preferred.lower() in cols_lower: return cols_lower[preferred.lower()]
        sys.exit(f"--attribute '{preferred}' not found. Available: {list(columns)}")
    for k in HCAT3_ALIASES:
        if k in columns: return k
        if k.lower() in cols_lower: return cols_lower[k.lower()]
    for k in FALLBACK_NUMERIC:
        if k in columns: return k
        if k.lower() in cols_lower: return cols_lower[k.lower()]
    for k in BDF_ALIASES:
        if k in columns: return k
        if k.lower() in cols_lower: return cols_lower[k.lower()]
    for c in columns:
        if c.lower() != "geometry": return c
    sys.exit("No usable attribute field found.")

def coerce_to_int_codes(series: pd.Series):
    s = series.copy()
    # already integer?
    if pd.api.types.is_integer_dtype(s):
        arr = s.fillna(0).astype("int32").to_numpy()
        uniq = np.unique(arr)
        legend = pd.DataFrame({"code_int": uniq, "label": uniq.astype(str)})
        if 0 not in legend["code_int"].values:
            legend = pd.concat([pd.DataFrame({"code_int":[0],"label":["<nodata>"]}), legend], ignore_index=True)
        return arr, legend
    # numeric strings?
    s_num = pd.to_numeric(s, errors="coerce")
    if not s_num.isna().all():
        arr = s_num.fillna(0).astype("uint32").to_numpy()
        arr = arr.astype("uint32") if arr.max() <= np.iinfo(np.uint32).max else arr
        uniq = pd.unique(s_num.dropna().astype(int))
        legend = pd.DataFrame({"code_int": np.r_[0, uniq], "label": np.r_[["<nodata>"], uniq.astype(str)]})
        return arr, legend
    # general categorical
    cats = pd.Categorical(s.astype("string"))
    codes = cats.codes.astype("uint32") + 1
    codes[s.isna().to_numpy()] = 0
    legend = pd.DataFrame({"code_int": np.arange(0, len(cats.categories)+1, dtype="uint32"),
                           "label": pd.Index(["<nodata>"]).append(cats.categories)})
    return codes, legend

def grid_from_reference(ref_path, nodata_default):
    with rasterio.open(ref_path) as ref:
        return ref.crs, ref.transform, ref.width, ref.height, (ref.nodata if ref.nodata is not None else nodata_default)

def main():
    ap = argparse.ArgumentParser(description="Vector → GeoTIFF rasterization with HCAT3-first attribute selection.")
    ap.add_argument("vector", help="Input vector (SHP/GPKG/GeoJSON)")
    ap.add_argument("out_tif", help="Output GeoTIFF")
    ap.add_argument("--layer", help="Layer name for multi-layer sources (e.g., GPKG)")
    ap.add_argument("--attribute", help="Attribute to burn (defaults to HCAT3 alias if present)", default="EC_hcat_c")
    ap.add_argument("--pixel-size", type=float, help="Pixel size (units of output CRS). Required if no --ref.", default=30)
    ap.add_argument("--ref", help="Reference raster to align (CRS/grid). Overrides pixel size/extent.")
    ap.add_argument("--out-crs", help="Force output CRS (e.g., EPSG:32633).")
    ap.add_argument("--all-touched", action="store_true")
    ap.add_argument("--nodata", type=float, default=0)
    args = ap.parse_args()

    lyr = resolve_layer(args.vector, args.layer)
    info = read_info(args.vector, layer=lyr)
    cols = list(info["fields"])
    attr = select_attribute_field(cols, preferred=args.attribute)

    # fast read: only geometry + chosen attribute
    gdf = read_dataframe(args.vector, layer=lyr, columns=["geometry", attr])
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=info["crs"])

    if gdf.crs is None and not args.out_crs and not args.ref:
        sys.exit("Input has no CRS. Supply --out-crs or use --ref to inherit from a raster.")

    # Decide output CRS
    if args.ref:
        with rasterio.open(args.ref) as ref:
            out_crs = ref.crs
        if gdf.crs is None or CRS.from_user_input(gdf.crs) != out_crs:
            gdf = gdf.to_crs(out_crs)
    else:
        if args.out_crs:
            out_crs = CRS.from_user_input(args.out_crs)
            gdf = gdf.to_crs(out_crs)
        else:
            out_crs = pick_utm_crs_from_gdf(gdf) if is_geographic(gdf.crs) else gdf.crs
            if out_crs != gdf.crs:
                gdf = gdf.to_crs(out_crs)

    # Grid
    if args.ref:
        out_crs, transform, width, height, nodata = grid_from_reference(args.ref, args.nodata)
        rb = rasterio.transform.array_bounds(height, width, transform)
        gdf = gpd.overlay(gdf, gpd.GeoDataFrame(geometry=[box(*rb)], crs=out_crs),
                          how="intersection", keep_geom_type=False)
    else:
        if args.pixel_size is None:
            sys.exit("--pixel-size is required when --ref is not provided.")
        xmin, ymin, xmax, ymax = gdf.total_bounds
        width  = int(np.ceil((xmax - xmin) / args.pixel_size))
        height = int(np.ceil((ymax - ymin) / args.pixel_size))
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
        nodata = args.nodata

    # Attribute → int codes + legend
    codes, legend = coerce_to_int_codes(gdf[attr])
    gdf["__code__"] = codes

    arr = rasterize(
        shapes=zip(gdf.geometry, gdf["__code__"]),
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        all_touched=args.all_touched,
        dtype="uint32",
    )

    profile = dict(driver="GTiff", height=height, width=width, count=1, dtype="uint32",
                   crs=out_crs, transform=transform, nodata=nodata,
                   compress="deflate", tiled=True, blockxsize=512, blockysize=512)

    with rasterio.open(args.out_tif, "w", **profile) as dst:
        dst.write(arr, 1)

    legend_path = args.out_tif.rsplit(".", 1)[0] + "_legend.csv"
    legend.rename(columns={"label": attr}).to_csv(legend_path, index=False)

    print(f"[OK] Raster: {args.out_tif}")
    print(f"[OK] Legend: {legend_path}")
    print(f"[INFO] Layer used: {lyr if lyr else '(none / single-layer driver)'}")
    print(f"[INFO] Attribute burned: {attr}")

if __name__ == "__main__":
    main()