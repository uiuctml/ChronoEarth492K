import glob
import os
import numpy as np
from typing import Tuple, Union, Mapping, Optional
from shapely.geometry import Polygon, shape
from shapely.prepared import prep
from shapely.ops import transform as shp_transform
from shapely.geometry import box
from pyproj import Transformer
import pandas as pd
import geopandas as gpd
from pathlib import Path
import zipfile
from tqdm import tqdm, trange
from rasterio.vrt import WarpedVRT

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from rasterio.crs import CRS
import ast

from rasterio.warp import transform_bounds


NLCD_DIR_TEMPLATE = "Annual_NLCD_{patch_id}_{product}_CU_C1V1"
NLCD_TIF_FILE_NAME_TEMPLATE = "Annual_NLCD_{patch_id}_{product}_{year}_CU_C1V1.tif"
NLCD_AUX_FILE_NAME_TEMPLATE = "Annual_NLCD_{patch_id}_{product}_{year}_CU_C1V1.tif.aux.xml"
NLCD_NODATA = 250  # uint8 codes; 250 is nodata in NLCD Land Cover
NLCD_PRODUCTS = ["LndCov", "LndCovChg", "SpecChgDOY", "FracImp"]
YEARS = list(range(2001, 2018))
LABEL_DTYPE = "uint8"

def unzip_nlcd_data(root_dir):
    zip_files = glob.glob(f"{root_dir}/*.zip")
    logger.info(f"Find {len(zip_files)} zip files.")
    for file_path in tqdm(zip_files):
        save_path = file_path.split(".")[0]
        os.makedirs(save_path, exist_ok=True)
        # unzip file
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(save_path)
        except:
            print(file_path)
        for file in os.listdir(save_path):
                _, _, patch_id, product, year, _, post_fix = file.split("_")
                if int(year) not in YEARS:
                        os.remove(os.path.join(save_path, file))
                elif not post_fix.endswith("aux.xml") and not post_fix.endswith("tif"):
                        os.remove(os.path.join(save_path, file))
        # remove zip file
        os.remove(file_path)
    logger.info(f"Unzip {len(os.listdir(root_dir))} files.")

def build_label_index(root_path, product_key='land_cover', year=2001):
    label_paths = glob.glob(f"{root_path}/{product_key}/*/{tif_file_name_template.format(patch_id='*', product='*', year=year, post_fix='.tif')}")
    label_paths = sorted(label_paths)
    rows=[]
    for p in label_paths:
        file_name = os.path.basename(p)
        _, _, patch_id, product, year, _, _ = file_name.split("_")
        with rasterio.open(p) as src:
            b = src.bounds
            rows.append({"path": str(p), "geometry": box(b.left, b.bottom, b.right, b.top), "patch_id": patch_id, "product": product, "year": int(year)})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=src.crs)  # NLCD CRS
    _ = gdf.sindex  # build R-tree
    return gdf

def build_label_index_from_path(label_root: str, label_template: str, year = None, ee_patch_id = False):
    label_paths = glob.glob(f"{label_root}/{label_template}")
    label_paths = sorted(label_paths)
    rows=[]
    
    # choose a anchor crs
    with rasterio.open(label_paths[0]) as src:
        anchor_crs = src.crs
    
    for p in label_paths:
        file_name = os.path.basename(p).replace(".tif", "")
        if ee_patch_id:
            _, pid_1, pid_2 = file_name.split("-")
            patch_id = f"{pid_1}-{pid_2}"
        else:
           patch_id = file_name
        with rasterio.open(p) as src:
            b = src.bounds
            if src.crs != anchor_crs:
                # convert to anchor crs
                tfm = Transformer.from_crs(src.crs, anchor_crs, always_xy=True).transform
                b = shp_transform(tfm, box(b.left, b.bottom, b.right, b.top))
            else:
                b = box(b.left, b.bottom, b.right, b.top)
            rows.append({"path": str(p), "geometry": b, "patch_id": patch_id, "year": int(year)})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=anchor_crs)
    _ = gdf.sindex  # build R-tree
    return gdf

def match_tile_by_bounds(hsi_meta_row, label_index_gdf, year=None):
    """
    bounds: [xmin, ymin, xmax, ymax] from your HSI metadata (image CRS).
    crs_epsg: int EPSG of the HSI image (e.g., 32606).
    label_index_gdf: GeoDataFrame from build_label_index().
    year: optional int to restrict to the matching NLCD year.

    Returns: path (str) to the single best tile, or None if no overlap.
    """
    # 0) Read from meta
    bounds = hsi_meta_row['bounds'][:6]
    crs_epsg = hsi_meta_row['crs_epsg']
    
    # 1) Build query polygon in image CRS
    img_poly = box(*bounds)
    img_crs = CRS.from_epsg(int(crs_epsg))
    idx_crs = label_index_gdf.crs

    # 2) Reproject query polygon to NLCD CRS (index CRS) if needed
    if CRS.from_user_input(idx_crs) != img_crs:
        tfm = Transformer.from_crs(img_crs, idx_crs, always_xy=True).transform
        q_poly = shp_transform(tfm, img_poly)
    else:
        q_poly = img_poly

    # 3) Optional year filter to shrink search space
    cand = label_index_gdf
    if year is not None and "year" in cand.columns:
        cand = cand[cand["year"] == int(year)]
        if cand.empty:
            return None

    # 4) R-tree shortlist on bbox, then exact intersection
    idxs = list(cand.sindex.intersection(q_poly.bounds))
    if not idxs:
        return None
    cand = cand.iloc[idxs]
    hits = cand[cand.intersects(q_poly)].copy()
    if hits.empty:
        return None

    # 5) Disambiguate near tile seams: choose max overlap area
    if len(hits) > 1:
        return None
        
    extended_hsi_meta_row = hsi_meta_row.copy()
    extended_hsi_meta_row["label_tile_patch_id"] = hits["patch_id"].tolist()[0]
    return extended_hsi_meta_row

def target_grid_from_meta(meta):
    """Return (crs, transform, width, height) from a dict/Series meta row."""
    # CRS
    if "crs_epsg" in meta and meta["crs_epsg"] is not None:
        crs = CRS.from_epsg(int(meta["crs_epsg"]))
    elif "crs_wkt" in meta and meta["crs_wkt"]:
        crs = CRS.from_wkt(str(meta["crs_wkt"]))
    else:
        raise ValueError("Metadata missing crs_epsg/crs_wkt")

    # Transform (first 6 vals; some metadata stores a 3x3)
    transform = meta["transform"]
    T = Affine(*transform[:6])

    # Width/Height
    W, H = 128, 128

    return crs, T, W, H

def cut_label_to_meta_grid(dst_crs, dst_T, W, H, label_tile_path, out_path=None,
                           NODATA=0, LABEL_DTYPE=np.uint8):
    """
    Align label tile to the target grid (dst_crs, dst_T, W, H) using windowed IO.
    Returns (labels, transform, crs). If out_path is given, writes a GeoTIFF.
    """
    with rasterio.open(label_tile_path) as src:
        # Wrap the source in a virtual raster aligned to your target grid
        vrt_opts = dict(
            crs=dst_crs,
            transform=dst_T,
            width=W,
            height=H,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=src.nodata
        )
        with WarpedVRT(src, **vrt_opts) as vrt:
            out = vrt.read(
                out_shape=(vrt.count, H, W),
                # boundless=True,
                fill_value=src.nodata
            )

    # If everything is nodata, signal empty
    if (out == src.nodata).all():
        return None, None, None
    
    # if everything is nan, signal empty
    if np.isnan(out).all():
        return None, None, None

    if out_path:
        profile = {
            "driver": "GTiff",
            "height": H, "width": W, "count": out.shape[0],
            "dtype": out.dtype.name,
            "crs": dst_crs, "transform": dst_T,
            "compress": "DEFLATE", "tiled": True,
            "blockxsize": min(256, W), "blockysize": min(256, H),
            "nodata": src.nodata,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out)

    # If you prefer single-band return for labels:
    return (out[0] if out.shape[0] == 1 else out), dst_T, dst_crs

def match_label_to_hsi_meta(hsi_meta: pd.DataFrame, data_dir: str, product: str = "land_cover"):
    df_label_index = pd.DataFrame()
    for year in tqdm(YEARS):
        hsi_meta_year = hsi_meta[hsi_meta["year"] == year]
        label_index_gdf_year = build_label_index(data_dir, product, year)
        for _, hsi_meta_row in tqdm(hsi_meta_year.iterrows(), total=len(hsi_meta_year), desc=f"Processing year {year}"):
            updated_row = match_tile_by_bounds(hsi_meta_row, label_index_gdf_year)
            if updated_row is not None:
                df_label_index = pd.concat([df_label_index, pd.DataFrame([updated_row])])
    df_label_index = df_label_index.reset_index(drop=True)
    #save as parquet
    df_label_index.to_parquet(f"{data_dir}/nlcd_metadata.parquet")
    
def _is_tile_in_bounds(hsi_meta_row, label_bounds, label_crs, year=None):
    """
    bounds: [xmin, ymin, xmax, ymax] from your HSI metadata (image CRS).
    crs_epsg: int EPSG of the HSI image (e.g., 32606).
    label_index_gdf: GeoDataFrame from build_label_index().
    year: optional int to restrict to the matching NLCD year.

    Returns: True if the tile is in the bounds, False otherwise.
    """
    # 0) Read from meta
    bounds = hsi_meta_row['bounds']
    crs_epsg = hsi_meta_row['crs']
    
    # 1) Build query polygon in image CRS
    img_poly = box(*bounds)
    img_crs = CRS.from_epsg(int(crs_epsg))
    idx_crs = label_crs

    # 2) Reproject query polygon to NLCD CRS (index CRS) if needed
    if CRS.from_user_input(idx_crs) != img_crs:
        tfm = Transformer.from_crs(img_crs, idx_crs, always_xy=True).transform
        q_poly = shp_transform(tfm, img_poly)
    else:
        q_poly = img_poly
        
    k_poly = box(*label_bounds)

    if k_poly.contains(q_poly):
        return True
    else:
        return False
    
def filter_metadata_by_label_bounds(source_meta: pd.DataFrame, label_bounds: Tuple[float, float, float, float], label_crs: str):
    df_filtered = pd.DataFrame()
    for _, row in tqdm(source_meta.iterrows(), total=len(source_meta)):
        if _is_tile_in_bounds(row, label_bounds, label_crs):
            df_filtered = pd.concat([df_filtered, row.to_frame().T], ignore_index=True)
    df_filtered = df_filtered.reset_index(drop=True)
    return df_filtered

def tif_bbox_geojson(tif_path, out_geojson, to_epsg=4326):
    with rasterio.open(tif_path) as src:
        xmin, ymin, xmax, ymax = src.bounds
        poly = Polygon([(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)])
        gdf = gpd.GeoDataFrame({"name":["aoi_bbox"]}, geometry=[poly], crs=src.crs)
        if to_epsg:
            gdf = gdf.to_crs(epsg=to_epsg)
        gdf.to_file(out_geojson, driver="GeoJSON")