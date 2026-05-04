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
    
    # if outpath is already exists, load the existing file and combine them by directly setting the nodata values to the new values
    if out_path and os.path.exists(out_path):
        with rasterio.open(out_path) as src:
            existing_out = src.read()
        out = np.where(out == src.nodata, existing_out, out)

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

def tif_bbox_geojson(tif_path, out_geojson, to_epsg=4326):
    with rasterio.open(tif_path) as src:
        xmin, ymin, xmax, ymax = src.bounds
        poly = Polygon([(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)])
        gdf = gpd.GeoDataFrame({"name":["aoi_bbox"]}, geometry=[poly], crs=src.crs)
        if to_epsg:
            gdf = gdf.to_crs(epsg=to_epsg)
        gdf.to_file(out_geojson, driver="GeoJSON")