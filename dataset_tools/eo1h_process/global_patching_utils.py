from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import math, os
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon
from affine import Affine
from pyproj import CRS
from rasterio.warp import reproject, Resampling
from rasterio.warp import transform as warp_transform
from rasterio.transform import xy
import glob
from dataset_tools.eo1h_process.utils import parse_metadata, update_metadata_json, patch_corners
from rasterio.transform import array_bounds
import json
from rasterio.vrt import WarpedVRT

# CRS_WGS84 = "EPSG:4326"
P = 30.0             # meters per pixel (exact)
N = 128              # patch size
S = N * P            # 3840 m per patch
s = 128              # stride in pixels (use 64 if you want 50% overlap)
Ss = s * P           # stride in meters
AX, AY = 0.0, 0.0    # anchor (grid origin) in target CRS

def ij_range_from_bounds(xmin, ymin, xmax, ymax):
    """Which patch indices (i along +x, j along +y) intersect this bbox?"""
    i_min = math.floor((xmin - AX) / Ss)
    i_max = math.floor((xmax - AX - 1e-9) / Ss)
    j_min = math.floor((ymin - AY) / Ss)
    j_max = math.floor((ymax - AY - 1e-9) / Ss)
    return range(i_min, i_max + 1), range(j_min, j_max + 1)

def patch_affine(i, j):
    """North-up affine for patch (i,j) on the fixed lattice."""
    x_ul = AX + i * Ss
    y_ul = AY + j * Ss
    return Affine(P, 0.0, x_ul, 0.0, -P, y_ul)

def cut_patch(src, A_patch, resampling=Resampling.bilinear, nodata_rate_max=0.1):
    """Reproject src -> patch array (C,H,W) on the fixed patch grid."""
    H = W = N
    # Wrap the source in a virtual raster aligned to your target grid
    out = np.full((src.count, H, W), np.nan, np.float32)
    for b in range(1, src.count + 1):
        reproject(
            source=rasterio.band(src, b),
            destination=out[b-1],
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=A_patch, dst_crs=src.crs,   # src already EPSG:32619
            dst_width=W, dst_height=H,
            resampling=resampling,
            src_nodata=src.nodata, dst_nodata=np.nan
        )
    nodata_rate = 0
    nodata_rate += np.sum(out == 0) / (H * W * src.count) 
    nodata_rate += np.sum(np.isnan(out)) / (H * W * src.count)
    nodata_rate += np.sum(out == src.nodata) / (H * W * src.count)
    if nodata_rate > nodata_rate_max:
        return None, 1-nodata_rate
    return out, 1-nodata_rate

def cut_raster_to_tile(src:rasterio.DatasetReader,
                        nodata_rate_max:float=0.1,
                        resampling_data:Resampling=Resampling.bilinear):
    patched_dict = {}
    metadata_rows = pd.DataFrame()
    # 1) compute all lattice indices overlapping this scene
    i_rng, j_rng = ij_range_from_bounds(*src.bounds)
    dropped_patches = 0
    kept_patches = 0
    
    # 2) iterate patches on the fixed global grid
    for i in i_rng:
        for j in j_rng:
            A = patch_affine(i, j)
            # quick reject: patch bbox vs scene bbox overlap test
            x0, y0 = A.c, A.f
            x1, y1 = x0 + S, y0 - S
            if (x1 <= src.bounds.left or x0 >= src.bounds.right or
                y1 >= src.bounds.top  or y0 <= src.bounds.bottom):
                continue
            # 3) cut (reproject) this patch
            patch, data_rate = cut_patch(src, A, resampling=resampling_data, nodata_rate_max=nodata_rate_max)
            # -> save or feed to your pipeline; name with a deterministic ID:
            if patch is not None:
                kept_patches += 1
                patch_id = f"{src.crs.to_epsg()}:{i}:{j}"
                patched_dict[patch_id] = patch
                metadata = pd.DataFrame({
                    'location_uid': [patch_id],
                    'transform': [A],
                    'crs': [src.crs.to_epsg()],
                    'bounds': [array_bounds(height=N, width=N, transform=A)],
                    'data_rate': [data_rate],
                    'nodata': [src.nodata]
                })
                metadata_rows = pd.concat([metadata_rows, metadata], ignore_index=True)
            else:
                dropped_patches += 1
    assert dropped_patches != 0, "Something went wrong"
    return patched_dict, metadata_rows, kept_patches, dropped_patches

def cut_raster_to_tile_with_transform(src:rasterio.DatasetReader,
                                      save_path:str,
                        dst_transform:Affine,
                        dst_crs:CRS,
                        nodata_rate_max:float=0.1,
                        resampling_data:Resampling=Resampling.bilinear,
                        write:bool=True):
    patch, _ = cut_patch(src, dst_transform, resampling=resampling_data, nodata_rate_max=nodata_rate_max*2) # allow 2x the nodata rate 
    assert patch is not None, "Something went wrong"
    if write:
        write_patch_tif(save_path, patch, dst_transform, dst_crs)
    return patch
    

def write_patch_tif(path:str, data:np.ndarray, transform:Affine, crs:CRS):
    bands,H,W = data.shape
    profile = {
        "driver":"GTiff","height":H,"width":W,"count":bands,"dtype":str(data.dtype),
        "crs":crs,"transform":transform,"compress":"deflate",
        "tiled":True,"blockxsize":min(256,W),"blockysize":min(256,H),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

# # --------------------------
# # 3) Orchestrator (one GeoTIFF)
# # --------------------------
def tile_image_with_grids(geotiff_dir:str, patch_save_dir:str,
                        nodata_rate_max:float=0.1,
                        write:bool=True) -> Dict:
    """
    1) Read UTM zone/CRS from image; 2) use its footprint as AOI; 3) build N grids; 4) cut patches.
    If pixel_size=None, infer from image transform (abs(pixel width)).
    """
    save_top_dir_template: str='EO1H{locationId}'
    save_dir_template: str='EO1H{locationId}_{year}{day}'
    save_name_template: str='EO1H{locationId}_{year}{day}_B{band}_L1T.TIF'
    save_metadata_template: str='EO1H{locationId}_{year}{day}_L1T.json'
    
    geotiffs = glob.glob(os.path.join(geotiff_dir, "*.TIF"))
    geotiffs.sort()
    
    metadata_path = glob.glob(os.path.join(geotiff_dir, "*.TXT"))[0]
    metadata = parse_metadata(metadata_path, patch_size=N)
    
    # Get the first geotiff
    first_geotiff = geotiffs[0]
    first_name = os.path.basename(first_geotiff)
    year = first_name[10:14]
    day = first_name[14:17]
    
    metadata_dict: Dict[int, Dict[str, Any]] = {}
    with rasterio.open(first_geotiff) as src:
        epsg = src.crs.to_epsg()
        patched_dict, metadata_rows, kept_patches, dropped_patches = cut_raster_to_tile(src, nodata_rate_max=nodata_rate_max)
    out_stats = {"epsg": epsg, "kept":kept_patches, "dropped":dropped_patches, "patch_paths":[]}
    
    if kept_patches == 0:
        return out_stats
    
    for _, row in metadata_rows.iterrows():   
        location_id = row["location_uid"]
        save_top_dir = save_top_dir_template.format(locationId=location_id)
        save_dir = save_dir_template.format(locationId=location_id, year=year, day=day)
        save_metadata = save_metadata_template.format(locationId=location_id, year=year, day=day)
        save_top_dir = os.path.join(patch_save_dir, save_top_dir)
        save_dir = os.path.join(save_top_dir, save_dir)
        out_stats["patch_paths"].append(save_dir) 
        if write:
            os.makedirs(save_top_dir, exist_ok=True)
            os.makedirs(save_dir, exist_ok=True)
        
        corners_latlon, corners_proj = patch_corners(row["transform"], N, N, CRS.from_epsg(row["crs"]))
        new_metadata = json.loads(json.dumps(metadata))  # deep copy via JSON
        new_metadata = update_metadata_json(
            new_metadata, corners_latlon, corners_proj
        )
        new_metadata["L1_METADATA_FILE"]["PRODUCT_METADATA"]["METADATA_FILE_NAME"] = save_metadata
        metadata_dict[location_id] = {
            "save_dir": save_dir,
            "save_metadata": save_metadata,
            "metadata": new_metadata
        }
        
    
    for geotiff in geotiffs:
        band_name = os.path.basename(geotiff)
        band = band_name.split("_")[1][1:]  # after 'B'
        with rasterio.open(geotiff) as src:
            for _, row in metadata_rows.iterrows():
                location_id = row["location_uid"]
                save_name = save_name_template.format(locationId=location_id, year=year, day=day, band=band)
                aff = row["transform"]
                crs = CRS.from_epsg(row["crs"])
                out_path = os.path.join(metadata_dict[location_id]["save_dir"], save_name)
                metadata_dict[location_id]["metadata"]["L1_METADATA_FILE"]["PRODUCT_METADATA"][f"BAND{int(band)}_FILE_NAME"] = save_name
                cut_raster_to_tile_with_transform(src, out_path, aff, crs, nodata_rate_max=nodata_rate_max, write=write)
        
    # add the dir_name column to the metadata_rows
    scene_id = geotiff_dir.split("/")[-1]
    metadata_rows["scene_id"] = scene_id
    metadata_rows["year"] = year
    metadata_rows["day"] = day
            
    if write:
        for _, entry in metadata_dict.items():
            save_path = os.path.join(entry["save_dir"], entry["save_metadata"])
            with open(save_path, "w") as f:
                json.dump(entry["metadata"], f, indent=4)
                
        # save the metadata_rows to a csv file
        metadata_rows.to_parquet(os.path.join(patch_save_dir, f"{scene_id}.parquet"))

    return out_stats, metadata_rows
