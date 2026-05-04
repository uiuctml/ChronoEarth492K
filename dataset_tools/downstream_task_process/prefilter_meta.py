import rasterio
import pandas as pd
from loguru import logger
import argparse
import glob
from typing import List
import os
from shapely.geometry import box
import geopandas as gpd
from rasterio.crs import CRS
from tqdm import tqdm
from pyproj import Transformer
from shapely.ops import transform as shp_transform

def build_gdf_index(label_paths: List[str], dataset_name: str):
    rows=[]
    # choose a anchor crs
    with rasterio.open(label_paths[0]) as src:
        anchor_crs = src.crs

    for label_path in label_paths:
        if dataset_name == "NLCD":
            label_patch_uid = label_path.split("/")[-2].split("_")[2]
        elif dataset_name in ["TreeMap", "ISDASoil"]:
            _,pid_1, pid_2 = label_path.split("/")[-1].strip(".tif").split("-")
            label_patch_uid = f"{pid_1}-{pid_2}"
        else:
            label_patch_uid = label_path.split("/")[-1]
        p = os.path.basename(label_path)
            
        with rasterio.open(label_path) as src:
            b = src.bounds
            if src.crs != anchor_crs:
                # convert to anchor crs
                tfm = Transformer.from_crs(src.crs, anchor_crs, always_xy=True).transform
                b = shp_transform(tfm, box(b.left, b.bottom, b.right, b.top))
            else:
                b = box(b.left, b.bottom, b.right, b.top)
                
            rows.append({"path": str(p), "geometry": b, "patch_id": label_patch_uid})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=anchor_crs)
    _ = gdf.sindex  # build R-tree
    return gdf

# Filter out rows with any negative coordinate in geometry bounds
def has_negative_coordinate(geom):
    # Try both for Polygon and MultiPolygon
    try:
        coords = list(geom.exterior.coords)
    except AttributeError:
        # MultiPolygon: combine all coords
        coords = []
        for part in geom.geoms:
            coords.extend(list(part.exterior.coords))
    return any((x < 0 or y < 0) for x, y in coords)

def generate_meta_from_label_patches(df_region: pd.DataFrame, label_paths: List[str], dataset_name: str, allow_multiple_hits: bool = False):
    crss = df_region['crs'].unique()
    gdf = build_gdf_index(label_paths, dataset_name)
    df_ic = []
    for crs in crss:
        df_crs = df_region.loc[df_region['crs'] == crs]
        img_crs = CRS.from_epsg(int(crs))
        idx_crs = gdf.crs.to_string()

        # 2) Reproject NLCD CRS (index CRS) to query polygon if needed
        if idx_crs != img_crs:
            gdf_dst = gdf.to_crs(img_crs)
        else:
            gdf_dst = gdf
            
        k_polys = gdf_dst
        
        # Filter out rows with any negative coordinate in geometry bounds
        k_polys = k_polys[~k_polys.geometry.astype(object).apply(has_negative_coordinate)].reset_index(drop=True)
            
        for _, row in tqdm(df_crs.iterrows(), total=len(df_crs)):
            img_bounds = row['bounds']
            q_poly = box(*img_bounds)
            
            idxs = list(k_polys.sindex.intersection(q_poly.bounds))
            if not idxs:
                continue
            cand = k_polys.iloc[idxs]
            hits = cand[cand.intersects(q_poly)].copy()
            if hits.empty:
                continue
            
            if not allow_multiple_hits:
                if len(hits) > 1:
                    continue
            
            for patch_id in hits["patch_id"].tolist():
                extended_row = row.copy()
                extended_row["label_tile_patch_id"] = patch_id
                df_ic.append(extended_row.to_frame().T)

    df_ic = pd.concat(df_ic, ignore_index=True)
    return df_ic

def generate_meta_from_label_tile(df_region: pd.DataFrame, example_tile: str):
    crss = df_region['crs'].unique()
    
    with rasterio.open(example_tile) as lab:
        label_crs = lab.crs
        label_bounds = lab.bounds
    
    lab_poly = box(*label_bounds)
    df_ic = []
    for crs in crss:
        df_crs = df_region.loc[df_region['crs'] == crs]
        img_crs = CRS.from_epsg(int(crs))
        idx_crs = label_crs

        # 2) Reproject NLCD CRS (index CRS) to query polygon if needed
        if CRS.from_user_input(idx_crs) != img_crs:
            tfm = Transformer.from_crs(idx_crs, img_crs, always_xy=True).transform
            k_poly = shp_transform(tfm, lab_poly)
        else:
            k_poly = lab_poly
        
        for _, row in tqdm(df_crs.iterrows(), total=len(df_crs)):
            img_bounds = row['bounds']
            q_poly = box(*img_bounds)
            
            if k_poly.contains(q_poly):
                df_ic.append(row.to_frame().T)
    df_ic = pd.concat(df_ic, ignore_index=True)
    return df_ic

def main(args):
    df_region = pd.read_parquet(f"{args.data_root}/EO1H/{args.region}_metadata.parquet")
    df_region = df_region.sort_values(by=['crs']).reset_index(drop=True)
    
    data_file = f"{args.data_root}/{args.dataset_name}"
    if args.subset_name is not None:
        data_file = f"{data_file}/{args.subset_name}"
    logger.info(f"Data file: {data_file}")
    
    try:
        tile_paths = glob.glob(f"{data_file}/{args.label_keyword}.tif")
        assert len(tile_paths) > 0
    except:
        tile_paths = glob.glob(f"{data_file}/**/{args.label_keyword}.tif")
        assert len(tile_paths) > 0
        
    tile_paths.sort()
        
    if args.single_tile:
        example_tile = tile_paths[0]
        logger.info(f"Example tile: {example_tile}")
        df_ic = generate_meta_from_label_tile(df_region, example_tile)
    else:
        logger.info(f"Generating metadata from {len(tile_paths)} label tiles")
        df_ic = generate_meta_from_label_patches(df_region, tile_paths, args.dataset_name, args.allow_multiple_hits)
    
    df_ic = df_ic.rename(columns={'crs': 'crs_epsg'})
    if args.dataset_name == "GFC":
        save_path = f"{args.data_root}/{args.dataset_name}/{args.dataset_name}_{args.region}_landcover.parquet"
    else:
        save_path = f"{args.data_root}/{args.dataset_name}/{args.dataset_name}_landcover.parquet"
    df_ic.to_parquet(save_path)
    unique_location_uids = df_ic['location_uid'].unique()
    logger.info(f"Filtered metadata: {len(unique_location_uids)}")
    logger.info(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/haozhesi/EO1H-313K/data/")
    parser.add_argument("--single_tile", action="store_true", help="Use a single tile to generate metadata")
    parser.add_argument("--dataset_name", "-d", type=str, required=True)
    parser.add_argument("--subset_name", "-s", type=str, default=None)
    parser.add_argument("--label_keyword", "-l", type=str, default="*")
    parser.add_argument("--region", "-r", type=str, required=True)
    parser.add_argument("--allow_multiple_hits", "-m", action="store_true", help="Allow multiple hits for a single image")
    args = parser.parse_args()
    main(args)

