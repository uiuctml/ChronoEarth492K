import sys
import os

# Add the project root to Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

import pandas as pd
import geopandas as gpd
from patching_utils import *
import argparse
from tqdm import tqdm
from loguru import logger

YEARS = {
    "ISDASoil": [2017],
    "TreeMap": [2016, 2020, 2022],
    "BDF": [2017],
    "EuroCrop": [2021],
    "NLCD": range(2001, 2018),
    "FRCrop": [2018],
    "GFC": [2017],
}

LABEL_TEMPLATE = {
    "ISDASoil": "ISDASOIL_img_0-{patch_id}.tif",
    "TreeMap": "USFS_TreeMap_landcover_{year}-{patch_id}.tif",
    "BDF": "{patch_id}",
    "EuroCrop": "{patch_id}.tif",
    "NLCD_FctImp": "fractional_impervious_surface/Annual_NLCD_{patch_id}_FctImp_CU_C1V1/Annual_NLCD_{patch_id}_FctImp_{year}_CU_C1V1.tif",
    "NLCD_LndCov": "land_cover/Annual_NLCD_{patch_id}_LndCov_CU_C1V1/Annual_NLCD_{patch_id}_LndCov_{year}_CU_C1V1.tif",
    "NLCD_LndChg": "land_cover_change/Annual_NLCD_{patch_id}_LndChg_CU_C1V1/Annual_NLCD_{patch_id}_LndChg_{year}_CU_C1V1.tif",
    "NLCD_SpcChg": "spectral_change_day_of_year/Annual_NLCD_{patch_id}_SpcChg_CU_C1V1/Annual_NLCD_{patch_id}_SpcChg_{year}_CU_C1V1.tif",
}

SAVE_DIR = "{DATA_ROOT}/land_cover"
META_DATA_PATH = "{DATA_ROOT}/{PRODUCT}_landcover.parquet"

def main(args):
    if args.product != "GFC":
        metadata_path = META_DATA_PATH.format(DATA_ROOT=args.data_root, PRODUCT=args.product)
    else:
        metadata_path = META_DATA_PATH.format(DATA_ROOT=args.data_root, PRODUCT=f"{args.product}_{args.sub_product}")
    df_ic = pd.read_parquet(metadata_path)
    df_ic = df_ic.drop_duplicates(subset=['location_uid', 'label_tile_patch_id']).reset_index(drop=True)
    logger.info(f"Metadata path: {metadata_path}")
    save_dir = SAVE_DIR.format(DATA_ROOT=args.data_root)
    save_dir = save_dir if args.sub_product is None else save_dir + f"/{args.sub_product}"
    logger.info(f"Save directory: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    label_root = args.data_root
    if args.product in ["FRCrop"]:
        label_root = os.path.join(label_root, "FRCrop_raw")
    elif args.product.startswith("NLCD"):
        label_root = os.path.join(label_root, "nlcd_raw")
    logger.info(f"Label root: {label_root}")
    
    if args.sub_product is not None:
        product = f"{args.product}_{args.sub_product}"
    else:
        product = args.product

    for _, row in tqdm(df_ic.iterrows(), total=len(df_ic)):
        dst_crs, dst_T, W, H = target_grid_from_meta(row)
        location_uid = row['location_uid']
        out_dir = f"{save_dir}/{location_uid}"
        os.makedirs(out_dir, exist_ok=True)
        for year in YEARS[args.product]:
            out_path = os.path.join(out_dir, f'{product}_{year}.tif')
            label_patch_id = row['label_tile_patch_id']
            try:
                label_path = LABEL_TEMPLATE[product].format(year=year, patch_id=label_patch_id)
            except:
                try:
                    label_path = LABEL_TEMPLATE[product].format(patch_id=label_patch_id)
                except:
                    label_path = label_patch_id
            label_path = os.path.join(label_root, label_path)
            out, _, _ = cut_label_to_meta_grid(dst_crs, dst_T, W, H, label_path, out_path)
        if len(os.listdir(out_dir)) == 0:
            os.rmdir(out_dir)
    logger.info(f"Saved {len(os.listdir(save_dir))} patches")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/haozhesi/EO1H-313K/data")
    parser.add_argument("--product", "-p", type=str, required=True)
    parser.add_argument("--sub_product", "-sp", type=str, default=None)
    parser.add_argument("--region", "-r", type=str, default=None)
    args = parser.parse_args()
    args.data_root = os.path.join(args.data_root, args.product)
    main(args)