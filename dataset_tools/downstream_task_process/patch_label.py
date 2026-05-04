import sys
import os

# # Add the project root to Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

import pandas as pd
from patching_utils import cut_label_to_meta_grid, target_grid_from_meta
import argparse
from tqdm import tqdm
from loguru import logger
import glob

DATASETS = ['BNETD', 'CLCD', 'CDL', 'CORINE']

YEARS= {
    'BNETD': [2020],
    "CLCD": range(2001, 2018),
    "CDL": range(2008, 2018),
    "CORINE": [2000, 2006, 2012, 2018],
    "ARG_LANDCOVER": [2013],
    "ARG_FOREST": [2000, 2007, 2013],
    "CHI_LANDCOVER": [2013],
    "CHI_FOREST": [2000, 2007, 2013],
    "FRCrop": [2018],
}

LABEL_PATH_TEMPLATE = {
    'BNETD': "BNETD_landcover.tif",
    'CLCD': "CLCD_v01_{year}.tif",
    'CDL': "{year}_30m_cdls/{year}_30m_cdls.tif",
    'CORINE': "clc{year}/DATA/*.tif",
    'ARG_LANDCOVER': "d2-lcm2013_arg.tif",
    'ARG_FOREST': "d2-fcm{year}_arg.tif",
    'CHI_LANDCOVER': "d2-lcm2013_chi.tif",
    'CHI_FOREST': "d2-fcm{year}_chi.tif",
    'FRCrop': "FR_{year}_EC21.tif",
}

# Default paths on Euler
PRODUCT = "BNETD"
DATA_ROOT = "/home/haozhesi/EO1H-313K/data"
SAVE_DIR = "{DATA_ROOT}/land_cover"
META_DATA_PATH = "{DATA_ROOT}/{PRODUCT}_landcover.parquet"

def main(args):
    metadata_path = META_DATA_PATH.format(DATA_ROOT=args.data_root, PRODUCT=args.product)
    logger.info(f"Metadata path: {metadata_path}")
    save_dir = SAVE_DIR.format(DATA_ROOT=args.data_root)
    logger.info(f"Save directory: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    df_ic = pd.read_parquet(metadata_path)
    df_ic = df_ic.drop_duplicates(subset=['location_uid'])
    for _, row in tqdm(df_ic.iterrows(), total=len(df_ic)):
        dst_crs, dst_T, W, H = target_grid_from_meta(row)
        location_uid = row['location_uid']
        out_dir = f"{save_dir}/{location_uid}"
        os.makedirs(out_dir, exist_ok=True)
        for year in YEARS[args.product]:
            out_path = os.path.join(out_dir, f'{args.product}_{year}.tif')
            try:
                label_path = LABEL_PATH_TEMPLATE[args.product].format(year=year)
            except:
                label_path = LABEL_PATH_TEMPLATE[args.product]
            label_path = os.path.join(args.data_root, label_path)
            label_path = glob.glob(label_path)[0]
            out, _, _ = cut_label_to_meta_grid(dst_crs, dst_T, W, H, label_path, out_path)
        if len(os.listdir(out_dir)) == 0:
            os.rmdir(out_dir)
    logger.info(f"Saved {len(os.listdir(save_dir))} patches")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/haozhesi/EO1H-313K/data")
    parser.add_argument("--product", "-p", type=str, required=True)
    args = parser.parse_args()
    args.data_root = os.path.join(args.data_root, args.product)
    main(args)