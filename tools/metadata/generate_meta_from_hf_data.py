import json
import re
import os
from tqdm.auto import tqdm, trange
import numpy as np

import sys
# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import rasterio
import matplotlib.pyplot as plt
from rasterio.vrt import WarpedVRT
from rasterio.transform import xy
from rasterio.warp import transform as coord_transform

import pandas as pd
import glob

import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

import requests
import datetime
import threading
from loguru import logger

from EO1H313K import get_eo1h313k_metadata, EO1H313K_dev, NUM_CHANNELS, EO1H313K
from dataset_tools.nlcd_patching_utils import *
from rasterio.transform import Affine
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch):
    frame_ids, regions, location_uids, patch_ids, years, days, crs_epsgs, transforms, bounds, res, nodatas = [], [], [], [], [], [], [], [], [], [], []
    for example in batch:
        frame_id = example['frame_id']
        frame_ids.append(frame_id)
        region, location_uid, patch_id = frame_id.split('/') 
        regions.append(region)
        location_uids.append(location_uid)
        patch_ids.append(patch_id)
        date = patch_id[4:11]
        year, day = date[:4], date[4:]
        years.append(int(year))
        days.append(int(day))
        crs_epsgs.append(example['geo']['crs_epsg'])
        transforms.append(Affine(*example['geo']['transform'][:6]))
        bounds.append(example['geo']['bounds'])
        res.append(example['geo']['res'])
        nodatas.append(example['geo']['nodata'])
    return dict(frame_id=frame_ids, region=regions, location_uid=location_uids, patch_id=patch_ids,\
        year=years, day=days, crs_epsg=crs_epsgs, transform=transforms, bounds=bounds, res=res, nodata=nodatas)

data_root = "./data/EO1H-313K"

dataset = EO1H313K_dev(
        split="train",
        cache_dir=data_root,
        regions=['AC', 'AF', 'EA', 'EU', 'LA', 'NA', 'OC', 'SEA', 'SWA'],
        channel_groups=["VNIR"]
    )
dl = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=16)

df_metadata = pd.DataFrame()
for data in tqdm(dl):
    df_row = pd.DataFrame(data)
    df_metadata = pd.concat((df_metadata, df_row))
df_metadata = df_metadata.reset_index(drop=True)
df_metadata_ = df_metadata.copy()
df_metadata_['location_uid'] = df_metadata_['location_uid'].str[4:-8]
df_metadata_.sort_values(by=['location_uid', 'year', 'day'])

# save as parquet
df_metadata_.to_parquet(f"{data_root}/geo_metadata.parquet")