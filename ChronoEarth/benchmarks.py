from torch.utils.data import Dataset
from typing import List
import os
import pandas as pd
import rasterio
import numpy as np
from scipy.ndimage import binary_fill_holes
from huggingface_hub import hf_hub_download
import tarfile
import torch
from datasets import load_dataset
import numpy as np
from ChronoEarth.ChronoEarth import channel_metadata, ALL_BANDS, RGB_BANDS

WV_MAX = 2365.20
WV_MIN = 447.17

# EO1H-313K data statistics
NUM_CHANNELS = {
    "VNIR": 48,
    "SWIR1": 17,
    "SWIR2": 19,
    "SWIR3": 31,
    "SWIR4": 40,
}

SWIR1_META = channel_metadata['SWIR1']
SWIR2_META = channel_metadata['SWIR2']
SWIR3_META = channel_metadata['SWIR3']
SWIR4_META = channel_metadata['SWIR4']
VNIR_META = channel_metadata['VNIR']

ALL_BANDS_MEAN = VNIR_META['mean'] + SWIR1_META['mean'] + SWIR2_META['mean'] + SWIR3_META['mean'] + SWIR4_META['mean']
ALL_BANDS_STD = VNIR_META['std'] + SWIR1_META['std'] + SWIR2_META['std'] + SWIR3_META['std'] + SWIR4_META['std']

BANDS_TO_IDX = np.zeros(max(ALL_BANDS) + 1, dtype=np.int32)
for i, band in enumerate(ALL_BANDS):
    BANDS_TO_IDX[band] = i

BANDS = {
    "RGB": RGB_BANDS,
    "VNIR": VNIR_META['band_index'],
    "SWIR1": SWIR1_META['band_index'],
    "SWIR2": SWIR2_META['band_index'],
    "SWIR3": SWIR3_META['band_index'],
    "SWIR4": SWIR4_META['band_index'],
    "ALL": ALL_BANDS,
}

class StaticTask(Dataset):
    DATASET_NAME = "StaticTask"
    BAND_TEMPLATE = "_B{:03d}_L1T.TIF"

    channel_metadata = channel_metadata
    
    def __init__(self, dataset_name: str, split: str, metadata_path: str, image_dir: str, label_dir: str, BANDS: List[int] = RGB_BANDS, transform=None, **kwargs):
        self.metadata = pd.read_parquet(metadata_path)
        if 'CORINE' in dataset_name:
            assert split in ["train", "ood_space_test", "ood_all_test", "ood_temp_test", 'val', 'test', 'all'], "split must be one of train, ood_space_test, ood_all_test, ood_temp_test, val, test, or all"
        elif 'GFC' in dataset_name:
            assert split in ["train", "val", "test", "ood_test", "all"], "split must be one of train, val, test, ood_test, or all"
        else:
            assert split in ["train", "val", "test", "all"], "split must be one of train, val, test, or all"    
        if split == "all":
            self.metadata = self.metadata
        else:
            self.metadata = self.metadata.loc[self.metadata["split"] == split].reset_index(drop=True)
        self.BANDS = BANDS
        if not os.path.exists(label_dir):
            os.makedirs(image_dir, exist_ok=True)
            downloaded_file_path = hf_hub_download(
                repo_id=f"GFM-Bench/{dataset_name}", 
                filename=f"{dataset_name}.tar.gz", 
                repo_type="dataset", 
                cache_dir=label_dir
            )
            with tarfile.open(downloaded_file_path, "r:gz") as tf:
                tf.extractall(label_dir)
            print("Labels extracted to:", label_dir)

        self.image_dir = image_dir
        self.label_dir = label_dir

        self.transform = transform

        self._metadata = self._get_metadata(["VNIR", "SWIR1", "SWIR2", "SWIR3", "SWIR4"])
        BAND_IDX = [BANDS_TO_IDX[band] for band in BANDS]
        self._metadata = list(np.array(self._metadata)[BAND_IDX])
        
    def __len__(self):
        return len(self.metadata)
    
    def _load_image(self, image_dir: str, BANDS: List[int]):
        # print(image_dir)
        # image_path = [os.path.join(image_dir, image_dir.split('/')[-1]+ self.BAND_TEMPLATE.format(band)) for band in BANDS]
        image_path = os.path.join(image_dir, image_dir.split('/')[-1]+ '.TIF')
        # imgs = []
        with rasterio.open(image_path) as src:
            img = src.read()
            img = np.where(np.isnan(img), 0.0, img)
            # imgs.append(img)
        # img = np.concatenate(imgs, axis=0)
        return img
    
    def _load_label(self, label_path: str):
        with rasterio.open(label_path) as src:
            label = src.read()
        label = label - 1 # no data -> -1   
        label = np.where(np.isnan(label), 0, label).astype(np.uint8) # 0 -> 255
        return label

    def _get_metadata(self, channel_groups: List[str]):
        channel_wv = []
        for channel_group in channel_groups:
            channel_wv += self.channel_metadata[channel_group]["channel_wv"]

        return channel_wv

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]
        image_dir = os.path.join(self.image_dir, row['image_dir'])
        img = self._load_image(image_dir, self.BANDS)
        img = np.nan_to_num(img, nan=0.0)
        label_path = os.path.join(self.label_dir, row['label_path'])
        label = self._load_label(label_path)
        label = np.nan_to_num(label, nan=255)
        nondata_mask = img[0] != 0
        nondata_mask = binary_fill_holes(nondata_mask)
        # apply nondata mask to label
        label = np.where(nondata_mask, label, 255)

        # img, label, nondata_mask = torch.tensor(img), torch.tensor(label), torch.tensor(nondata_mask)
        if self.transform is not None:
            img, _ , label, spatial_resolution = self.transform(
                optical=img,
                radar=None,
                label=label,
                spatial_resolution=30,
            )
        data = {
            "optical": img,
            "label": label,
            "nondata_mask": nondata_mask,
            "optical_channel_wv": self._metadata,
            "spatial_resolution": 30,
        }
        return data
    
class ShortHorizonTemporalTask(StaticTask):
    DATASET_NAME = "ShortHorizonTemporalTask"
    
    def __init__(self, dataset_name: str, split: str, metadata_path: str, image_dir: str, label_dir: str, num_frames: int = 4, 
                 BANDS: List[int] = RGB_BANDS, transform=None, group_keyword: str = "label_path", frames_lb: int = 1):
        super().__init__(dataset_name, split, metadata_path, image_dir, label_dir, BANDS, transform)
        label_counts = self.metadata['label_path'].value_counts()
        dense_label_paths = label_counts[label_counts >= frames_lb].index
        self.metadata = self.metadata.loc[self.metadata['label_path'].isin(dense_label_paths)].reset_index(drop=True)
        self.num_frames = num_frames
        self.metadata_groups = list(self.metadata.groupby(group_keyword))
        self.group_keyword = group_keyword
        self.split = split
        
    def __len__(self):
        return len(self.metadata_groups)
    
    def __getitem__(self, idx):
        # row = self.metadata.iloc[idx]
        # given the label path, retrive the key and df from the metadata_groups
        # keyword = row[self.group_keyword]
        # df = self.metadata_groups.get_group(keyword).sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        # given the label path, retrieve the df from the metadata_groups
        keyword, df = self.metadata_groups[idx]
        df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        img_dirs = df['image_dir'].tolist()
        img_dirs = [os.path.join(self.image_dir, img_dir) for img_dir in img_dirs]
        img_dirs = np.array(img_dirs)
        label_paths = df['label_path'].tolist()[-1]
        label_path = os.path.join(self.label_dir, label_paths)
        timestamps = np.array(df['timestamp'].tolist())
        n_timestamps = len(timestamps)
        
        label = self._load_label(label_path)
        
        if self.num_frames > 0 and n_timestamps > self.num_frames:
            # keep index by equally spacing the timestamps
            # kept_idx = np.linspace(0, n_timestamps - 1, self.num_frames).astype(int)
            if self.split == "train":
                kept_idx = np.random.choice(n_timestamps, self.num_frames, replace=False)
            else:
                if self.num_frames == 1:
                    kept_idx = np.array([(n_timestamps - 1) // 2], dtype=int)
                else:
                    kept_idx = np.linspace(0, n_timestamps - 1, self.num_frames).astype(int)
            kept_idx = np.sort(kept_idx)
            img_dirs = img_dirs[kept_idx]
            timestamps = timestamps[kept_idx]
        
        imgs = []
        nondata_masks = None
        for img_dir in img_dirs:
            img = self._load_image(img_dir, self.BANDS)
            imgs.append(img)
            nondata_mask = img[0] != 0
            nondata_mask = binary_fill_holes(nondata_mask)
            if nondata_masks is None:
                nondata_masks = nondata_mask
            else:
                nondata_masks = nondata_masks | nondata_mask
        # apply nondata mask to label
        label = label * nondata_masks
        imgs = np.stack(imgs, axis=0)
        
        if self.transform is not None:
            imgs, _ , label, spatial_resolution = self.transform(
                optical=imgs,
                radar=None,
                label=label,
                spatial_resolution=30,
            )
        if imgs.ndim == 3:
            imgs = imgs[None, :, :, :]
    
        if self.num_frames > 0 and n_timestamps < self.num_frames:
            diff = self.num_frames - n_timestamps
            # add diff number of -1 to the beginning of timestamps
            pad = np.array([-1] * diff)
            timestamps = np.concatenate([pad, timestamps], axis=0)
            # pad the imgs with 0
            imgs = np.concatenate([np.zeros((diff, *imgs.shape[1:])), imgs], axis=0)
            imgs = imgs.astype(np.float32)
            
        data = {
            "optical": imgs,
            "label": label,
            "nondata_mask": nondata_masks,
            "optical_channel_wv": self._metadata,
            "spatial_resolution": 30,
            "timestamp": timestamps,
            "label_timestamp": df['label_timestamp'].iloc[-1],
        }
        return data
        
class LongHorizonTemporalTask(StaticTask):
    DATASET_NAME = "LongHorizonTemporalTask"
    
    def __init__(self, dataset_name: str, split: str, metadata_path: str, image_dir: str, label_dir: str, num_frames: int = 4, 
                 BANDS: List[int] = RGB_BANDS, transform=None, group_keyword: str = "location_uid", keep_last_n_frames: bool = True, frames_lb: int = 1):
        super().__init__(dataset_name, split, metadata_path, image_dir, label_dir, BANDS, transform)
        label_counts = self.metadata[group_keyword].value_counts()
        dense_label_paths = label_counts[label_counts >= frames_lb].index
        self.metadata = self.metadata.loc[self.metadata[group_keyword].isin(dense_label_paths)].reset_index(drop=True)
        self.num_frames = num_frames
        self.metadata_groups = list(self.metadata.groupby(group_keyword))
        self.group_keyword = group_keyword
        self.keep_last_n_frames = keep_last_n_frames
        self.split = split
        
    def __len__(self):
        return len(self.metadata_groups)
    
    def __getitem__(self, idx):
        keyword, df = self.metadata_groups[idx]
        df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        img_dirs = df['image_dir'].tolist()
        img_dirs = [os.path.join(self.image_dir, img_dir) for img_dir in img_dirs]
        img_dirs = np.array(img_dirs)
        label_paths = np.array(df['label_path'].tolist())
        timestamps = np.array(df['timestamp'].tolist())
        n_timestamps = len(timestamps)
        
        if self.num_frames > 0 and n_timestamps > self.num_frames:
            if self.keep_last_n_frames or self.split != "train": # keep last n frames for validation and test and for train if specified
                kept_idx = np.arange(n_timestamps - self.num_frames, n_timestamps)
            else:
                kept_idx = np.random.choice(n_timestamps, self.num_frames, replace=False)
                kept_idx = np.sort(kept_idx)
            img_dirs = img_dirs[kept_idx]
            label_paths = label_paths[kept_idx]
            timestamps = timestamps[kept_idx]
            
        label_path = os.path.join(self.label_dir, label_paths[-1])
        label = self._load_label(label_path)
        label_timestamp = df['label_timestamp'].iloc[-1]
            
        imgs = []
        nondata_masks = None
        for img_dir in img_dirs:
            img = self._load_image(img_dir, self.BANDS)
            imgs.append(img)
            nondata_mask = img[0] != 0
            nondata_mask = binary_fill_holes(nondata_mask)
            if nondata_masks is None:
                nondata_masks = nondata_mask
            else:
                nondata_masks = nondata_masks | nondata_mask
        # apply nondata mask to label 
        label = label * nondata_masks # Non-data regions are set to 0
        imgs = np.stack(imgs, axis=0)
        
        if self.transform is not None:
            imgs, _ , label, spatial_resolution = self.transform(
                optical=imgs,
                radar=None,
                label=label,
                spatial_resolution=30,
            )
        
        if self.num_frames > 0 and n_timestamps < self.num_frames:
            diff = self.num_frames - n_timestamps
            # add diff number of 0 to the beginning of timestamps
            pad = np.array([-1] * diff)
            timestamps = np.concatenate([pad, timestamps], axis=0)
            # pad the imgs with 0
            imgs = np.concatenate([np.zeros((diff, *imgs.shape[1:])), imgs], axis=0)
            imgs = imgs.astype(np.float32)
            
        data = {
            "optical": imgs,
            "label": label,
            "nondata_mask": nondata_masks,
            "optical_channel_wv": self._metadata,
            "spatial_resolution": 30,
            "timestamp": timestamps,
            "label_timestamp": label_timestamp,
        }
        return data
