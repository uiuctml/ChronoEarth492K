from ChronoEarth.benchmarks import *

class_sets = {
        19: [
            'Urban fabric',
            'Industrial or commercial units',
            'Arable land',
            'Permanent crops',
            'Pastures',
            'Complex cultivation patterns',
            'Agriculture and Vegetation',
            'Agro-forestry areas',
            'Broad-leaved forest',
            'Coniferous forest',
            'Mixed forest',
            'Natural grassland and sparsely vegetated areas',
            'Moors, heathland and sclerophyllous vegetation',
            'Transitional woodland, shrub',
            'Beaches, dunes, sands',
            'Inland wetlands',
            'Coastal wetlands',
            'Inland waters',
            'Marine waters',
        ],
        44: [
            'Continuous urban fabric',
            'Discontinuous urban fabric',
            'Industrial or commercial units',
            'Road and rail networks and associated land',
            'Port areas',
            'Airports',
            'Mineral extraction sites',
            'Dump sites',
            'Construction sites',
            'Green urban areas',
            'Sport and leisure facilities',
            'Non-irrigated arable land',
            'Permanently irrigated land',
            'Rice fields',
            'Vineyards',
            'Fruit trees and berry plantations',
            'Olive groves',
            'Pastures',
            'Annual crops associated with permanent crops',
            'Complex cultivation patterns',
            'Land principally occupied by agriculture, with significant areas of natural vegetation',
            'Agro-forestry areas',
            'Broad-leaved forest',
            'Coniferous forest',
            'Mixed forest',
            'Natural grassland',
            'Moors and heathland',
            'Sclerophyllous vegetation',
            'Transitional woodland/shrub',
            'Beaches, dunes, sands',
            'Bare rock',
            'Sparsely vegetated areas',
            'Burnt areas',
            "Glacier and perpetual snow",
            'Inland marshes',
            'Peatbogs',
            'Salt marshes',
            'Salines',
            'Intertidal flats',
            'Water courses',
            'Water bodies',
            'Coastal lagoons',
            'Estuaries',
            'Sea and ocean',
        ],
    }

CORINE_LABEL_REPROJECT = {
        1: 0,
        2: 0,
        3: 1,
        12: 2,
        13: 2,
        14: 2,
        15: 3,
        16: 3,
        17: 3,
        19: 3,
        18: 4,
        20: 5,
        21: 6,
        22: 7,
        23: 8,
        24: 9,
        25: 10,
        26: 11,
        32: 11,
        27: 12,
        28: 12,
        29: 13,
        30: 14,
        35: 15,
        36: 15,
        37: 16,
        38: 16,
        40: 17,
        41: 17,
        42: 18,
        43: 18,
        44: 18,
        128: 255
    }

class CORINEMixin:
    NUM_CLASSES = 19
    TASK_TYPE = "multilabel"
    LABEL_PATH = "CORINE/land_cover"
    IMAGE_DIR = "EO1H/EU"

class StaticCORINE(CORINEMixin, StaticTask):
    DATASET_NAME = "StaticCORINE"
    METADATA_PATH = "CORINE/CORINE_metadata_static_v2.parquet"
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, transform=None, frames_lb: int = 1, **kwargs):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, BANDS, transform, frames_lb=frames_lb)
        
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_dir = os.path.join(self.image_dir, row['image_dir'])
        img = self._load_image(image_dir, self.BANDS)
        label = row['label'].astype(np.uint8)
        label_timestamp = row['label_timestamp']
        if self.transform is not None:
            img, _, spatial_resolution = self.transform(
                optical=img,
                radar=None,
                spatial_resolution=30,
            )

        data = {
            "optical": img,
            "label": label,
            "optical_channel_wv": self._metadata,
            "spatial_resolution": 30,
            "label_timestamp": label_timestamp,
        }
        return data
    
class LongHorizonCORINE(CORINEMixin, LongHorizonTemporalTask):
    DATASET_NAME = "LongHorizonCORINE"
    METADATA_PATH = "CORINE/CORINE_metadata_lh_v2.parquet"
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, num_frames: int = 4, transform=None, frames_lb: int = 1):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, num_frames, BANDS, transform, frames_lb=frames_lb, group_keyword="label_path")
    
    def __getitem__(self, idx):
        keyword, df = self.metadata_groups[idx]
        df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        img_dirs = df['image_dir'].tolist()
        img_dirs = [os.path.join(self.image_dir, img_dir) for img_dir in img_dirs]
        img_dirs = np.array(img_dirs)
        label = df['label'].iloc[-1]
        label_timestamp = df['label_timestamp'].iloc[-1]
        timestamps = np.array(df['timestamp'].tolist())
        n_timestamps = len(timestamps)
        
        imgs = []
        for img_dir in img_dirs:
            img = self._load_image(img_dir, self.BANDS)
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)
        
        if self.transform is not None:
            imgs, _, spatial_resolution = self.transform(
                optical=imgs,
                radar=None,
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
            
        elif self.num_frames > 0 and n_timestamps > self.num_frames:
            if self.keep_last_n_frames or self.split != "train": # keep last n frames for validation and test and for train if specified
                kept_idx = np.arange(n_timestamps - self.num_frames, n_timestamps)
            else:
                kept_idx = np.random.choice(n_timestamps, self.num_frames, replace=False)   
                kept_idx = np.sort(kept_idx)
            imgs = imgs[kept_idx]
            timestamps = np.array(timestamps)[kept_idx]
            
        data = {
            "optical": imgs,
            "label": label,
            "optical_channel_wv": self._metadata,
            "spatial_resolution": 30,
            "timestamp": timestamps,
            "label_timestamp": label_timestamp,
        }
        return data