from ChronoEarth.benchmarks import *
from functools import cached_property 

class GFCMixin:
    """Provide LUT construction and label loading using a projection dict."""
    NUM_RAW_CLASSES = 25
    NUM_CLASSES = 1
    TASK_TYPE = "segmentation"

    @cached_property
    def lut(self) -> np.ndarray:
        """Build once per instance; fast index map old_id -> new_id."""
        lut = np.zeros(self.NUM_RAW_CLASSES, dtype=np.uint8)
        for i in range(18):
            lut[i] = i
        return lut

    def _load_label(self, label_path: str, year_start: int, year_end: int) -> np.ndarray:
        """Load single-band label and project via LUT."""
        with rasterio.open(label_path) as src:
            # NLCD labels are single band
            label = src.read(1)
        label = self.lut[label]
        year_label_after_start = label > year_start
        year_label_before_end = label < year_end
        year_label = year_label_after_start & year_label_before_end
        return year_label  
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        region = row['region']
        image_dir = os.path.join(self.image_dir, region)
        label_dir = os.path.join(self.label_dir, region)
        start_image_dir = os.path.join(image_dir, row['image_dir_start'])
        end_image_dir = os.path.join(image_dir, row['image_dir_end'])
        timestamp_start = row['timestamp_start']
        timestamp_end = row['timestamp_end']
        label_timestamp = row['label_timestamp']
        label_path = os.path.join(label_dir, row['label_path'])
        year_start = int(row['year_start']) - 2000
        year_end = int(row['year_end']) - 2000
        
        img_start = self._load_image(start_image_dir, self.BANDS)
        img_end = self._load_image(end_image_dir, self.BANDS)
        imgs = np.stack([img_start, img_end], axis=0)
        
        timestamps = np.array([timestamp_start, timestamp_end])
        
        nondata_mask = (img_start[0] != 0) & (img_end[0] != 0)
        nondata_mask = binary_fill_holes(nondata_mask)
        
        label = self._load_label(label_path, year_start, year_end)
        label = label * nondata_mask
        
        if self.transform is not None:
            imgs, _ , label, spatial_resolution = self.transform(
                optical=imgs,
                radar=None,
                label=label,
                spatial_resolution=30,
            )

        data = {
            "optical": imgs,
            "label": label.unsqueeze(0),
            "timestamp": timestamps,
            "label_timestamp": label_timestamp,
            "nondata_mask": nondata_mask,
            "optical_channel_wv": self._metadata,
            "spatial_resolution": 30,
        }
        return data  

class ChangeDetectionGFC(GFCMixin, StaticTask):
    DATASET_NAME = "ChangeDetectionGFC"
    LABEL_PATH = "GFC/land_cover"
    IMAGE_DIR = "EO1H"
    METADATA_PATH = "GFC/GFC_metadata_seasonal.parquet"
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, transform=None, **kwargs):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, BANDS, transform)