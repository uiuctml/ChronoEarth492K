from ChronoEarth.benchmarks import *
from functools import cached_property 

CLCD_CLASS_MAP = {
    0:  [0, 3, 9],   
    1:  [1],                                  
    2:  [2],                                 
    3:  [4],                                  
    4:  [5],                                 
    5:  [6],                                 
    6:  [7],                                 
    7:  [8],                                 
}

CLCD_COLOR_MAP = {
    0: ("Background",      "#000000"),
    1: ("Cropland",        "#CDB33B"),
    2: ("Forest",          "#009900"),
    3: ("Grassland",       "#91AF40"),
    4: ("Water",           "#4D70A3"),
    5: ("Snow/Ice",        "#D7CDCC"),
    6: ("Barren",          "#F7E084"),
    7: ("Built-up",        "#CC0013"),
}

CLCD_LABEL_REPROJECT = {
    0: 0,
    1: 1,
    2: 2,
    3: 0,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 0,
}

class CLCDMixin:
    NUM_CLASSES = 7
    NUM_RAW_CLASSES = 10
    TASK_TYPE = "segmentation"
    LABEL_PATH = "CLCD/land_cover"
    IMAGE_DIR = "EO1H/EA"
    CLCD_LABEL_PROJECTION_DICT = CLCD_LABEL_REPROJECT
    COLOR_MAP = CLCD_COLOR_MAP
    
    @cached_property
    def lut(self) -> np.ndarray:
        """Build once per instance; fast index map old_id -> new_id."""
        lut = np.zeros(self.NUM_RAW_CLASSES, dtype=np.uint8)
        for old_id, new_id in self.CLCD_LABEL_PROJECTION_DICT.items():
            # Defensive: ignore out-of-range keys
            if 0 <= int(old_id) < self.NUM_RAW_CLASSES:
                lut[int(old_id)] = np.uint8(new_id)
        return lut

    def _load_label(self, label_path: str) -> np.ndarray:
        """Load single-band label and project via LUT."""
        with rasterio.open(label_path) as src:
            # CDL labels are single band
            label = src.read(1)
        # Ensure uint8 index for LUT; unknowns fall to 0 due to LUT init
        label = label.astype(np.uint8, copy=False)
        label = self.lut[label]
        label = label - 1 # no data -> -1   
        label = label.astype(np.uint8, copy=False) # 0 -> 255
        return label

class StaticCLCD(CLCDMixin, StaticTask):
    DATASET_NAME = "StaticCLCD"
    METADATA_PATH = "CLCD/CLCD_metadata_static.parquet"
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, transform=None, **kwargs):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, BANDS, transform)
    
class ShortHorizonCLCD(CLCDMixin, ShortHorizonTemporalTask):
    DATASET_NAME = "ShortHorizonCLCD"
    METADATA_PATH = "CLCD/CLCD_metadata_sh.parquet"
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, num_frames: int = 4, transform=None, frames_lb: int = 1):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, num_frames, BANDS, transform, frames_lb=frames_lb)
    
class LongHorizonCLCD(CLCDMixin, LongHorizonTemporalTask):
    DATASET_NAME = "LongHorizonCLCD"
    METADATA_PATH = "CLCD/CLCD_metadata_lh.parquet"
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, num_frames: int = 4, transform=None, frames_lb: int = 1):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, num_frames, BANDS, transform, frames_lb=frames_lb)
