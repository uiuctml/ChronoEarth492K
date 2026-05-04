from ChronoEarth.benchmarks import *
from functools import cached_property 

CDL_SUPER_CLASS = {
    0:  [0, 63, 64, 65, 81, 82, 83, 87, 88, 
         111, 112, 121, 122, 123, 124, 131, 
         141, 142, 143, 152, 176, 190, 195],   # Background and Non-cropland
    1:  [75],                                  # Almonds
    2:  [3],                                   # Rice
    3:  [5],                                   # Soybeans
    4:  [69],                                  # Grapes
    5:  [26],                                  # Dbl Crop WinWht/Soybeans
    6:  [37],                                  # Other Hay/Non Alfalfa
    7:  [24, 236, 238],                        # Winter Wheat
    8: [61],                                  # Fallow/Idle Cropland
    9: [23],                                  # Spring Wheat
    10: [36],                                  # Alfalfa
    11: [1, 225, 226, 228, 237, 241],          # Corn
    # 12 Other Crops
}

CDL_COLOR_MAP = {
    0: ("Background", "#000000"),
    1: ("Almonds", "#00a884"), #
    2: ("Rice", "#00a9e6"), 
    3: ("Soybeans", "#267300"), #
    4: ("Grapes", "#704489"), 
    5: ("Dbl Crop WinWht/Soybeans", "#737300"),
    6: ("Other Hay/Non Alfalfa", "#a5f58d"),
    7: ("Winter Wheat", "#a87000"), #
    8: ("Fallow/Idle Cropland", "#bfbf7a"), #
    9: ("Spring Wheat", "#d9b56c"), #
    10: ("Alfalfa", "#ffa8e3"),
    11: ("Corn", "#ffd400"), #
    12: ("Other Crops", "#e0a60f"), #
}


CDL_SUPER_CLASS_LH = {
    0:  [0, 63, 64, 65, 81, 82, 83, 87, 88, 
         111, 112, 121, 122, 123, 124, 131, 
         141, 142, 143, 152, 176, 190, 195],   # Background and Non-cropland
    1:  [3],                                   # Rice
    2:  [5],                                   # Soybeans
    3:  [69],                                  # Grapes
    4:  [26],                                  # Dbl Crop WinWht/Soybeans
    5:  [37],                                  # Other Hay/Non Alfalfa
    6:  [24, 236, 238],                        # Winter Wheat
    7: [61],                                  # Fallow/Idle Cropland
    8: [23],                                  # Spring Wheat
    9: [36],                                  # Alfalfa
    10: [1, 225, 226, 228, 237, 241],          # Corn
    # 11 Other Crops
}

CDL_COLOR_MAP_LH = {
    0: ("Background", "#000000"),
    1: ("Rice", "#00a9e6"), 
    2: ("Soybeans", "#267300"), #
    3: ("Grapes", "#704489"), 
    4: ("Dbl Crop WinWht/Soybeans", "#737300"),
    5: ("Other Hay/Non Alfalfa", "#a5f58d"),
    6: ("Winter Wheat", "#a87000"), #
    7: ("Fallow/Idle Cropland", "#bfbf7a"), #
    8: ("Spring Wheat", "#d9b56c"), #
    9: ("Alfalfa", "#ffa8e3"),
    10: ("Corn", "#ffd400"), #
    11: ("Other Crops", "#e0a60f"), #
}

CDL_LABEL_REPROJECT = {k: 12 for k in range(255)}
for k, v in CDL_SUPER_CLASS.items():
    for v in v:
        CDL_LABEL_REPROJECT[v] = k

CDL_LABEL_REPROJECT_LH = {k: 11 for k in range(255)}
for k, v in CDL_SUPER_CLASS_LH.items():
    for v in v:
        CDL_LABEL_REPROJECT_LH[v] = k

class CDLMixin:
    NUM_RAW_CLASSES = 255
    NUM_CLASSES = 12
    CDL_LABEL_PROJECTION_DICT = CDL_LABEL_REPROJECT
    TASK_TYPE = "segmentation"
    LABEL_PATH = "CDL/land_cover"
    IMAGE_DIR = "EO1H/NA"

    @cached_property
    def lut(self) -> np.ndarray:
        """Build once per instance; fast index map old_id -> new_id."""
        lut = np.zeros(self.NUM_RAW_CLASSES, dtype=np.uint8)
        for old_id, new_id in self.CDL_LABEL_PROJECTION_DICT.items():
            # Defensive: ignore out-of-range keys
            if 0 <= int(old_id) < self.NUM_RAW_CLASSES:
                lut[int(old_id)] = np.uint8(new_id)
        return lut

    def _load_label(self, label_path: str) -> np.ndarray:
        """Load single-band label and project via LUT."""
        with rasterio.open(label_path) as src:
            # CDL labels are single band
            label = src.read(1)
        label = label.astype(np.uint8, copy=False)
        label = self.lut[label]
        # Ensure uint8 index for LUT; unknowns fall to 0 due to LUT init
        label = label - 1 # no data -> -1   
        label = label.astype(np.uint8, copy=False) # 0 -> 255
        return label
    
class StaticCDL(CDLMixin, StaticTask):
    DATASET_NAME = "StaticCDL"
    METADATA_PATH = "CDL/CDL_metadata_static.parquet"
    COLOR_MAP = CDL_COLOR_MAP
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, transform=None, **kwargs):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, BANDS, transform)
    
class ShortHorizonCDL(CDLMixin, ShortHorizonTemporalTask):
    DATASET_NAME = "ShortHorizonCDL"
    METADATA_PATH = "CDL/CDL_metadata_static.parquet"
    COLOR_MAP = CDL_COLOR_MAP
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, num_frames: int = 4, frames_lb: int = 1, transform=None):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        # if frames_lb > 1:
            # self.CDL_LABEL_PROJECTION_DICT = CDL_LABEL_REPROJECT_LH
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, num_frames, BANDS, frames_lb=frames_lb, transform=transform)
        
class LongHorizonCDL(CDLMixin, LongHorizonTemporalTask):
    DATASET_NAME = "LongHorizonCDL"
    METADATA_PATH = "CDL/CDL_metadata_lh.parquet"
    CDL_LABEL_PROJECTION_DICT = CDL_LABEL_REPROJECT_LH
    NUM_CLASSES = 11
    COLOR_MAP = CDL_COLOR_MAP_LH
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, num_frames: int = 4, transform=None, frames_lb: int = 1):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, num_frames, BANDS, transform=transform, frames_lb=frames_lb)
