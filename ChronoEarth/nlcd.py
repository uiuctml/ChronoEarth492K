from ChronoEarth.benchmarks import *
from functools import cached_property 

NLCD_LndCov_LABEL_REPROJECT = {11: 1, 12: 2, 21: 3, 22: 4, 23: 5, 24: 6,
                        31: 7, 41: 8, 42: 9, 43: 10, 52: 11,
                        71: 12, 81: 13, 82: 14,
                        90: 15, 95: 16}

NLCD_LndCov_COLOR_MAP = {
    0:  ("Background", "#000000"),
    1:  ("Open Water", "#466b9f"),
    2:  ("Perennial Ice/Snow", "#d1def8"),
    3:  ("Developed, Open Space", "#dec5c5"),
    4:  ("Developed, Low Intensity", "#d99282"),
    5:  ("Developed, Medium Intensity", "#eb0000"),
    6:  ("Developed, High Intensity", "#ab0000"),
    7:  ("Barren Land (Rock/Sand/Clay)", "#b3ac9f"),
    8:  ("Deciduous Forest", "#68ab5f"),
    9:  ("Evergreen Forest", "#1c5f2c"),
    10: ("Mixed Forest", "#b5c58f"),
    11: ("Shrub/Scrub", "#ccb879"),
    12: ("Grassland/Herbaceous", "#dfdfc2"),
    13: ("Pasture/Hay", "#dcd939"),
    14: ("Cultivated Crops", "#ab6c28"),
    15: ("Woody Wetlands", "#b8d9eb"),
    16: ("Emergent Herbaceous Wetlands", "#6c9fb8"),
}

NLCD_LndCov_LABEL_REVERSE_REPROJECT = {v: k for k, v in NLCD_LndCov_LABEL_REPROJECT.items()}

NLCD_LndCov_LABEL_REPROJECT.update({0: 0, 250: 0, 255: 0, 127: 0, 51: 0, 72: 0, 73: 0, 74: 0})
NLCD_LndCov_LABEL_REVERSE_REPROJECT.update({0: [0, 250, 255, 127, 51, 72, 73, 74]})

class NLCDLabelMixin:
    """Provide LUT construction and label loading using a projection dict."""
    NUM_RAW_CLASSES = 256
    NUM_CLASSES = 16
    NLCD_LndCov_LABEL_PROJECTION_DICT = NLCD_LndCov_LABEL_REPROJECT
    TASK_TYPE = "segmentation"
    COLOR_MAP = NLCD_LndCov_COLOR_MAP

    @cached_property
    def lut(self) -> np.ndarray:
        """Build once per instance; fast index map old_id -> new_id."""
        lut = np.zeros(self.NUM_RAW_CLASSES, dtype=np.uint8)
        for old_id, new_id in self.NLCD_LndCov_LABEL_PROJECTION_DICT.items():
            # Defensive: ignore out-of-range keys
            if 0 <= int(old_id) < self.NUM_RAW_CLASSES:
                lut[int(old_id)] = np.uint8(new_id)
        return lut

    def _load_label(self, label_path: str) -> np.ndarray:
        """Load single-band label and project via LUT."""
        with rasterio.open(label_path) as src:
            # NLCD labels are single band
            label = src.read(1)
        # Ensure uint8 index for LUT; unknowns fall to 0 due to LUT init
        label = label.astype(np.uint8, copy=False)
        label = self.lut[label]
        label = label - 1 # no data -> -1   
        label = label.astype(np.uint8, copy=False) # 0 -> 255
        return label
    
class NLCDRegressionMixin:
    TASK_TYPE = "regression"
    NUM_CLASSES = 1
    
    def _load_label(self, label_path: str) -> np.ndarray:
        with rasterio.open(label_path) as src:
            label = src.read(1)
        return label.astype(np.float32) / 100.0

class StaticNLCDLndCov(NLCDLabelMixin, StaticTask):
    DATASET_NAME = "StaticNLCDLndCov"
    LABEL_PATH = "NLCD/land_cover/LndCov"
    IMAGE_DIR = "EO1H/NA"
    METADATA_PATH = "NLCD/NLCD_LndCov_metadata_static.parquet"
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, transform=None, **kwargs):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, BANDS, transform=transform)
    
class ShortHorizonNLCDLndCov(NLCDLabelMixin, ShortHorizonTemporalTask):
    DATASET_NAME = "ShortHorizonNLCDLndCov"
    METADATA_PATH = "NLCD/NLCD_LndCov_metadata_sh.parquet"
    IMAGE_DIR = "EO1H/NA"
    LABEL_PATH = "NLCD/land_cover/LndCov"
    NUM_FRAMES = 8
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, num_frames: int = NUM_FRAMES, transform=None, frames_lb: int = 1):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, num_frames, BANDS, transform=transform, frames_lb=frames_lb)
    
class LongHorizonNLCDLndCov(NLCDLabelMixin, LongHorizonTemporalTask):
    DATASET_NAME = "LongHorizonNLCDLndCov"
    METADATA_PATH = "NLCD/NLCD_LndCov_metadata_lh.parquet"
    IMAGE_DIR = "EO1H/NA"
    LABEL_PATH = "NLCD/land_cover/LndCov"
    NUM_FRAMES = 8
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, num_frames: int = NUM_FRAMES, transform=None, frames_lb: int = 1):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, num_frames, BANDS, transform=transform, frames_lb=frames_lb)

class StaticNLCDFctImp(NLCDRegressionMixin, StaticTask):
    DATASET_NAME = "StaticNLCDFctImp"
    METADATA_PATH = "NLCD/NLCD_FctImp_metadata_static.parquet"
    IMAGE_DIR = "EO1H/NA"
    LABEL_PATH = "NLCD/land_cover/FctImp"
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, transform=None, **kwargs):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, BANDS, transform=transform)

class ShortHorizonNLCDFctImp(NLCDRegressionMixin, ShortHorizonTemporalTask):
    DATASET_NAME = "ShortHorizonNLCDFctImp"
    METADATA_PATH = "NLCD/NLCD_FctImp_metadata_sh.parquet"
    IMAGE_DIR = "EO1H/NA"
    LABEL_PATH = "NLCD/land_cover/FctImp"
    NUM_FRAMES = 8
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, num_frames: int = NUM_FRAMES, transform=None, frames_lb: int = 1):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, num_frames, BANDS, transform=transform, frames_lb=frames_lb)
    
class LongHorizonNLCDFctImp(NLCDRegressionMixin, LongHorizonTemporalTask):
    DATASET_NAME = "LongHorizonNLCDFctImp"
    METADATA_PATH = "NLCD/NLCD_FctImp_metadata_lh.parquet"
    IMAGE_DIR = "EO1H/NA"
    LABEL_PATH = "NLCD/land_cover/FctImp"
    NUM_FRAMES = 8
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, num_frames: int = NUM_FRAMES, transform=None, frames_lb: int = 1):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, num_frames, BANDS, transform=transform, frames_lb=frames_lb)