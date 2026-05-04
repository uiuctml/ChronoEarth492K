from ChronoEarth.benchmarks import *

ISDA_KEPT_LABELS = [0, 2, 3, 5, 6, 8, 10, 11]

class ISDASoilMixin:
    NUM_CLASSES = 8
    TASK_TYPE = "multilabel"
    LABEL_PATH = "ISDASoil/land_cover"
    IMAGE_DIR = "EO1H/AF"

class StaticISDASoil(ISDASoilMixin, StaticTask):
    DATASET_NAME = "StaticISDASoil"
    METADATA_PATH = "ISDASoil/ISDASoil_metadata_static.parquet"
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, use_fraction_labels: bool = True, use_segmentation_labels: bool = False, transform=None, **kwargs):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, BANDS, transform=transform)
        self.use_fraction_labels = use_fraction_labels
        self.use_segmentation_labels = use_segmentation_labels
        assert use_fraction_labels and use_segmentation_labels is False, "use_fraction_labels and use_segmentation_labels cannot be both True"
        assert use_fraction_labels or use_segmentation_labels is True, "use_fraction_labels and use_segmentation_labels cannot be both False"
        
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_dir = os.path.join(self.image_dir, row['image_dir'])
        img = self._load_image(image_dir, self.BANDS)
        
        if self.use_segmentation_labels:
            label_path = os.path.join(self.label_dir, row['label_path'])
            label = self._load_label(label_path)
            nondata_mask = img[0] != 0
            nondata_mask = binary_fill_holes(nondata_mask)
            # apply nondata mask to label
            label = label * nondata_mask
        else:
            label = None
        if self.use_fraction_labels: 
            frac_labels = row['fraction_labels'][1:] # first class is background
            frac_labels = np.array(frac_labels) > 0.001
            frac_labels = frac_labels[ISDA_KEPT_LABELS]
            frac_labels = frac_labels.astype(np.uint8)

        # img, label, nondata_mask = torch.tensor(img), torch.tensor(label), torch.tensor(nondata_mask)
        if self.transform is not None and self.use_segmentation_labels:
            img, _ , label, spatial_resolution = self.transform(
                optical=img,
                radar=None,
                label=label,
                spatial_resolution=30,
            )
        elif self.transform is not None and self.use_fraction_labels:
            img, _ , spatial_resolution = self.transform(
                optical=img,
                radar=None,
                spatial_resolution=30,
            )
        
        data = {
            "optical": img,
            "optical_channel_wv": self._metadata,
            "spatial_resolution": 30,
        }
        
        if self.use_segmentation_labels:
            data['label'] = label
            data['nondata_mask'] = nondata_mask
        if self.use_fraction_labels:
            data['label'] = frac_labels
        
        return data
    
class LongHorizonISDASOIL(ISDASoilMixin, LongHorizonTemporalTask):
    DATASET_NAME = "LongHorizonISDASOIL"
    METADATA_PATH = "ISDASoil/ISDASoil_metadata_lh.parquet"
    
    def __init__(self, data_root: str, image_root: str, split: str, BANDS: List[int] = RGB_BANDS, num_frames: int = 4, 
                 use_fraction_labels: bool = True, use_segmentation_labels: bool = False, transform=None, frames_lb: int = 1):
        metadata_path = os.path.join(data_root, self.METADATA_PATH)
        image_dir = os.path.join(image_root, self.IMAGE_DIR)
        label_dir = os.path.join(data_root, self.LABEL_PATH)
        super().__init__(self.DATASET_NAME, split, metadata_path, image_dir, label_dir, num_frames, BANDS, transform=transform, frames_lb=frames_lb, group_keyword="label_path")
        self.use_fraction_labels = use_fraction_labels
        self.use_segmentation_labels = use_segmentation_labels
        assert use_fraction_labels and use_segmentation_labels is False, "use_fraction_labels and use_segmentation_labels cannot be both True"
        assert use_fraction_labels or use_segmentation_labels is True, "use_fraction_labels and use_segmentation_labels cannot be both False"

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
        
        if self.use_segmentation_labels:
            label_path = os.path.join(self.label_dir, label_paths[-1])
            label = self._load_label(label_path)
        if self.use_fraction_labels:
            frac_labels = df['fraction_labels'].iloc[-1][1:] # first class is background
            frac_labels = np.array(frac_labels) > 0.001
            frac_labels = frac_labels[ISDA_KEPT_LABELS]
            frac_labels = frac_labels.astype(np.uint8)
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
        if self.use_segmentation_labels:
            label = label * nondata_masks
        else:
            label = None
        imgs = np.stack(imgs, axis=0)
        
        if self.transform is not None and self.use_segmentation_labels:
            imgs, _ , label, spatial_resolution = self.transform(
                optical=imgs,
                radar=None,
                label=label,
                spatial_resolution=30,
            )
        elif self.transform is not None and self.use_fraction_labels:
            imgs, _ , spatial_resolution = self.transform(
                optical=imgs,
                radar=None,
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
            "optical_channel_wv": self._metadata,
            "spatial_resolution": 30,
            "timestamp": timestamps,
            "label_timestamp": label_timestamp,
        }
        if self.use_segmentation_labels:
            data['label'] = label
            data['nondata_mask'] = nondata_masks
        if self.use_fraction_labels:
            data['label'] = frac_labels
        return data