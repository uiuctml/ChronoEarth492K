from .ChronoEarth import ChronoEarth, channel_metadata, ALL_BANDS

# supported downstream datasets
from .clcd import StaticCLCD, ShortHorizonCLCD, LongHorizonCLCD
from .cdl import StaticCDL, ShortHorizonCDL, LongHorizonCDL
from .nlcd import StaticNLCDLndCov, ShortHorizonNLCDLndCov, LongHorizonNLCDLndCov
from .nlcd import StaticNLCDFctImp, ShortHorizonNLCDFctImp, LongHorizonNLCDFctImp
from .isdasoil import StaticISDASoil, LongHorizonISDASOIL
from .corine import StaticCORINE, LongHorizonCORINE
from .gfc import ChangeDetectionGFC

STATIC_DOWNSTREAM_DATASETS = {
    "CLCD": StaticCLCD,
    "CDL": StaticCDL,
    "NLCDLndCov": StaticNLCDLndCov,
    "NLCDFctImp": StaticNLCDFctImp,
    "ISDASoil": StaticISDASoil,
    "CORINE": StaticCORINE,
}

SHORT_HORIZON_TEMPORAL_DOWNSTREAM_DATASETS = {
    "CLCD": ShortHorizonCLCD,
    "CDL": ShortHorizonCDL,
    "NLCDLndCov": ShortHorizonNLCDLndCov,
    # "NLCDFctImp": ShortHorizonNLCDFctImp,
    "ISDASoil": LongHorizonISDASOIL,  # The task is actually a short horizon temporal task despite the name
    # "CORINE": LongHorizonCORINE, # The task is actually a short horizon temporal task despite the name
}

LONG_HORIZON_TEMPORAL_DOWNSTREAM_DATASETS = {
    "CLCD": LongHorizonCLCD,
    "CDL": LongHorizonCDL,
    "NLCDLndCov": LongHorizonNLCDLndCov,
    # "NLCDFctImp": LongHorizonNLCDFctImp,
}

CHANGE_DETECTION_DOWNSTREAM_DATASETS = {
    "GFC": ChangeDetectionGFC,
}

TEMPORAL_CONFIGS = {
    "S": STATIC_DOWNSTREAM_DATASETS,
    "SH": SHORT_HORIZON_TEMPORAL_DOWNSTREAM_DATASETS,
    "LH": LONG_HORIZON_TEMPORAL_DOWNSTREAM_DATASETS,
    "CD": CHANGE_DETECTION_DOWNSTREAM_DATASETS,
}

NUM_CLASSES = {
    "CLCD": StaticCLCD.NUM_CLASSES,
    "CDL": StaticCDL.NUM_CLASSES,
    "NLCDLndCov": StaticNLCDLndCov.NUM_CLASSES,
    "NLCDFctImp": StaticNLCDFctImp.NUM_CLASSES,
    "ISDASoil": StaticISDASoil.NUM_CLASSES,
    "CORINE": StaticCORINE.NUM_CLASSES,
    "GFC": ChangeDetectionGFC.NUM_CLASSES,
}

def get_chronoearth_metadata(channel_groups=['SWIR1', 'SWIR2', 'SWIR3', 'SWIR4', 'VNIR']):
    metadata = channel_metadata

    metadata_dict = {}
    for channel_group in channel_groups:
        metadata_dict['mean'] = metadata_dict.get('mean', []) + metadata[channel_group]['mean']
        metadata_dict['std'] = metadata_dict.get('std', []) + metadata[channel_group]['std']
        metadata_dict['channel_wv'] = metadata_dict.get('channel_wv', []) + metadata[channel_group]['channel_wv']

    return metadata_dict

def get_downstream_dataset(args, train_transform, eval_transform):
    if args.image_dir is None:
        args.image_dir = args.data_dir
    dataset_dict = {}
    if args.dataset_name == "CORINE":
        splits = ['train', 'ood_space_test', 'ood_all_test', 'ood_temp_test', 'val', 'test']
    elif args.dataset_name == "GFC":
        splits = ['train', 'val', 'test', 'ood_test']
    else:
        splits = ['train', 'val', 'test']
    assert args.dataset_name in TEMPORAL_CONFIGS[args.temporal_config], "Dataset not supported under this task type"
    for split in splits:
        transform = train_transform if split == "train" else eval_transform
        dataset_dict[split] = TEMPORAL_CONFIGS[args.temporal_config][args.dataset_name](
            data_root=args.data_dir,
            image_root=args.image_dir,
            split=split,
            BANDS=ALL_BANDS, # Default to all bands
            transform=transform,
            frames_lb = args.frames_fliter, 
            num_frames = args.num_frames
        )
    return dataset_dict