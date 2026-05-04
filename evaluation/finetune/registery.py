# Dataset-specific configurations
from dataclasses import dataclass
from typing import Optional

MODEL_NAMES_MAP = {
    "satmae": "SatMAE",
    "specvit": "SpectralViT",
    "dinov3": "DINOv3",
    "dofa": "DOFA",
    "spatsigma": "SpatSigma",
    "lessvit": "LESSViT",
}

STATIC_SEGMENTATION_MODELS = [
    {"model_name": "lessvit", "decoder_model": "upernet", "temporal_model": False, "temporal_embedding": False, "temporal_pooling": None},
    {"model_name": "satmae", "decoder_model": "upernet", "temporal_model": False, "temporal_embedding": False, "temporal_pooling": None},
    {"model_name": "specvit", "decoder_model": "upernet", "temporal_model": False, "temporal_embedding": False, "temporal_pooling": None},
    {"model_name": "dinov3", "decoder_model": "upernet", "temporal_model": False, "temporal_embedding": False, "temporal_pooling": None},
    {"model_name": "dofa", "decoder_model": "upernet", "temporal_model": False, "temporal_embedding": False, "temporal_pooling": None},
    {"model_name": "spatsigma", "decoder_model": None, "temporal_model": False, "temporal_embedding": False, "temporal_pooling": None},
]

STATIC_MULTILABEL_MODELS = [
    {"model_name": "lessvit", "decoder_model": "linear", "temporal_model": False, "temporal_embedding": False, "temporal_pooling": None},
    {"model_name": "satmae", "decoder_model": "linear", "temporal_model": False, "temporal_embedding": False, "temporal_pooling": None},
    {"model_name": "specvit", "decoder_model": "linear", "temporal_model": False, "temporal_embedding": False, "temporal_pooling": None},
    {"model_name": "dinov3", "decoder_model": "linear", "temporal_model": False, "temporal_embedding": False, "temporal_pooling": None},
    {"model_name": "dofa", "decoder_model": "linear", "temporal_model": False, "temporal_embedding": False, "temporal_pooling": None},
    {"model_name": "spatsigma", "decoder_model": None, "temporal_model": False, "temporal_embedding": False, "temporal_pooling": None},
]

TEMPORAL_SEGMENTATION_MODELS = [
    # stage-2 temporal pretraining pooling
    {"model_name": "lessvit", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "pretrain"},
    {"model_name": "satmae", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "pretrain"},
    {"model_name": "specvit", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "pretrain"},
    {"model_name": "dofa", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "pretrain"},
    # # attention pooling
    # {"model_name": "satmae", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "attention"},
    # {"model_name": "specvit", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "attention"},
    # {"model_name": "dinov3", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "attention"},
    # {"model_name": "dofa", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "attention"},
    # mean pooling
    # {"model_name": "satmae", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "mean"},
    # {"model_name": "specvit", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "mean"},
    # {"model_name": "dinov3", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "mean"},
    # {"model_name": "dofa", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "mean"},
    # max pooling
    {"model_name": "lessvit", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "max"},
    {"model_name": "satmae", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "max"},
    {"model_name": "specvit", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "max"},
    {"model_name": "dinov3", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "max"},
    {"model_name": "dofa", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "max"},
]

TEMPORAL_MULTILABEL_MODELS = [
    # stage-2 temporal pretraining pooling
    {"model_name": "lessvit", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "pretrain"},
    {"model_name": "dofa", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "pretrain"},
    {"model_name": "satmae", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "pretrain"},
    {"model_name": "specvit", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "pretrain"},
    # attention pooling
    # {"model_name": "satmae", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "attention"},
    # {"model_name": "specvit", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "attention"},
    # {"model_name": "dinov3", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "attention"},
    # {"model_name": "dofa", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "attention"},
    # mean pooling
    # {"model_name": "satmae", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "mean"},
    # {"model_name": "specvit", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "mean"},
    # {"model_name": "dinov3", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "mean"},
    # {"model_name": "dofa", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "mean"},
    # max pooling
    {"model_name": "lessvit", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "max"},
    {"model_name": "dinov3", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "max"},
    {"model_name": "dofa", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "max"},
    {"model_name": "satmae", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "max"},
    {"model_name": "specvit", "decoder_model": "linear", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "max"},
   
]

CHANGE_DETECTION_MODELS = [
    {"model_name": "lessvit", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "pretrain"},
    {"model_name": "satmae", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "pretrain"},
    {"model_name": "specvit", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "pretrain"},
    {"model_name": "dofa", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "pretrain"},
    {"model_name": "lessvit", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "diff"},
    {"model_name": "satmae", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "diff"},
    {"model_name": "specvit", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "diff"},
    {"model_name": "dinov3", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "diff"},
    {"model_name": "dofa", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "diff"},
    # {"model_name": "satmae", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "attention"},
    # {"model_name": "specvit", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "attention"},
    # {"model_name": "dinov3", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "attention"},
    # {"model_name": "dofa", "decoder_model": "upernet", "temporal_model": True, "temporal_embedding": True, "temporal_pooling": "attention"},
]

STATIC_TASKS = {
    "segmentation": ['CLCD', 'CDL', "NLCDLndCov",],
    "regression": ['NLCDFctImp'],
    "multilabel": ['CORINE', "ISDASoil"]
}

SHORT_HORIZON_SEGMENTATION_TASKS = {
    "segmentation": ['CLCD', 'CDL', "NLCDLndCov"],
    "multilabel": ["ISDASoil"]
}

LONG_HORIZON_SEGMENTATION_TASKS = {
    "segmentation": ['CLCD', 'CDL', "NLCDLndCov"],
}

CHANGE_DETECTION_TASKS = {
    "segmentation": ['GFC']
}

TEMPORAL_CONFIGS = {
    "S": STATIC_TASKS,
    "SH": SHORT_HORIZON_SEGMENTATION_TASKS,
    "LH": LONG_HORIZON_SEGMENTATION_TASKS,
    "CD": CHANGE_DETECTION_TASKS,
}

STATIC_MODEL_CONFIGS = {
    "segmentation": STATIC_SEGMENTATION_MODELS,
    "multilabel": STATIC_MULTILABEL_MODELS,
    "regression": STATIC_SEGMENTATION_MODELS,
}

TEMPORAL_MODEL_CONFIGS = {
    "segmentation": TEMPORAL_SEGMENTATION_MODELS,
    "multilabel": TEMPORAL_MULTILABEL_MODELS,
    "regression": TEMPORAL_SEGMENTATION_MODELS,
}

CHANGE_DETECTION_MODEL_CONFIGS = {
    "segmentation": CHANGE_DETECTION_MODELS,
}

MODEL_CONFIGS = {
    "S": STATIC_MODEL_CONFIGS,
    "SH": TEMPORAL_MODEL_CONFIGS,
    "LH": TEMPORAL_MODEL_CONFIGS,
    "CD": CHANGE_DETECTION_MODEL_CONFIGS,
}
