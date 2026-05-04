from GFM_Baselines.wrappers.satmae_wrapper import SatMAEEncoder, SatMAEConfig
from GFM_Baselines.wrappers.lessvit_wrapper import LESSViTEncoder, LESSViTConfig
from GFM_Baselines.wrappers.specvit_wrapper import SpecViTEncoder, SpecViTConfig
from GFM_Baselines.wrappers.dinov3_wrapper import DINOv3Encoder, DINOv3Config
from GFM_Baselines.wrappers.dofa_wrapper import DOFAEncoder, DOFAConfig
from GFM_Baselines.wrappers.spatsigma_wrapper import SpatSigmaClsEncoder, SpatSigmaSegEncoder, SpatSigmaConfig

ENCODER_CONFIGS = {
    "lessvit": LESSViTConfig,
    "satmae": SatMAEConfig,
    "specvit": SpecViTConfig,
    "dinov3": DINOv3Config,
    "dofa": DOFAConfig,
    "spatsigma": SpatSigmaConfig,
}

ENCODER_MODELS = {
    "lessvit": LESSViTEncoder,
    "satmae": SatMAEEncoder,
    "specvit": SpecViTEncoder,
    "dinov3": DINOv3Encoder,
    "dofa": DOFAEncoder,
    "spatsigma_cls": SpatSigmaClsEncoder,
    "spatsigma_seg": SpatSigmaSegEncoder,
}