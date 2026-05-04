from GFM_Baselines.models.LESSViT.spatial_spectral_low_rank_vit import SpatialSpectralLowRankViTEncoder
from transformers import PretrainedConfig, PreTrainedModel
import glob
import os
from loguru import logger
from typing import Optional, Union, Tuple, List

class LESSViTConfig(PretrainedConfig):
    model_type = "lessvit"

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 768,
        channel_embed_dims_per_head: int = 2,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: Optional[float] = 1.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pos_chan_embed_residual: bool = True,
        return_dict: bool = False,
        use_perception_field_mask: bool = True,
        attention_radius: int = 640,
        use_rope_embed: bool = True,
        rope_embed_base: float = 100.0,
        rank: int = 1,
        num_channel_groups: Optional[List[int]] = None,
        channel_groups: Optional[List[int]] = None,
        num_patches: Optional[int] = 64,
        spatial_resolution: Optional[int] = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.channel_dim = channel_embed_dims_per_head * num_heads
        self.spatial_dim = embed_dim // self.channel_dim * num_heads  
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.drop_path_rate = drop_path_rate
        self.drop_path_uniform = drop_path_uniform
        self.init_values = init_values
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.num_tokens = 1
        self.return_dict = return_dict
        self.mask_ratio = 0
        self.channel_mask_ratio = 0
        self.pretrain = False
        self.rank = rank
        self.num_channel_groups = num_channel_groups
        self.channel_groups = channel_groups
        
        # Perception field mask
        self.use_perception_field_mask = use_perception_field_mask
        self.attention_radius = attention_radius
        self.num_patches = num_patches
        self.spatial_resolution = spatial_resolution
        
        # Positional channel embedding residual
        self.pos_chan_embed_residual = pos_chan_embed_residual
        
        # RoPe embedding
        self.use_rope_embed = use_rope_embed
        self.rope_embed_base = rope_embed_base

class LESSViTEncoder(PreTrainedModel):
    config_class = LESSViTConfig
    model_type = "lessvit"
    def __init__(self, config: LESSViTConfig):
        super().__init__(config=config)
        self.encoder = SpatialSpectralLowRankViTEncoder(config)
        
    def forward_encoder(self, x, **kwargs):
        # Get encoder outputsp
        channel_wv = kwargs.get('channel_wv', None)
        spatial_resolution = kwargs.get('spatial_resolution', 10)
        outputs = self.encoder(x, optical_channel_wv=channel_wv, spatial_resolution=spatial_resolution)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        else:
            outputs = outputs.last_hidden_state
            
        return outputs[:, 0, :]
    
    def load_pretrained_weights(self, pretrained_model_dir):
        model_path = os.path.join(pretrained_model_dir, "model.safetensors")
        from safetensors import safe_open
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("encoder."):
                    # Get the corresponding key in target model
                    param = f.get_tensor(key)
                    self.state_dict()[key].copy_(param)
        # logger.info("Load pretrained LESSViT Encoder successfully!")

