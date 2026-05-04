from GFM_Baselines.models.SatMAE.mae_group_channels import MaskedAutoencoderGroupChannelViT, MaskedAutoencoderGroupChannelViTConfig
from ChronoEarth import NUM_CHANNELS
import numpy as np
import os
import glob
import torch
from loguru import logger

grouped_bands = []
channel_count = 0
for group in ["VNIR", "SWIR1", "SWIR2", "SWIR3", "SWIR4"]:
    n_channels = NUM_CHANNELS[group]
    grouped_bands.append(tuple(np.arange(channel_count, channel_count + n_channels).tolist()))
    channel_count += n_channels

class SatMAEConfig(MaskedAutoencoderGroupChannelViTConfig):
    def __init__(self, 
        img_size=128,
        in_chans=channel_count,
        channel_groups=tuple(grouped_bands),
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_pix_loss=False,
        **kwargs):
        super().__init__(img_size=img_size, 
                         in_chans=in_chans, 
                         channel_groups=channel_groups, 
                         embed_dim=embed_dim, 
                         depth=depth, 
                         num_heads=num_heads, 
                         decoder_embed_dim=decoder_embed_dim, 
                         decoder_depth=decoder_depth, 
                         decoder_num_heads=decoder_num_heads, 
                         mlp_ratio=mlp_ratio, 
                         norm_pix_loss=norm_pix_loss, 
                         **kwargs)

class SatMAEEncoder(MaskedAutoencoderGroupChannelViT):
    config_class = SatMAEConfig
    model_type = "satmae"
    def __init__(self, config: SatMAEConfig):
        super().__init__(config=config)
        # delete decoder related modules
        del self.decoder_embed
        del self.decoder_pos_embed
        del self.decoder_channel_embed
        del self.decoder_blocks
        del self.decoder_norm
        del self.decoder_pred
        del self.mask_token
        
    def forward_encoder(self, x, **kwargs):
        b = x.shape[0]
        G = len(self.channel_groups)
        latent = super().forward_encoder(x, mask_ratio=0.0)[0] # return the patch encodings
        D = latent.shape[-1]
        cls_token, patch_encodings = latent[:, :1, :], latent[:, 1:, :] # N, 1, D; N, GL, D 
        patch_encodings = patch_encodings.view(b, G, -1, D).mean(dim=1) # N, L, D
        features = torch.cat([cls_token, patch_encodings], dim=1) # N, L+1, D
        return features
      
    def load_pretrained_weights(self, pretrained_model_dir):
        from safetensors import safe_open
        model_path = glob.glob(os.path.join(pretrained_model_dir, "model.safetensors"))
        model_path.sort()
        model_path = model_path[-1]
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("decoder"):
                    continue
                if key.startswith("mask_token"):
                    continue
                if key.startswith("pos_embed"):
                    continue
                param = f.get_tensor(key)
                self.state_dict()[key].copy_(param)
        # logger.info("Load pretrained SatMAE Encoder successfully!")
