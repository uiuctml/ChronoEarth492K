import torch.nn as nn
import numpy as np
import torch
from transformers import PreTrainedModel
from .spatial_spectral_low_rank_vit import SpatialSpectralLowRankViTEncoder, SpatialSpectralLowRankViTDecoder, SpatialSpectralLowRankViTConfig, SpatialViTDecoder
from typing import Dict, Union, Any
import logging
import random
logger = logging.getLogger(__name__)

class SpatialSpectralMAEViT(PreTrainedModel):
    config_class = SpatialSpectralLowRankViTConfig
    main_input_name = ['optical']
    def __init__(self, config):
        super().__init__(config)
        self.encoder = SpatialSpectralLowRankViTEncoder(config)
        self.decoder = SpatialSpectralLowRankViTDecoder(config)
        # self.decoder = SpatialViTDecoder(config)

    def _forward(self, optical, radar, optical_channel_wv, radar_channel_wv, mask_ratio=None, channel_mask_ratio=None, spatial_resolution=10, prefix=''):
        latent, channel_mask, channel_ids_restore, pos_mask, pos_ids_restore = self.encoder(optical=optical, radar=radar, optical_channel_wv=optical_channel_wv, radar_channel_wv=radar_channel_wv, 
                                                                       spatial_resolution=spatial_resolution, mask_ratio=mask_ratio, channel_mask_ratio=channel_mask_ratio)
        recon, hidden_states = self.decoder(latent, pos_ids_restore, channel_ids_restore, optical_channel_wv, spatial_resolution, restore_input_dim=False)

        # return dict
        return_dict= {f'{prefix}_channel_mask': channel_mask, 
                      f'{prefix}_recon': recon, 
                      f'{prefix}_pos_mask': pos_mask,
                      f'{prefix}_hidden_states': hidden_states
                      }
        return return_dict
    
    def forward(self, optical, optical_channel_wv, mask_ratio=None, channel_mask_ratio=None, spatial_resolution=10, modal=None):
        if self.config.channel_dropout is not None and self.training:
            assert len(self.config.channel_dropout) in [1, 2], f"channel_dropout should be a float or a list of two floats, but got {self.config.channel_dropout}"
            for i in range(len(self.config.channel_dropout)):
                assert 0.0 <= self.config.channel_dropout[i] < 1.0, f"channel_dropout should be between 0 and 1, but got {self.config.channel_dropout}"
            channel_dropout = sorted(self.config.channel_dropout)
            start = 0
            idx_to_keep = []
            num_channel_groups_new = []
            for num_channel in self.config.num_channel_groups:
                # C = max(1, int(num_channel * (1 - self.config.channel_dropout))) # number of channels to keep
                C_min = max(1, int(num_channel * (1 - channel_dropout[1]))) if len(channel_dropout) == 2 else C
                C_max = max(1, int(num_channel * (1 - channel_dropout[0]))) if len(channel_dropout) == 2 else C
                C = random.randint(C_min, C_max)
                to_keep = sorted(random.sample(range(num_channel), C))

                num_channel_groups_new.append(C)
                idx_to_keep.extend([i + start for i in to_keep])
                start += num_channel
            optical = optical[:, idx_to_keep, :, :]
            optical_channel_wv = optical_channel_wv[:, idx_to_keep]
            if self.config.num_channel_groups != num_channel_groups_new:
                self.encoder.num_channel_groups = num_channel_groups_new
            

        assert modal in ['optical', None]
        target = self.decoder.forward_target(optical)
        return_dict = dict(target=target)
        n_optical_channels = optical_channel_wv.shape[1]

        if modal is None or modal == 'optical':
            optical_dict = self._forward(optical, None, optical_channel_wv, None, mask_ratio, channel_mask_ratio, spatial_resolution, prefix='optical')
            return_dict.update(optical_dict)
        return return_dict

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        if isinstance(self.main_input_name, list):
            tokens = 0
            for main_input_name in self.main_input_name:
                if main_input_name in input_dict:
                    tokens += input_dict[main_input_name].numel()
            return tokens
        elif self.main_input_name in input_dict:
            return input_dict[self.main_input_name].numel()
        elif "estimate_tokens" not in self.warnings_issued:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            self.warnings_issued["estimate_tokens"] = True
        return 0