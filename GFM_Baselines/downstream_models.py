import math
import torch
import logging
import safetensors.torch
import glob
import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, List, Optional
from ChronoEarth import NUM_CHANNELS
from .upernet import UPerNet
from typing import Optional, Tuple, Union
from transformers import PretrainedConfig, PreTrainedModel
from .convhead import ConvHead
from .temporal_attn import CausalAttentionBlock, PMA
from .pos_embed import get_temporal_sincos_pos_embed
from functools import partial
from .registery import ENCODER_CONFIGS, ENCODER_MODELS
from temporal_pretrain.temporal_embed import TimestampEmbedding
from temporal_pretrain.temporal_mae import CausalTemporalBlock


logger = logging.getLogger(__name__)

def get_encoder(model_name, task_type=None, num_labels=None):
    assert model_name in ENCODER_CONFIGS, f"Model {model_name} not supported"
    config = ENCODER_CONFIGS[model_name](classes=num_labels)
    if model_name == "leastvit":
        raise NotImplementedError(f"Model {model_name} not implemented yet")
        # TODO: implement LESSViT encoder
    
    if model_name == "spatsigma":
        assert task_type is not None, "Task type is required for SpatSigma"
        if task_type in ["multilabel", "classification"]:
            model_name = "spatsigma_cls"
        elif task_type in ["segmentation", "regression"]:
            model_name = "spatsigma_seg"
        else:
            raise NotImplementedError(f"Task type {task_type} not supported for SpatSigma")
            
    encoder = ENCODER_MODELS[model_name](config=config)
    return encoder

def get_task_head(num_labels, embed_dim, task_type, image_size=128, patch_size=None, decoder_model="upernet"):
    if decoder_model is None:
        return None
    elif decoder_model == "linear" and task_type in ["classification", "multilabel"]:
        return LinearHead(embed_dim, num_labels)
    elif decoder_model == "upernet" and task_type in ["segmentation", "regression"]:
        return UPerNet(num_labels, image_size, embed_dim)
    elif decoder_model == "convhead" and task_type in ["segmentation", "regression"]:
        return ConvHead(embed_dim, num_labels, patch_size)
    else:
        raise NotImplementedError(f"Decoder model {decoder_model} not supported for task type {task_type}")

def resolve_checkpoint_file(path):
    if path is None:
        return None
    if os.path.isfile(path):
        return path
    for patterns in (
        ("model.safetensors", "pytorch_model.bin"),
        ("checkpoint-*/model.safetensors", "checkpoint-*/pytorch_model.bin"),
        ("*.pth",),
        ("checkpoint-*/*.pth",),
    ):
        candidates = []
        for pattern in patterns:
            candidates.extend(glob.glob(os.path.join(path, pattern)))
        if candidates:
            candidates.sort()
            return candidates[-1]
    return None

def load_checkpoint_state(path):
    checkpoint_file = resolve_checkpoint_file(path)
    if checkpoint_file is None:
        raise FileNotFoundError(f"No checkpoint file found under {path}")
    if checkpoint_file.endswith(".safetensors"):
        return safetensors.torch.load_file(checkpoint_file, device="cpu")
    state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
    return state_dict

def strip_prefix_state_dict(state_dict, prefix):
    prefix = prefix + "."
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

def yyyydoy_to_year_doy(timestamps):
    timestamps = timestamps.long()
    invalid = timestamps < 0
    year = (timestamps // 1000).masked_fill(invalid, -1)
    doy = (timestamps % 1000).masked_fill(invalid, -1)
    return torch.stack([year, doy], dim=-1)

class TemporalFusion(nn.Module):
    def __init__(self, temporal_pooling=None, embed_dim=768, use_temporal_embedding=False):
        super().__init__()
        self.temporal_pooling = temporal_pooling
        self.embed_dim = embed_dim
        if temporal_pooling == "attention":
            self.target_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
            self.attention = PMA(embed_dim, num_heads=12)
            self.use_temporal_embedding = use_temporal_embedding
        
    def forward(self, features, timestamps=None):
        # features: B, T, N, D 
        B, T, N, D = features.shape
        if self.temporal_pooling == "attention":
            padding_mask = timestamps.eq(-1) # B, T, True for padded positions
            # expand target token to be B, 1, N, D
            target_token = self.target_token.unsqueeze(0).repeat(B, 1, N, 1)
            features = torch.cat([features, target_token], dim=1) # B, T+1, N, D
            # apply temporal embedding to target token
            if self.use_temporal_embedding:
                print(f"Using temporal embedding for target token")
                temporal_embedding = get_temporal_sincos_pos_embed(self.embed_dim, timestamps)
                features = features + temporal_embedding 
            return self.attention(features, padding_mask=padding_mask)
        elif self.temporal_pooling == "mean":
            padding_mask = ~ timestamps.eq(-1)[:, :-1] # B, T
            features_mask = padding_mask.view(B, T, 1, 1).repeat(1, 1, N, D)
            features = features * features_mask # B, T, N, D
            pooled = features.sum(dim=1) # B, N, D
            return pooled / padding_mask.sum(dim=1, keepdim=True).unsqueeze(-1) # B, N, D
        elif self.temporal_pooling == "max":
            # padding mask for max pooling
            padding_mask = timestamps.eq(-1)[:, :-1] # B, T
            features = features.masked_fill(padding_mask.view(B, T, 1, 1).repeat(1, 1, N, D), -float('inf'))
            return features.max(dim=1)[0]
        elif self.temporal_pooling == "diff": # for change detection
            return (features[:, -1, :] - features[:, 0, :]).abs()
        elif self.temporal_pooling is None:
            return features
        else:
            raise NotImplementedError(f"Temporal pooling method {self.temporal_pooling} not implemented")

class TaskModelConfig(PretrainedConfig):
    model_type = "encoder_with_task_head"

    def __init__(
        self,
        model_name: str = "lessvit",
        task_type: str = "classification",
        num_labels: int = 2,
        image_size: int = 128,
        embed_dim: int = 768,
        # decoder config
        decoder_model: str = "upernet",
        patch_size: int = 4, # only used for convhead
        # temporal config
        temporal_model: bool = False,
        temporal_pooling: str = None,
        temporal_embedding: bool = False,
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.decoder_model = decoder_model
        self.patch_size = patch_size
        self.temporal_model = temporal_model
        self.temporal_pooling = temporal_pooling
        self.temporal_embedding = temporal_embedding

class LinearHead(nn.Module):
    def __init__(self, embed_dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_labels)
        
    def forward(self, features, labels=None):
        if len(features.shape) == 2:
            logits = self.classifier(features)
        else:
            logits = self.classifier(features[:, 0, :])
            
        return logits

class TaskModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self._gradient_checkpointing_enabled = False
        self.num_labels = config.num_labels
        self.encoder = get_encoder(config.model_name, config.task_type, config.num_labels)
        self.decoder = get_task_head(config.num_labels, config.embed_dim, config.task_type, 
                                     config.image_size, config.patch_size, config.decoder_model)
        
        self.temporal_pooling = TemporalFusion(config.temporal_pooling, config.embed_dim)
        if config.temporal_embedding and config.temporal_model:
            self.get_temporal_embedding = partial(get_temporal_sincos_pos_embed, embed_dim=config.embed_dim)
        else:
            self.get_temporal_embedding = None
            
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the encoder.
        
        This method allows the model to support Trainer's gradient_checkpointing parameter.
        It enables checkpointing on the encoder's transformer blocks if available.
        """
        if self._gradient_checkpointing_enabled:
            return  # Already enabled
        
        if hasattr(self, 'encoder'):
            try:
                # First, try to enable on encoder blocks directly (for SatMAE, SpecViT, etc.)
                # This is more reliable than calling gradient_checkpointing_enable which may not exist
                
                # For SatMAE models (MaskedAutoencoderGroupChannelViT) with blocks
                if hasattr(self.encoder, 'blocks') and isinstance(self.encoder.blocks, nn.ModuleList):
                    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
                    for i, block in enumerate(self.encoder.blocks):
                        self.encoder.blocks[i] = checkpoint_wrapper(block)
                    self._gradient_checkpointing_enabled = True
                    import logging
                    logging.info("Enabled gradient checkpointing on encoder blocks (SatMAE) using checkpoint_wrapper")
                    return
                # For SpecViT models with vit_core (timm VisionTransformer)
                elif hasattr(self.encoder, 'vit_core') and hasattr(self.encoder.vit_core, 'blocks'):
                    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
                    for i, block in enumerate(self.encoder.vit_core.blocks):
                        self.encoder.vit_core.blocks[i] = checkpoint_wrapper(block)
                    self._gradient_checkpointing_enabled = True
                    import logging
                    logging.info("Enabled gradient checkpointing on encoder vit_core blocks using checkpoint_wrapper")
                    return
                # For timm VisionTransformer directly (fallback)
                elif hasattr(self.encoder, 'blocks'):
                    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
                    for i, block in enumerate(self.encoder.blocks):
                        self.encoder.blocks[i] = checkpoint_wrapper(block)
                    self._gradient_checkpointing_enabled = True
                    import logging
                    logging.info("Enabled gradient checkpointing on encoder blocks using checkpoint_wrapper")
                    return
                # Try to enable on encoder directly (last resort)
                elif hasattr(self.encoder, 'gradient_checkpointing_enable'):
                    self.encoder.gradient_checkpointing_enable()
                    self._gradient_checkpointing_enabled = True
                    import logging
                    logging.info("Enabled gradient checkpointing on encoder")
                    return
            except Exception as e:
                import logging
                logging.warning(f"Could not enable gradient checkpointing on encoder: {e}")
        
        # If we reach here, checkpointing couldn't be enabled
        # Instead of raising an error, just log a warning and continue
        # This allows training to proceed even if checkpointing is not available
        import logging
        logging.warning(f"Gradient checkpointing requested but {self.__class__.__name__} does not support it. Continuing without checkpointing.")
        # Mark as enabled to prevent repeated warnings, even though it's not actually enabled
        self._gradient_checkpointing_enabled = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing_enabled = False
        # Note: We can't easily unwrap checkpoint_wrapper, so we just mark it as disabled
    
    def load_pretrained_encoder(self, pretrained_model_dir):
        logger.info(f"Loading pretrained encoder from {pretrained_model_dir}")
        self.encoder.load_pretrained_weights(pretrained_model_dir)
        
    def forward(
        self, 
        optical=None, radar=None, optical_channel_wv=None, radar_channel_wv=None, spatial_resolution=30, labels=None, nondata_mask=None, timestamps=None, label_timestamps=None
    ) -> Union[Tuple, dict]:
        
        wave_list = (optical_channel_wv.squeeze(dim=0) / 1000).cpu().tolist()
        
        if self.decoder is None:
            if self.config.temporal_model:
                raise NotImplementedError("Temporal Adaption is not compatible with decoder-less models")
            else:
                assert optical is not None and len(optical.shape) == 4, "Static model requires 4D input"
                B, C, H, W = optical.shape
                logits = self.encoder.forward_encoder(optical, wave_list=wave_list)
        else:
            if self.config.temporal_model:
                assert optical is not None and len(optical.shape) == 5, "Temporal model requires 5D input"
                assert self.temporal_pooling is not None, "Temporal pooling method is required for temporal model"
                assert timestamps is not None and label_timestamps is not None, "Timestamps and label timestamps are required for temporal model"
                all_timestamps = torch.cat([timestamps, label_timestamps], dim=-1)

                B, T, C, H, W = optical.shape
                optical = optical.view(B*T, C, H, W)
                features = self.encoder.forward_encoder(optical, wave_list=wave_list, channel_wv=optical_channel_wv, spatial_resolution=spatial_resolution)
                features = features.view(B, T, *features.shape[1:])
                # Temporal pooling
                features = self.temporal_pooling(features, all_timestamps) # [B, T, N, D] -> [B, N, D]
            else:
                assert optical is not None and len(optical.shape) == 4, "Static model requires 4D input"
                B, C, H, W = optical.shape
                features = self.encoder.forward_encoder(optical, wave_list=wave_list, channel_wv=optical_channel_wv, spatial_resolution=spatial_resolution)
        
            assert len(features.shape) == 3, f"Expected features to be of shape (B, N, D), but got {features.shape}"
            assert (
                features.shape[-1] == self.config.embed_dim
            ), f"Expected embedding dimension {self.config.embed_dim}, but got {features.shape[2]}"
            
            if self.config.task_type in ["segmentation", "regression"]:
                features = features[:, 1:, :]
                # N must be a perfect square to form HxW grid
                assert int(math.sqrt(features.shape[1])) ** 2 == features.shape[1], (
                    f"Number of patches {features.shape[1]} is not a perfect square"
                )
                B, N, D = features.shape
                H = W = int(math.sqrt(N))
                assert H * W == N, "N is not a perfect square"
                features = features.permute(0, 2, 1).reshape(B, D, H, W)
                logits = self.decoder(features)
            else:
                features = features[:, 0, :] # B, D
                logits = self.decoder(features)
                
        # sanity check for logits

        if self.config.task_type == "segmentation" or self.config.task_type == "regression":
            assert logits.shape == (B, self.config.num_labels, self.config.image_size, self.config.image_size), "Not valid Segmentation Map"
        elif self.config.task_type == "classification" or self.config.task_type == "multilabel":
            assert logits.shape == (B, self.config.num_labels), "Not valid Classification Logits"
        else:
            raise NotImplementedError(f"Task type {self.config.task_type} not supported")
        if self.config.num_labels == 1:
            logits = logits.squeeze(dim=1)
        return {"logits": logits} if self.config.return_dict else logits 


class PretrainedTemporalTaskModel(PreTrainedModel):
    """
    Downstream model for stage-2 temporal-pretrained checkpoints.

    This path is intentionally separate from TemporalFusion: it loads the frozen
    single-frame encoder plus temporal_embed / temporal_blocks / temporal_norm
    directly from a temporal_pretrain checkpoint and uses the resulting target
    time representation for SH, LH, and CD heads.
    """
    config_class = TaskModelConfig

    def __init__(self, config):
        super().__init__(config)
        self._gradient_checkpointing_enabled = False
        self.num_labels = config.num_labels
        self.encoder = get_encoder(config.model_name, config.task_type, config.num_labels)
        self.decoder = get_task_head(
            config.num_labels,
            config.embed_dim,
            config.task_type,
            config.image_size,
            config.patch_size,
            config.decoder_model,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, config.embed_dim))
        self.temporal_embed = TimestampEmbedding(config.embed_dim, year_range=(2000, 2020))
        self.temporal_blocks = nn.ModuleList([
            CausalTemporalBlock(config.embed_dim, num_heads=12)
            for _ in range(4)
        ])
        self.temporal_norm = nn.LayerNorm(config.embed_dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def load_pretrained_encoder(self, pretrained_model_dir):
        logger.info(f"Loading temporal pretrained model from {pretrained_model_dir}")
        state_dict = load_checkpoint_state(pretrained_model_dir)

        encoder_state = strip_prefix_state_dict(state_dict, "encoder")
        if not encoder_state:
            raise RuntimeError(
                f"Checkpoint {pretrained_model_dir} does not contain encoder.* keys. "
                "temporal_pooling=pretrain expects a temporal_pretrain checkpoint."
            )
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if unexpected:
            logger.warning(f"Unexpected encoder keys while loading temporal pretrain checkpoint: {unexpected}")
        logger.info(f"Loaded temporal-pretrained encoder tensors: {len(encoder_state)}")

        temporal_prefixes = ("mask_token", "temporal_embed.", "temporal_blocks.", "temporal_norm.")
        temporal_state = {k: v for k, v in state_dict.items() if k.startswith(temporal_prefixes)}
        missing, unexpected = self.load_state_dict(temporal_state, strict=False)
        if unexpected:
            logger.warning(f"Unexpected temporal keys while loading temporal pretrain checkpoint: {unexpected}")
        missing_temporal = [
            k for k in missing
            if k.startswith(("mask_token", "temporal_embed", "temporal_blocks", "temporal_norm"))
        ]
        if missing_temporal:
            raise RuntimeError(f"Missing temporal pretrain keys: {missing_temporal}")

    def _apply_temporal_pretrain(self, features, timestamps, label_timestamps):
        assert timestamps is not None, "timestamps are required for temporal pretrained model"
        B, T, N, D = features.shape

        if label_timestamps is not None:
            target_token = self.mask_token.expand(B, 1, N, D)
            features = torch.cat([features, target_token], dim=1)
            frame_timestamps = torch.cat([timestamps, label_timestamps], dim=1)
            target_index = torch.full((B,), T, dtype=torch.long, device=features.device)
        else:
            frame_timestamps = timestamps
            valid = frame_timestamps.ne(-1)
            target_index = valid.long().sum(dim=1).clamp(min=1) - 1
            features = features.clone()
            features[torch.arange(B, device=features.device), target_index] = self.mask_token.expand(B, N, D)

        timestamp_pairs = yyyydoy_to_year_doy(frame_timestamps).to(features.device)
        features = features + self.temporal_embed(timestamp_pairs).unsqueeze(2)

        T_out = features.shape[1]
        x = features.permute(0, 2, 1, 3).reshape(B * N, T_out, D)
        for block in self.temporal_blocks:
            x = block(x)
        x = self.temporal_norm(x)
        x = x.reshape(B, N, T_out, D).permute(0, 2, 1, 3)
        return x[torch.arange(B, device=x.device), target_index]

    def _decode_features(self, features, batch_size):
        assert len(features.shape) == 3, f"Expected features to be of shape (B, N, D), but got {features.shape}"
        assert features.shape[-1] == self.config.embed_dim, (
            f"Expected embedding dimension {self.config.embed_dim}, got {features.shape[-1]}"
        )

        if self.config.task_type in ["segmentation", "regression"]:
            features = features[:, 1:, :]
            assert int(math.sqrt(features.shape[1])) ** 2 == features.shape[1], (
                f"Number of patches {features.shape[1]} is not a perfect square"
            )
            B, N, D = features.shape
            H = W = int(math.sqrt(N))
            features = features.permute(0, 2, 1).reshape(B, D, H, W)
            logits = self.decoder(features)
        else:
            logits = self.decoder(features[:, 0, :])

        if self.config.task_type == "segmentation" or self.config.task_type == "regression":
            assert logits.shape == (
                batch_size,
                self.config.num_labels,
                self.config.image_size,
                self.config.image_size,
            ), "Not valid Segmentation Map"
        elif self.config.task_type == "classification" or self.config.task_type == "multilabel":
            assert logits.shape == (batch_size, self.config.num_labels), "Not valid Classification Logits"
        else:
            raise NotImplementedError(f"Task type {self.config.task_type} not supported")

        if self.config.num_labels == 1:
            logits = logits.squeeze(dim=1)
        return logits

    def forward(
        self,
        optical=None,
        radar=None,
        optical_channel_wv=None,
        radar_channel_wv=None,
        spatial_resolution=30,
        labels=None,
        nondata_mask=None,
        timestamps=None,
        label_timestamps=None,
    ) -> Union[Tuple, dict]:
        assert optical is not None and len(optical.shape) == 5, "Temporal pretrained model requires 5D optical input"
        assert self.decoder is not None, "Temporal pretrained model requires a downstream decoder/head"

        wave_list = (optical_channel_wv.squeeze(dim=0) / 1000).cpu().tolist()
        B, T, C, H, W = optical.shape
        optical = optical.view(B * T, C, H, W)
        features = self.encoder.forward_encoder(
            optical,
            wave_list=wave_list,
            channel_wv=optical_channel_wv,
            spatial_resolution=spatial_resolution,
        )
        features = features.view(B, T, *features.shape[1:])
        features = self._apply_temporal_pretrain(features, timestamps, label_timestamps)
        logits = self._decode_features(features, B)
        return {"logits": logits} if self.config.return_dict else logits



# class ClassificationModel(TaskModel):
#     def __init__(self, config):
#         super().__init__(config)
#         # TODO: Initialize encoder based on config

# class SegmentationModel(BaselineWithTaskHead):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.encoder = get_encoder(self.config.model_name, self.config.task_type, self.config.num_labels)
#         # Align decoder with static BaselineWithUPerNet: use ConvHead / DINOHead
#         if self.config.decoder_model == "upernet":
#             self.decoder = UPerNet(
#                 num_classes=config.num_labels,
#                 image_size=config.image_size,
#                 debug=False,
#                 embed_dim=config.embed_dim
#             )
#         elif self.config.decoder_model == "convhead":
#             self.decoder = ConvHead(
#                 embedding_size=self.config.embed_dim,
#                 num_classes=self.config.num_labels,
#                 patch_size=self.config.patch_size,
#             ) # TODO: add options for ConvHead and UperNet
#         elif self.config.decoder_model == None:
#             self.decoder = None
#         else:
#             raise NotImplementedError(f"Decoder model {self.config.decoder_model} not supported")
        
#         if self.config.temporal_model:
#             self.temporal_pooling = self.config.temporal_pooling
#             self.temporal_embedding = self.config.temporal_embedding if self.config.temporal_embedding else False
#         total_params = sum(p.numel() for p in self.encoder.parameters())
#         print(f"Total parameters: {total_params}")
#         print(f"Temporal pooling method: {self.temporal_pooling}")

#     def apply_temporal_pooling(self, features, num_frames):
#         """Apply different temporal pooling methods for temporal features [B*T, N, D]"""
#         features = features.view(-1, num_frames, *features.shape[1:]) # [B*T, N, D] -> [B, T, N, D]
        
#         if self.temporal_pooling == 'mean':
#             return features.mean(dim=1)
#         elif self.temporal_pooling == 'max':
#             return features.max(dim=1)[0]
        
#         elif self.temporal_pooling == 'attention':
#             attention_weights = torch.softmax(features.mean(dim=2), dim=1)
#             attention_weights = attention_weights.unsqueeze(2)
#             pooled = (features * attention_weights).sum(dim=1)
#             return pooled
#         elif self.temporal_pooling == 'none':
#             features = features
#         else:
#             raise NotImplementedError(f"Temporal pooling method {self.temporal_pooling} not implemented")
    
#     return features
    
#     def forward(
#         self, 
#         optical=None, radar=None, optical_channel_wv=None, radar_channel_wv=None, spatial_resolution=30, labels=None, nondata_mask=None
#     ) -> Union[Tuple, dict]:
        
#         wave_list = (optical_channel_wv.squeeze(dim=0) / 1000).cpu().tolist()
        
#         if self.config.temporal_model:
#             assert optical is not None and len(optical.shape) == 5, "Temporal model requires 5D input"
#             B, T, C, H, W = optical.shape
#             optical = optical.view(B*T, C, H, W)
#             # TODO: Apply temporal embedding to the optical tensor
            
            
#             features = self.encoder.forward_encoder(optical, wave_list=wave_list)
#             features = self.apply_temporal_pooling(features, T)
#         else:
#             assert optical is not None and len(optical.shape) == 4, "Static model requires 4D input"
#             features = self.encoder.forward_encoder(optical, wave_list=wave_list)
        
#         if self.decoder is None:
#             assert features.shape == (B, self.config.num_labels, self.config.image_size, self.config.image_size), "Not valid Segmentation Map"
#             logits = features
#         else:
#             assert len(features.shape) == 3, f"Expected features to be of shape (B, N, D), but got {features.shape}"
#             assert (
#                 features.shape[-1] == self.config.embed_dim
#             ), f"Expected embedding dimension {self.config.embed_dim}, but got {features.shape[2]}"
#             # N must be a perfect square to form HxW grid
#             assert int(math.sqrt(features.shape[1])) ** 2 == features.shape[1], (
#                 f"Number of patches {features.shape[1]} is not a perfect square"
#             )
#             B, N, D = features.shape
#             H = W = int(math.sqrt(N))
#             assert H * W == N, "N is not a perfect square"
#             features = features.permute(0, 2, 1).reshape(B, D, H, W)
#             logits = self.decoder(features)

#         return {"logits": logits} if self.config.return_dict else logits
