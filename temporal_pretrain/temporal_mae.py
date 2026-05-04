import torch
import torch.nn as nn
import numpy as np
from functools import partial
from typing import Callable

from .temporal_embed import TimestampEmbedding


class CausalTemporalBlock(nn.Module):
    """
    Pre-norm transformer block with built-in causal attention mask.
    Input: (B, T, D) — attends along T with each position seeing only past tokens.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        # Causal mask: upper triangle = -inf so position t cannot attend to t+1..T-1
        causal = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=causal, need_weights=False)
        x = residual + x
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    """Pre-norm ViT decoder block over spatial patch tokens."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + x
        x = x + self.mlp(self.norm2(x))
        return x


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, patch_size**2 * C)"""
    B, C, H, W = x.shape
    h, w = H // patch_size, W // patch_size
    x = x.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, patch_size * patch_size * C)
    return x


class TemporalMAEWrapper(nn.Module):
    """
    Stage-2 temporal MAE wrapper for frozen single-frame EO encoders.

    Pipeline:
      1. Frozen encoder encodes each frame independently → (B, T, N+1, D)
      2. Timestamp embedding added to each frame's tokens          → inject temporal context
      3. The last valid frame is masked (mask_token replaces tokens)
      4. Temporal transformer blocks attend across T for each spatial position
      5. The last frame is decoded back to pixel patches via a shallow ViT decoder
      6. MSE loss on non-zero (valid) patches

    Compatible with DOFA, SatMAE, SpectralViT — any encoder whose
    forward_encoder() returns (B, N+1, D) patch features.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encode_fn: Callable,            # model-specific: (x: B*T,C,H,W) → (B*T, N+1, D)
        channel_wv: torch.Tensor,       # (1, C) wavelengths — registered as buffer
        embed_dim: int = 768,
        num_patches: int = 64,          # (img_size // patch_size) ** 2
        patch_size: int = 16,
        in_chans: int = 155,
        n_temporal_layers: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        max_frames: int = 32,
        temporal_mask_ratio: float = 0.5,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 2,
        decoder_num_heads: int = 8,
        decoder_mlp_ratio: float = 4.0,
        norm_pix_loss: bool = True,
        year_range: tuple = (2000, 2020),
    ):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("channel_wv", channel_wv)  # moves with model.to(device)
        self.encode_fn = encode_fn
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.temporal_mask_ratio = temporal_mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # Freeze the stage-1 encoder entirely and put it in eval mode so
        # models with training-time-only behaviour (e.g. LESSViT masking) are
        # consistent during encoding.
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # --- Temporal embedding (trainable) ---
        # Added to each frame's features after frozen encoding.
        # Encodes actual acquisition year + day-of-year so the model can learn
        # seasonal patterns and multi-year change dynamics.
        self.temporal_embed = TimestampEmbedding(embed_dim, year_range=year_range)

        # Mask token replaces all (N+1) tokens of a masked frame
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # --- Temporal transformer blocks (trainable) ---
        # For each spatial position i in [0, N], we attend across all T frames.
        # Reshape (B, T, N+1, D) -> (B*(N+1), T, D), apply blocks, reshape back.
        self.temporal_blocks = nn.ModuleList([
            CausalTemporalBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(n_temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(embed_dim)

        # --- Shallow ViT decoder over spatial patch tokens ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        for layer in [self.decoder_embed, self.decoder_pred]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_frames(self, optical: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        """
        optical: (B, T, C, H, W)
        returns: (B, T, N+1, D)

        Processes encoder in chunks of `chunk` frames to avoid stalling
        on very large batches (e.g. 256 samples × 2 frames = 512 at once).
        """
        self.encoder.eval()
        B, T, C, H, W = optical.shape
        x_flat = optical.reshape(B * T, C, H, W)
        feats = torch.cat([
            self.encode_fn(x_flat[i:i + chunk])
            for i in range(0, B * T, chunk)
        ], dim=0)                                   # (B*T, N+1, D)
        return feats.reshape(B, T, feats.shape[1], feats.shape[2])

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def _last_frame_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Mask exactly the last valid frame per sample.

        Causal design: the model always predicts later observations from earlier
        context — matching real-world deployment where only past frames are available.

        valid_mask: (B, T) True = real frame, False = padding
        returns:    (B, T) True = masked frame
        """
        B, T = valid_mask.shape
        device = valid_mask.device
        temporal_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            valid_idx = valid_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_idx) > 0:
                temporal_mask[b, valid_idx[-1]] = True
        return temporal_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        optical: torch.Tensor,              # (B, T, C, H, W)
        timestamps: torch.Tensor,           # (B, T, 2): year and day-of-year
        valid_mask: torch.Tensor,           # (B, T) True = real frame
        **kwargs,
    ) -> dict:
        B, T, C, H, W = optical.shape
        frame_feats = self._encode_frames(optical)                  # (B, T, N+1, D)

        # 2. Inject timestamp embeddings
        #    temp_emb: (B, T, D) — broadcast over the N+1 spatial positions
        temp_emb = self.temporal_embed(timestamps)              # B, T, D
        frame_feats = frame_feats + temp_emb.unsqueeze(2)       # B, T, N+1, D

        # 3. Causal temporal masking — always mask only the last valid frame
        temporal_mask = self._last_frame_mask(valid_mask)  # B, T

        mask_exp = temporal_mask[:, :, None, None].expand_as(frame_feats)
        mask_tokens = self.mask_token.expand(B, T, frame_feats.shape[2], self.embed_dim)
        frame_feats = torch.where(mask_exp, mask_tokens, frame_feats)  # B, T, N+1, D

        # 4. Temporal attention: reshape to (B*(N+1), T, D), causal mask inside block
        N1 = frame_feats.shape[2]
        x = frame_feats.permute(0, 2, 1, 3).reshape(B * N1, T, self.embed_dim)

        for blk in self.temporal_blocks:
            x = blk(x)
        x = self.temporal_norm(x)

        # Reshape back: (B*(N+1), T, D) → (B, T, N+1, D)
        x = x.reshape(B, N1, T, self.embed_dim).permute(0, 2, 1, 3)

        # 5. Decode patch tokens of the last frame with a shallow ViT decoder
        x_masked     = x[temporal_mask]            # M, N+1, D
        patch_tokens = x_masked[:, 1:, :]          # M, N, D

        decoder_tokens = self.decoder_embed(patch_tokens) + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            decoder_tokens = blk(decoder_tokens)
        pred = self.decoder_pred(self.decoder_norm(decoder_tokens))  # M, N, patch_size**2 * in_chans

        # 6. Build patchified reconstruction target
        optical_flat = optical.reshape(B * T, C, H, W)[:, :self.in_chans]
        target = patchify(optical_flat, self.patch_size)    # B*T, N, p**2*C
        target = target.reshape(B, T, self.num_patches, -1)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        target_masked = target[temporal_mask]               # M, N, p**2*C

        # 7. MSE on valid (non-zero) patches only
        valid_patches = (target_masked.abs().sum(dim=-1) > 0)   # M, N
        n_valid = valid_patches.sum()
        if n_valid == 0:
            loss = pred.sum() * 0.0
        else:
            err  = ((pred - target_masked) ** 2).mean(dim=-1)
            loss = (err * valid_patches).sum() / n_valid

        return {
            "loss": loss,
            "temporal_mask": temporal_mask,          # (B, T)
            "n_masked_frames": temporal_mask.sum(),  # keep as tensor for DataParallel
        }
