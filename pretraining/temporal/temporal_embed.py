import torch
import torch.nn as nn
import math


class TimestampEmbedding(nn.Module):
    """
    Encodes acquisition timestamps into a D-dimensional embedding.

    Each timestamp is represented as (year, day-of-year). We embed these
    separately with sinusoidal functions and combine them, so the model
    can learn seasonal patterns (DOY) and multi-year trends (year).

    Usage:
        embed = TimestampEmbedding(embed_dim=768)
        # timestamps: (B, T, 2) where [..., 0]=year, [..., 1]=day-of-year
        out = embed(timestamps)  # (B, T, D)
    """

    def __init__(self, embed_dim: int, year_range: tuple = (2000, 2020), max_doy: int = 366):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        self.embed_dim = embed_dim
        self.year_min = year_range[0]
        self.year_max = year_range[1]
        self.max_doy = max_doy

        half = embed_dim // 2

        # Sinusoidal frequencies — fixed, not learned
        freq = torch.arange(0, half, 2).float()
        self.register_buffer("year_freq", freq)
        self.register_buffer("doy_freq", freq)

        # Learnable projection to mix year and DOY embeddings
        self.proj = nn.Linear(embed_dim, embed_dim)

    def _sincos(self, x: torch.Tensor, freq: torch.Tensor, period: float) -> torch.Tensor:
        """x: (...), freq: (F,) → (..., 2F)"""
        x = x.unsqueeze(-1).float()                      # ..., 1
        angles = x * (2 * math.pi / period) / (10000 ** (freq / freq.shape[0]))
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # ..., 2F

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        timestamps: (B, T, 2) — [..., 0] = year (e.g. 2008), [..., 1] = day-of-year (1-366)
                    Padding frames have year=-1 and are handled safely by clamping.
        returns: (B, T, embed_dim)
        """
        year = timestamps[..., 0].float().clamp(min=self.year_min)
        doy = timestamps[..., 1].float().clamp(min=1)

        # Normalize year to [0, 1] relative to dataset range
        year_norm = (year - self.year_min) / max(self.year_max - self.year_min, 1)

        half = self.embed_dim // 2
        year_emb = self._sincos(year_norm, self.year_freq, period=1.0)   # B, T, half
        doy_emb = self._sincos(doy, self.doy_freq, period=self.max_doy)  # B, T, half

        emb = torch.cat([year_emb, doy_emb], dim=-1)  # B, T, embed_dim
        return self.proj(emb)
