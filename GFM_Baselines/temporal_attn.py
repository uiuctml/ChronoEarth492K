import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from timm.layers import use_fused_attn

class CausalAttentionBlock(nn.Module):
    """Self-attention with both causal masking and optional padding mask."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.norm = LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: B T N D, padding_mask: B T (True for padded positions)
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        padding_mask = padding_mask.unsqueeze(1).repeat(1, N, 1)
        padding_mask = padding_mask.reshape(B * N, T)
        
        qkv = self.qkv(self.norm(x)).reshape(B * N, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: B, num_heads, T, head_dim

        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask[:, None, None, :].to(torch.bool)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).reshape(B, N, T, -1).permute(0, 2, 1, 3)
        last_frame = out[:, -1, :, :]
        return self.proj(last_frame)

class PMA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        proj_drop: float = 0.,
    ) -> None:
        """
        Pooling Multi-Head Attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add bias to query, key, value projections.
            qk_norm (bool): If True, apply normalization to query and key.
            attn_drop (float): Dropout rate for attention weights.
            proj_drop (float): Dropout rate for output.
            norm_layer (nn.Module): Normalization layer to use.

        This method will use the cls token to pool the feature.
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5  # Scaling factor for attention scores
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: B, T, N, D, padding_mask: B, T, True for padded positions
        """
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
            
        x_cls = x[:, -2:-1, :] # B*N, 1, D

        q = self.q(x_cls).reshape(B * N, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B*N, num_heads, 1, head_dim
        kv = self.kv(x).reshape(B * N, T, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # 2, B*N, num_heads, T, head_dim
        k, v = kv.unbind(0) # B*N, num_heads, T, head_dim
        q, k = self.q_norm(q), self.k_norm(k)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale # B*N, num_heads, 1, T
        # apply padding mask
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).repeat(1, N, 1).reshape(B * N, T)
            attn = attn.masked_fill(padding_mask[:, None, None, :], float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2) # B*N, num_heads, 1, D
            
        x = x.reshape(B, N, self.dim) # B N dim, take only cls token
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x