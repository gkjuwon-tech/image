"""Temporal Trajectory Transformer Decoder Backbone (~16.83M Parameters).

Employs learnable knot queries and multi-head cross-attention over directorial conditioning
memory to synthesize continuous camera control knot features.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from config import ModelConfig


class TrajectoryDecoderLayer(nn.Module):
    """Single layer of temporal knot decoder with self-attention, cross-attention, and FFN."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.backbone_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.backbone_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.backbone_ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.backbone_ffn_dim, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.norm3 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        norm_x = self.norm2(x)
        cross_out, _ = self.cross_attn(norm_x, memory, memory, key_padding_mask=memory_mask)
        x = x + cross_out
        
        norm_x = self.norm3(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        return x


class TrajectoryTransformerBackbone(nn.Module):
    """Multi-layer temporal transformer synthesizing knot embeddings for Lie-spline predictor."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_knots = config.num_knots
        self.d_model = config.d_model
        
        self.knot_queries = nn.Parameter(torch.randn(1, config.num_knots, config.d_model) * 0.02)
        self.layers = nn.ModuleList([
            TrajectoryDecoderLayer(config) for _ in range(config.backbone_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B = memory.shape[0]
        x = self.knot_queries.expand(B, -1, -1)
        
        for layer in self.layers:
            x = layer(x, memory, memory_mask=memory_mask)
            
        return self.final_norm(x)
