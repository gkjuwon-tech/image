"""Directorial Conditioning Encoder Module (~8.71M Parameters).

Encodes textual directorial commands and continuous camera trajectory control parameters
into rich conditioning tokens for the trajectory backbone.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from config import ModelConfig


class DirectorialConditioningEncoder(nn.Module):
    """Encodes directorial text prompts and continuous camera parameters into conditioning memory."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # 1. Text Token Embedding: 4096 vocab * 512 = ~2.10M params
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_text_len, config.d_model) * 0.02)
        
        # 2. Continuous Directorial Parameter Projection (16D -> 512D) (~0.27M params)
        self.directorial_mlp = nn.Sequential(
            nn.Linear(config.directorial_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model)
        )
        
        # 3. Directorial Text Transformer Encoder: 2 layers (~6.30M params)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.cond_heads,
            dim_feedforward=config.cond_ffn_dim,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.cond_layers
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        text_tokens: torch.Tensor,
        directorial_params: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, S = text_tokens.shape
        
        tok_emb = self.token_embedding(text_tokens)
        pos_emb = self.pos_embedding[:, :S, :]
        h_text = tok_emb + pos_emb
        
        h_text = self.transformer_encoder(h_text, src_key_padding_mask=text_mask)
        ctrl_token = self.directorial_mlp(directorial_params).unsqueeze(1)
        
        memory = torch.cat([ctrl_token, h_text], dim=1)
        return self.output_norm(memory)
