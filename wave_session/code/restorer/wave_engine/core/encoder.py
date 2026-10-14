"""
Micro-Wavefront Coordinate Encoder (Model 1: ~3.5M Parameters)
=============================================================
Maps 3D spatial coordinates and low-rank tensor features into
continuous complex wave parameters: amplitude A, optical phase phi, and density sigma.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    """Fourier frequency positional encoding."""
    def __init__(self, in_dim: int = 3, num_freqs: int = 10):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        self.out_dim = in_dim * (2 * num_freqs + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * np.pi * x))
            out.append(torch.cos(freq * np.pi * x))
        return torch.cat(out, dim=-1)


class MicroWaveEncoder(nn.Module):
    """
    Micro-Wavefront Coordinate Encoder.
    
    Predicts continuous wave amplitude A, optical phase phi, and density sigma.
    """
    def __init__(
        self,
        in_spatial_dim: int = 64,  # From factorizer
        hidden_dim: int = 256,
        num_layers: int = 6
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(in_dim=3, num_freqs=8)  # 3 * 17 = 51 dims
        total_in_dim = in_spatial_dim + self.pos_enc.out_dim

        # Input projection
        self.input_layer = nn.Linear(total_in_dim, hidden_dim)

        # Residual backbone layers (~2.8M parameters)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output heads
        self.density_head = nn.Linear(hidden_dim, 1)
        self.amplitude_head = nn.Linear(hidden_dim, 3)
        self.phase_head = nn.Linear(hidden_dim, 3)
        self.feature_head = nn.Linear(hidden_dim, 32)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize phase head weights small
        nn.init.uniform_(self.phase_head.weight, -0.01, 0.01)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        coords: torch.Tensor,
        factorizer_features: torch.Tensor
    ) -> dict:
        """
        Forward pass.
        coords: (B, N_pts, 3) in [-1, 1]
        factorizer_features: (B, N_pts, 64)
        """
        enc_coords = self.pos_enc(coords)
        x = torch.cat([factorizer_features, enc_coords], dim=-1)
        h = F.silu(self.input_layer(x))

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            residual = h
            h = F.silu(norm(layer(h)))
            if i % 2 == 1:
                h = h + residual

        density = F.softplus(self.density_head(h) + 0.5) * 2.0
        amplitude = F.softplus(self.amplitude_head(h))
        phase = torch.pi * torch.tanh(self.phase_head(h))  # Optical phase [-pi, pi]
        features = self.feature_head(h)

        return {
            "density": density,        # (B, N_pts, 1)
            "amplitude": amplitude,    # (B, N_pts, 3)
            "phase": phase,            # (B, N_pts, 3)
            "features": features       # (B, N_pts, 32)
        }
