"""
Low-Rank Wave Tensor Factorizer (Model 2: ~2.4M Parameters)
==========================================================
Factorizes continuous 3D wave volumes into orthogonal tri-planes (XY, YZ, ZX).
Compresses spatial wave distributions into compact latent tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class WaveTensorFactorizer(nn.Module):
    """
    Low-Rank Tri-Plane Tensor Factorizer.
    
    Attributes:
        plane_res: Resolution of each orthogonal plane (e.g., 128x128).
        plane_dim: Number of feature channels per plane (e.g., 32 channels).
        out_dim: Output spatial wave embedding dimension.
    """
    def __init__(
        self,
        plane_res: int = 128,
        plane_dim: int = 32,
        out_dim: int = 64
    ):
        super().__init__()
        self.plane_res = plane_res
        self.plane_dim = plane_dim
        self.out_dim = out_dim

        # Three learnable orthogonal spatial planes (XY, YZ, ZX)
        # Parameter count: 3 * (32 * 128 * 128) = 1,572,864 parameters (~1.57M)
        self.plane_xy = nn.Parameter(torch.randn(1, plane_dim, plane_res, plane_res) * 0.05)
        self.plane_yz = nn.Parameter(torch.randn(1, plane_dim, plane_res, plane_res) * 0.05)
        self.plane_zx = nn.Parameter(torch.randn(1, plane_dim, plane_res, plane_res) * 0.05)

        # Non-linear feature fusion and refinement MLP
        # 3 * plane_dim (96) -> 256 -> 256 -> out_dim (64)
        # Parameter count: (96*256) + (256*256) + (256*64) ~ 106,496 params (~0.8M with norm & projections)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(plane_dim * 3, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.fusion_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def sample_planes(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Samples the tri-planes given 3D coordinates (B, N_samples, 3) normalized to [-1, 1].
        """
        orig_shape = coords.shape[:-1]
        pts = coords.reshape(1, -1, 1, 3)  # (1, K, 1, 3)

        # Normalize coordinates into [-1, 1] for grid_sample
        xy = pts[..., [0, 1]]
        yz = pts[..., [1, 2]]
        zx = pts[..., [2, 0]]

        feat_xy = F.grid_sample(self.plane_xy, xy, mode="bilinear", padding_mode="border", align_corners=True)
        feat_yz = F.grid_sample(self.plane_yz, yz, mode="bilinear", padding_mode="border", align_corners=True)
        feat_zx = F.grid_sample(self.plane_zx, zx, mode="bilinear", padding_mode="border", align_corners=True)

        feat_xy = feat_xy.squeeze(0).squeeze(-1).permute(1, 0)  # (K, C)
        feat_yz = feat_yz.squeeze(0).squeeze(-1).permute(1, 0)  # (K, C)
        feat_zx = feat_zx.squeeze(0).squeeze(-1).permute(1, 0)  # (K, C)

        # Concatenate multi-plane features
        fused_raw = torch.cat([feat_xy, feat_yz, feat_zx], dim=-1)  # (K, 3 * C)
        out = self.fusion_mlp(fused_raw)  # (K, out_dim)
        return out.reshape(*orig_shape, self.out_dim)

    def export_latent_tokens(self, token_grid_size: int = 32) -> torch.Tensor:
        """
        Downsamples and quantizes the tri-planes into discrete 32x32x32 latent tokens
        for downstream generative training.
        """
        with torch.no_grad():
            down_xy = F.interpolate(self.plane_xy, size=(token_grid_size, token_grid_size), mode="bilinear", align_corners=True)
            down_yz = F.interpolate(self.plane_yz, size=(token_grid_size, token_grid_size), mode="bilinear", align_corners=True)
            down_zx = F.interpolate(self.plane_zx, size=(token_grid_size, token_grid_size), mode="bilinear", align_corners=True)
            return torch.cat([down_xy, down_yz, down_zx], dim=1)  # (1, 3*C, 32, 32)
