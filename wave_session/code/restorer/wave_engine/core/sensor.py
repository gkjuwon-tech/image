"""
Optical Sensor Integration Head (Model 4: ~2.2M Parameters)
===========================================================
Integrates the volumetric wave field along camera rays, simulating
physical light transport, wave irradiance, and sensor tone-mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DirectionalEncoding(nn.Module):
    """Encodes viewing directions using harmonic frequencies."""
    def __init__(self, in_dim: int = 3, num_freqs: int = 4):
        super().__init__()
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        self.out_dim = in_dim * (2 * num_freqs + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * torch.pi * x))
            out.append(torch.cos(freq * torch.pi * x))
        return torch.cat(out, dim=-1)


class SensorIntegrationHead(nn.Module):
    """
    Optical Sensor Integration Head.
    
    Performs volume rendering over wave irradiance and computes view-dependent radiance.
    """
    def __init__(
        self,
        in_feat_dim: int = 32,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.dir_enc = DirectionalEncoding(in_dim=3, num_freqs=4)  # 3 * 9 = 27 dims
        in_dim = in_feat_dim + self.dir_enc.out_dim + 3  # features + dir + amplitude

        # View-dependent radiance shading network (~2.2M parameters)
        self.shading_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()  # Outputs RAW RGB in [0, 1]
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.shading_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def integrate_rays(
        self,
        densities: torch.Tensor,     # (B, N_pts, 1)
        amplitudes: torch.Tensor,    # (B, N_pts, 3)
        features: torch.Tensor,      # (B, N_pts, 32)
        dists: torch.Tensor,         # (B, N_pts)
        view_dirs: torch.Tensor,     # (B, 3)
        white_bkgd: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Volume rendering integration over physical wave samples.
        """
        B, N_pts, _ = densities.shape
        sigma = densities.squeeze(-1)  # (B, N_pts)

        # Alpha blending weights
        # alpha = 1 - exp(-sigma * dists)
        sigma_delta = sigma * dists
        alpha = 1.0 - torch.exp(-sigma_delta)  # (B, N_pts)

        # Transmittance T_i = prod_{j=1}^{i-1} (1 - alpha_j)
        # Numerical stability via cumsum
        transmittance = torch.cumprod(
            torch.cat([torch.ones(B, 1, device=densities.device), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[:, :-1]  # (B, N_pts)

        weights = transmittance * alpha  # (B, N_pts)

        # Directional radiance evaluation
        dirs_expanded = view_dirs.unsqueeze(1).expand(B, N_pts, -1)
        dir_embed = self.dir_enc(dirs_expanded)
        shading_in = torch.cat([features, dir_embed, amplitudes], dim=-1)
        radiance = self.shading_net(shading_in)  # (B, N_pts, 3)

        # Integrate color
        rgb = torch.sum(weights.unsqueeze(-1) * radiance, dim=1)  # (B, 3)

        if white_bkgd:
            accum_weight = torch.sum(weights, dim=1, keepdim=True)
            rgb = rgb + (1.0 - accum_weight)

        return {
            "rgb": rgb,
            "weights": weights,
            "opacity": torch.sum(weights, dim=1)
        }
