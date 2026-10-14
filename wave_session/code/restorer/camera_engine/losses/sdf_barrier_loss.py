"""Differentiable Signed Distance Field (SDF) Obstacle Collision Barrier Loss.

Penalizes camera trajectory intrusion into scene obstacle boundaries with smooth
quadratic barrier penalty ReLU(d_safe - Phi(x))^2 and epsilon-smoothed Euclidean norm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class DifferentiableSphereSDF(nn.Module):
    """Analytic Signed Distance Field for multiple spherical obstacles."""
    def __init__(self, centers: torch.Tensor, radii: torch.Tensor):
        super().__init__()
        self.register_buffer("centers", centers)
        self.register_buffer("radii", radii)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate minimum signed distance to all spheres with epsilon-smoothed norm.
        
        Args:
            points: (B, T, 3) camera translation points
        Returns:
            min_sdf: (B, T) minimum signed distance in meters
        """
        # (B, T, 1, 3) - (1, 1, N_obs, 3) -> (B, T, N_obs, 3)
        diff = points.unsqueeze(-2) - self.centers.unsqueeze(0).unsqueeze(0)
        # Epsilon-smoothed Euclidean norm prevents center-penetration gradient NaN
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
        sdf = dist - self.radii.unsqueeze(0).unsqueeze(0)  # (B, T, N_obs)
        min_sdf, _ = torch.min(sdf, dim=-1)                # (B, T)
        return min_sdf


class SDFBarrierLoss(nn.Module):
    """Quadratic Barrier Loss penalizing camera path proximity to obstacle surfaces."""
    def __init__(self, safe_margin: float = 0.5):
        super().__init__()
        self.safe_margin = safe_margin

    def forward(self, trajectory_trans: torch.Tensor, sdf_evaluator: nn.Module) -> torch.Tensor:
        """Compute smooth quadratic barrier loss.
        
        Args:
            trajectory_trans: (B, T, 3) predicted camera translations
            sdf_evaluator: Differentiable module mapping (B, T, 3) -> (B, T)
        Returns:
            barrier_loss: Scalar penalty
        """
        min_sdf = sdf_evaluator(trajectory_trans)  # (B, T)
        intrusion = F.relu(self.safe_margin - min_sdf)
        barrier_loss = torch.mean(intrusion ** 2)
        return barrier_loss
