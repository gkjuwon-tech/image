"""SE(3) Lie-Knot, Focal Length, and Vertigo Prediction Heads (~1.67M Parameters).

Predicts Lie-group SO(3) rotation quaternions, R^3 translation knots,
continuous focal lengths, and Lie-algebra tangent space residual refinements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from geometry.lie_algebra import so3_exp


class LieKnotHead(nn.Module):
    """Predicts SO(3) unit quaternion and R^3 translation knots (~1.05M params)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 1024),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 7)
        )
        
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def forward(self, knot_features: torch.Tensor):
        out = self.net(knot_features)
        q_raw = out[..., :4]
        t = out[..., 4:]
        q = F.normalize(q_raw, p=2, dim=-1)
        return q, t


class DynamicFocalHead(nn.Module):
    """Predicts dynamic focal length (fx, fy) and principal point (cx, cy) (~0.40M params)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 4)
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.copy_(torch.tensor([9.275, 9.275, 0.0, 0.0]))

    def forward(self, knot_features: torch.Tensor):
        out = self.net(knot_features)
        focal = torch.exp(torch.clamp(out[..., :2], 7.0, 11.0))
        pp_offset = torch.tanh(out[..., 2:]) * 50.0
        return focal, pp_offset


class VertigoAuxiliaryHead(nn.Module):
    """Predicts auxiliary vertigo confidence in [0, 1] (~0.13M params)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        pooled = memory.mean(dim=1)
        return self.net(pooled)


class TemporalRefinementBlock(nn.Module):
    """Lie-algebra Tangent Space 1D Conv Residual Refinement Block (~0.09M params)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        C = config.refinement_channels
        K = config.refinement_kernel_size
        P = K // 2
        
        self.conv1 = nn.Conv1d(3, C, kernel_size=K, padding=P)
        self.conv2 = nn.Conv1d(C, C, kernel_size=K, padding=P)
        self.conv3 = nn.Conv1d(C, 6, kernel_size=K, padding=P)
        self.act = nn.GELU()
        
        with torch.no_grad():
            self.conv3.weight.zero_()
            self.conv3.bias.zero_()

    def forward(self, R_spline: torch.Tensor, t_spline: torch.Tensor):
        x = t_spline.transpose(1, 2)
        res = self.conv3(self.act(self.conv2(self.act(self.conv1(x)))))
        res = res.transpose(1, 2)
        
        delta_omega = res[..., :3] * 0.1
        delta_trans = res[..., 3:] * 0.1
        
        delta_R = so3_exp(delta_omega)
        R_refined = torch.matmul(R_spline, delta_R)
        t_refined = t_spline + delta_trans
        return R_refined, t_refined
