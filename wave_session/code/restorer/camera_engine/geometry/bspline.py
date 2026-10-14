"""Continuous-time Cumulative B-Spline on Lie Algebra SO(3) x R^3.

Implements differentiable uniform cumulative cubic B-spline interpolation
guaranteeing C^2 kinematic continuity across 300 output frames.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple

from geometry.lie_algebra import (
    so3_exp,
    so3_log,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion
)


class CumulativeLieBSpline(nn.Module):
    """Uniform Cubic Cumulative B-spline Evaluator on Lie Group SO(3) x R^3."""
    def __init__(self, num_knots: int = 32, num_output_frames: int = 300, fps: float = 30.0):
        super().__init__()
        self.num_knots = num_knots
        self.num_frames = num_output_frames
        self.fps = fps
        self.dt = 1.0 / fps
        
        t_eval = torch.linspace(0.0, 1.0, num_output_frames)
        self.register_buffer("t_eval", t_eval)
        
        self.num_segments = num_knots - 3
        
        weights_cum, weights_trans, segment_indices = self._compute_evaluation_weights(t_eval)
        self.register_buffer("weights_cum", weights_cum)
        self.register_buffer("weights_trans", weights_trans)
        self.register_buffer("segment_indices", segment_indices)

    def _compute_evaluation_weights(self, t_eval: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute uniform cubic B-spline cumulative basis functions with safe boundary clamp."""
        T = t_eval.shape[0]
        eps = 1e-6
        s = torch.clamp(t_eval * float(self.num_segments), 0.0, float(self.num_segments) - eps)
        
        seg_idx = s.long()
        u = s - seg_idx.float()
        u2 = u ** 2
        u3 = u ** 3
        
        # 1. Standard Cubic B-Spline Basis for Translation (sum = 1)
        b0 = (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0
        b1 = (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0
        b2 = (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0
        b3 = u3 / 6.0
        weights_trans = torch.stack([b0, b1, b2, b3], dim=-1)
        
        # 2. Cumulative Cubic B-Spline Basis for Rotation on Lie Algebra SO(3)
        cb1 = (5.0 + 3.0 * u - 3.0 * u2 + u3) / 6.0
        cb2 = (1.0 + 3.0 * u + 3.0 * u2 - 2.0 * u3) / 6.0
        cb3 = u3 / 6.0
        weights_cum = torch.stack([cb1, cb2, cb3], dim=-1)
        
        return weights_cum, weights_trans, seg_idx

    def forward(
        self,
        knots_rot: torch.Tensor,
        knots_trans: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = knots_trans.device
        
        if knots_rot.shape[-1] == 4:
            R_knots = quaternion_to_rotation_matrix(knots_rot)
        else:
            R_knots = knots_rot
            
        seg_idx = self.segment_indices.to(device)
        w_cum = self.weights_cum.to(device)
        w_trans = self.weights_trans.to(device)
        
        idx0 = seg_idx
        idx1 = seg_idx + 1
        idx2 = seg_idx + 2
        idx3 = seg_idx + 3
        
        # 1. Translation Spline Evaluation
        t_k0 = knots_trans[:, idx0, :]
        t_k1 = knots_trans[:, idx1, :]
        t_k2 = knots_trans[:, idx2, :]
        t_k3 = knots_trans[:, idx3, :]
        
        w0 = w_trans[:, 0].unsqueeze(0).unsqueeze(-1)
        w1 = w_trans[:, 1].unsqueeze(0).unsqueeze(-1)
        w2 = w_trans[:, 2].unsqueeze(0).unsqueeze(-1)
        w3 = w_trans[:, 3].unsqueeze(0).unsqueeze(-1)
        
        t_dense = w0 * t_k0 + w1 * t_k1 + w2 * t_k2 + w3 * t_k3
        
        # 2. Rotation Cumulative Lie Spline Evaluation
        R_k0 = R_knots[:, idx0, :, :]
        R_k1 = R_knots[:, idx1, :, :]
        R_k2 = R_knots[:, idx2, :, :]
        R_k3 = R_knots[:, idx3, :, :]
        
        R_rel1 = torch.matmul(R_k0.transpose(-1, -2), R_k1)
        R_rel2 = torch.matmul(R_k1.transpose(-1, -2), R_k2)
        R_rel3 = torch.matmul(R_k2.transpose(-1, -2), R_k3)
        
        twist1 = so3_log(R_rel1)
        twist2 = so3_log(R_rel2)
        twist3 = so3_log(R_rel3)
        
        cb1 = w_cum[:, 0].unsqueeze(0).unsqueeze(-1)
        cb2 = w_cum[:, 1].unsqueeze(0).unsqueeze(-1)
        cb3 = w_cum[:, 2].unsqueeze(0).unsqueeze(-1)
        
        delta_R1 = so3_exp(cb1 * twist1)
        delta_R2 = so3_exp(cb2 * twist2)
        delta_R3 = so3_exp(cb3 * twist3)
        
        R_dense = torch.matmul(R_k0, delta_R1)
        R_dense = torch.matmul(R_dense, delta_R2)
        R_dense = torch.matmul(R_dense, delta_R3)
        
        # 3. Kinematic Derivatives
        dt = self.dt
        v_dense = (t_dense[:, 1:, :] - t_dense[:, :-1, :]) / dt
        a_dense = (v_dense[:, 1:, :] - v_dense[:, :-1, :]) / dt
        j_dense = (a_dense[:, 1:, :] - a_dense[:, :-1, :]) / dt
        
        return R_dense, t_dense, v_dense, a_dense, j_dense
