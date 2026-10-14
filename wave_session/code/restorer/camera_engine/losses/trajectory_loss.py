"""Multi-Objective Trajectory Loss Suite.

Implements Riemannian geodesic SO(3) distance, Huber translation loss,
Lie-algebra C^2 kinematic penalties (velocity, acceleration, jerk),
sensor-scale normalized focal loss, and relative variance Hitchcock vertigo loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from config import LossConfig, TrajectoryConfig
from geometry.lie_algebra import geodesic_distance_so3, so3_log


class TrajectoryMultiLoss(nn.Module):
    """Complete multi-objective loss module for Camera Trajectory Engine."""
    def __init__(self, loss_config: LossConfig, traj_config: TrajectoryConfig):
        super().__init__()
        self.loss_cfg = loss_config
        self.traj_cfg = traj_config

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        device = preds["translations"].device
        loss_dict = {}
        
        # 1. SO(3) Riemannian Geodesic Rotation Loss
        R_pred = preds["rotations"]
        R_gt = targets["rotations"]
        rot_dist = geodesic_distance_so3(R_pred, R_gt, eps=self.loss_cfg.geodesic_eps)
        loss_rot = torch.mean(rot_dist)
        loss_dict["loss_rot_geodesic"] = loss_rot
        
        # 2. Translation Huber Loss
        t_pred = preds["translations"]
        t_gt = targets["translations"]
        loss_trans = F.huber_loss(t_pred, t_gt, delta=self.loss_cfg.huber_delta, reduction="mean")
        loss_dict["loss_trans_huber"] = loss_trans
        
        # 3. Kinematic Smoothness (C^2 Continuity Penalties)
        # Linear Jerk Penalty
        jerk_pred = preds["jerks"]
        loss_jerk = torch.mean(torch.sum(jerk_pred ** 2, dim=-1))
        loss_dict["loss_jerk_smoothness"] = loss_jerk
        
        # Linear Velocity & Acceleration Consistency
        v_pred = preds["velocities"]
        v_gt = targets.get("velocities", None)
        if v_gt is not None:
            loss_vel = F.mse_loss(v_pred, v_gt)
        else:
            loss_vel = torch.mean(torch.sum(v_pred ** 2, dim=-1))
        loss_dict["loss_vel_smoothness"] = loss_vel
        
        a_pred = preds["accelerations"]
        loss_acc = torch.mean(torch.sum(a_pred ** 2, dim=-1))
        loss_dict["loss_acc_smoothness"] = loss_acc
        
        # 4. Lie Algebra Vector Angular Acceleration Loss
        # Avoids scalar arccos gradient explosion by using Lie algebra omega vectors in so(3)
        R_curr = R_pred[:, :-1, :, :]
        R_next = R_pred[:, 1:, :, :]
        R_diff = torch.matmul(R_curr.transpose(-1, -2), R_next)
        omega = so3_log(R_diff) * self.traj_cfg.fps  # Angular velocity vectors (B, T-1, 3)
        
        loss_ang_vel = torch.mean(torch.sum(omega ** 2, dim=-1))
        loss_dict["loss_angular_vel"] = loss_ang_vel
        
        alpha = (omega[:, 1:, :] - omega[:, :-1, :]) * self.traj_cfg.fps  # Angular acceleration (B, T-2, 3)
        loss_ang_acc = torch.mean(torch.sum(alpha ** 2, dim=-1))
        loss_dict["loss_angular_acc"] = loss_ang_acc
        
        # 5. Sensor-Scale Normalized Focal Length Loss
        # Normalizes 8K pixels by image width so loss magnitude is O(0.01) - O(0.1)
        scale_px = float(self.traj_cfg.image_width)
        f_pred_norm = preds["focal_lengths"] / scale_px
        f_gt_norm = targets["focal_lengths"] / scale_px
        loss_focal = F.l1_loss(f_pred_norm, f_gt_norm)
        loss_dict["loss_focal_normalized"] = loss_focal
        
        # 6. Hitchcock Vertigo Coupling Loss (Relative Variance Formulation)
        target_center = targets.get("subject_target_pos", None)
        if target_center is not None:
            dist_to_subj = torch.norm(t_pred - target_center.unsqueeze(1), dim=-1)
            ratio = preds["focal_lengths"][..., 0] / torch.clamp(dist_to_subj, min=0.1)
            mean_ratio = torch.mean(ratio, dim=-1, keepdim=True)
            rel_ratio = ratio / torch.clamp(mean_ratio, min=1e-4)
            loss_vertigo = torch.mean((rel_ratio - 1.0) ** 2)
        else:
            loss_vertigo = torch.tensor(0.0, device=device)
        loss_dict["loss_vertigo_coupling"] = loss_vertigo
        
        # 7. Auxiliary Vertigo Classification Head Loss
        if "vertigo_factor" in preds:
            vertigo_pred = preds["vertigo_factor"]
            is_vertigo_gt = targets.get("is_vertigo", None)
            if is_vertigo_gt is not None:
                if not torch.is_tensor(is_vertigo_gt):
                    is_vertigo_gt = torch.tensor(is_vertigo_gt, dtype=torch.float32, device=device)
                else:
                    is_vertigo_gt = is_vertigo_gt.to(device=device, dtype=torch.float32)
                if is_vertigo_gt.ndim == 1:
                    is_vertigo_gt = is_vertigo_gt.unsqueeze(-1)
                loss_vertigo_cls = F.binary_cross_entropy(vertigo_pred, is_vertigo_gt)
            else:
                loss_vertigo_cls = torch.mean(vertigo_pred ** 2)
        else:
            loss_vertigo_cls = torch.tensor(0.0, device=device)
        loss_dict["loss_vertigo_cls"] = loss_vertigo_cls
        
        # Total Weighted Multi-Objective Loss
        total_loss = (
            self.loss_cfg.w_rot_geodesic * loss_rot +
            self.loss_cfg.w_trans_huber * loss_trans +
            self.loss_cfg.w_jerk_smoothness * loss_jerk +
            self.loss_cfg.w_vel_smoothness * loss_vel +
            self.loss_cfg.w_acc_smoothness * loss_acc +
            self.loss_cfg.w_angular_vel_smoothness * loss_ang_vel +
            self.loss_cfg.w_angular_acc_smoothness * loss_ang_acc +
            self.loss_cfg.w_focal_normalized * loss_focal +
            self.loss_cfg.w_vertigo_coupling * loss_vertigo +
            1.0 * loss_vertigo_cls
        )
        loss_dict["total_trajectory_loss"] = total_loss
        return loss_dict
