"""Stage 1: Camera Trajectory Engine Full Model (~27.21M Parameters).

Integrates Directorial Conditioning, Temporal Knot Transformer Decoder,
Lie-algebra Cumulative B-Spline Evaluator, and Dynamic Intrinsics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from config import ModelConfig, TrajectoryConfig
from geometry.lie_algebra import (
    rotation_matrix_to_quaternion
)
from geometry.bspline import CumulativeLieBSpline
from geometry.camera_intrinsics import build_intrinsic_matrix
from models.conditioning import DirectorialConditioningEncoder
from models.trajectory_backbone import TrajectoryTransformerBackbone
from models.knot_predictor import (
    LieKnotHead,
    DynamicFocalHead,
    VertigoAuxiliaryHead,
    TemporalRefinementBlock
)


class CameraTrajectoryEngine(nn.Module):
    """Complete 27.21M Parameter Camera Trajectory Engine for 8K 300-Frame Neural Rendering."""
    def __init__(self, model_config: Optional[ModelConfig] = None, traj_config: Optional[TrajectoryConfig] = None):
        super().__init__()
        self.model_config = model_config or ModelConfig()
        self.traj_config = traj_config or TrajectoryConfig()
        
        # 1. Conditioning Encoder (~8.71M params)
        self.conditioning_encoder = DirectorialConditioningEncoder(self.model_config)
        
        # 2. Temporal Knot Transformer Decoder (~16.83M params)
        self.backbone = TrajectoryTransformerBackbone(self.model_config)
        
        # 3. Knot Prediction Heads (~1.67M params)
        self.lie_knot_head = LieKnotHead(self.model_config)
        self.focal_knot_head = DynamicFocalHead(self.model_config)
        self.vertigo_head = VertigoAuxiliaryHead(self.model_config)
        
        # 4. Continuous Differentiable Cumulative Lie-Spline Evaluator (0 params)
        self.spline_evaluator = CumulativeLieBSpline(
            num_knots=self.model_config.num_knots,
            num_output_frames=self.model_config.num_frames,
            fps=self.traj_config.fps
        )
        
        # 5. Temporal Lie-Tangent Residual Refinement Block (~0.09M params)
        self.refinement_block = TemporalRefinementBlock(self.model_config)

    def get_parameter_count(self) -> Dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
            
        counts = {
            "conditioning_encoder": count(self.conditioning_encoder),
            "backbone": count(self.backbone),
            "lie_knot_head": count(self.lie_knot_head),
            "focal_knot_head": count(self.focal_knot_head),
            "vertigo_head": count(self.vertigo_head),
            "refinement_block": count(self.refinement_block),
            "total_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        return counts

    def forward(
        self,
        text_tokens: torch.Tensor,
        directorial_params: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # 1. Directorial Conditioning
        memory = self.conditioning_encoder(text_tokens, directorial_params, text_mask=text_mask)
        
        # 2. Knot Backbone
        knot_features = self.backbone(memory)
        
        # 3. Knot Heads
        knots_rot_quat, knots_trans = self.lie_knot_head(knot_features)
        knots_focal, knots_pp = self.focal_knot_head(knot_features)
        vertigo_factor = self.vertigo_head(memory)
        
        # 4. Continuous Lie-Spline Interpolation to 300 Frames
        R_spline, t_spline, _, _, _ = self.spline_evaluator(knots_rot_quat, knots_trans)
        
        # 5. Lie-Tangent Temporal Residual Refinement
        R_refined, t_refined = self.refinement_block(R_spline, t_spline)
        q_refined = rotation_matrix_to_quaternion(R_refined)
        
        # Kinematics on refined trajectory
        dt = 1.0 / self.traj_config.fps
        v_refined = (t_refined[:, 1:, :] - t_refined[:, :-1, :]) / dt
        a_refined = (v_refined[:, 1:, :] - v_refined[:, :-1, :]) / dt
        j_refined = (a_refined[:, 1:, :] - a_refined[:, :-1, :]) / dt
        
        # 6. Dense Focal Length & Intrinsics Interpolation
        focal_dense = F.interpolate(
            knots_focal.transpose(1, 2),
            size=self.model_config.num_frames,
            mode="linear",
            align_corners=True
        ).transpose(1, 2)
        
        pp_offset_dense = F.interpolate(
            knots_pp.transpose(1, 2),
            size=self.model_config.num_frames,
            mode="linear",
            align_corners=True
        ).transpose(1, 2)
        
        cx_base = float(self.traj_config.image_width) * 0.5
        cy_base = float(self.traj_config.image_height) * 0.5
        
        fx_dense = focal_dense[..., 0]
        fy_dense = focal_dense[..., 1]
        cx_dense = cx_base + pp_offset_dense[..., 0]
        cy_dense = cy_base + pp_offset_dense[..., 1]
        
        K_dense = build_intrinsic_matrix(fx_dense, fy_dense, cx_dense, cy_dense)
        
        return {
            "rotations": R_refined,
            "quaternions": q_refined,
            "translations": t_refined,
            "intrinsics": K_dense,
            "focal_lengths": focal_dense,
            "velocities": v_refined,
            "accelerations": a_refined,
            "jerks": j_refined,
            "knots_rot": knots_rot_quat,
            "knots_trans": knots_trans,
            "vertigo_factor": vertigo_factor
        }
