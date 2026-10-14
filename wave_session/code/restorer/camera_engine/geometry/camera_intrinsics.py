"""Dynamic Pinhole Camera Intrinsics and Optical Mathematics.

Implements K-matrix generation across time, physical millimeter to pixel conversion
for 8K full-frame sensors, and optical coupling for the Hitchcock vertigo effect.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple


def focal_length_mm_to_px(focal_mm: float, sensor_width_mm: float = 36.0, image_width_px: int = 7680) -> float:
    """Convert physical lens focal length in millimeters to pixel focal length.
    
    fx = f_mm * (image_width_px / sensor_width_mm)
    """
    return focal_mm * (float(image_width_px) / float(sensor_width_mm))


def focal_length_px_to_mm(focal_px: float, sensor_width_mm: float = 36.0, image_width_px: int = 7680) -> float:
    """Convert pixel focal length to physical lens focal length in millimeters."""
    return float(focal_px) * (float(sensor_width_mm) / float(image_width_px))


def fov_deg_to_focal_px(fov_deg: torch.Tensor, image_width_px: int = 7680) -> torch.Tensor:
    """Convert horizontal Field of View in degrees to focal length in pixels."""
    fov_rad = fov_deg * (math.pi / 180.0)
    return (image_width_px * 0.5) / torch.clamp(torch.tan(fov_rad * 0.5), min=1e-5)


def focal_px_to_fov_deg(focal_px: torch.Tensor, image_width_px: int = 7680) -> torch.Tensor:
    """Convert focal length in pixels to horizontal Field of View in degrees."""
    fov_rad = 2.0 * torch.atan((image_width_px * 0.5) / torch.clamp(focal_px, min=1.0))
    return fov_rad * (180.0 / math.pi)


def build_intrinsic_matrix(
    fx: torch.Tensor,
    fy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor
) -> torch.Tensor:
    """Build pinhole 3x3 intrinsic matrix K across batch and frames."""
    zero = torch.zeros_like(fx)
    one = torch.ones_like(fx)
    
    row0 = torch.stack([fx, zero, cx], dim=-1)
    row1 = torch.stack([zero, fy, cy], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def dolly_zoom_coupling_focal(
    distance_curr: torch.Tensor,
    distance_ref: torch.Tensor,
    focal_ref: torch.Tensor
) -> torch.Tensor:
    """Compute Hitchcock vertigo focal length maintaining constant target scale."""
    ratio = distance_curr / torch.clamp(distance_ref, min=1e-4)
    return focal_ref * ratio


def project_points(
    points_world: torch.Tensor,
    R_cam_world: torch.Tensor,
    t_cam_world: torch.Tensor,
    K: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D world points into 2D camera coordinates."""
    p_cam = torch.matmul(points_world, R_cam_world.transpose(-1, -2)) + t_cam_world.unsqueeze(-2)
    depth = p_cam[..., 2]
    
    p_homo = torch.matmul(p_cam, K.transpose(-1, -2))
    u = p_homo[..., 0] / torch.clamp(p_homo[..., 2], min=1e-5)
    v = p_homo[..., 1] / torch.clamp(p_homo[..., 2], min=1e-5)
    uv = torch.stack([u, v], dim=-1)
    return uv, depth
