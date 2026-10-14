"""Lie Algebra and Lie Group SO(3) / SE(3) Operations in PyTorch.

Implements numerically stable exponential/logarithm maps with Taylor expansions,
hat/vee operators, robust 4-branch Shepperd-style quaternion conversions,
SLERP, and Riemannian geodesic metrics on SO(3).
"""

import math
import torch
import torch.nn.functional as F


def hat(v: torch.Tensor) -> torch.Tensor:
    """Compute the skew-symmetric matrix (hat operator) for 3D vectors."""
    v0, v1, v2 = v[..., 0], v[..., 1], v[..., 2]
    zero = torch.zeros_like(v0)
    row0 = torch.stack([zero, -v2, v1], dim=-1)
    row1 = torch.stack([v2, zero, -v0], dim=-1)
    row2 = torch.stack([-v1, v0, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def vee(Omega: torch.Tensor) -> torch.Tensor:
    """Extract 3D vector from skew-symmetric matrix (vee operator)."""
    return torch.stack([
        Omega[..., 2, 1],
        Omega[..., 0, 2],
        Omega[..., 1, 0]
    ], dim=-1)


def so3_exp(v: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Exponential map from Lie algebra so(3) to Lie group SO(3) via Rodrigues formula."""
    theta_sq = torch.sum(v ** 2, dim=-1, keepdim=True)
    theta = torch.sqrt(torch.clamp(theta_sq, min=1e-12))
    
    Omega = hat(v)
    Omega2 = torch.matmul(Omega, Omega)
    
    I = torch.eye(3, dtype=v.dtype, device=v.device).expand(v.shape[:-1] + (3, 3))
    
    near_zero = theta_sq < (eps ** 2)
    theta_sq_val = theta_sq.unsqueeze(-1)
    
    A_series = 1.0 - theta_sq_val / 6.0 + (theta_sq_val ** 2) / 120.0
    B_series = 0.5 - theta_sq_val / 24.0 + (theta_sq_val ** 2) / 720.0
    
    A_exact = torch.sin(theta.unsqueeze(-1)) / torch.clamp(theta.unsqueeze(-1), min=1e-12)
    B_exact = (1.0 - torch.cos(theta.unsqueeze(-1))) / torch.clamp(theta_sq_val, min=1e-12)
    
    A = torch.where(near_zero.unsqueeze(-1), A_series, A_exact)
    B = torch.where(near_zero.unsqueeze(-1), B_series, B_exact)
    
    R = I + A * Omega + B * Omega2
    return R


def so3_log(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Logarithm map from Lie group SO(3) to Lie algebra so(3) via Shepperd quaternion.
    
    Globally numerically stable across the entire manifold [0, pi], eliminating
    the classic sin(theta) ~ 0 Rodrigues axis-angle singularity at 180 degrees.
    """
    orig_dtype = R.dtype
    q = rotation_matrix_to_quaternion(R)  # [..., 4] as [w, x, y, z]
    
    # Enforce w >= 0 so principal rotation angle is in [0, pi]
    w = q[..., 0:1]
    v_q = q[..., 1:4]
    sign = torch.where(w < 0.0, -torch.ones_like(w), torch.ones_like(w))
    w = w * sign
    v_q = v_q * sign
    
    w_clamped = torch.clamp(w, 0.0, 1.0 - eps)
    theta = 2.0 * torch.acos(w_clamped)  # [..., 1]
    sin_half_theta = torch.sqrt(torch.clamp(1.0 - w_clamped ** 2, min=1e-12))
    
    near_zero = (theta < 1e-4)
    factor_series = 2.0 + (theta ** 2) / 12.0
    factor_exact = theta / sin_half_theta
    factor = torch.where(near_zero, factor_series, factor_exact)
    
    res = v_q * factor
    return res.to(orig_dtype)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions [w, x, y, z] to SO(3) rotation matrices."""
    q_norm = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q_norm[..., 0], q_norm[..., 1], q_norm[..., 2], q_norm[..., 3]
    
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)
    
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def rotation_matrix_to_quaternion(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Convert SO(3) rotation matrices to unit quaternions via Shepperd's algorithm."""
    orig_dtype = R.dtype
    R_32 = R.to(torch.float32)
    
    r00, r01, r02 = R_32[..., 0, 0], R_32[..., 0, 1], R_32[..., 0, 2]
    r10, r11, r12 = R_32[..., 1, 0], R_32[..., 1, 1], R_32[..., 1, 2]
    r20, r21, r22 = R_32[..., 2, 0], R_32[..., 2, 1], R_32[..., 2, 2]
    
    tr = r00 + r11 + r22
    
    w2 = 1.0 + tr
    x2 = 1.0 + r00 - r11 - r22
    y2 = 1.0 - r00 + r11 - r22
    z2 = 1.0 - r00 - r11 + r22
    
    # Branch 0: w dominant
    w_0 = 0.5 * torch.sqrt(torch.clamp(w2, min=eps))
    x_0 = (r21 - r12) / (4.0 * torch.clamp(w_0, min=1e-6))
    y_0 = (r02 - r20) / (4.0 * torch.clamp(w_0, min=1e-6))
    z_0 = (r10 - r01) / (4.0 * torch.clamp(w_0, min=1e-6))
    q_0 = torch.stack([w_0, x_0, y_0, z_0], dim=-1)
    
    # Branch 1: x dominant
    x_1 = 0.5 * torch.sqrt(torch.clamp(x2, min=eps))
    w_1 = (r21 - r12) / (4.0 * torch.clamp(x_1, min=1e-6))
    y_1 = (r01 + r10) / (4.0 * torch.clamp(x_1, min=1e-6))
    z_1 = (r02 + r20) / (4.0 * torch.clamp(x_1, min=1e-6))
    q_1 = torch.stack([w_1, x_1, y_1, z_1], dim=-1)
    
    # Branch 2: y dominant
    y_2 = 0.5 * torch.sqrt(torch.clamp(y2, min=eps))
    w_2 = (r02 - r20) / (4.0 * torch.clamp(y_2, min=1e-6))
    x_2 = (r01 + r10) / (4.0 * torch.clamp(y_2, min=1e-6))
    z_2 = (r12 + r21) / (4.0 * torch.clamp(y_2, min=1e-6))
    q_2 = torch.stack([w_2, x_2, y_2, z_2], dim=-1)
    
    # Branch 3: z dominant
    z_3 = 0.5 * torch.sqrt(torch.clamp(z2, min=eps))
    w_3 = (r10 - r01) / (4.0 * torch.clamp(z_3, min=1e-6))
    x_3 = (r02 + r20) / (4.0 * torch.clamp(z_3, min=1e-6))
    y_3 = (r12 + r21) / (4.0 * torch.clamp(z_3, min=1e-6))
    q_3 = torch.stack([w_3, x_3, y_3, z_3], dim=-1)
    
    cond_0 = (tr > 0.0).unsqueeze(-1)
    cond_1 = ((r00 > r11) & (r00 > r22)).unsqueeze(-1)
    cond_2 = (r11 > r22).unsqueeze(-1)
    
    q = torch.where(
        cond_0,
        q_0,
        torch.where(
            cond_1,
            q_1,
            torch.where(cond_2, q_2, q_3)
        )
    )
    
    q_norm = F.normalize(q, p=2, dim=-1)
    return q_norm.to(orig_dtype)


def geodesic_distance_so3(R1: torch.Tensor, R2: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute true Riemannian geodesic rotation distance on SO(3) in float32.
    
    Clamps cos_angle within [-1.0 + eps, 1.0 - eps] to eliminate the acos(x)
    derivative singularity (-1 / sqrt(1 - x^2) -> -inf) at x = 1, ensuring zero NaN gradients.
    """
    R1_32 = R1.to(torch.float32)
    R2_32 = R2.to(torch.float32)
    
    R_rel = torch.matmul(R1_32.transpose(-1, -2), R2_32)
    tr = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_angle = torch.clamp((tr - 1.0) * 0.5, -1.0 + eps, 1.0 - eps)
    dist = torch.acos(cos_angle)
    return dist.to(R1.dtype)


def quaternion_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Spherical linear interpolation (SLERP) between two unit quaternions."""
    q0 = F.normalize(q0, p=2, dim=-1)
    q1 = F.normalize(q1, p=2, dim=-1)
    
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.clamp(torch.abs(dot), -1.0 + eps, 1.0 - eps)
    
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    
    near_zero = sin_theta_0 < eps
    scale0_exact = torch.sin((1.0 - t) * theta_0) / torch.clamp(sin_theta_0, min=eps)
    scale1_exact = torch.sin(t * theta_0) / torch.clamp(sin_theta_0, min=eps)
    
    scale0 = torch.where(near_zero, 1.0 - t, scale0_exact)
    scale1 = torch.where(near_zero, t, scale1_exact)
    
    return F.normalize(scale0 * q0 + scale1 * q1, p=2, dim=-1)
