"""Configuration definitions for Stage 1: Camera Trajectory Engine.

Defines dataclasses for model architecture, trajectory dynamics,
loss weights, procedural dataset generation, and the training pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class TrajectoryConfig:
    """Configuration for camera trajectory physical dynamics."""
    num_frames: int = 300                 # 10 seconds at 30 fps
    fps: float = 30.0                     # Target frame rate
    duration_sec: float = 10.0            # Total trajectory duration
    default_fov_deg: float = 60.0         # Default horizontal Field of View
    sensor_width_mm: float = 36.0         # Standard full-frame 35mm sensor width
    sensor_height_mm: float = 24.0        # Standard full-frame 35mm sensor height
    image_width: int = 7680               # 8K resolution width
    image_height: int = 4320              # 8K resolution height
    near_clip: float = 0.1                # Near clipping plane (meters)
    far_clip: float = 1000.0              # Far clipping plane (meters)
    safe_collision_distance: float = 0.5  # Safety distance for SDF obstacle avoidance (meters)


@dataclass
class ModelConfig:
    """Hyperparameters for Camera Trajectory Engine architecture (~27.21M params)."""
    # Text / Directorial Conditioning Head (~8.71M params)
    vocab_size: int = 4096                # Directorial token vocabulary size
    max_text_len: int = 64                # Maximum directorial prompt token length
    directorial_dim: int = 16             # Continuous directorial control metadata vector dimension
    d_model: int = 512                    # Main transformer hidden feature dimension
    cond_layers: int = 2                  # Number of directorial text transformer layers
    cond_heads: int = 8                   # Attention heads in conditioning encoder
    cond_ffn_dim: int = 2048              # Feedforward inner dimension in conditioning encoder
    
    # Trajectory Backbone (~16.83M params)
    num_knots: int = 32                   # Lie-algebra B-spline control knots
    num_frames: int = 300                 # Output temporal trajectory frame resolution
    backbone_layers: int = 4              # Temporal cross-attention decoder layers
    backbone_heads: int = 8               # Backbone attention heads
    backbone_ffn_dim: int = 2048          # Backbone feedforward dimension
    dropout: float = 0.1                  # Dropout probability
    activation: str = "gelu"              # Non-linear activation function
    
    # Spline & Lie Tangent Refinement (~0.09M params)
    use_spline_interpolation: bool = True # Differentiable Lie-algebra cumulative B-spline
    refinement_channels: int = 128        # Temporal residual 1D convolution refinement channels
    refinement_kernel_size: int = 5       # Kernel size for residual refinement convs
    
    # Focal / Vertigo Head (~0.53M params)
    predict_dynamic_focal: bool = True    # Predict dynamic focal length curves
    predict_vertigo_scale: bool = True    # Auxiliary vertigo / dolly-zoom head


@dataclass
class LossConfig:
    """Weighting factors and hyper-parameters for multi-objective trajectory loss."""
    # Rotation & Translation
    w_rot_geodesic: float = 10.0          # Geodesic SO(3) rotation loss weight
    w_trans_huber: float = 5.0            # Translation Huber loss weight
    huber_delta: float = 0.05             # Delta parameter for Huber translation loss
    
    # Kinematic Smoothness (C^2 continuity)
    w_vel_smoothness: float = 1.0         # Linear velocity continuity weight
    w_acc_smoothness: float = 0.5         # Linear acceleration smoothness weight
    w_jerk_smoothness: float = 0.2        # Linear jerk (d^3 x / dt^3) penalty weight
    w_angular_vel_smoothness: float = 0.5 # Angular velocity penalty weight
    w_angular_acc_smoothness: float = 0.5 # Angular acceleration penalty weight
    
    # Scene Obstacle Barrier
    w_sdf_barrier: float = 5.0            # SDF collision avoidance barrier penalty
    sdf_safety_margin: float = 0.5        # Safety distance threshold (meters)
    
    # Focal Length & Dolly-Zoom Vertigo (normalized scale)
    w_focal_normalized: float = 5.0       # Sensor-normalized focal length loss weight
    w_vertigo_coupling: float = 2.0       # Hitchcock vertigo relative variance loss weight
    
    # Numerical stability
    geodesic_eps: float = 1e-6            # Clamping epsilon for arccos in SO(3) geodesic distance


@dataclass
class TrainConfig:
    """Training loop configuration and optimization hyperparameters."""
    exp_name: str = "stage1_camera_engine_w_ghnr"
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    
    # Optimization
    batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    max_steps: int = 10000
    eval_every_steps: int = 500
    save_every_steps: int = 1000
    grad_clip_norm: float = 1.0
    
    # Hardware & Precision
    use_amp: bool = False                 # FP32 native precision for Lie-algebra numerical stability
    num_workers: int = 2
    device: str = "cuda"                  # Fallback to "cpu" if CUDA unavailable
    
    # Synthetic Dataset
    num_train_samples: int = 2000
    num_val_samples: int = 200
