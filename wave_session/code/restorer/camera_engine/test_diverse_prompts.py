"""Comprehensive Multi-Style Cinematic Directorial Prompt Testing Suite.

Evaluates the trained Camera Trajectory Engine across 6 distinct cinematic styles:
1. Counter-Clockwise Telephoto Orbit (85mm Lens)
2. Reverse Vertigo Dolly-Out (Dolly Back & Zoom In: 35mm -> 100mm)
3. Ultra-Wide Descending Inward Spiral (24mm Lens)
4. Cinematic Ground-to-Overhead Crane Shot (50mm Lens)
5. High-Speed Dynamic FPV Fly-Through (Dynamic Tilt & Roll)
6. Expansive Wide Panoramic Orbit (35mm Lens)
"""

import math
import time
import torch
import numpy as np

from config import TrajectoryConfig, ModelConfig
from models.camera_engine import CameraTrajectoryEngine
from data.procedural_generator import SimpleDirectorialTokenizer
from geometry.camera_intrinsics import focal_length_px_to_mm, focal_px_to_fov_deg


def rotation_matrix_to_euler_deg(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to Euler angles (pitch, yaw, roll in degrees)."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z]) * (180.0 / math.pi)


def evaluate_prompt(model, tokenizer, traj_cfg, device, prompt: str, directorial_vec: np.ndarray, title: str):
    tokens, mask = tokenizer.encode(prompt, max_len=64)
    tokens_t = tokens.unsqueeze(0).to(device)
    mask_t = mask.unsqueeze(0).to(device)
    directorial_t = torch.tensor(directorial_vec, dtype=torch.float32).unsqueeze(0).to(device)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model(tokens_t, directorial_t, text_mask=mask_t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    trans = preds["translations"][0].cpu().numpy()     # (300, 3)
    rots = preds["rotations"][0].cpu().numpy()         # (300, 3, 3)
    focals = preds["focal_lengths"][0].cpu().numpy()   # (300, 2)
    vels = preds["velocities"][0].cpu().numpy()        # (299, 3)
    jerks = preds["jerks"][0].cpu().numpy()            # (297, 3)
    vertigo_prob = preds["vertigo_factor"][0].item()
    
    speeds = np.linalg.norm(vels, axis=-1)
    mean_speed = float(np.mean(speeds))
    max_speed = float(np.max(speeds))
    mean_jerk = float(np.mean(np.linalg.norm(jerks, axis=-1)))
    
    # Keyframe timestamps: 0s (f0), 2s (f60), 4s (f120), 6s (f180), 8s (f240), 10s (f299)
    keyframes = [0, 59, 119, 179, 239, 299]
    timestamps = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    
    print("\n" + "=" * 80)
    print(f"🎬 [SHOT DEMO] {title}")
    print(f"   Prompt: \"{prompt}\"")
    print(f"   Inference: {latency_ms:.2f} ms | Vertigo Head: {vertigo_prob * 100:.1f}% | Avg Speed: {mean_speed:.2f} m/s | Smoothness: {mean_jerk:.4f} m/s^3")
    print("-" * 80)
    print(f"   {'Time':^6} | {'X (m)':^7} | {'Y (m)':^7} | {'Z (m)':^7} | {'Pitch':^7} | {'Yaw':^7} | {'Roll':^7} | {'Focal':^8} | {'H-FOV':^7}")
    print("-" * 80)
    
    for t_sec, idx in zip(timestamps, keyframes):
        p = trans[idx]
        eul = rotation_matrix_to_euler_deg(rots[idx])
        f_px = focals[idx, 0]
        f_mm = focal_length_px_to_mm(f_px, traj_cfg.sensor_width_mm, traj_cfg.image_width)
        fov_deg = 2.0 * math.atan((traj_cfg.image_width * 0.5) / f_px) * (180.0 / math.pi)
        print(f"   {t_sec:4.1f}s  | {p[0]:+7.2f} | {p[1]:+7.2f} | {p[2]:+7.2f} | {eul[0]:+6.1f}° | {eul[1]:+6.1f}° | {eul[2]:+6.1f}° | {f_mm:6.1f}mm | {fov_deg:5.1f}°")
    
    print("-" * 80)
    x_span = trans[:, 0].max() - trans[:, 0].min()
    y_span = trans[:, 1].max() - trans[:, 1].min()
    z_span = trans[:, 2].max() - trans[:, 2].min()
    f_start = focal_length_px_to_mm(focals[0, 0], traj_cfg.sensor_width_mm, traj_cfg.image_width)
    f_end = focal_length_px_to_mm(focals[-1, 0], traj_cfg.sensor_width_mm, traj_cfg.image_width)
    print(f"   📊 Trajectory Bounds: ΔX={x_span:.2f}m, ΔY={y_span:.2f}m, ΔZ={z_span:.2f}m | Optical Transition: {f_start:.1f}mm -> {f_end:.1f}mm")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Stage 1: Camera Trajectory Engine - Diverse Cinematic Multi-Prompt Test Suite")
    print(f"Running on: {device} | Checkpoint: checkpoints/checkpoint_final.pt")
    print("=" * 80)
    
    traj_cfg = TrajectoryConfig()
    model_cfg = ModelConfig()
    
    model = CameraTrajectoryEngine(model_cfg, traj_cfg)
    ckpt = torch.load("checkpoints/checkpoint_final.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    tok = SimpleDirectorialTokenizer(model_cfg.vocab_size)
    
    # 1. Counter-Clockwise Telephoto Orbit (85mm)
    p1 = "Slow cinematic counter clockwise orbit around hero statue with 85mm portrait telephoto lens"
    v1 = np.array([-0.5, 4.5, 1.4, 0.0, 85.0, 85.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    evaluate_prompt(model, tok, traj_cfg, device, p1, v1, "Shot 1: Counter-Clockwise Telephoto Orbit (85mm)")
    
    # 2. Reverse Vertigo Dolly-Out (Zooming telephoto while moving back)
    p2 = "Dramatic reverse vertigo dolly zoom tracking backward while zooming telephoto"
    v2 = np.array([-1.0, 2.0, 1.2, 1.0, 35.0, 105.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    evaluate_prompt(model, tok, traj_cfg, device, p2, v2, "Shot 2: Reverse Vertigo Dolly-Zoom (Dolly Back & Zoom In)")
    
    # 3. Ultra-Wide Descending Inward Spiral (24mm)
    p3 = "Fast descending inward spiral towards central frozen hero with 24mm ultra wide lens"
    v3 = np.array([2.0, 5.5, -1.2, 0.0, 24.0, 24.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    evaluate_prompt(model, tok, traj_cfg, device, p3, v3, "Shot 3: Ultra-Wide Descending Spiral (24mm)")
    
    # 4. Cinematic Ground-to-Overhead Crane Shot (50mm)
    p4 = "Smooth crane shot rising from ground level to high angle overview with 50mm lens"
    v4 = np.array([0.0, 3.0, 2.5, 0.0, 50.0, 50.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    evaluate_prompt(model, tok, traj_cfg, device, p4, v4, "Shot 4: Ground-to-Overhead Crane Elevation (50mm)")
    
    # 5. High-Speed Dynamic FPV Fly-Through
    p5 = "High speed fpv drone fly through past frozen statue with dynamic roll and tilt"
    v5 = np.array([2.5, 6.0, 0.8, 0.0, 28.0, 35.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    evaluate_prompt(model, tok, traj_cfg, device, p5, v5, "Shot 5: High-Speed FPV Fly-Through")
    
    # 6. Expansive Wide Panoramic Orbit (35mm)
    p6 = "Wide cinematic 360 degree panoramic orbit revealing the frozen landscape"
    v6 = np.array([1.0, 6.5, 2.0, 0.0, 35.0, 35.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    evaluate_prompt(model, tok, traj_cfg, device, p6, v6, "Shot 6: Expansive Panoramic Orbit (35mm)")
    
    print("\n" + "=" * 80)
    print("✨ ALL 6 CINEMATIC DIRECTORIAL PROMPT TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
