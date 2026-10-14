"""Interactive Prompt Testing and Inference Demo for Trained Camera Trajectory Engine.

Loads the trained checkpoint and executes multiple diverse natural language
directorial prompts to inspect continuous 300-frame trajectories, camera positions,
orientations, and dynamic focal lengths.
"""

import sys
import math
import time
import torch
import numpy as np

from config import TrajectoryConfig, ModelConfig
from models.camera_engine import CameraTrajectoryEngine
from data.procedural_generator import SimpleDirectorialTokenizer
from geometry.camera_intrinsics import focal_length_px_to_mm


def run_prompt_test(model, tokenizer, traj_cfg, device, prompt: str, directorial_vec: np.ndarray, title: str):
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
    
    translations = preds["translations"][0].cpu().numpy()  # (300, 3)
    rotations = preds["rotations"][0].cpu().numpy()        # (300, 3, 3)
    focals_px = preds["focal_lengths"][0].cpu().numpy()    # (300, 2)
    jerks = preds["jerks"][0].cpu().numpy()                # (297, 3)
    vertigo_factor = preds["vertigo_factor"][0].item()
    
    focals_mm = [
        focal_length_px_to_mm(f[0], traj_cfg.sensor_width_mm, traj_cfg.image_width)
        for f in focals_px
    ]
    mean_jerk = float(np.mean(np.linalg.norm(jerks, axis=-1)))
    
    print("-" * 75)
    print(f"[*] Prompt Test: {title}")
    print(f"    Prompt Text: \"{prompt}\"")
    print(f"    Inference Latency: {latency_ms:.2f} ms | Vertigo Confidence: {vertigo_factor:.4f} | Smoothness Jerk: {mean_jerk:.4f} m/s^3")
    print(f"    [Camera Spatial Position Progression (10 Seconds @ 30 FPS)]:")
    print(f"      - Frame   1 (0.0s): X={translations[0, 0]:+6.2f}m, Y={translations[0, 1]:+6.2f}m, Z={translations[0, 2]:+6.2f}m | Focal={focals_mm[0]:5.1f}mm")
    print(f"      - Frame  75 (2.5s): X={translations[74, 0]:+6.2f}m, Y={translations[74, 1]:+6.2f}m, Z={translations[74, 2]:+6.2f}m | Focal={focals_mm[74]:5.1f}mm")
    print(f"      - Frame 150 (5.0s): X={translations[149, 0]:+6.2f}m, Y={translations[149, 1]:+6.2f}m, Z={translations[149, 2]:+6.2f}m | Focal={focals_mm[149]:5.1f}mm")
    print(f"      - Frame 225 (7.5s): X={translations[224, 0]:+6.2f}m, Y={translations[224, 1]:+6.2f}m, Z={translations[224, 2]:+6.2f}m | Focal={focals_mm[224]:5.1f}mm")
    print(f"      - Frame 300 (10.s): X={translations[299, 0]:+6.2f}m, Y={translations[299, 1]:+6.2f}m, Z={translations[299, 2]:+6.2f}m | Focal={focals_mm[299]:5.1f}mm")
    print(f"    [3D Bounding Envelope]:")
    print(f"      X range: [{translations[:, 0].min():+.2f}m ~ {translations[:, 0].max():+.2f}m], "
          f"Y range: [{translations[:, 1].min():+.2f}m ~ {translations[:, 1].max():+.2f}m], "
          f"Z range: [{translations[:, 2].min():+.2f}m ~ {translations[:, 2].max():+.2f}m]")
    print(f"    [Optical Dynamics]: Lens transition from {focals_mm[0]:.1f}mm to {focals_mm[-1]:.1f}mm")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Stage 1: Camera Trajectory Engine - Multi-Prompt Inference Demonstration")
    print(f"Device: {device} | Loading: checkpoints/checkpoint_final.pt")
    print("=" * 80)
    
    traj_cfg = TrajectoryConfig()
    model_cfg = ModelConfig()
    
    model = CameraTrajectoryEngine(model_cfg, traj_cfg)
    ckpt = torch.load("checkpoints/checkpoint_final.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    tokenizer = SimpleDirectorialTokenizer(model_cfg.vocab_size)
    
    # Test Shot 1: 360-degree Orbit
    prompt_1 = "Cinematic clockwise 360 degree orbit around frozen center with 50mm lens"
    vec_1 = np.array([1.0, 3.5, 1.2, 0.0, 50.0, 50.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    run_prompt_test(model, tokenizer, traj_cfg, device, prompt_1, vec_1, "1. 360-Degree Orbit Shot")
    
    # Test Shot 2: Hitchcock Vertigo Dolly-Zoom
    prompt_2 = "Dramatic Hitchcock vertigo dolly zoom tracking forward while zooming wide"
    vec_2 = np.array([1.0, 6.0, 1.2, 1.0, 100.0, 30.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    run_prompt_test(model, tokenizer, traj_cfg, device, prompt_2, vec_2, "2. Hitchcock Vertigo (Dolly-Zoom) Shot")
    
    # Test Shot 3: Inward Ascending Spiral
    prompt_3 = "Dynamic ascending spiral inwards towards central statue with 35mm wide lens"
    vec_3 = np.array([1.5, 5.0, 1.5, 0.0, 35.0, 35.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    run_prompt_test(model, tokenizer, traj_cfg, device, prompt_3, vec_3, "3. Ascending Spiral Shot")
    
    # Test Shot 4: Fast FPV Drone Dive & Reveal
    prompt_4 = "Fast fpv drone aerial reveal tracking backward and tilting up"
    vec_4 = np.array([-1.0, 4.0, 2.5, 0.0, 24.0, 50.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    run_prompt_test(model, tokenizer, traj_cfg, device, prompt_4, vec_4, "4. FPV Aerial Reveal Shot")
    
    print("-" * 75)
    print("[+] All prompt inference tests executed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
