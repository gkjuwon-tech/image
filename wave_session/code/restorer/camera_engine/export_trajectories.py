"""Export 300-Frame Trajectories from Trained Model to JSON for Three.js Visualization.
"""

import json
import math
import torch
import numpy as np

from config import TrajectoryConfig, ModelConfig
from models.camera_engine import CameraTrajectoryEngine
from data.procedural_generator import SimpleDirectorialTokenizer
from geometry.camera_intrinsics import focal_length_px_to_mm


def export():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traj_cfg = TrajectoryConfig()
    model_cfg = ModelConfig()
    
    model = CameraTrajectoryEngine(model_cfg, traj_cfg)
    ckpt = torch.load("checkpoints/checkpoint_final.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    tok = SimpleDirectorialTokenizer(model_cfg.vocab_size)
    
    shots_spec = [
        {
            "id": "orbit",
            "title": "360° Cinematic Orbit",
            "prompt": "Cinematic clockwise 360 degree orbit around frozen center with 50mm lens",
            "vec": [1.0, 3.5, 1.2, 0.0, 50.0, 50.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "id": "vertigo",
            "title": "Hitchcock Vertigo (Dolly-Zoom)",
            "prompt": "Dramatic Hitchcock vertigo dolly zoom tracking forward while zooming wide",
            "vec": [1.0, 6.0, 1.2, 1.0, 100.0, 30.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "id": "spiral",
            "title": "Ascending Inward Spiral",
            "prompt": "Dynamic ascending spiral inwards towards central statue with 35mm wide lens",
            "vec": [1.5, 5.0, 1.5, 0.0, 35.0, 35.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "id": "fpv",
            "title": "FPV Drone Aerial Reveal",
            "prompt": "Fast fpv drone aerial reveal tracking backward and tilting up",
            "vec": [-1.0, 4.0, 2.5, 0.0, 24.0, 50.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    ]
    
    exported_data = {
        "metadata": {
            "fps": 30,
            "duration_sec": 10.0,
            "total_frames": 300,
            "resolution": [7680, 4320]
        },
        "shots": {}
    }
    
    with torch.no_grad():
        for s in shots_spec:
            tokens, mask = tok.encode(s["prompt"], max_len=64)
            tokens_t = tokens.unsqueeze(0).to(device)
            mask_t = mask.unsqueeze(0).to(device)
            vec_t = torch.tensor(s["vec"], dtype=torch.float32).unsqueeze(0).to(device)
            
            preds = model(tokens_t, vec_t, text_mask=mask_t)
            
            trans = preds["translations"][0].cpu().numpy().tolist()       # [300, 3]
            quats = preds["quaternions"][0].cpu().numpy().tolist()         # [300, 4] as [w, x, y, z]
            focals_px = preds["focal_lengths"][0].cpu().numpy()            # [300, 2]
            
            focals_mm = [
                round(focal_length_px_to_mm(f[0], traj_cfg.sensor_width_mm, traj_cfg.image_width), 2)
                for f in focals_px
            ]
            fovs_deg = [
                round(2.0 * math.atan((traj_cfg.image_width * 0.5) / f[0]) * (180.0 / math.pi), 2)
                for f in focals_px
            ]
            
            # Clean rounding for compact JSON
            trans_rounded = [[round(coord, 4) for coord in p] for p in trans]
            quats_rounded = [[round(q, 4) for q in quat] for quat in quats]
            
            exported_data["shots"][s["id"]] = {
                "title": s["title"],
                "prompt": s["prompt"],
                "positions": trans_rounded,
                "quaternions": quats_rounded,
                "focal_mm": focals_mm,
                "fov_deg": fovs_deg
            }
            
    with open("trajectories_export.json", "w") as f:
        json.dump(exported_data, f, indent=2)
    print("[+] Successfully exported trajectories to trajectories_export.json")


if __name__ == "__main__":
    export()
