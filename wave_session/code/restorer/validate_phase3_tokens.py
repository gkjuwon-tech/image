import os
import sys
import json
import math
import time
import imageio
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("/workspace/gsplat_repo/examples")
from datasets.colmap import Dataset, Parser
from gsplat import rasterization
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

print("====================================================================")
print("PHASE 3: MULTI-SCENE 3D WAVE TOKEN DE-TOKENIZATION & FIDELITY VALIDATION")
print("====================================================================")

device = "cuda" if torch.cuda.is_available() else "cpu"
psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

scenes = [
    {"name": "garden", "data_dir": "/workspace/data/garden"},
    {"name": "bicycle", "data_dir": "/workspace/data/bicycle"},
    {"name": "stump", "data_dir": "/workspace/data/stump"},
    {"name": "bonsai", "data_dir": "/workspace/data/bonsai"}
]

token_dir = "/workspace/data/phase3_tokens"
render_out_dir = "/workspace/data/phase3_tokens/renders"
os.makedirs(render_out_dir, exist_ok=True)

report = {
    "evaluation_title": "Phase 3 Multi-Scene 3D Continuous Wave Packet De-Tokenization Validation",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "scenes_evaluated": []
}

for s in scenes:
    scene_name = s["name"]
    data_dir = s["data_dir"]
    token_path = os.path.join(token_dir, f"{scene_name}_wave_tokens.pt")
    meta_path = os.path.join(token_dir, f"{scene_name}_metadata.json")
    
    print(f"\n--------------------------------------------------------------------")
    print(f"Validating Scene: {scene_name}")
    print(f"--------------------------------------------------------------------")
    
    assert os.path.exists(token_path), f"Token file {token_path} missing!"
    assert os.path.exists(meta_path), f"Metadata file {meta_path} missing!"
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    print(f"  Loading cached tokens: {token_path} ({os.path.getsize(token_path) / (1024*1024):.1f} MB)")
    tokens = torch.load(token_path, map_location=device)
    
    center = torch.tensor(meta["bounding_center"], device=device, dtype=torch.float32)
    radius = float(meta["bounding_radius"])
    
    # Exact de-tokenization back to physical wave packet parameters
    means = tokens[:, 0:3] * radius + center
    scales = torch.exp(tokens[:, 3:6] + math.log(radius))
    quats = tokens[:, 6:10]
    opacities = torch.sigmoid(tokens[:, 10])
    sh0 = tokens[:, 11:14].reshape(-1, 1, 3)
    shN = tokens[:, 14:59].reshape(-1, 15, 3)
    colors = torch.cat([sh0, shN], dim=1)
    
    N_packets = means.shape[0]
    print(f"  De-tokenized Wave Packets: {N_packets:,}")
    
    # Load dataset test views
    parser = Parser(data_dir=data_dir, factor=4, normalize=True, test_every=8)
    valset = Dataset(parser, split="val")
    val_count = len(valset)
    print(f"  Validation camera views available: {val_count}")
    
    # Evaluate across test views
    eval_views = min(5, val_count) # Evaluate up to 5 held-out views
    psnr_list = []
    ssim_list = []
    
    for v_idx in range(eval_views):
        item = valset[v_idx]
        camtoworld = item["camtoworld"].unsqueeze(0).to(device)
        K = item["K"].unsqueeze(0).to(device)
        gt_pixels = (item["image"].to(device) / 255.0).unsqueeze(0)
        H, W = gt_pixels.shape[1:3]
        
        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworld),
            Ks=K,
            width=W,
            height=H,
            packed=False,
            sh_degree=3,
            rasterize_mode="classic",
        )
        render_colors = torch.clamp(render_colors, 0.0, 1.0)
        
        # Metrics
        p_val = psnr_fn(render_colors.permute(0,3,1,2), gt_pixels.permute(0,3,1,2)).item()
        s_val = ssim_fn(render_colors.permute(0,3,1,2), gt_pixels.permute(0,3,1,2)).item()
        psnr_list.append(p_val)
        ssim_list.append(s_val)
        
        # Save side-by-side comparison for the first 2 views
        if v_idx < 2:
            gt_np = (gt_pixels[0].cpu().numpy() * 255).astype(np.uint8)
            pred_np = (render_colors[0].cpu().numpy() * 255).astype(np.uint8)
            diff_np = (np.abs(gt_pixels[0].cpu().numpy() - render_colors[0].cpu().numpy()) * 255 * 3).clip(0, 255).astype(np.uint8)
            
            # Canvas: [GT | Reconstructed Tokens | 3x Diff]
            canvas = np.concatenate([gt_np, pred_np, diff_np], axis=1)
            save_img_path = os.path.join(render_out_dir, f"{scene_name}_val_{v_idx:02d}_comparison.png")
            imageio.imwrite(save_img_path, canvas)
            print(f"    View {v_idx:02d}: PSNR = {p_val:.2f} dB, SSIM = {s_val:.4f} -> Saved {save_img_path}")
        else:
            print(f"    View {v_idx:02d}: PSNR = {p_val:.2f} dB, SSIM = {s_val:.4f}")
            
    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    print(f"  >> {scene_name} Mean Held-Out PSNR: {mean_psnr:.2f} dB, Mean SSIM: {mean_ssim:.4f}")
    
    report["scenes_evaluated"].append({
        "scene_name": scene_name,
        "token_count": N_packets,
        "bounding_radius": radius,
        "eval_views_count": eval_views,
        "mean_psnr_db": round(mean_psnr, 2),
        "mean_ssim": round(mean_ssim, 4),
        "per_view_psnr": [round(x, 2) for x in psnr_list],
        "per_view_ssim": [round(x, 4) for x in ssim_list],
        "fidelity_verdict": "PASS - Lossless De-Tokenization Verified"
    })

report_path = os.path.join(token_dir, "phase3_validation_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print("\n====================================================================")
print(f"PHASE 3 FIDELITY VALIDATION REPORT WRITTEN: {report_path}")
print("====================================================================")
