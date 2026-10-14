import os
import json
import math
import glob
import torch
import torch.nn.functional as F
import numpy as np

print("====================================================================")
print("PHASE 3: MULTI-SCENE 3D WAVE TOKEN EXTRACTION & PRE-CACHING")
print("====================================================================")

def morton_encode_3d(pos_norm):
    """
    Compute 3D Morton code (Z-order curve) for spatial locality sorting.
    pos_norm: [N, 3] tensor in [-1, 1]
    """
    # Map [-1, 1] to [0, 1023] (10-bit integer grid)
    grid_coords = torch.clamp(((pos_norm + 1.0) * 0.5 * 1023).long(), 0, 1023)
    x, y, z = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]
    
    def spread_bits(v):
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8))  & 0x0300F00F
        v = (v | (v << 4))  & 0x030C30C3
        v = (v | (v << 2))  & 0x09249249
        return v
    
    morton_codes = spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)
    return morton_codes

def extract_and_tokenize_scene(ckpt_path, scene_name, output_dir, seq_tokens=65536):
    """
    Loads trained 3D continuous wave packet checkpoint, normalizes coordinate frames,
    extracts canonical 59-dim wave tokens, sorts by spatial Morton code, and caches:
      1) Full lossless active wave tokens ({scene_name}_wave_tokens.pt)
      2) Fixed-length DiT sequence tokens ({scene_name}_tokens_65k.pt)
    """
    print(f"\n[Tokenizing Scene] {scene_name} from {ckpt_path}...")
    assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} does not exist!"
    
    data = torch.load(ckpt_path, map_location="cpu")
    step = data.get("step", 7000)
    splats = data["splats"]
    
    means = splats["means"].float()          # [N, 3]
    scales = splats["scales"].float()        # [N, 3]
    quats = splats["quats"].float()          # [N, 4]
    opacities = splats["opacities"].float()  # [N]
    sh0 = splats["sh0"].float().reshape(-1, 3) # [N, 3]
    shN = splats["shN"].float().reshape(-1, 45) # [N, 45]
    
    N_total = means.shape[0]
    print(f"  Raw 3D Continuous Wave Packets: {N_total:,}")
    
    # 1. Filter out near-zero opacity dead packets (opacity threshold 0.005)
    valid_mask = torch.sigmoid(opacities) > 0.005
    means = means[valid_mask]
    scales = scales[valid_mask]
    quats = quats[valid_mask]
    opacities = opacities[valid_mask]
    sh0 = sh0[valid_mask]
    shN = shN[valid_mask]
    N_active = means.shape[0]
    print(f"  Active Wave Packets (sigmoid opacity > 0.005): {N_active:,}")
    
    # 2. Compute Robust Spatial Bounding Sphere
    q_low = torch.quantile(means, 0.005, dim=0)
    q_high = torch.quantile(means, 0.995, dim=0)
    center = (q_low + q_high) * 0.5
    radius = torch.max(torch.norm(means - center, dim=1)).item()
    radius = max(radius, 1e-3)
    
    # Normalized coordinates in [-1, 1]
    means_norm = torch.clamp((means - center) / radius, -1.0, 1.0)
    
    # Normalize log scales by scene radius: s_norm = s / r => log(s_norm) = log(s) - log(r)
    scales_norm = scales - math.log(radius)
    
    # Normalize quaternions to unit norm
    quats_norm = F.normalize(quats, p=2, dim=-1)
    
    # 3. Concatenate Spherical Harmonics into 48-dim wave field vector
    # sh0: 3 channels (DC), shN: 45 channels (15 bands x 3 RGB)
    sh_all = torch.cat([sh0, shN], dim=-1) # [N, 48]
    
    # 4. Spatial Sorting via 3D Morton Space-Filling Curve
    morton_codes = morton_encode_3d(means_norm)
    sort_indices = torch.argsort(morton_codes)
    
    means_norm_sorted = means_norm[sort_indices]
    scales_norm_sorted = scales_norm[sort_indices]
    quats_norm_sorted = quats_norm[sort_indices]
    opacities_sorted = opacities[sort_indices]
    sh_all_sorted = sh_all[sort_indices]
    
    # 5. Assemble Full Canonical Lossless Token Tensor: [N_active, 59]
    token_full = torch.cat([
        means_norm_sorted,
        scales_norm_sorted,
        quats_norm_sorted,
        opacities_sorted.unsqueeze(-1),
        sh_all_sorted
    ], dim=-1).to(torch.float32)
    
    assert token_full.shape == (N_active, 59), f"Unexpected shape {token_full.shape}"
    
    # 6. Extract Fixed-Length DiT Sequence Tokens (Top seq_tokens by optical importance)
    if N_active > seq_tokens:
        sig_opac = torch.sigmoid(opacities_sorted)
        mean_scale = torch.exp(scales_norm_sorted).mean(dim=-1)
        importance = sig_opac / (mean_scale + 1e-4)
        top_k_indices = torch.topk(importance, k=seq_tokens, sorted=False).indices
        
        # Keep spatial Morton ordering for sequence model
        top_k_indices = torch.sort(top_k_indices).values
        token_seq = token_full[top_k_indices]
    else:
        token_seq = token_full
        
    K_seq = token_seq.shape[0]
    print(f"  Canonical Lossless Tokens: {N_active:,} (Dim: 59)")
    print(f"  Fixed Sequence Tokens (Morton-ordered): {K_seq:,} (Dim: 59)")
    
    # 7. Save Caching Files
    os.makedirs(output_dir, exist_ok=True)
    full_save_path = os.path.join(output_dir, f"{scene_name}_wave_tokens.pt")
    seq_save_path = os.path.join(output_dir, f"{scene_name}_tokens_65k.pt")
    meta_save_path = os.path.join(output_dir, f"{scene_name}_metadata.json")
    
    metadata = {
        "scene_name": scene_name,
        "source_checkpoint": ckpt_path,
        "training_step": step,
        "raw_wave_packets": N_total,
        "active_lossless_tokens": N_active,
        "fixed_sequence_tokens": K_seq,
        "token_feature_dim": 59,
        "feature_layout": {
            "means_norm": [0, 3],
            "log_scales_norm": [3, 6],
            "quats_unit": [6, 10],
            "logit_opacity": [10, 11],
            "spherical_harmonics_wave": [11, 59]
        },
        "bounding_center": center.tolist(),
        "bounding_radius": float(radius),
        "spatial_ordering": "morton_z_order_3d",
        "morton_grid_resolution": 1024,
        "paths": {
            "full_tokens": f"{scene_name}_wave_tokens.pt",
            "seq_tokens": f"{scene_name}_tokens_65k.pt",
            "metadata": f"{scene_name}_metadata.json"
        }
    }
    
    torch.save(token_full, full_save_path)
    torch.save(token_seq, seq_save_path)
    with open(meta_save_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"  Saved lossless tokens: {full_save_path} ({os.path.getsize(full_save_path) / (1024*1024):.2f} MB)")
    print(f"  Saved sequence tokens: {seq_save_path} ({os.path.getsize(seq_save_path) / (1024*1024):.2f} MB)")
    print(f"  Saved metadata: {meta_save_path}")
    return metadata

if __name__ == "__main__":
    out_dir = "/workspace/data/phase3_tokens"
    os.makedirs(out_dir, exist_ok=True)
    
    scenes = [
        {"name": "garden", "ckpt": "/workspace/output/phase2_garden/ckpts/ckpt_6999_rank0.pt"},
        {"name": "bicycle", "ckpt": "/workspace/output/phase2_wave_packets/ckpts/ckpt_29999_rank0.pt"},
        {"name": "stump", "ckpt": "/workspace/output/phase3_stump/ckpts/ckpt_6999_rank0.pt"},
        {"name": "bonsai", "ckpt": "/workspace/output/phase3_bonsai/ckpts/ckpt_6999_rank0.pt"}
    ]
    
    manifest = {
        "dataset_name": "Phase 3 Multi-Scene 3D Continuous Wave Packet Token Corpus",
        "format_version": "3.0.0",
        "token_dim": 59,
        "feature_description": "means(3) + log_scales(3) + quats(4) + opacity(1) + sh_harmonics(48)",
        "scenes": []
    }
    
    for s in scenes:
        meta = extract_and_tokenize_scene(s["ckpt"], s["name"], out_dir)
        manifest["scenes"].append(meta)
        
    manifest_path = os.path.join(out_dir, "phase3_dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n====================================================================")
    print("PHASE 3 TOKENIZATION COMPLETE! ALL 4 SCENES EXTRACTED & PRE-CACHED.")
    print(f"Master Manifest: {manifest_path}")
    print("====================================================================")
