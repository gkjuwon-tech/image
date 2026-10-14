import os
import sys
import json
import numpy as np
from plyfile import PlyData, PlyElement

def clean_scene(scene_dir, max_needle_ratio=3.5, min_opacity=0.05, max_scale_rel=0.05):
    ply_path = os.path.join(scene_dir, "point_cloud.ply")
    cam_path = os.path.join(scene_dir, "cameras.json")
    out_ply_path = os.path.join(scene_dir, "point_cloud_clean.ply")
    out_cam_path = os.path.join(scene_dir, "cameras_clean.json")

    if not os.path.exists(ply_path) or not os.path.exists(cam_path):
        print(f"Skipping {scene_dir}: missing ply or cameras.json")
        return None

    scene_name = os.path.basename(scene_dir)
    print(f"\n==========================================")
    print(f"Cleaning Scene: {scene_name}")
    print(f"==========================================")

    # 1. Load cameras to determine scene envelope
    with open(cam_path, "r") as f:
        cams = json.load(f)
    
    cam_positions = np.array([c["position"] for c in cams], dtype=np.float32)
    scene_center = cam_positions.mean(axis=0)
    cam_dists = np.linalg.norm(cam_positions - scene_center, axis=-1)
    r_cam = np.median(cam_dists)
    r_max = np.max(cam_dists)
    scene_radius = max(r_max * 2.0, r_cam * 3.0)
    max_abs_scale = max(0.15, float(scene_radius * max_scale_rel))

    print(f"Cameras: {len(cams)}, Scene Center: {scene_center.round(3)}, Scene Radius: {scene_radius:.2f}m, Max Scale: {max_abs_scale:.3f}m")

    # 2. Load Point Cloud
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    N_orig = len(v)

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales_raw = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)
    scales = np.exp(scales_raw)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1)
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"]))
    
    # Check distances to scene center
    dist_pts = np.linalg.norm(xyz - scene_center, axis=-1)

    # Filter 1: Ghost Opacity + Spatial Envelope
    mask_valid = (opacity >= min_opacity) & (dist_pts <= scene_radius)
    print(f"Initial: {N_orig:,} pts | Valid Opacity & Envelope: {mask_valid.sum():,} pts ({mask_valid.sum()/N_orig*100:.1f}%)")

    # Extract valid points
    xyz_val = xyz[mask_valid]
    scales_val = scales[mask_valid]
    rot_val = rot[mask_valid]
    opacity_val = v["opacity"][mask_valid]

    # Extract all spherical harmonics fields
    sh_fields = [f for f in v.data.dtype.names if f.startswith("f_")]
    sh_data = {f: v[f][mask_valid] for f in sh_fields}

    # Filter 2: Needle Regularization (clamp s1 <= max_needle_ratio * s2)
    s_sorted = np.sort(scales_val, axis=-1)[:, ::-1]
    s1, s2, s3 = s_sorted[:, 0], s_sorted[:, 1], s_sorted[:, 2]

    # Needle ratio
    needle_ratios = s1 / (s2 + 1e-8)
    num_needles = int((needle_ratios > max_needle_ratio).sum())
    print(f"Needle splats exceeding ratio {max_needle_ratio}: {num_needles:,} ({num_needles/len(scales_val)*100:.1f}%)")

    # Regularize scales: clamp needle elongation
    s1_clamped = np.minimum(s1, max_needle_ratio * s2)
    s1_clamped = np.minimum(s1_clamped, max_abs_scale)
    s2_clamped = np.minimum(s2, max_abs_scale)
    s3_clamped = np.minimum(s3, max_abs_scale)

    # Reconstruct scales with original orientation
    sort_order = np.argsort(scales_val, axis=-1)[:, ::-1]
    scales_reg = np.zeros_like(scales_val)
    np.put_along_axis(scales_reg, sort_order[:, 0:1], s1_clamped[:, None], axis=-1)
    np.put_along_axis(scales_reg, sort_order[:, 1:2], s2_clamped[:, None], axis=-1)
    np.put_along_axis(scales_reg, sort_order[:, 2:3], s3_clamped[:, None], axis=-1)

    # Convert back to log scale for PLY
    scales_log = np.log(np.maximum(scales_reg, 1e-8))

    # Construct new PLY elements
    N_clean = len(xyz_val)
    dtype_list = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ]
    for f in sh_fields:
        dtype_list.append((f, "f4"))
    dtype_list.append(("opacity", "f4"))
    dtype_list.extend([("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")])
    dtype_list.extend([("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")])

    vertex_clean = np.empty(N_clean, dtype=dtype_list)
    vertex_clean["x"] = xyz_val[:, 0]
    vertex_clean["y"] = xyz_val[:, 1]
    vertex_clean["z"] = xyz_val[:, 2]
    vertex_clean["nx"] = 0.0
    vertex_clean["ny"] = 0.0
    vertex_clean["nz"] = 0.0

    for f in sh_fields:
        vertex_clean[f] = sh_data[f]

    vertex_clean["opacity"] = opacity_val
    vertex_clean["scale_0"] = scales_log[:, 0]
    vertex_clean["scale_1"] = scales_log[:, 1]
    vertex_clean["scale_2"] = scales_log[:, 2]
    vertex_clean["rot_0"] = rot_val[:, 0]
    vertex_clean["rot_1"] = rot_val[:, 1]
    vertex_clean["rot_2"] = rot_val[:, 2]
    vertex_clean["rot_3"] = rot_val[:, 3]

    el = PlyElement.describe(vertex_clean, "vertex")
    PlyData([el]).write(out_ply_path)
    print(f"Saved Cleaned PLY ({N_clean:,} Gaussians, 0 needles, 0 ghost floaters) -> {out_ply_path}")

    # 3. Filter Valid Cameras (cameras facing the scene)
    valid_cams = []
    for c in cams:
        R = np.array(c["rotation"])
        pos = np.array(c["position"])
        view_dir = R[2, :]
        vec_to_scene = scene_center - pos
        vec_dist = np.linalg.norm(vec_to_scene)
        if vec_dist > 1e-4:
            vec_dir = vec_to_scene / vec_dist
            cos_angle = np.dot(view_dir, vec_dir)
            if cos_angle > -0.2:
                valid_cams.append(c)
        else:
            valid_cams.append(c)

    print(f"Cameras Filtered: {len(valid_cams)} / {len(cams)} valid viewing angles")
    with open(out_cam_path, "w") as f:
        json.dump(valid_cams, f, indent=2)

    return out_ply_path, out_cam_path

if __name__ == "__main__":
    for s in ["truck", "train", "drjohnson", "bicycle"]:
        s_dir = f"/workspace/dataset_pairs/{s}"
        if os.path.exists(s_dir):
            clean_scene(s_dir)
