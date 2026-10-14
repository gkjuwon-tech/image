import torch
import numpy as np
import imageio
import gsplat

device = "cuda"

# 1. Load real Garden splats (real grass & foliage)
ckpt_garden = torch.load("/workspace/output/phase2_garden/ckpts/ckpt_6999_rank0.pt", map_location=device)
g_splats = ckpt_garden["splats"]
g_means = g_splats["means"]
g_scales = torch.exp(g_splats["scales"])
g_quats = g_splats["quats"]
g_opacities = torch.sigmoid(g_splats["opacities"])
g_colors = torch.cat([g_splats["sh0"], g_splats["shN"]], dim=1) # [N, 16, 3]

# Extract green grass/foliage
g_rgb = g_splats["sh0"].squeeze(1) * 0.28209479 + 0.5
is_green = (g_rgb[:, 1] > g_rgb[:, 0] + 0.03) & (g_rgb[:, 1] > g_rgb[:, 2] + 0.03)
idx_green = torch.where(is_green)[0]
idx_grass = idx_green[:100000]

print(f"Loaded {len(idx_grass):,} real grass wave packets from Garden.")

# Extract wood from Stump
ckpt_stump = torch.load("/workspace/output/phase3_stump/ckpts/ckpt_6999_rank0.pt", map_location=device)
s_splats = ckpt_stump["splats"]
s_means = s_splats["means"]
s_scales = torch.exp(s_splats["scales"])
s_quats = s_splats["quats"]
s_opacities = torch.sigmoid(s_splats["opacities"])
s_colors = torch.cat([s_splats["sh0"], s_splats["shN"]], dim=1)

s_rgb = s_splats["sh0"].squeeze(1) * 0.28209479 + 0.5
is_wood = (s_rgb[:, 0] > s_rgb[:, 1] * 1.05) & (s_rgb[:, 0] > s_rgb[:, 2] * 1.2)
idx_wood = torch.where(is_wood)[0][:50000]
print(f"Loaded {len(idx_wood):,} real wood wave packets from Stump.")

# Let's see: we map real grass to lawn (Y in [+0.8, +1.2], Z in [2, 15])
m_grass = g_means[idx_grass]
# Center and scale grass to lawn volume
m_grass_norm = (m_grass - m_grass.mean(dim=0)) / (m_grass.std(dim=0) + 1e-4)
lawn_means = torch.stack([
    m_grass_norm[:, 0] * 2.5,          # X width
    0.95 + m_grass_norm[:, 1] * 0.08,  # Y ground level (positive Y = bottom)
    3.0 + (m_grass_norm[:, 2] + 2.0) * 2.5 # Z depth
], dim=-1)

# Real wood to bench (X in [-0.7, -0.1], Y in [0.5, 0.85], Z in [2.8, 3.5])
m_wood = s_means[idx_wood]
m_wood_norm = (m_wood - m_wood.mean(dim=0)) / (m_wood.std(dim=0) + 1e-4)
bench_means = torch.stack([
    -0.45 + m_wood_norm[:, 0] * 0.28,
    0.68 + m_wood_norm[:, 1] * 0.15,
    3.1 + m_wood_norm[:, 2] * 0.2
], dim=-1)

# Combine
all_m = torch.cat([lawn_means, bench_means], dim=0)
all_s = torch.cat([g_scales[idx_grass] * 0.05, s_scales[idx_wood] * 0.02], dim=0)
all_q = torch.cat([g_quats[idx_grass], s_quats[idx_wood]], dim=0)
all_o = torch.cat([g_opacities[idx_grass], s_opacities[idx_wood]], dim=0)
all_c = torch.cat([g_colors[idx_grass], s_colors[idx_wood]], dim=0)

# Render test
c2w = torch.eye(4, device=device).unsqueeze(0)
K = torch.tensor([[1100.0, 0.0, 512.0], [0.0, 1100.0, 512.0], [0.0, 0.0, 1.0]], device=device).unsqueeze(0)

out, _, _ = gsplat.rasterization(
    means=all_m, quats=all_q, scales=all_s, opacities=all_o, colors=all_c,
    viewmats=torch.linalg.inv(c2w), Ks=K, width=1024, height=1024, packed=False, sh_degree=3
)
img = torch.clamp(out[0], 0.0, 1.0).cpu().numpy()
imageio.imwrite("/workspace/test_real_render.png", (img * 255).astype(np.uint8))
print("Saved /workspace/test_real_render.png successfully!")
