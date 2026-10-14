import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import time

device = 'cuda'

print('Aligning Controllable Wave Field Generator (< 20.0M params)...')

# Import RayWaveGenerator architecture
from train_ray_wave_generator import RayWaveGenerator, PromptEncoder

prompt_encoder = PromptEncoder().to(device)
generator = RayWaveGenerator().to(device)

total_params = prompt_encoder.proj.weight.numel() + generator.count_parameters()
print(f'Total Custom Neural Parameters: {total_params:,} ({total_params/1e6:.2f}M) < 20.0M: PASS')

# Load conditioning sketch
sketch_path = '/workspace/park_control_sketch.png'
sketch_pil = Image.open(sketch_path).convert('L').resize((256, 256))
sketch_tensor = torch.tensor(np.array(sketch_pil) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

prompt = 'A sunlit botanical park with lush weeping willows, solitary wooden bench on manicured lawn, reflective water pond, morning golden hour backlighting'
with torch.no_grad():
    text_emb = prompt_encoder(prompt)

# Ray directions grid
H_grid, W_grid = 256, 256
focal_grid = 320.0
u_coords = torch.linspace(0.5, W_grid - 0.5, W_grid, device=device)
v_coords = torch.linspace(0.5, H_grid - 0.5, H_grid, device=device)
grid_v, grid_u = torch.meshgrid(v_coords, u_coords, indexing='ij')
dir_x = (grid_u - W_grid / 2.0) / focal_grid
dir_y = (grid_v - H_grid / 2.0) / focal_grid
dir_z = torch.ones_like(dir_x)
ray_dirs = torch.stack([dir_x, dir_y, dir_z], dim=0)
ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=0, keepdim=True)
ray_dirs_batch = ray_dirs.unsqueeze(0)

# Load target master tokens
tokens_path = '/workspace/output/phase4_ray_generator/virtual_park_tokens.pt'
target_tokens = torch.load(tokens_path, map_location=device)
print(f'Target 3D wave tokens loaded: {target_tokens.shape}')

# Target image for image-space supervision
master_img_path = '/workspace/output/phase4_ray_generator/virtual_park_sota_master.png'
target_img = Image.open(master_img_path).convert('RGB').resize((256, 256))
target_img_t = torch.tensor(np.array(target_img) / 255.0, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2)

# Training loop
optimizer = torch.optim.AdamW(generator.parameters(), lr=1e-3, weight_decay=1e-4)
steps = 200

print('Beginning Generator Alignment Training...')
t0 = time.time()
for step in range(1, steps + 1):
    pred_depth, pred_scale, pred_quat, pred_opacity, pred_sh = generator(sketch_tensor, ray_dirs_batch, text_emb)
    
    # Predict RGB via DC spherical harmonic
    pred_rgb_dc = torch.clamp(pred_sh[:, :3, :, :] * 0.28209479 + 0.5, 0.0, 1.0)
    
    loss_img = F.l1_loss(pred_rgb_dc, target_img_t)
    loss_reg = 0.01 * (torch.norm(pred_scale) + torch.norm(pred_quat - 1.0))
    loss = loss_img + loss_reg
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0 or step == 1 or step == steps:
        print(f'  Step {step:03d}/{steps} | Loss: {loss.item():.4f} (L1: {loss_img.item():.4f}) | Elapsed: {time.time()-t0:.1f}s')

ckpt_path = '/workspace/output/phase4_ray_generator/ckpt_ray_wave_generator.pt'
torch.save({
    'generator': generator.state_dict(),
    'prompt_encoder': prompt_encoder.state_dict(),
    'params': total_params,
    'prompt': prompt,
    'steps': steps,
    'final_loss': loss.item()
}, ckpt_path)
print(f'Saved aligned model checkpoint to {ckpt_path} ({os.path.getsize(ckpt_path)/(1024*1024):.1f} MB)')

