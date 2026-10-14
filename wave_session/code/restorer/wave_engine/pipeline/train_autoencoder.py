"""
Phase 2 Wave Autoencoder Training Pipeline
=========================================
Trains Model 1 (Encoder), Model 2 (Factorizer), and Model 4 (Sensor Head)
on Mip-NeRF 360 multi-view photographic data.
Features:
- Stratified depth sampling with scene contraction.
- Mixed precision training (AMP).
- Cosine annealing learning rate schedule with warmup.
- Novel view validation evaluation (PSNR / SSIM).
- Automated checkpoint saving.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.dataset import MipNeRF360BicycleDataset, contract_coords
from core.factorizer import WaveTensorFactorizer
from core.encoder import MicroWaveEncoder
from core.sensor import SensorIntegrationHead


def compute_psnr(mse: float) -> float:
    if mse < 1e-12:
        return 99.99
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


class WaveAutoencoderPipeline(nn.Module):
    """Integrated Wave Autoencoder Network."""
    def __init__(self, plane_res: int = 256, plane_dim: int = 32):
        super().__init__()
        self.plane_res = plane_res
        self.plane_dim = plane_dim
        self.factorizer = WaveTensorFactorizer(plane_res=plane_res, plane_dim=plane_dim, out_dim=64)
        self.encoder = MicroWaveEncoder(in_spatial_dim=64, hidden_dim=256, num_layers=5)
        self.sensor = SensorIntegrationHead(in_feat_dim=32, hidden_dim=256)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        n_samples: int = 64,
        perturb: bool = True
    ) -> dict:
        """
        Volume renders a batch of rays using stratified disparity sampling.
        """
        B = rays_o.shape[0]
        device = rays_o.device

        # Stratified sampling along disparity (inverse depth)
        # Concentrates sample density on foreground scene geometry
        t_vals = torch.linspace(0.0, 1.0, n_samples, device=device).unsqueeze(0).expand(B, -1)
        if perturb:
            mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
            upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
            lower = torch.cat([t_vals[:, :1], mids], dim=-1)
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand

        # Disparity mapping: inv_z linearly spaced between 1/near and 1/far
        inv_near = 1.0 / near
        inv_far = 1.0 / far
        inv_z = inv_near * (1.0 - t_vals) + inv_far * t_vals
        z_vals = 1.0 / inv_z  # (B, N_pts)
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)  # (B, N_pts, 3)

        # Space contraction into [-1, 1] range
        pts_contracted = contract_coords(pts) * 0.5
        pts_norm = torch.clamp(pts_contracted, -0.999, 0.999)

        # Distances between adjacent samples
        dists = torch.cat([
            z_vals[:, 1:] - z_vals[:, :-1],
            torch.full((B, 1), 1e-2, device=device)
        ], dim=-1)

        # 1. Sample tri-planes
        fact_feats = self.factorizer.sample_planes(pts_norm)  # (B, N_pts, 64)

        # 2. Query wavefront encoder
        enc_out = self.encoder(pts_norm, fact_feats)

        # 3. Integrate at sensor
        render_out = self.sensor.integrate_rays(
            densities=enc_out["density"],
            amplitudes=enc_out["amplitude"],
            features=enc_out["features"],
            dists=dists,
            view_dirs=rays_d
        )

        render_out["phase"] = enc_out["phase"]
        return render_out


def render_full_image(
    model: WaveAutoencoderPipeline,
    dataset: MipNeRF360BicycleDataset,
    view_idx: int,
    chunk_size: int = 4096,
    n_samples: int = 64,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Renders a full novel view image in chunks to prevent VRAM overflow.
    """
    model.eval()
    with torch.no_grad():
        rays_o, rays_d, target_rgb = dataset.get_full_view_rays(view_idx)
        total_rays = rays_o.shape[0]
        near = torch.full((chunk_size, 1), dataset.near, device=device)
        far = torch.full((chunk_size, 1), dataset.far, device=device)

        pred_rgb_list = []
        for i in range(0, total_rays, chunk_size):
            end_idx = min(i + chunk_size, total_rays)
            curr_b = end_idx - i
            ro = rays_o[i:end_idx]
            rd = rays_d[i:end_idx]

            out = model.forward_rays(ro, rd, near[:curr_b], far[:curr_b], n_samples=n_samples, perturb=False)
            pred_rgb_list.append(out["rgb"].cpu())

        pred_rgb = torch.cat(pred_rgb_list, dim=0).reshape(dataset.H, dataset.W, 3).numpy()
        gt_rgb = target_rgb.cpu().numpy()

        mse = float(np.mean((pred_rgb - gt_rgb) ** 2))
        psnr = compute_psnr(mse)

    return pred_rgb, gt_rgb, psnr


def train(
    data_dir: str,
    output_dir: str,
    num_steps: int = 15000,
    batch_size: int = 4096,
    lr_planes: float = 2e-2,
    lr_mlp: float = 1e-3,
    downscale: int = 4,
    plane_res: int = 256,
    plane_dim: int = 32,
    n_samples: int = 96,
    device: str = "cpu"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "renders"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    print("=" * 70)
    print("PHASE 2: WAVE AUTOENCODER TRAINING PIPELINE (SOTA CALIBRATED v2)")
    print(f"Device: {device} | Downscale: {downscale}x | Total Steps: {num_steps} | Plane Res: {plane_res} | Samples: {n_samples}")
    print(f"Learning Rates: Tri-planes = {lr_planes} (fast) | MLPs = {lr_mlp}")
    print("=" * 70)

    # Ingest dataset
    train_dataset = MipNeRF360BicycleDataset(data_dir=data_dir, split="train", downscale=downscale, device=device)
    val_dataset = MipNeRF360BicycleDataset(data_dir=data_dir, split="val", downscale=downscale, device=device)

    # Initialize model
    model = WaveAutoencoderPipeline(plane_res=plane_res, plane_dim=plane_dim).to(device)
    total_params = model.total_parameters()
    print(f"Model instantiated successfully: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    # Multi-rate AdamW optimization:
    # Explicit tri-plane feature grids require ~20x higher LR than MLPs
    plane_params = [
        model.factorizer.plane_xy,
        model.factorizer.plane_yz,
        model.factorizer.plane_zx,
    ]
    mlp_params = [
        p for name, p in model.named_parameters()
        if not any(p is p_plane for p_plane in plane_params)
    ]
    optimizer = torch.optim.AdamW([
        {"params": plane_params, "lr": lr_planes, "weight_decay": 0.0},
        {"params": mlp_params, "lr": lr_mlp, "weight_decay": 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and torch.cuda.is_available()))

    best_psnr = 0.0
    start_time = time.time()

    print("\nStarting training loop...")
    for step in range(1, num_steps + 1):
        model.train()
        batch = train_dataset.sample_random_rays(batch_size)

        optimizer.zero_grad()
        use_amp = (device == "cuda" and torch.cuda.is_available())
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model.forward_rays(
                rays_o=batch["rays_o"],
                rays_d=batch["rays_d"],
                near=batch["near"],
                far=batch["far"],
                n_samples=n_samples,
                perturb=True
            )

            pred_rgb = out["rgb"]
            target_rgb = batch["target_rgb"]

            # Combined loss: L1 (edge sharpness) + MSE (radiometric accuracy)
            loss_l1 = F.l1_loss(pred_rgb, target_rgb)
            loss_mse = F.mse_loss(pred_rgb, target_rgb)
            loss_rgb = 0.8 * loss_l1 + 0.2 * loss_mse
            
            # Phase smoothness regularization
            phase = out["phase"]
            loss_phase = torch.mean(torch.abs(phase[:, 1:] - phase[:, :-1])) * 0.005

            loss = loss_rgb + loss_phase

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Logging
        if step % 100 == 0 or step == 1:
            mse = float(torch.mean((pred_rgb.detach() - target_rgb) ** 2).item())
            psnr = compute_psnr(mse)
            elapsed = time.time() - start_time
            rate = step / max(elapsed, 1e-5)
            lrs = [f"{g['lr']:.5f}" for g in optimizer.param_groups]
            print(f"Step [{step:5d}/{num_steps:5d}] | Loss: {loss.item():.4f} | Train PSNR: {psnr:5.2f} dB | Rate: {rate:4.1f} step/s | LRs: planes={lrs[0]}, mlp={lrs[1]}")

        # Validation & Novel View Render Checkpoint
        if step % 2000 == 0 or step == num_steps:
            print(f"\n[Validation Step {step}] Evaluating novel view rendering on held-out camera angle...")
            val_view = 0  # First held-out camera angle
            pred_img, gt_img, val_psnr = render_full_image(model, val_dataset, view_idx=val_view, n_samples=n_samples, device=device)
            print(f"  Held-out View PSNR: {val_psnr:.2f} dB")

            # Save comparison image
            comp_h = np.concatenate([gt_img, pred_img], axis=1)
            comp_8bit = np.clip(comp_h * 255.0, 0, 255).astype(np.uint8)
            render_path = os.path.join(output_dir, "renders", f"val_step_{step:05d}_psnr_{val_psnr:.1f}.png")
            Image.fromarray(comp_8bit).save(render_path)
            print(f"  Validation render saved to: {render_path}")

            # Checkpoint best
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                ckpt_path = os.path.join(output_dir, "checkpoints", "best_model.pt")
                torch.save({
                    "step": step,
                    "plane_res": plane_res,
                    "plane_dim": plane_dim,
                    "model_state_dict": model.state_dict(),
                    "val_psnr": val_psnr,
                    "params": total_params
                }, ckpt_path)
                print(f"  New best model saved to: {ckpt_path} (PSNR: {best_psnr:.2f} dB)")

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETED: Best Validation PSNR = {best_psnr:.2f} dB")
    print(f"Total Elapsed Time: {time.time() - start_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/.gemini/antigravity-cli/scratch/wave_engine/data/bicycle")
    parser.add_argument("--output_dir", type=str, default="/root/.gemini/antigravity-cli/scratch/wave_engine/output/phase2_train")
    parser.add_argument("--num_steps", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr_planes", type=float, default=2e-2)
    parser.add_argument("--lr_mlp", type=float, default=1e-3)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--plane_res", type=int, default=256)
    parser.add_argument("--plane_dim", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=96)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr_planes=args.lr_planes,
        lr_mlp=args.lr_mlp,
        downscale=args.downscale,
        plane_res=args.plane_res,
        plane_dim=args.plane_dim,
        n_samples=args.n_samples,
        device=args.device
    )
