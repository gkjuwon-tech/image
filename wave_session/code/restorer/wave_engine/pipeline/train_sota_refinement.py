"""
Phase 2.5 High-Frequency Wave Refinement Pipeline (Target: 30 dB)
================================================================
Takes pre-trained 3D geometric anchor from Phase 2 (Step 15,000),
upsamples tri-planes from 256x256 to 384x384 (14.9M parameters, < 20M limit),
and sharpens sub-pixel micro-structures using patch edge gradient supervision.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.dataset import MipNeRF360BicycleDataset
from pipeline.train_autoencoder import WaveAutoencoderPipeline, compute_psnr, render_full_image


def compute_edge_loss(pred_patch: torch.Tensor, target_patch: torch.Tensor) -> torch.Tensor:
    """Computes spatial finite-difference edge gradient loss across patches."""
    dx_p = pred_patch[:, :, 1:, :] - pred_patch[:, :, :-1, :]
    dx_t = target_patch[:, :, 1:, :] - target_patch[:, :, :-1, :]
    dy_p = pred_patch[:, 1:, :, :] - pred_patch[:, :-1, :, :]
    dy_t = target_patch[:, 1:, :, :] - target_patch[:, :-1, :, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


def train_refinement(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    num_steps: int = 8000,
    target_res: int = 384,
    n_samples: int = 112,
    downscale: int = 4,
    lr_planes: float = 8e-3,
    lr_mlp: float = 3e-4,
    device: str = "cpu"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "renders"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    print("=" * 70)
    print("PHASE 2.5: HIGH-FREQUENCY WAVE REFINEMENT (TARGET: 30 dB)")
    print(f"Loading Base Geometric Anchor: {checkpoint_path}")
    print(f"Upsampling Tri-Planes: 256x256 -> {target_res}x{target_res} | Samples/ray: {n_samples}")
    print("=" * 70)

    # Ingest dataset at 4x downscale
    train_dataset = MipNeRF360BicycleDataset(data_dir=data_dir, split="train", downscale=downscale, device=device)
    val_dataset = MipNeRF360BicycleDataset(data_dir=data_dir, split="val", downscale=downscale, device=device)

    # Load base checkpoint and upsample tri-planes
    ckpt = torch.load(checkpoint_path, map_location=device)
    base_state = ckpt["model_state_dict"]

    model = WaveAutoencoderPipeline(plane_res=target_res, plane_dim=32).to(device)

    # Bilinearly interpolate tri-plane weights to high-resolution grid
    for plane_name in ["factorizer.plane_xy", "factorizer.plane_yz", "factorizer.plane_zx"]:
        old_weight = base_state[plane_name]
        new_weight = F.interpolate(old_weight, size=(target_res, target_res), mode="bilinear", align_corners=True)
        base_state[plane_name] = new_weight

    model.load_state_dict(base_state)
    total_params = model.total_parameters()
    print(f"Model upsampled successfully: {total_params:,} parameters ({total_params/1e6:.2f}M, strictly < 20M)")

    # Multi-rate AdamW with fine-tuning learning rates
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

    best_psnr = ckpt.get("val_psnr", 0.0)
    start_time = time.time()

    print("\nStarting high-frequency refinement loop...")
    for step in range(1, num_steps + 1):
        model.train()

        # Alternate between patch-based edge sampling (70%) and random ray sampling (30%)
        use_patches = (step % 10 < 7)
        if use_patches:
            batch = train_dataset.sample_random_patches(num_patches=4, patch_size=32)
        else:
            batch = train_dataset.sample_random_rays(batch_size=4096)

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

            # Color loss
            loss_l1 = F.l1_loss(pred_rgb, target_rgb)
            loss_mse = F.mse_loss(pred_rgb, target_rgb)
            loss_color = 0.8 * loss_l1 + 0.2 * loss_mse

            # High-frequency edge gradient loss on patches
            if use_patches:
                p_pred = pred_rgb.reshape(4, 32, 32, 3)
                p_targ = target_rgb.reshape(4, 32, 32, 3)
                loss_edge = compute_edge_loss(p_pred, p_targ) * 0.15
            else:
                loss_edge = torch.tensor(0.0, device=device)

            # Phase smoothness
            phase = out["phase"]
            loss_phase = torch.mean(torch.abs(phase[:, 1:] - phase[:, :-1])) * 0.003

            loss = loss_color + loss_edge + loss_phase

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
            print(f"Refine [{step:5d}/{num_steps:5d}] | Loss: {loss.item():.4f} (Edge: {loss_edge.item():.4f}) | Train PSNR: {psnr:5.2f} dB | Rate: {rate:4.1f} step/s | LRs: {lrs[0]}")

        # Novel View Validation Checkpoint
        if step % 2000 == 0 or step == num_steps:
            print(f"\n[Validation Step {step}] Evaluating novel view rendering on held-out camera angle...")
            val_view = 0
            pred_img, gt_img, val_psnr = render_full_image(model, val_dataset, view_idx=val_view, n_samples=n_samples, device=device)
            print(f"  Held-out View PSNR: {val_psnr:.2f} dB")

            # Save comparison image
            comp_h = np.concatenate([gt_img, pred_img], axis=1)
            comp_8bit = np.clip(comp_h * 255.0, 0, 255).astype(np.uint8)
            render_path = os.path.join(output_dir, "renders", f"val_step_{step:05d}_psnr_{val_psnr:.1f}.png")
            Image.fromarray(comp_8bit).save(render_path)
            print(f"  Validation render saved to: {render_path}")

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                ckpt_path = os.path.join(output_dir, "checkpoints", "best_refined_model.pt")
                torch.save({
                    "step": step,
                    "plane_res": target_res,
                    "plane_dim": 32,
                    "model_state_dict": model.state_dict(),
                    "val_psnr": val_psnr,
                    "params": total_params
                }, ckpt_path)
                print(f"  New best refined model saved to: {ckpt_path} (PSNR: {best_psnr:.2f} dB)")

    print("\n" + "=" * 70)
    print(f"REFINEMENT COMPLETED: Best Validation PSNR = {best_psnr:.2f} dB")
    print(f"Total Elapsed Time: {time.time() - start_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/workspace/output/phase2_sota_v2/checkpoints/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="/workspace/data/bicycle")
    parser.add_argument("--output_dir", type=str, default="/workspace/output/phase2_5_refined")
    parser.add_argument("--num_steps", type=int, default=8000)
    parser.add_argument("--target_res", type=int, default=384)
    parser.add_argument("--n_samples", type=int, default=112)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--lr_planes", type=float, default=8e-3)
    parser.add_argument("--lr_mlp", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_refinement(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        target_res=args.target_res,
        n_samples=args.n_samples,
        downscale=args.downscale,
        lr_planes=args.lr_planes,
        lr_mlp=args.lr_mlp,
        device=args.device
    )
