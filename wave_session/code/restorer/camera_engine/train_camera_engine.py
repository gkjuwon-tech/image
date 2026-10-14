"""CLI Training Script for Camera Trajectory Engine.

Supports configurable hyperparameters, synthetic data training, dry-run checks,
and periodic validation and checkpointing.
"""

import argparse
import sys
import torch

from config import TrajectoryConfig, ModelConfig, LossConfig, TrainConfig
from models.camera_engine import CameraTrajectoryEngine
from data.dataset import create_dataloaders
from train.trainer import CameraEngineTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1: Camera Trajectory Engine")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per step")
    parser.add_argument("--max-steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--eval-every", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--save-every", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu")
    parser.add_argument("--dry-run", action="store_true", help="Execute 3 steps then exit cleanly")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("Stage 1: Camera Trajectory Engine (27.21M Parameters) - Training Initializer")
    print("=" * 80)
    
    traj_cfg = TrajectoryConfig()
    model_cfg = ModelConfig()
    loss_cfg = LossConfig()
    train_cfg = TrainConfig(
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        eval_every_steps=args.eval_every,
        save_every_steps=args.save_every
    )
    
    # 1. Initialize Model
    model = CameraTrajectoryEngine(model_cfg, traj_cfg)
    counts = model.get_parameter_count()
    print(f"Model architecture assembled successfully.")
    print(f"Total Trainable Parameters: {counts['total_trainable']:,} (Allocated Budget: 25M - 30M)")
    for name, cnt in counts.items():
        if name != "total_trainable":
            print(f"  - {name}: {cnt:,}")
            
    # 2. Build DataLoaders
    print("Generating procedural datasets...")
    train_loader, val_loader = create_dataloaders(
        traj_cfg, model_cfg,
        batch_size=train_cfg.batch_size,
        num_train=train_cfg.num_train_samples,
        num_val=train_cfg.num_val_samples,
        num_workers=0  # Safe in all environments
    )
    print(f"Datasets ready: {len(train_loader)} train batches, {len(val_loader)} val batches.")
    
    # 3. Initialize Trainer
    trainer = CameraEngineTrainer(model, train_loader, val_loader, train_cfg, loss_cfg, traj_cfg)
    print(f"Trainer configured on device: {trainer.device}")
    
    if args.dry_run:
        print("\n--- Running Dry-Run Verification (3 steps) ---")
        train_iter = iter(train_loader)
        for s in range(3):
            batch = next(train_iter)
            metrics = trainer.train_step(batch)
            print(f"Step {s+1}/3 | Total Loss: {metrics['total_loss']:.4f} | Grad Norm: {metrics['grad_norm']:.4f} | LR: {metrics['lr']:.6f}")
        eval_metrics = trainer.evaluate()
        print(f"Dry-run evaluation | Val Rot Err: {eval_metrics['val_rot_err_deg']:.2f}° | Val Trans Err: {eval_metrics['val_trans_err_m']:.4f}m")
        print("Dry run completed successfully.")
        return

    # 4. Main Training Loop
    print("\n--- Starting Training Loop ---")
    train_iter = iter(train_loader)
    for step in range(1, train_cfg.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        metrics = trainer.train_step(batch)
        
        if step % 50 == 0:
            print(f"Step {step:05d} | Loss: {metrics['total_loss']:.4f} | Rot: {metrics['loss_rot_geodesic']:.4f} | Trans: {metrics['loss_trans_huber']:.4f} | Jerk: {metrics['loss_jerk_smoothness']:.4f} | Barrier: {metrics['loss_barrier']:.4f}")
            
        if step % train_cfg.eval_every_steps == 0:
            eval_metrics = trainer.evaluate()
            print(f"[*] Eval @ Step {step:05d} | Rot Error: {eval_metrics['val_rot_err_deg']:.2f}° | Trans Error: {eval_metrics['val_trans_err_m']:.4f}m | Jerk: {eval_metrics['val_mean_jerk']:.4f}")
            
        if step % train_cfg.save_every_steps == 0:
            ckpt_path = f"{train_cfg.checkpoint_dir}/checkpoint_step_{step}.pt"
            trainer.save_checkpoint(ckpt_path)
            print(f"[+] Saved checkpoint to {ckpt_path}")
            
    print("\n" + "=" * 80)
    print("Training Completed! Running Final Validation & Performance Evaluation...")
    print("=" * 80)
    final_eval = trainer.evaluate()
    print(f"[FINAL EVALUATION METRICS]")
    print(f"  * Geodesic Rotation Error: {final_eval['val_rot_err_deg']:.3f}°")
    print(f"  * Euclidean Translation Error: {final_eval['val_trans_err_m']:.4f} m")
    print(f"  * Kinematic Jerk (C^2 smoothness): {final_eval['val_mean_jerk']:.5f} m/s^3")
    final_ckpt = f"{train_cfg.checkpoint_dir}/checkpoint_final.pt"
    trainer.save_checkpoint(final_ckpt, extra_info={"final_eval": final_eval})
    print(f"[+] Final checkpoint saved to {final_ckpt}")
    print("=" * 80)


if __name__ == "__main__":
    main()
