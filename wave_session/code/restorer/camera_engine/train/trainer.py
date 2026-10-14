"""Production-Grade Trainer Class for Camera Trajectory Engine.

Supports AdamW optimizer with decoupled weight decay, Cosine Annealing with linear warmup,
FP32-guarded loss computation, gradient norm clipping, checkpointing, and evaluation metrics.
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from config import TrainConfig, LossConfig, TrajectoryConfig, ModelConfig
from models.camera_engine import CameraTrajectoryEngine
from losses.trajectory_loss import TrajectoryMultiLoss
from losses.sdf_barrier_loss import SDFBarrierLoss, DifferentiableSphereSDF
from geometry.lie_algebra import geodesic_distance_so3


class CameraEngineTrainer:
    """Manages training, validation, checkpointing, and metric logging."""
    def __init__(
        self,
        model: CameraTrajectoryEngine,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: TrainConfig,
        loss_cfg: LossConfig,
        traj_cfg: TrajectoryConfig
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_cfg = train_cfg
        self.loss_cfg = loss_cfg
        self.traj_cfg = traj_cfg
        
        self.device = torch.device(train_cfg.device if torch.cuda.is_available() and train_cfg.device == "cuda" else "cpu")
        self.model.to(self.device)
        
        # Loss Modules
        self.criterion = TrajectoryMultiLoss(loss_cfg, traj_cfg).to(self.device)
        self.barrier_criterion = SDFBarrierLoss(safe_margin=loss_cfg.sdf_safety_margin).to(self.device)
        
        # Scene obstacle SDF for collision barrier testing
        centers = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        radii = torch.tensor([0.4], device=self.device)
        self.sdf_obstacle = DifferentiableSphereSDF(centers, radii).to(self.device)
        
        # Optimizer with weight decay grouping
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name.lower() or "bias" in name.lower() or "pos_embedding" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        self.optimizer = AdamW([
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ], lr=train_cfg.learning_rate)
        
        # Cosine Annealing with Warmup Scheduler
        self.scheduler = self._build_cosine_warmup_scheduler()
        
        # Mixed Precision Scaler
        self.use_amp = train_cfg.use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        
        self.step = 0
        self.best_val_rot_err = float("inf")
        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    def _build_cosine_warmup_scheduler(self):
        warmup_steps = self.train_cfg.warmup_steps
        max_steps = self.train_cfg.max_steps
        min_lr_ratio = self.train_cfg.min_learning_rate / self.train_cfg.learning_rate

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        text_tokens = batch["text_tokens"].to(self.device)
        directorial_params = batch["directorial_params"].to(self.device)
        text_mask = batch["text_mask"].to(self.device)
        
        targets = {
            "rotations": batch["rotations"].to(self.device),
            "translations": batch["translations"].to(self.device),
            "focal_lengths": batch["focal_lengths"].to(self.device),
            "velocities": batch["velocities"].to(self.device),
            "subject_target_pos": batch["subject_target_pos"].to(self.device),
            "is_vertigo": batch.get("is_vertigo", torch.zeros(text_tokens.shape[0], 1)).to(self.device)
        }
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            preds = self.model(text_tokens, directorial_params, text_mask=text_mask)
            
        with torch.amp.autocast('cuda', enabled=False):
            preds_fp32 = {
                k: v.float() if isinstance(v, torch.Tensor) else v
                for k, v in preds.items()
            }
            loss_dict = self.criterion(preds_fp32, targets)
            loss_barrier = self.barrier_criterion(preds_fp32["translations"], self.sdf_obstacle)
            total_loss = loss_dict["total_trajectory_loss"] + self.loss_cfg.w_sdf_barrier * loss_barrier
            
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip_norm)
            self.optimizer.step()
            
        self.scheduler.step()
        
        self.step += 1
        
        step_metrics = {k: v.item() for k, v in loss_dict.items()}
        step_metrics["loss_barrier"] = loss_barrier.item()
        step_metrics["total_loss"] = total_loss.item()
        step_metrics["grad_norm"] = grad_norm.item()
        step_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        return step_metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_rot_err_deg = 0.0
        total_trans_err_m = 0.0
        total_jerk = 0.0
        total_samples = 0
        
        for batch in self.val_loader:
            text_tokens = batch["text_tokens"].to(self.device)
            directorial_params = batch["directorial_params"].to(self.device)
            text_mask = batch["text_mask"].to(self.device)
            
            R_gt = batch["rotations"].to(self.device)
            t_gt = batch["translations"].to(self.device)
            B = text_tokens.shape[0]
            
            preds = self.model(text_tokens, directorial_params, text_mask=text_mask)
            
            # Rotation error in degrees
            rot_dist_rad = geodesic_distance_so3(preds["rotations"], R_gt)
            rot_err_deg = torch.mean(rot_dist_rad) * (180.0 / math.pi)
            
            # Translation error in meters
            trans_err_m = torch.mean(torch.norm(preds["translations"] - t_gt, dim=-1))
            
            # Kinematic jerk
            jerk_norm = torch.mean(torch.norm(preds["jerks"], dim=-1))
            
            total_rot_err_deg += rot_err_deg.item() * B
            total_trans_err_m += trans_err_m.item() * B
            total_jerk += jerk_norm.item() * B
            total_samples += B
            
        return {
            "val_rot_err_deg": total_rot_err_deg / total_samples,
            "val_trans_err_m": total_trans_err_m / total_samples,
            "val_mean_jerk": total_jerk / total_samples
        }

    def save_checkpoint(self, path: str, extra_info: Optional[Dict[str, Any]] = None):
        checkpoint = {
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.use_amp else None,
            "best_val_rot_err": self.best_val_rot_err,
            "extra_info": extra_info or {}
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if self.use_amp and checkpoint.get("scaler_state") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.step = checkpoint.get("step", 0)
        self.best_val_rot_err = checkpoint.get("best_val_rot_err", float("inf"))
