"""
Mip-NeRF 360 Dataset Loader & Ray Generator
===========================================
Loads calibrated 360-degree camera poses, performs scene centering,
unbounded space contraction, and generates stratified 3D camera rays.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Optional, Dict


def contract_coords(x: torch.Tensor) -> torch.Tensor:
    """
    Mip-NeRF 360 contraction function mapping unbounded space R^3 to [-2, 2]^3.
    contract(x) = x if ||x|| <= 1 else (2 - 1/||x||) * (x / ||x||)
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    mask = norm <= 1.0
    contracted = torch.where(mask, x, (2.0 - 1.0 / torch.clamp(norm, min=1e-6)) * (x / torch.clamp(norm, min=1e-6)))
    return contracted


class MipNeRF360BicycleDataset:
    """
    Dataset loader for Mip-NeRF 360 `bicycle` scene.
    Supports train/test split (every 8th view is held out for novel view evaluation).
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        downscale: int = 8,
        holdout: int = 8,
        device: str = "cpu"
    ):
        self.data_dir = data_dir
        self.split = split
        self.downscale = downscale
        self.device = device

        # Load poses and bounds
        poses_path = os.path.join(data_dir, "poses_bounds.npy")
        if not os.path.exists(poses_path):
            raise FileNotFoundError(f"poses_bounds.npy not found in {data_dir}")

        poses_arr = np.load(poses_path)  # (N, 17)
        self.raw_poses = poses_arr[:, :15].reshape(-1, 3, 5)  # (N, 3, 5)
        self.bounds = poses_arr[:, 15:17]  # (N, 2)
        self.num_images = self.raw_poses.shape[0]

        # Determine image folder
        img_folder_name = f"images_{downscale}" if downscale in [4, 8] else "images_8"
        img_dir = os.path.join(data_dir, img_folder_name)
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory {img_dir} does not exist.")

        # Match images sorted alphabetically
        all_img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        if len(all_img_files) != self.num_images:
            raise ValueError(f"Found {len(all_img_files)} images, expected {self.num_images} from poses.")

        # Train / Test split
        test_indices = list(range(0, self.num_images, holdout))
        train_indices = [i for i in range(self.num_images) if i not in test_indices]

        self.indices = train_indices if split == "train" else test_indices
        print(f"[{split.upper()}] Ingested {len(self.indices)} images (downscale={downscale}x, holdout={holdout}).")

        # Load selected images into memory (images_8 is ~44MB, images_4 is ~162MB)
        self.images = []
        for idx in self.indices:
            img = Image.open(all_img_files[idx]).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            self.images.append(arr)
        self.images = np.stack(self.images, axis=0)  # (N_split, H, W, 3)

        self.H, self.W = self.images.shape[1], self.images.shape[2]

        # Compute camera intrinsics and extrinsics
        # In LLFF: poses[:, :, 4] is [H_orig, W_orig, f_orig]
        orig_h, orig_w, orig_f = self.raw_poses[0, :, 4]
        scale_f = self.W / float(orig_w)
        self.focal = float(orig_f) * scale_f

        # Camera intrinsic matrix K
        self.K = np.array([
            [self.focal, 0.0, self.W * 0.5],
            [0.0, self.focal, self.H * 0.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Process camera poses:
        # LLFF pose orientation: [down, right, backward, t]
        # Convert to standard camera coordinate frame: [right, up, backward, t]
        poses = self.raw_poses[self.indices, :3, :4]  # (N_split, 3, 4)
        # Build full c2w matrices for ALL poses
        all_c2w_raw = np.concatenate([
            self.raw_poses[:, :, 1:2],       # right
            -self.raw_poses[:, :, 0:1],      # up
            self.raw_poses[:, :, 2:3],       # backward
            self.raw_poses[:, :, 3:4]        # translation
        ], axis=2)  # (N_total, 3, 4)

        # Normalize camera positions using training poses so all splits share identical coordinate frame
        train_translations = all_c2w_raw[train_indices, :, 3]
        center = np.mean(train_translations, axis=0, keepdims=True)
        dist = np.linalg.norm(train_translations - center, axis=1)
        scale = 1.0 / np.percentile(dist, 90)

        # Apply global training normalization to selected split
        c2w_matrices = all_c2w_raw[self.indices].copy()
        c2w_matrices[:, :, 3] = (c2w_matrices[:, :, 3] - center) * scale
        # Calibrate near bound to match nearest real surface (avoid wasting samples in empty air)
        self.near = float(np.min(self.bounds[:, 0]) * scale * 0.95)
        self.far = float(np.percentile(self.bounds[:, 1], 95) * scale * 1.05)
        self.c2w = torch.from_numpy(c2w_matrices).float().to(device)
        self.images = torch.from_numpy(self.images).float().to(device)

    def sample_random_rays(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Samples a random batch of rays across all training viewpoints.
        """
        num_views = len(self.indices)
        img_idx = torch.randint(0, num_views, (batch_size,), device=self.device)
        y = torch.randint(0, self.H, (batch_size,), device=self.device)
        x = torch.randint(0, self.W, (batch_size,), device=self.device)

        # Ray directions in camera space
        cx, cy, f = self.W * 0.5, self.H * 0.5, self.focal
        u = (x.float() - cx) / f
        v = -(y.float() - cy) / f
        dirs_cam = torch.stack([u, v, -torch.ones_like(u)], dim=-1)  # (B, 3)

        # Transform to world space
        c2w_selected = self.c2w[img_idx]  # (B, 3, 4)
        rot = c2w_selected[:, :3, :3]     # (B, 3, 3)
        trans = c2w_selected[:, :3, 3]    # (B, 3)

        rays_d = torch.bmm(rot, dirs_cam.unsqueeze(-1)).squeeze(-1)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = trans

        target_rgb = self.images[img_idx, y, x]  # (B, 3)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "target_rgb": target_rgb,
            "near": torch.full_like(rays_o[:, :1], self.near),
            "far": torch.full_like(rays_o[:, :1], self.far)
        }

    def get_full_view_rays(self, view_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates all rays for a single complete camera frame (for validation rendering).
        """
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.H, device=self.device),
            torch.arange(self.W, device=self.device),
            indexing="ij"
        )
        cx, cy, f = self.W * 0.5, self.H * 0.5, self.focal
        u = (x_grid.float() - cx) / f
        v = -(y_grid.float() - cy) / f
        dirs_cam = torch.stack([u, v, -torch.ones_like(u)], dim=-1)  # (H, W, 3)

        c2w_mat = self.c2w[view_idx]  # (3, 4)
        rot = c2w_mat[:3, :3]
        trans = c2w_mat[:3, 3]

        dirs_flat = dirs_cam.reshape(-1, 3)
        rays_d = torch.matmul(dirs_flat, rot.T)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = trans.unsqueeze(0).expand(rays_d.shape[0], -1)

        target_rgb = self.images[view_idx]  # (H, W, 3)
        return rays_o, rays_d, target_rgb

    def sample_random_patches(self, num_patches: int = 4, patch_size: int = 32) -> Dict[str, torch.Tensor]:
        """
        Samples contiguous square pixel patches for spatial gradient and edge supervision.
        """
        num_views = len(self.indices)
        img_idx = torch.randint(0, num_views, (num_patches,), device=self.device)
        y0 = torch.randint(0, self.H - patch_size, (num_patches,), device=self.device)
        x0 = torch.randint(0, self.W - patch_size, (num_patches,), device=self.device)

        dy = torch.arange(patch_size, device=self.device)
        dx = torch.arange(patch_size, device=self.device)
        grid_y, grid_x = torch.meshgrid(dy, dx, indexing="ij")

        all_y = y0.view(num_patches, 1, 1) + grid_y.unsqueeze(0)
        all_x = x0.view(num_patches, 1, 1) + grid_x.unsqueeze(0)

        flat_img_idx = img_idx.view(num_patches, 1, 1).expand(-1, patch_size, patch_size).reshape(-1)
        flat_y = all_y.reshape(-1)
        flat_x = all_x.reshape(-1)

        cx, cy, f = self.W * 0.5, self.H * 0.5, self.focal
        u = (flat_x.float() - cx) / f
        v = -(flat_y.float() - cy) / f
        dirs_cam = torch.stack([u, v, -torch.ones_like(u)], dim=-1)

        c2w_selected = self.c2w[flat_img_idx]
        rot = c2w_selected[:, :3, :3]
        trans = c2w_selected[:, :3, 3]

        rays_d = torch.bmm(rot, dirs_cam.unsqueeze(-1)).squeeze(-1)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = trans

        target_rgb = self.images[flat_img_idx, flat_y, flat_x]

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "target_rgb": target_rgb,
            "near": torch.full_like(rays_o[:, :1], self.near),
            "far": torch.full_like(rays_o[:, :1], self.far),
            "num_patches": num_patches,
            "patch_size": patch_size
        }
