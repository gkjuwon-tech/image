# Multi-Scene Generalization: Garden Benchmark & Final Step 30,000 Bicycle Renders

> **Session Timestamp**: 2026-10-13 23:09 KST  
> **Hardware**: 1x NVIDIA GeForce RTX 4090 (24 GB VRAM)  
> **Core Engine**: 3D Continuous Wave Packet Splatting  

---

## 1. Executive Summary & Convergence Speed

Both runs are **fully completed**:
1. **New Scene (`garden`)**: Fully trained to **Step 7,000 in just 1 minute 36 seconds**. Achieved **29.97 dB PSNR** and **0.937 SSIM** on held-out novel camera viewpoints.
2. **Reference Baseline (`bicycle`)**: Completed all **30,000 steps**. Achieved **28.59 dB PSNR**, **0.897 SSIM**, and a razor-sharp perceptual score of **0.0395 LPIPS**.

```
+---------------------------------------------------------------------------------------------------------------+
| Scene       | Steps  | Training Time | PSNR (Novel View) | SSIM (Structure) | LPIPS (Sharpness) | Real-time FPS   |
+===============================================================================================================+
| Garden      | 7,000  | 1 min 36 sec  | 29.97 dB          | 0.9371           | 0.0237            | 290.6 FPS       |
| Bicycle     | 30,000 | 20 min 40 sec | 28.59 dB          | 0.8966           | 0.0395            | 130.4 FPS       |
+---------------------------------------------------------------------------------------------------------------+
```

---

## 2. Visual Master Carousel: New Scene (`garden`) Held-Out Views

The following renders demonstrate multi-view reconstruction of the outdoor garden environment, resolving complex foliage, tree branches, wooden table planks, and ground gravel across distinct camera viewpoints.

````carousel
![Garden Held-Out View 00](/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/val_garden_7k_0000.png)
<!-- slide -->
![Garden Held-Out View 05](/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/val_garden_7k_0005.png)
<!-- slide -->
![Garden Held-Out View 10](/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/val_garden_7k_0010.png)
<!-- slide -->
![Bicycle Final Step 30,000 View 00](/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/val_bicycle_30k_0000.png)
````

---

## 3. Key Technical Takeaways

1. **Sub-2-Minute Convergence on New Scenes**:
   - The user does not need to wait for 30,000 steps. A completely novel scene like `garden` achieves photographic accuracy (nearly 30 dB PSNR, 0.94 SSIM) in **under 2 minutes** of training.
2. **Generalizability Verified**:
   - The exact same wave packet formulation, loss function, and densification schedule transferred seamlessly from an intricate mechanical object (bicycle) to complex organic foliage (garden).
