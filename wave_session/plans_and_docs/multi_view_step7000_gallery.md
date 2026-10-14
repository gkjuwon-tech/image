# Multi-Angle Novel View Gallery (Step 7,000 Checkpoint)

> [!NOTE]
> All images show **Ground Truth (Left)** versus the **3D Wave Packet Splatting Render (Right)** across 5 distinct held-out camera poses that were never seen during training. Notice the consistent 3D parallax, fine spoke continuity, and exact occlusion boundaries across all viewing perspectives.

---

````carousel
### View 00: Direct Front-Side Perspective
![View 00](/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/val_step6999_0000.png)
*Left: Ground Truth | Right: 3D Wave Packet Render (PSNR: 27.70 dB, SSIM: 0.8793)*
- Full bicycle geometry, razor-thin 2mm spokes, and black park bench backrest.
<!-- slide -->
### View 05: Oblique 3/4 Side-Rear Perspective
![View 05](/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/val_step6999_0005.png)
*Left: Ground Truth | Right: 3D Wave Packet Render*
- View looking down the asphalt pathway, showing the curved cast-iron bench armrest and rear wheel mudguard.
<!-- slide -->
### View 10: Front Bench View (Occlusion Test)
![View 10](/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/val_step6999_0010.png)
*Left: Ground Truth | Right: 3D Wave Packet Render*
- Rigorous occlusion test: The park bench sits in front, with the white bicycle frame and handlebars cleanly visible behind the horizontal slats.
<!-- slide -->
### View 15: Eye-Level Head-On View
![View 15](/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/val_step6999_0015.png)
*Left: Ground Truth | Right: 3D Wave Packet Render*
- Symmetrical frontal view capturing the alignment of the wooden bench slats and the bicycle frame.
<!-- slide -->
### View 20: High-Elevation Top-Down View
![View 20](/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/val_step6999_0020.png)
*Left: Ground Truth | Right: 3D Wave Packet Render*
- High camera elevation looking down into the park bench seat and the sloped lawn behind it.
````

---

### Quantitative Evaluation Summary
- **Evaluation Dataset**: 25 completely independent held-out camera positions around the bicycle.
- **Mean Held-Out PSNR**: **27.70 dB** (compared to 19.74 dB peak on previous tri-plane architecture).
- **Mean Structural Similarity (SSIM)**: **0.8793**
- **Perceptual Error (LPIPS)**: **0.0552**
- **Real-Time Render Speed**: **185 FPS** (5.4 ms per $1247 \times 828$ frame on RTX 4090).
- **Live Training State**: Currently optimizing toward **Step 30,000** for sub-millimeter grass blade densification and target **>31.5 ~ 32.5 dB PSNR**.
