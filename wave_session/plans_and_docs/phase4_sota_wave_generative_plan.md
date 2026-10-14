# Phase 4: Controllable 3D Continuous Wave Packet Generative Park Plan

> [!IMPORTANT]
> **Architectural Integrity & True 3D Wave Provenance**:
> Phase 4 will **NOT** utilize 2D Stable Diffusion, SDXL, or any 2D image-to-image generative models. It will **NOT** resort to static patchwork collage ("짜집기") of old assets, nor will it rely on procedural mathematical primitives that yield synthetic 3D video game graphics.
> 
> The system implements a **legitimate feed-forward 3D Continuous Wave Field Generator** using a frozen self-supervised visual representation encoder (DINOv2) for rich semantic priors, coupled with a custom **3D Wave DiT Flow Matcher** ($\mathbf{< 20.0\text{ M}}$ trainable parameters). The output is canonical 59-dimensional continuous 3D wave packets ($\mathbb{R}^{K \times 59}$) in physical Euclidean space, rasterized via `gsplat` (Phase 2) and diffracted via the Rayleigh-Sommerfeld Angular Spectrum Method (Phase 1).

---

## 1. System Architecture: The Legitimate 3D Wave Pipeline

```
+-------------------------------------------------------------------------------------------------------------------+
|                                 PHASE 4: SOTA 3D OPTICAL WAVE GENERATIVE SYSTEM                                   |
|                                                                                                                   |
|  [USER INPUT]                                                                                                     |
|  + Text Prompt: "A sunlit botanical park with lush weeping willows, solitary wooden bench..."                     |
|  + Spatial Layout: park_control_sketch.png (Frustum layout: Willow left, Bench center, Pond right)               |
|                                                   |                                                               |
|                                                   v                                                               |
|  [LEGITIMATE VISUAL BACKBONE]                                                                                     |
|  + Frozen DINOv2-small Encoder (Self-Supervised Vision Transformer, ~21M frozen weights)                          |
|  + GENERATES ZERO 2D PIXELS. Extracts dense spatial semantic/geometric feature tokens F in R^{H x W x D}         |
|  + Provides rich photographic prior (surface normals, material boundaries, natural chlorophyll/wood features)    |
|                                                   |                                                               |
|                                                   v                                                               |
|  [OUR CUSTOM 3D WAVE DIT GENERATOR] (< 20.0M Trainable Parameters)                                                |
|  + 8-Layer Transformer with Cross-Attention over text tokens & DINOv2 spatial features                           |
|  + Camera Ray Field Conditioning: R(u, v) = K^-1 [u, v, 1]^T normalized                                          |
|  + Directly predicts K = 65,536 Continuous 3D Wave Tokens in physical Euclidean space:                            |
|    - 3D Position mu in R^3 (Depth along ray + spatial offset)                                                    |
|    - Anisotropic scale s in R^3 (Continuous millimeter Gaussian bounds: 1mm to 8mm)                               |
|    - Unit quaternion q in S^3                                                                                     |
|    - Opacity alpha in [0, 1]                                                                                      |
|    - Degree-3 Spherical Harmonics c_wave in R^{16 x 3} (48 wave phase/amplitude coefficients)                     |
|                                                   |                                                               |
|                                                   v                                                               |
|  [PHASE 2: PHYSICAL 3D WAVE RASTERIZATION] (`gsplat`)                                                             |
|  + Hardware-accelerated GPU rasterization directly from 3D wave tokens                                            |
|  + Evaluated across 4 novel camera viewpoints (Master 0 deg, Orbit Left +22 deg, Orbit Right -22 deg, Closeup)   |
|  + 100% rigid physical parallax (zero 2D warping, zero hallucination)                                             |
|                                                   |                                                               |
|                                                   v                                                               |
|  [PHASE 1: RAYLEIGH-SOMMERFELD ASM WAVE OPTICS]                                                                   |
|  + Depth-coupled complex transfer function H(fx, fy; lambda, Delta z) for circular pupil aperture f/2.8          |
|  + RGB multi-wavelength dispersion (650nm, 532nm, 450nm) synthesizing authentic Bessel J1(x)/x Airy disk bokeh   |
|  + Native 1024x1024 photographic sensor response & ISO 100 film grain                                             |
|                                                   |                                                               |
|                                                   v                                                               |
|  [DELIVERABLES] SOTA Photorealistic Virtual Park Master & Multi-View Trajectory Images                            |
+-------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formalization: 59-Dimensional Continuous Wave Packets

The neural generator operates strictly in the canonical continuous wave packet space established in Phase 3. For each of the $K = 65,536$ rays:

$$\mathbf{t}_i = \left[ \mathbf{\mu}_i \in \mathbb{R}^3, \; \log(\mathbf{s}_i) \in \mathbb{R}^3, \; \mathbf{q}_i \in \mathbb{S}^3, \; \text{logit}(\alpha_i) \in \mathbb{R}, \; \mathbf{c}_{lm, i} \in \mathbb{R}^{48} \right]$$

1. **Spatial Anchor $\mathbf{\mu}_i$**:
   $$\mathbf{\mu}_i = d(u, v) \cdot \mathbf{r}(u, v) + \Delta \mathbf{\mu}_i$$
   where $d(u, v)$ is the predicted depth along the calibrated camera ray $\mathbf{r}(u, v)$, grounded in physical meters.
2. **Millimeter Scale Bound $\mathbf{s}_i$**:
   $$\mathbf{s}_i = \exp\left(\text{clamp}(\mathbf{s}_{\text{pred}}, \ln(0.001), \ln(0.008))\right)$$
   Constrained to $1\text{ mm} \dots 8\text{ mm}$ to match the optical pixel footprint at scene depth ($Z = 1.5 \dots 5.0\text{m}$). This mathematically eliminates both the olive-mud blur (caused by oversized Gaussians) and the screen-door grid (by properly overlapping with adjacent ray footprints).
3. **Complex Wave Optics Spherical Harmonics $\mathbf{c}_{lm, i}$**:
   $16 \times 3 = 48$ coefficients representing degree-3 spherical harmonics:
   $$I(\mathbf{v}) = \sum_{l=0}^3 \sum_{m=-l}^l \mathbf{c}_{lm} Y_{lm}(\mathbf{v})$$
   capturing view-dependent specular highlights on varnish, water Fresnel reflections, and forward chlorophyll scattering in weeping willow foliage.

---

## 3. Analysis of Previous Failure Modes & How This Plan Eliminates Them

| Failure Mode in Prior Runs | Root Cause Identified | Engineering Fix in New Plan |
| :--- | :--- | :--- |
| **Attempt 1: Rainbow Neon Static** | Unordered 3D point cloud diffusion trained from white noise without spatial inductive bias. | Replaced by Ray-Anchored Frustum Transformer: every wave packet is geometrically grounded to a physical camera ray. |
| **Attempt 2: Black Void with Clump** | Procedural box placement of isolated assets ("짜집기") leaving 95% of space empty. | Permanently banned. The generator predicts a continuous dense 3D wave field spanning ground, water, tree, and sky. |
| **Attempt 3: Olive-Green Mud Blur** | Gaussian scales clamped up to $37\text{ cm}$ ($100\times$ too large) + pure $L_1$ loss minimizing to the median color. | Gaussian scales bounded to $1\dots 8\text{ mm}$; loss upgraded to multi-scale SSIM + Fourier high-frequency spectral loss. |
| **Attempt 4: 90s CG Video Game Asset** | Handcrafted procedural math (sine waves, random scatter) producing artificial, sterile frequencies. | Replaced by DINOv2 self-supervised visual prior, embedding natural texture frequencies into the wave generator. |
| **Attempt 5: Screen-Door Dot-Matrix Grid** | Discrete ray grid with undersized, non-overlapping points trained for only 85 seconds from scratch. | Continuous ray expansion with bilateral footprint overlap, trained for 5,000 steps with DINOv2 semantic guidance. |

---

## 4. Training & Supervision Strategy

### 4.1 Dataset
- **Active Training Corpus**: 504 paired outdoor park nature images in `/workspace/data/park_nature_corpus/` (multi-view camera captures of botanical foliage, trees, lawns, and outdoor structures).
- **Conditioning**: Edge/sketch maps extracted via Canny filtering paired with photographic ground truth.

### 4.2 Multi-Component Loss Function
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{1} + 0.35 \cdot \mathcal{L}_{\text{SSIM}} + 0.20 \cdot \mathcal{L}_{\text{Fourier}} + 0.15 \cdot \mathcal{L}_{\text{DINO}} + 0.01 \cdot \mathcal{L}_{\text{TV}}$$

1. **$\mathcal{L}_{1}$**: Pixel-wise color fidelity.
2. **$\mathcal{L}_{\text{SSIM}}$**: Multi-scale structural consistency ($1.0 - \text{SSIM}$).
3. **$\mathcal{L}_{\text{Fourier}}$**: 2D Fourier power spectrum loss ($\| |\mathcal{F}(I_{\text{pred}})| - |\mathcal{F}(I_{\text{GT}})| \|_1$) to enforce natural photographic frequency distribution.
4. **$\mathcal{L}_{\text{DINO}}$**: Cosine distance between patch-level DINOv2 feature maps of predicted and ground-truth images, enforcing high-level semantic photorealism (real foliage and wood textures).
5. **$\mathcal{L}_{\text{TV}}$**: Total variation loss on depth map to prevent noisy depth tears.

---

## 5. Specification of Final Deliverables

1. **Master Photorealistic Virtual Park Render (`virtual_park_sota_master.png`)**:
   - Resolution: Native $1024 \times 1024$ PNG.
   - Composition:
     - Foreground: Wooden park bench with authentic longitudinal wood grain, cast-iron frame, and crisp contact shadow on lawn.
     - Midground: Weeping willow tree with cascading foliage framing the bench, lit by morning sunlight.
     - Right: Reflective pond surface mirroring golden hour sky with subtle Fresnel water ripples.
     - Background: Distant atmospheric hills and morning golden hour sky with natural exposure bloom.
2. **Multi-View Novel Views (Proving 3D Rigid Geometry)**:
   - `virtual_park_orbit_left.png` (+22 deg azimuth)
   - `virtual_park_orbit_right.png` (-22 deg azimuth)
   - `virtual_park_bench_closeup.png` (Low-angle bench detail)
3. **Physical Diagnostics**:
   - `virtual_park_depth_map.png`: Physical scene depth map $Z(x, y)$ in metric units.
   - `virtual_park_wave_airy_psf.png`: Rayleigh-Sommerfeld circular pupil Airy disk PSF ($f/2.8$).
   - `virtual_park_tokens.pt`: Canonical continuous 3D wave tokens ($65,536 \times 59$).
   - `ckpt_ray_wave_generator.pt`: Trained neural generator checkpoint ($< 20.0\text{M}$ trainable parameters).

---

## 6. Definition of Done & Quality Acceptance Criteria

To be approved and certified, the deliverables must satisfy all of the following criteria:

- [ ] **Zero 2D Generative Diffusion**: No Stable Diffusion, SDXL, or 2D image models invoked.
- [ ] **Parameter Limit**: Trainable parameters of custom neural layers strictly $< 20.0\text{M}$.
- [ ] **Zero Patchwork Collage**: No copy-pasting of static bounding box patches from old scenes.
- [ ] **Zero Screen-Door Artifacts**: Continuous, dense surfaces without pixelated mesh or flyscreen grids.
- [ ] **Photographic Visual Quality**: Inspected directly with human eyes. Crisp parallel bench slats with authentic wood grain, organic weeping willow foliage, reflective water surface, and natural golden hour lighting.
- [ ] **Multi-View Parallax**: 3D geometric consistency across novel camera views without 2D warping or temporal popping.
- [ ] **Wave Optics Integration**: True Rayleigh-Sommerfeld ASM circular pupil Airy disk depth-of-field bokeh applied.

---

## 7. Execution Roadmap & Milestones

| Stage | Milestone | Key Actions | Est. Time |
| :---: | :--- | :--- | :---: |
| **Stage 1** | **DINOv2 Feature Backbone Setup** | Load frozen DINOv2-small encoder; extract dense feature representations for park corpus and user sketch. | 5 min |
| **Stage 2** | **3D Wave DiT Architecture Build** | Assemble the Ray-Anchored 3D Wave DiT (< 20.0M params) with DINOv2 feature cross-attention & millimeter Gaussian scaling. | 10 min |
| **Stage 3** | **End-to-End Differentiable Training** | Train for 5,000 steps on RTX 4090 using multi-scale SSIM, Fourier spectral loss, and DINO feature supervision. | 15 min |
| **Stage 4** | **Sketch Inference & Multi-View Wave Rendering** | Run forward inference on user sketch; rasterize native 1024x1024 views via `gsplat`; apply Rayleigh-Sommerfeld ASM bokeh. | 5 min |
| **Stage 5** | **Forensic Visual Inspection & Gatekeeper Audit** | Directly inspect all renders with `view_file`; verify zero artifacts; present verified deliverables. | 5 min |
