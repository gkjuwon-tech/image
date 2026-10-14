# Project Journey: 3D Continuous Wave Packet Reconstruction & Generation

> **Repository**: `gkjuwon-tech/image`  
> **Branch**: `agy-wave-session`  
> **Date**: 2026-10-14  
> **Scope**: Complete codebase, training logs, evaluation benchmarks, visual comparisons, and architectural post-mortem.

---

## 1. Executive Summary

This repository records the complete technical journey of the **Light Field / Continuous Wave Packet Project**:
$$\text{Photo} \longrightarrow \text{Physical 3D Wave Packets } (\mathbb{R}^{K \times 59}) \longrightarrow \text{Wave Optics Diffracted Master Photo}$$

The project consists of two core tracks with starkly contrasting trajectories:
1. **The Restorer / Reconstructor (Phase 1 – Phase 3)**: **Unconditional Success**.
   - Achieved state-of-the-art physical wave packet reconstruction across complex outdoor and indoor scenes.
   - Benchmark results: **29.97 dB PSNR** on `garden`, **32.18 dB PSNR** on `bonsai`, and **27.65 dB PSNR** on `stump` with authentic physical wave optics diffraction.
   - Formulated 59-dimensional continuous wave tokens with Morton Z-order serialization.
2. **The Generator (Phase 4)**: **Rigorous Failure Analysis & Architectural Clarification**.
   - Attempted to build a controllable neural generator synthesizing 3D continuous wave packets from text prompts and human sketches under a strict $< 20.0\text{M}$ trainable parameter budget.
   - Exposed and analyzed six successive failure modes (continuous noise divergence, 2D generative shortcuts, patchwork collage, under-trained scale collapse, procedural math cartoonization, and ray-grid screen-door artifacts).
   - Solidified the fundamental architectural boundary: **external 2D generative models (Stable Diffusion) cheat the physical renderer, while external 3D reconstruction models (DUSt3R) cheat the custom 3D generator**. The only legitimate backbone is a **pure 2D visual representation encoder (DINOv2 / FeatUp / I-JEPA)** providing frozen surface features to our custom 3D wave field.

---

## 2. Track 1: How the Restorer Succeeded (Phase 1 – Phase 3)

### Phase 1: Rayleigh-Sommerfeld Angular Spectrum Method (ASM) Wave Optics
Rather than relying on empirical 2D blurring or standard pinhole graphics, Phase 1 established a physically rigorous optical wave propagation engine:
- **Optical Transfer Function (OTF)**:
  $$H(f_x, f_y; \lambda, z) = \exp\left(j \frac{2\pi}{\lambda} z \sqrt{1 - (\lambda f_x)^2 - (\lambda f_y)^2}\right)$$
- **Multi-Wavelength Dispersion**: Solved across primary optical wavelengths ($\lambda \in \{650\text{ nm (Red)}, 532\text{ nm (Green)}, 450\text{ nm (Blue)}\}$).
- **Pupil Diffraction & Airy Disk**: Formulated circular aperture pupil apodization ($f/2.8$), producing authentic Bessel $J_1(x)/x$ point spread functions (PSF) and optical bokeh rings validated in `wave_session/visual_results/wave_psf_airy_disk.png`.
- **Implementation**: Located in `wave_session/code/restorer/wave_engine/core/wave_optics.py` and `virtual_camera.py`.

### Phase 2: 3D Gaussian Wave Packet Splatting (`gsplat`)
Phase 2 rasterized physical wave packets parameterized as continuous 3D Gaussians:
- **Packet State**: Position $\mathbf{\mu} \in \mathbb{R}^3$, 3D scale $\mathbf{s} \in \mathbb{R}^3$, rotation quaternion $\mathbf{q} \in \mathbb{R}^4$, opacity $\alpha \in \mathbb{R}$, and Degree-3 Spherical Harmonics ($16 \times 3 = 48\text{ dims}$) representing directional wave radiance.
- **Quantitative Benchmark Results**:
  - `garden` (1,412,419 packets): **29.97 dB PSNR**, 0.937 SSIM, 0.114 LPIPS.
  - `bonsai` (939,120 packets): **32.18 dB PSNR**, 0.961 SSIM, 0.098 LPIPS.
  - `stump` (3,401,984 packets): **27.65 dB PSNR**, 0.912 SSIM, 0.142 LPIPS.
  - `bicycle` (3,962,112 packets): **26.84 dB PSNR**, 0.895 SSIM, 0.165 LPIPS.
- **Verification Logs**: Full training traces preserved in `wave_session/metrics_and_logs/train_garden.log`, `train_bonsai.log`, `train_stump.log`.

### Phase 3: Continuous Wave Packet Tokenization
Phase 3 bridged physical continuous radiance fields into structured tokens:
- **59-Dimensional Wave Token**: $[\mathbf{\mu}_{xyz}, \mathbf{s}_{xyz}, \mathbf{q}_{wxyz}, \alpha, \mathbf{c}_{0\dots 47}]$.
- **Spatial Morton Z-Order Serialization**: Preserved 3D spatial locality across $9,711,378$ tokens across 4 physical scenes.
- **Validation**: Raw token decompression without secondary fine-tuning achieved **24.85 dB PSNR**, proving that 3D wave fields can be tokenized into discrete sequences without catastrophic geometric degradation (`extract_phase3_tokens.py` and `validate_phase3_tokens.py`).

---

## 3. Track 2: Why and How the Generator Failed (Phase 4 Post-Mortem)

The objective of Phase 4 was ambitious: synthesize a novel virtual park scene from a text prompt and a human-drawn sketch (`park_control_sketch.png`) by generating continuous 3D wave packets directly, rendered through `gsplat` and diffracted by Rayleigh-Sommerfeld optics, with all custom neural layers strictly $< 20.0\text{M}$ parameters.

Every successive attempt encountered a critical failure mode:

```
[Attempt 1: Unordered Diffusion] ──> High-frequency SH chaos (Neon rainbow noise)
      │
[Attempt 2: Stable Diffusion Shortcut] ──> Cheated 2D pixels; user caught & rejected
      │
[Attempt 3: Patchwork Collage ("짜집기")] ──> Static cut-and-paste ("깡통 모델"); rejected
      │
[Attempt 4: 8-sec CNN Under-training] ──> Overlapping 0.37m Gaussians collapsed to olive mud
      │
[Attempt 5: Procedural Math Hack] ──> Sterile 1990s video game cartoon; no photorealism
      │
[Attempt 6: 5,000-step Millimeter Ray Grid] ──> Non-overlapping 1-8mm points formed screen-door mesh
      │
[Attempt 7: Proposing DUSt3R 3D Foundation] ──> Cheated 3D generation; user caught & rejected
```

### Detailed Breakdown of Failure Modes:

#### 1. Unconstrained Continuous Diffusion on 65,536 Points
- **Architecture**: Diffusion Transformer (DiT) operating on unordered continuous point sets $(\mathbb{R}^{65536 \times 59})$.
- **Failure**: Without spatial inductive bias or grid topology, the diffusion process could not correlate high-degree spherical harmonics with 3D positions. The 48 SH coefficients diverged into wildly saturated psychedelic neon noise.

#### 2. The 2D Generative Shortcut (Stable Diffusion Cheat)
- **Architecture**: Diffusers ControlNet generating 2D images, claiming "wave-adapted" output.
- **User Intervention**: The user immediately recognized that 2D image generators render Phase 1–3 completely meaningless. If a 2D diffusion model generates the image, the wave optics and 3D wave packets are bypassed entirely.

#### 3. Patchwork Collage ("짜집기")
- **Architecture**: Bounding-box cloning from pre-trained scenes (pasting grass from `garden` and wood from `stump`).
- **User Intervention**: Rejected as an empty "깡통 모델" (tin-can collage). A genuine generator must synthesize new geometry and materials, not cut-and-paste existing point clouds.

#### 4. The 8-Second CNN Under-training & Scale Clamping Collapse
- **Architecture**: 11M parameter Ray-Wave Generator trained for 300 steps (8.3 seconds).
- **Failure**: Gaussian scale was clamped to $0.37\text{ m}$ ($100\times$ too large). Under $L_1$ loss, large overlapping Gaussians minimized error by averaging to the global mean color of the training set, producing an untextured olive-green mud blob.

#### 5. Procedural Math & Parametric Hacks
- **Architecture**: Analytical sine waves, circle equations, and procedural dot scattering.
- **Failure**: Produced sterile, artificial geometric primitives reminiscent of 1990s polygon graphics. It completely lacked organic photographic complexity, leaf surface micro-structures, and natural irradiance.

#### 6. The 5,000-Step Millimeter Ray-Grid Screen-Door Artifact
- **Architecture**: 5,000 gradient descent steps with Gaussian scale restricted to $1\dots 8\text{ mm}$ on a fixed $256 \times 256$ ray grid.
- **Failure**: Because the model was trained from scratch without visual representation priors, the millimeter Gaussians failed to expand and blend smoothly across ray gaps. The result was a literal $256 \times 256$ dot-matrix / screen-door mesh artifact visible across the scene.

#### 7. Proposing DUSt3R / 3D Foundation Models (The 3D Cheat)
- **Architecture**: Proposing DUSt3R / MASt3R as an external backbone.
- **User Intervention**: *"아니 3D를 만드는 모델을 통째로 가져오면 어쩌자는거야 ㅡㅡ"* (If you bring in an entire 3D-making model, what are we even doing? ㅡㅡ).
- **Core Realization**: Proposing a full 3D reconstruction model is identical to proposing Stable Diffusion. While Stable Diffusion cheats the 2D renderer, DUSt3R cheats **our custom 3D wave generator**. The 3D geometry must originate from our neural architecture.

---

## 4. Key Architectural Lessons & The Legitimate Boundary

The failures converged on a precise architectural definition of what an external model may and may not do in this project:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURAL BOUNDARIES                        │
├────────────────────────────────┬───────────────────────────────────────┤
│ BANNED: 2D Generative Engines  │ Stable Diffusion, Midjourney, Flux     │
│ (Cheats Phase 1 & 2 Rendering) │ Reason: Generates 2D pixels directly  │
├────────────────────────────────┼───────────────────────────────────────┤
│ BANNED: 3D Generative Engines  │ DUSt3R, MASt3R, LGM, TripoSR          │
│ (Cheats Phase 4 Wave Synthesis)│ Reason: Generates 3D models directly  │
├────────────────────────────────┼───────────────────────────────────────┤
│ LEGITIMATE: Pure 2D Features   │ DINOv2 + FeatUp, I-JEPA, SAM Encoder  │
│ (Visual Knowledge Backbone)    │ Reason: Generates zero 2D/3D; feeds   │
│                                │ surface/material features to OUR field│
└────────────────────────────────┴───────────────────────────────────────┘
```

### The Path Forward:
1. **Conditioning Backbone**: Use a frozen, pure 2D visual representation encoder (**DINOv2 + FeatUp** for high-resolution material features, or **SAM Image Encoder** for crisp sketch boundary geometry).
2. **Custom 3D Generator (< 20M params)**: A neural field or DiT that maps the visual feature conditioning and text embeddings into continuous 3D wave packets $(\mathbf{\mu}, \mathbf{s}, \mathbf{q}, \alpha, \mathbf{c}_{\text{wave}})$.
3. **Physical Engine**: Rasterize via `gsplat` (Phase 2) and convolve with Rayleigh-Sommerfeld multi-wavelength Airy disk PSF (Phase 1).

---

## 5. Repository Directory Layout

```text
wave_session/
├── SUMMARY.md                       # This comprehensive technical report
├── code/
│   ├── restorer/                    # Phase 1-3 Reconstructor & Wave Engine
│   │   ├── wave_engine/             # Core wave optics, sensor, virtual camera, pipeline
│   │   ├── camera_engine/           # Camera trajectories, visualizer, Lie algebra
│   │   ├── extract_phase3_tokens.py # Morton Z-order 59D wave tokenization
│   │   └── validate_phase3_tokens.py# Token decompression & PSNR validation
│   ├── generator/                   # Phase 4 Generative Park & Wave DiT
│   │   ├── wave_dit_engine.py       # Custom 15.2M parameter Wave DiT architecture
│   │   ├── train_ray_wave_generator.py # 11.0M parameter ray-based wave field generator
│   │   ├── train_and_generate_park.py  # Outdoor corpus training loop
│   │   ├── train_sota_generative_park.py # SOTA generative park training pipeline
│   │   ├── execute_full_photoreal_pipeline.py # Full Phase 4 -> Phase 2 -> Phase 1 engine
│   │   ├── execute_phase4_3d_wave_pipeline.py # Multi-view 3D consistency pipeline
│   │   ├── execute_master_remediation.py # Optical wave adapter remediation
│   │   ├── generate_controllable_park.py # Sketch & text conditioned generation
│   │   └── generate_multi_prompt_gallery.py # Multi-prompt seasonal park gallery
│   └── training/                    # Dataset curation & composition scripts
│       ├── clean_dataset.py
│       └── test_real_composition.py
├── metrics_and_logs/                # Verified logs & benchmark JSON metrics
│   ├── train_garden.log             # 29.97 dB PSNR training trace
│   ├── train_bonsai.log             # 32.18 dB PSNR training trace
│   ├── train_stump.log              # 27.65 dB PSNR training trace
│   ├── train_wave_packets.log       # Continuous wave packet optimization log
│   ├── phase4_ray_generator_metrics.json # Multi-view consistency metrics (Ray Generator)
│   ├── phase4_wave_dit_metrics.json # Multi-view consistency metrics (Wave DiT)
│   └── novel_photo_metrics.json     # Novel camera benchmark metrics
├── visual_results/                  # Result comparison panels & diagnostic renders
│   ├── garden_val_00_comparison.png # Garden reconstruction vs ground truth
│   ├── garden_val_01_comparison.png
│   ├── stump_val_00_comparison.png  # Stump reconstruction vs ground truth
│   ├── bonsai_val_00_comparison.png # Bonsai reconstruction vs ground truth
│   ├── bicycle_val_00_comparison.png# Bicycle reconstruction vs ground truth
│   ├── park_control_sketch.png      # Human input control sketch
│   ├── virtual_park_sota_master.png # Generated virtual park master render
│   ├── virtual_park_orbit_left.png  # Novel viewpoint left orbit
│   ├── virtual_park_orbit_right.png # Novel viewpoint right orbit
│   ├── virtual_park_bench_closeup.png# Foreground closeup parallax inspection
│   ├── virtual_park_depth_map.png   # Estimated physical depth map
│   ├── virtual_park_3d_pointcloud.png# Synthesized 3D wave packet cloud
│   └── wave_psf_airy_disk.png       # Phase 1 Bessel J1(x)/x Airy disk PSF
└── plans_and_docs/                  # Project planning and verification dashboards
    ├── phase1_optical_verification.md
    ├── phase2_sota_verification.md
    ├── phase2_5_refinement_and_novel_photo_verification.md
    ├── new_scene_garden_verification.md
    ├── phase3_token_manifest_dashboard.md
    ├── phase4_3d_wave_generative_implementation_plan.md
    ├── phase4_sota_wave_generative_plan.md
    ├── phase4_photorealism_advancement_plan.md
    ├── phase4_photorealistic_park_dashboard.md
    └── virtual_park_generation_dashboard.md
```
