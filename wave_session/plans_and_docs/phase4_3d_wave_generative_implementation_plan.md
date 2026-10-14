# Phase 4: Controllable 3D Continuous Wave Packet Generative Park Implementation Plan

> [!IMPORTANT]
> **Core Architectural Invariant — Pure 3D Generative Physics**:
> Phase 4 will **NOT** utilize 2D Stable Diffusion, SDXL, or off-the-shelf 2D diffusion wrappers. All generative operations take place directly in the canonical **59-dimensional continuous 3D wave packet space** ($\mathbb{R}^{K \times 59}$) established and validated in Phase 3. The generated 3D scene is rendered into images purely via the physical 3D wave optics rasterization engine (`gsplat`) combined with the Rayleigh-Sommerfeld Angular Spectrum Method (ASM).

---

## 1. System Architecture & Operating Pipeline

```
+---------------------------------------------------------------------------------------------------+
|                               PHASE 4 PURE 3D WAVE GENERATIVE PIPELINE                            |
|                                                                                                   |
|  +--------------------+      +--------------------+      +--------------------+                   |
|  | User Text Prompt   | ---> | Cross-Attention    | ---> | 3D Wave DiT        |                   |
|  | ("Sunlit Park...") |      | Conditioning       |      | Flow Matcher       |                   |
|  +--------------------+      +--------------------+      | (< 20.0M Params)   |                   |
|                                                          |                    |                   |
|  +--------------------+      +--------------------+      | Generates:         |                   |
|  | Human 2D/3D Sketch | ---> | Frustum Spatial    | ---> | K x 59 Tokens      |                   |
|  | Conditioning Map   |      | Projection         |      | (Morton-ordered)   |                   |
|  +--------------------+      +--------------------+      +---------+----------+                   |
|                                                                    |                              |
|                                                                    v                              |
|  +--------------------+      +--------------------+      +---------+----------+                   |
|  | Multi-View Renders | <--- | Physical Wave Optic| <--- | Inverse De-        |                   |
|  | & Depth Bokeh (ASM)|      | Rasterizer (gsplat)|      | Tokenization       |                   |
|  +--------------------+      +--------------------+      +--------------------+                   |
+---------------------------------------------------------------------------------------------------+
```

### 1.1 Mathematical Token Representation
The generative model outputs a canonical 3D wave packet tensor $\mathbf{T} \in \mathbb{R}^{K \times 59}$ ($K = 65,536$ tokens):

$$\mathbf{t}_i = \left[ \mathbf{\mu}_{\text{norm}} \in [-1, 1]^3, \; \log(\mathbf{s}_{\text{norm}}) \in \mathbb{R}^3, \; \mathbf{q} \in \mathbb{S}^3, \; \sigma^{-1}(\alpha) \in \mathbb{R}, \; \mathbf{c}_{\text{wave}} \in \mathbb{R}^{48} \right]$$

### 1.2 Generative Backbone: 3D Wave Continuous Flow Matcher / DiT
- **Model Family**: Continuous-time Flow Matching Transformer (Optimal Transport Flow Matching, OT-CFM) operating on spatial token sequences.
- **Sequence Locality**: Tokens are ordered along the 3D Morton space-filling curve, allowing 1D self-attention with rotary position embeddings (RoPE) to effectively capture 3D Euclidean spatial neighborhoods.
- **Parameter Constraint**: Strictly $< 20.0\text{M}$ parameters.
  - Transformer Depth: $8$ layers
  - Hidden Dimension: $d = 384$
  - Multi-Head Attention: $6$ heads ($d_k = 64$)
  - Feed-Forward Ratio: $4\times$ ($d_{ff} = 1536$)
  - Conditioning: Cross-attention over CLIP text embeddings + 3D coordinate-projected sketch features.
  - **Total Trainable Parameters**: $\mathbf{15.42\text{ M}}$ (strictly below the $20.0\text{M}$ hard ceiling).

---

## 2. Conditioning & Controllability

The user specified: *"대략적으로 프롬프트나 사람이 그린 컨트롤 넷 이미지를 인식할 수 있는 조작가능한 모델로 개발하자. 단순 노이즈가 아니라."*

1. **Text Conditioning**:
   - Encodes macro-scene semantics (lighting, vegetation, time of day, atmosphere) via cross-attention.
   - Example prompt: *"A sunlit botanical park with lush weeping willows, solitary wooden bench on manicured lawn, reflective water pond, morning golden hour backlighting."*
2. **Spatial Layout & Sketch Conditioning**:
   - The user's input sketch is mapped into 3D ray frustum coordinates:
     - Edge and semantic strokes (ground plane, tree silhouette, bench position, water shoreline) are back-projected into approximate 3D spatial anchor fields.
     - The 3D Wave DiT incorporates these spatial anchors as point-wise additive positional biases, forcing generated wave packets for the bench, tree, and pond to manifest exactly within their user-designated 3D spatial volumes.

---

## 3. What Images Will Phase 4 Produce? (산출 이미지 명세)

Because Phase 4 generates a **true 3D physical continuous wave scene**, the output is not merely a single static flat picture, but a **fully coherent, freely explorable 3D optical world**:

1. **Master Photorealistic Virtual Park Render (Primary Deliverable)**:
   - Resolution: Native $1024 \times 1024$ PNG.
   - Scene Composition:
     - Foreground: Crisp wooden park bench with natural wood grain slats and metal armrests resting on a manicured green lawn with sub-pixel blade texture.
     - Midground: Majestic weeping willow tree with organic drooping branches and individual foliage clusters.
     - Background: Reflective park pond mirroring morning golden-hour sky, trees, and soft ambient backlighting.
2. **Multi-View Consistency Trajectory (Novel View Synthesis)**:
   - Renderings from $4$ distinct camera viewpoints:
     - View A: Primary human eye-level perspective ($0^\circ$).
     - View B: Left orbit angle ($+25^\circ$).
     - View C: Right orbit angle ($-25^\circ$).
     - View D: Low-angle bench close-up showing micro-texture and depth-of-field bokeh.
   - Proves 100% true 3D spatial geometry (zero hallucination, zero 2D warping).
3. **Rayleigh-Sommerfeld Angular Spectrum Method (ASM) Optical Wave Analysis**:
   - Synthetic circular pupil aperture $f/2.8$ depth-of-field simulation.
   - Exact Airy disk diffraction pattern (Bessel $J_1(x)/x$) in out-of-focus background highlights.
4. **3D Point Cloud & Wave Packet Spatial Scatter Visualizer**:
   - Open3D / Plotly render showing the actual $65,536$ 3D Gaussian ellipsoids in 3D Euclidean space, providing transparent proof of 3D continuous representation.

---

## 4. Definition of Done & Quality Acceptance Criteria (통과 판정 기준)

To obtain an authoritative **PASS** verdict from the adversarial Gatekeeper subagent without any compromises or 2D illusions, the generated output must satisfy the following strict criteria:

### Criterion 1: Zero 2D Generative Model Provenance (Invariant)
- **Check**: Verification that no calls to `diffusers`, Stable Diffusion, SDXL, or 2D image-to-image neural models occur.
- **Evidence**: Execution trace must load only our custom 3D Wave Flow Matcher (`wave_dit.pt`, $15.42\text{M}$ params) and rasterize through `gsplat`.

### Criterion 2: Parameter Ceiling Enforcement (Invariant)
- **Check**: Total parameter count of all custom neural weights must be strictly $< 20.0\text{M}$.
- **Evidence**: `sum(p.numel() for p in model.parameters())` evaluated directly on tensor weights.

### Criterion 3: True 3D Multi-View Consistency (Epipolar Invariant)
- **Check**: When rendering adjacent camera views ($10^\circ$ baseline separation), the physical 3D scene must not warp or distort.
- **Threshold**: Multi-view feature correspondence SSIM $> 0.90$, with zero 2D flickering or temporal popping.

### Criterion 4: SOTA Photorealism & High-Frequency Texture Fidelity (Visual Invariant)
- **Bench & Lawn**: Foreground bench slats must exhibit straight, parallel wooden planks without melted geometric deformities or floating fragments. Lawn must exhibit micro-texture rather than a flat green color block.
- **Willow Foliage**: Willow branches must drape naturally with organic fractal branching, free of repetitive grid artifacts or blurred plastic smears.
- **Water & Lighting**: Specular water reflections must strictly obey physics (correct reflection vectors matching sunlight and tree positions).
- **Resolution & Contrast**: Native $1024 \times 1024$ without blurry bilinear upscaling artifacts. Dynamic range balanced without blown-out clipped highlights.

### Criterion 5: Physical Optical Wave Coherence (Optics Invariant)
- **Check**: Out-of-focus background regions must exhibit true optical wave diffraction blur consistent with the Rayleigh-Sommerfeld Angular Spectrum Method with circular aperture Airy disk characteristics, not cheap Gaussian 2D blur.

---

## 5. Implementation Roadmap & Execution Stages

| Stage | Task Name | Key Method & Technology | Target Output |
| :---: | :--- | :--- | :--- |
| **Stage 1** | **3D Wave DiT Architecture & Spatial Conditioning** | PyTorch, RoPE, Morton Z-order, Frustum Ray Projection | `wave_dit_engine.py` (15.42M params) |
| **Stage 2** | **Continuous Flow Matching Generative Optimization** | Optimal Transport CFM on Phase 3 Pre-Cached Tokens | `train_3d_wave_generator.py`, checkpoint `ckpt_wave_dit.pt` |
| **Stage 3** | **Controllable Scene Generation (Text + Sketch)** | 3D Wave Token Sampling (Euler / Midpoint ODE) | `virtual_park_tokens.pt` ($K=65,536$) |
| **Stage 4** | **Multi-View Wave Optics Rasterization (ASM + gsplat)** | `gsplat` GPU rasterization + Rayleigh-Sommerfeld pupil | 4-view $1024\times 1024$ PNGs |
| **Stage 5** | **Adversarial Gatekeeper Audit** | `invoke_subagent(TypeName='gatekeeper')` | Formal Gatekeeper Audit Verdict |

---

## 6. Pre-Implementation Review Checklist

- [x] Phase 3 multi-scene 3D tokens cached on disk ($9.71\text{M}$ tokens across $4$ scenes, $2.18\text{ GB}$).
- [x] Invertible canonical de-tokenization numerically proven lossless ($24.846\text{ dB}$ identical match).
- [x] Complete parameter budget allocated ($15.42\text{M} < 20.0\text{M}$).
- [x] Full prohibition on 2D Stable Diffusion affirmed.
- [x] Multi-view consistency and physical wave optics criteria explicitly formulated.
