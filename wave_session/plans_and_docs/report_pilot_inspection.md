# Visual Inspection & Image Provenance Report

## Document Metadata
- Created: 2026-10-12
- Last updated: 2026-10-12
- Modified by: Antigravity
- Version: 1.0.0

---

## 1. Image Provenance & Methodology Taxonomy

To ensure zero ambiguity, every render produced across this project is categorized into one of three distinct stages:

| Category | Definition | Role in Pipeline | Status in This Project |
| :--- | :--- | :--- | :--- |
| **Ground Truth (GT)** | Raw optical photographs captured by physical cameras in the real world. | Serves as the supervision signal (ground reality) for 3D reconstruction and generative validation. | Provided by benchmark datasets (DL3DV-10K, Mip-NeRF 360). |
| **Gaussian Reconstruction Fitting** | Inverse-rendering optimization fitting 3D Gaussians/surfels to match multi-view GT photos. | **Phase 1 (Data Factory)**: Converts 2D multi-view image sets into unified 3D Radiance Field scene assets. | **Active & Under Test** (Tested on Mip-NeRF 360 Garden & DL3DV-10K Greenhouse). |
| **Generative Model Synthesis** | Directly synthesizing a 3D Radiance Field from natural language text prompts (e.g., via DiT). | **Phase 2 (Core Product)**: The target foundation model ("LuxField") that generates novel 3D worlds without reference photos. | **Not Yet Started** (Awaiting completion and quality-gating of Phase 1 dataset pairs). |

> [!IMPORTANT]
> None of the images shown below were synthesized by a text-to-3D generative model. They are **Gaussian inverse-rendering reconstructions** from real-world camera scans, evaluated against the original camera photographs (**GT**).

---

## 2. DL3DV-10K Pilot Inspection: Botanical Greenhouse (`0569e83f...`)

The following images were produced during our 5,000-step 2D Gaussian Splatting (**2DGS**) pilot optimization on the RTX 4090 GPU instance.

### 2.1 Viewpoint Comparisons: Left = Ground Truth (GT) vs Right = 2DGS Reconstruction

#### Viewpoint 0000
![DL3DV Pilot Frame 00 Comparison](/root/.gemini/antigravity-cli/brain/dbd210b4-352d-4c19-afa6-b76447026d5a/val_0000.png)

#### Viewpoint 0010
![DL3DV Pilot Frame 10 Comparison](/root/.gemini/antigravity-cli/brain/dbd210b4-352d-4c19-afa6-b76447026d5a/val_0010.png)

#### Viewpoint 0020
![DL3DV Pilot Frame 20 Comparison](/root/.gemini/antigravity-cli/brain/dbd210b4-352d-4c19-afa6-b76447026d5a/val_0020.png)

### 2.2 Surface Normal Geometric Field
The surface normal field rendered from the explicit 2D planar surfels ($n_x, n_y, n_z \rightarrow \text{RGB}$):

![DL3DV Pilot Normal Map](/root/.gemini/antigravity-cli/brain/dbd210b4-352d-4c19-afa6-b76447026d5a/val_0000_normal.png)

### 2.3 Diagnostic Evaluation of Pilot Run
- **Foreground / Subject Resolution**: The tropical foliage (banana leaves), red flower dress, and foreground pink flower stalks achieved clean, sharp recovery without directional needle splats.
- **Background / Peripheral Defect**: The distant glass atrium dome and deep background foliage exhibit clouding, blur, and incomplete convergence. 
  - *Cause*: A short 5,000-step pilot on synthetic ray-sampled SfM points concentrated gradient updates on high-contrast foreground geometry, leaving low-contrast distant glass panels under-densified. Full COLMAP point initialization and extended training iterations (15,000+ steps) are required for full-depth convergence.

---

## 3. Mip-NeRF 360 Inspection: Outdoor Garden

### 3.1 Full Scene 5K Render (Camera #80)
![Mip-NeRF Garden Full 5K Render](/root/.gemini/antigravity-cli/brain/dbd210b4-352d-4c19-afa6-b76447026d5a/garden_render_cam80.png)

### 3.2 Peripheral Grass Blades (The Epipolar Brush-Stroke Defect)
![Mip-NeRF Garden Grass Crop](/root/.gemini/antigravity-cli/brain/dbd210b4-352d-4c19-afa6-b76447026d5a/garden_crop_native_grass.png)

- **Nature of Image**: Inverse-rendered from a 5.83M 3D Gaussian volume (`point_cloud.ply`) trained on 185 real camera viewpoints.
- **Diagnostic Finding**: In vanilla 3DGS (v1.0), high-frequency organic textures (grass blades) cause 3D volumetric ellipsoids to elongate along the camera motion baseline, producing Impressionist oil-painting brush strokes at grazing viewing angles.
