# Generator design v2: Structured Latent Splat Diffusion (SLSD)

Assumption: the restorer works, i.e. photo -> 3D Gaussian reconstruction -> render is faithful on held-out views
(being audited separately; the design only needs per-scene reconstructions as training data).

## What 16 failed attempts taught (ours v0-v8, Gemini attempts 1-7)

| Failure | Seen in | Lesson |
|---|---|---|
| Regressing fine detail -> blur | v0-v4 | Any single-answer loss averages high-entropy texture. Sharpness needs a perceptual + adversarial term (the Stable Diffusion VAE recipe). |
| Sampling raw splat parameters -> haze, speckle, neon SH | v5, v6, Gemini #1 | Raw splat space is not a space to generate in: tiny parameter noise = huge screen error. Generate in a learned, smooth latent; decode with a network trained through the renderer. |
| Fixed branching per uniform voxel -> dense regions starved | v3 | Every structural unit needs a capacity matched to its content. |
| Fixed ray grid, tiny splats -> screen-door | Gemini #6 | Splat size must be free within its cell; no fixed screen-space lattice. |
| Under-training -> mean-colour mud | Gemini #4 | Gate every stage on measured quality, never on "it ran". |
| Collage / 2D or 3D foundation generators | Gemini #2, #3, #7 | Rejected by the owner. Allowed: frozen encoders that do not generate (text encoder, image feature encoders). |
| Unverified "done" reports | Gemini | Scoring is done only by a fixed, independent scorer (infra/eval_ckpt.py) on held-out views. |

## Architecture (all trained parts < 100M parameters)

```
text / sketch (BEV semantic map)
      |
[G1] Structure painter (~20M)      dense 2D flow-matching on a 128x128 bird's-eye grid:
      |                            per cell -> occupancy over height bins (2.5D parks)
      v
 sparse voxel scaffold (occupied cells of a 128x128xH grid)
      |
[G2] Latent painter (~40M)         sparse DiT with flow matching, windowed attention over occupied voxels,
      |                            conditioned on the scaffold + text/sketch; generates a 16-ch latent per voxel
      v
 per-voxel latents (KL-regularized, smooth)
      |
[D]  Splat decoder (~25M)          sparse-conv/transformer: latent -> K=32 Gaussians per voxel
      |                            (offsets bounded inside the voxel, free size/rotation/opacity/colour)
      v
 3D Gaussians -> gsplat render -> virtual camera (lens PSF, sensor noise) -> photo
```

### Why each choice

- **Voxel-anchored Gaussians (scaffold)**: gives the unordered splat set a grid structure so a generator can work
  on it, with a fixed slot budget per *occupied* voxel (empty space costs nothing; dense voxels can be subdivided).
  Positions are offsets bounded inside the voxel, which removes the "fly-away/haze" failure by construction.
- **VAE = encoder E + decoder D trained like the Stable Diffusion autoencoder**: L1 + SSIM + LPIPS on renders from
  zoomed novel cameras, a patch discriminator for sharpness (never tried in v0-v8; v8 was cancelled), and a small KL
  so the latent space is smooth enough for G2 to sample from.
- **Generation only in the latent**: G2 never touches raw splat parameters. Detail is sampled where it is cheap and
  smooth; D turns it into splats with screen-space-trained judgement.
- **2.5D structure first (G1)**: parks are height fields plus vertical objects; generating the bird's-eye occupancy
  first is a small, well-understood 2D problem and is also the sketch interface (ControlNet-style conditioning).
- **World-anchored noise**: G2's initial noise per voxel is hashed from the voxel coordinate, so one world stays
  identical from every camera and can be extended chunk by chunk (outpainting).

## Data

1. Per-scene reconstructions re-fitted as voxel-anchored Gaussians (teacher -> anchored teacher, distillation from
   the free 3DGS renders; gate: <= 0.5 dB loss vs the free reconstruction on held-out views).
2. Scenes: Mip-NeRF 360 outdoor (5) + DL3DV parks from the factory (target 100+).
3. Training samples are random sub-volumes (e.g. 32x32xH voxel crops) with 90-degree rotations and flips around the
   up axis: thousands of crops from ~100 scenes. Up-axis and metric scale are normalized per scene first
   (camera up-vectors -> gravity, median camera height -> 1.6 m).

## Gates (scored only by the fixed scorer, held-out views, unseen scene included)

| Gate | Stage | Pass condition |
|---|---|---|
| G0 | anchored teacher | held-out PSNR within 0.5 dB of free 3DGS; LPIPS within 0.01 |
| G1 | VAE (E+D) | on an unseen scene: PSNR >= teacher - 1.5 dB, LPIPS <= teacher + 0.03, high-frequency energy >= 85% of the photo |
| G2 | structure painter | samples: plausible occupancy (no floating mass), nearest training crop clearly different |
| G3 | latent painter | renders pass the AI-look meter range of real photos; zero flicker across a 30-frame camera move; blind A/B vs real photos |

A stage that fails its gate is not "done", whatever the training log says.

## Work split

- Claude: this design, the scorer and gate tests, job orchestration on the GPU box, verification and reports.
- agy (Gemini): implementation of one component at a time against a written interface + the gate test it must pass.

Order: G0 (anchored teacher) -> G1 (VAE) -> G2 -> G3. G1 is the step that previously failed (v0-v8); it now has the
two missing ingredients (adversarial sharpness, latent regularization) plus a structured scaffold.

## Budget

Rough 4090 hours: anchored teachers 15h, VAE 15h, structure painter 5h, latent painter 15h, evaluation 5h
(~55h, about $22). Current Vast credit (~$4) covers the restorer audit and G0 only.
