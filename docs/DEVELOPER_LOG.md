# Developer (발생기) experiment log

Protocol: Mip-NeRF 360 outdoor scenes. Teachers = per-scene 3M-splat reconstructions. The developer must grow a
coarsened scene back to full detail. Train on garden/stump/flowers/treehill, bicycle held out.
PSNR on held-out test views vs real photos. "highpass" = fine-detail energy of the render relative to the real photo
(garden view 16). Target: within 1.5 dB of the teacher, highpass >= 75%, shapes kept on the unseen scene.

| run | change | garden PSNR (teacher 27.83) | bicycle unseen (teacher 25.88) | highpass garden | verdict |
|---|---|---|---|---|---|
| v0 (018) | 64x merge, 2 levels, 32x32 genome grid | 20.13 | 17.76 (genome fitted) | 18% | blur; too little info |
| v1 (021) | per-splat latent + neighbours + LPIPS | collapsed | - | - | splats vanished in 1k steps |
| v1.1 (025) | bounded edits, LayerNorm, grad clip | 20.92 | 17.90 (fitted) | 24% | stable, still blur |
| c8 (026) | 8x merge, 1 level | 23.83 | 20.57 (fitted) | 48% | shapes back, out of focus |
| v2 (028) | set encoder writes latents | 22.31 | 19.27 (zero-shot) | 39% | zero-shot works, not sharper |
| v3 (030) | real representative coarse + teacher zoom views + per-voxel set loss | 21.62 | 18.77 (zero-shot) | 50% | sharper edges, lower PSNR; loss flat after 1.5k |

Diagnosis after v3: uniform voxels hold 1..hundreds of splats but the developer always grows 7-8, so dense grass and
foliage are starved. v4 switches to Z-order groups of exactly 8 with slot-wise supervision.
| v4 (031) | Z-order groups of exactly 8, slot-wise loss | 20.39 | 19.89 (zero-shot, best) | 50% | slot loss plateaus at 1.83 |

Diagnosis after v4: v0-v4 all regress ONE answer. Fine texture is high-entropy, so the best single answer is the
average of plausible ones = blur, whatever the loss. v5 makes the developer generative (flow matching over the 8
children), starting from world-anchored noise so sampled detail stays identical across views. Success = highpass and
LPIPS up (PSNR may not rise: it samples plausible detail, not the exact original).
| v5 preview (035) | flow matching, 4k steps, no render loss | 11.83 | 15.01 (zero-shot) | 31% | haze + long streaks: sampled outlier splats; sharp lines appear but structure lost |
| v6 (037) | frozen v4 mean + flow-sampled residual | 17.57 | 17.63 (zero-shot) | 45% | worse than v4: flow found little structure (loss 0.81); random residuals in splat space = speckle, not texture |

Lesson after v5/v6: randomness in splat-parameter space is not texture on screen. Sharp detail has to be learned
through the renderer (screen-space losses). Next: v8 = v7 restorer + patch discriminator on zoomed renders.

## Status when paused

Both GPU boxes were paused (stopped, disks kept) at the user's call. v7 (window transformer, 8x) was still training;
the 2x restorer (plan A) never started.

What worked:
- Reconstruction (photo -> splats -> photo): bicycle PSNR 25.9 (above the original 3DGS paper), 87% fine detail with
  pose refinement. DL3DV parks via the factory: PSNR 27-29.6, LPIPS 0.05-0.08.
- The developer recovers layout and shapes from 8x-coarsened scenes, also zero-shot on an unseen scene (v4: 19.9).

What did not:
- Fine detail after 8x coarsening stayed at ~40-50% of the real photo's high-frequency energy across v0-v4
  (regression = average = blur). Generating splat parameters directly (v5, v6) produced haze/speckle.

Untested next steps: 2x final restorer + cascade of latent painters; screen-space adversarial fine-tuning (v8);
hybrid neural rendering (3D structure + 2D texture network).
