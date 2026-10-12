"""Developer v2: encoder + developer (a splat autoencoder) instead of auto-decoded latents.

The encoder reads the teacher splats inside each coarse voxel and writes that voxel's latent; the developer grows
children from (coarse splat, latent, neighbours). Trained end-to-end through the renderer against real photos.
Held-out scene: no fitting at all, the encoder computes its latents directly (zero-shot).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .developer import DeveloperV1, SplatEncoder
from .train_v0 import Scene, photo_loss, render
from .train_v1 import crop, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--ratio", type=int, default=8)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--lpips_w", type=float, default=0.1)
    ap.add_argument("--lpips_start", type=int, default=2000)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    model = DeveloperV1(latent=args.latent, levels=1).to(dev)
    enc = SplatEncoder(latent=args.latent).to(dev)
    scenes = []
    for t in args.teachers:
        name, ckpt = t.split("=")
        sc = Scene(name, f"{args.data}/{name}", ckpt, args.ratio, dev)
        sc.nbr = model.knn(sc.coarse["means"], model.n_nbr)
        sc.inv = torch.unique(torch.floor(sc.teacher["means"] / sc.voxel).long(), dim=0, return_inverse=True)[1]
        scenes.append(sc)
    train_ids = [i for i, s in enumerate(scenes) if s.name != args.holdout]
    n_params = sum(p.numel() for m in (model, enc) for p in m.parameters()) / 1e6
    print(f"developer+encoder params {n_params:.2f}M | latent numbers per scene "
          f"{len(scenes[0].coarse['means']) * args.latent / 1e6:.2f}M", flush=True)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)

    params = list(model.parameters()) + list(enc.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1))))

    # only the active scene's teacher lives on the GPU; recompute activations in backward to fit in 24GB
    for sc in scenes:
        sc.teacher = {k: v.cpu().pin_memory() for k, v in sc.teacher.items()}

    def on_gpu(sp):
        return {k: v.to(dev, non_blocking=True) for k, v in sp.items()}

    def grow(sc):
        fine = on_gpu(sc.teacher)
        lat = checkpoint(enc, fine, sc.coarse, sc.inv, use_reentrant=False)
        del fine
        sp, _, _ = checkpoint(model, sc.coarse, lat, 0, sc.nbr, use_reentrant=False)
        return sp, lat

    t0 = time.time()
    for step in range(args.steps):
        sc = scenes[train_ids[step % len(train_ids)]]
        gt, c2w, K = sc.item(sc.train, np.random.randint(len(sc.train)), dev)
        sp, _ = grow(sc)
        out = render(sp, c2w, K, gt.shape[1], gt.shape[0])
        loss = photo_loss(out, gt)
        if args.lpips_w > 0 and step >= args.lpips_start:
            a, b = crop(out.clamp(0, 1), gt)
            loss = loss + args.lpips_w * lpips(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f} opacity {torch.sigmoid(sp['opacities']).mean():.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)

    results = {}
    with torch.no_grad():
        for i, sc in enumerate(scenes):
            _, lat = grow(sc)
            sc.teacher = on_gpu(sc.teacher)  # evaluate() renders the teacher too
            tag = "train" if i in train_ids else "holdout"
            results[sc.name if tag == "train" else f"{sc.name}_zeroshot"] = evaluate(
                model, sc, lat, 1, lpips, dev, args.out, tag)
            sc.teacher = {k: v.cpu() for k, v in sc.teacher.items()}
            torch.cuda.empty_cache()
    torch.save({"developer": model.state_dict(), "encoder": enc.state_dict()}, f"{args.out}/lightseed_v2.pt")
    json.dump(results, open(f"{args.out}/results.json", "w"), indent=1)
    print("RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
