"""Developer v0: can ONE small shared network restore fine detail across several scenes?

For each scene, a teacher reconstruction is coarsened 64x (voxel merge). The shared Developer splits every
coarse splat into 8 children, twice (64x back to full count), conditioned on a per-scene genome grid and
world-anchored noise. Loss = photo loss against the real training photos. Coarse splats stay frozen, so all
recovered detail must come from the Developer + genome.

Phase 2 (generalization): freeze the Developer, fit only a fresh genome for a held-out scene.

usage: python -m lightseed.train_v0 --data /workspace/data/mip360 --teachers bicycle=ckpt.pt garden=ckpt.pt ... \
         --holdout bicycle --out /workspace/results/lightseed_v0
"""
import argparse
import json
import math
import os
import sys
import time

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("GSPLAT_EXAMPLES", "/workspace/gsplat/examples"))
from datasets.colmap import Dataset, Parser  # noqa: E402
from fused_ssim import fused_ssim  # noqa: E402
from gsplat import rasterization  # noqa: E402

from .developer import Developer, Genome  # noqa: E402
from .gaussians import coarsen_to, load_ckpt  # noqa: E402


def render(sp, c2w, K, w, h):
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)
    img, _, _ = rasterization(
        means=sp["means"], quats=sp["quats"], scales=sp["scales"].exp(), opacities=torch.sigmoid(sp["opacities"]),
        colors=colors, viewmats=torch.linalg.inv(c2w)[None], Ks=K[None], width=w, height=h,
        sh_degree=3, rasterize_mode="antialiased")
    return img[0]


def psnr(a, b):
    return (-10 * torch.log10(F.mse_loss(a.clamp(0, 1), b))).item()


class Scene:
    def __init__(self, name, data_dir, ckpt, ratio, dev):
        self.name = name
        p = Parser(data_dir=data_dir, factor=4, normalize=True, test_every=8)
        self.train, self.test = Dataset(p, split="train"), Dataset(p, split="val")
        self.teacher = load_ckpt(ckpt, dev)
        n = len(self.teacher["means"])
        self.coarse, voxel = coarsen_to(self.teacher, n // ratio)
        print(f"[{name}] teacher {n} -> coarse {len(self.coarse['means'])} (voxel {voxel:.4f}), "
              f"train {len(self.train)} test {len(self.test)}", flush=True)
        self.cache = {}

    def item(self, ds, i, dev):
        key = (id(ds), i)
        if key not in self.cache:
            d = ds[i]
            self.cache[key] = (d["image"].to(torch.uint8), d["camtoworld"], d["K"])
        img, c2w, K = self.cache[key]
        return img.to(dev).float() / 255, c2w.to(dev), K.to(dev)


def develop(model, genome, gi, sp, levels):
    for lv in range(levels):
        sp = model(sp, genome(gi, sp["means"]), lv)
    return sp


def photo_loss(out, gt):
    l1 = F.l1_loss(out, gt)
    ss = 1 - fused_ssim(out.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], padding="valid")
    return 0.8 * l1 + 0.2 * ss


@torch.no_grad()
def evaluate(model, genome, gi, sc, levels, dev, out_dir, tag):
    res = {"coarse": [], "developed": [], "teacher": []}
    dsp = develop(model, genome, gi, sc.coarse, levels)
    os.makedirs(f"{out_dir}/{sc.name}/renders", exist_ok=True)
    for i in range(len(sc.test)):
        gt, c2w, K = sc.item(sc.test, i, dev)
        h, w = gt.shape[:2]
        for k, sp in (("coarse", sc.coarse), ("developed", dsp), ("teacher", sc.teacher)):
            r = render(sp, c2w, K, w, h)
            res[k].append(psnr(r, gt))
            if k == "developed" and i in (0, 8, 16):
                canvas = torch.cat([gt, r.clamp(0, 1)], 1)
                imageio.imwrite(f"{out_dir}/{sc.name}/renders/val_step{tag}_{i:04d}.png",
                                (canvas.cpu().numpy() * 255).astype(np.uint8))
    out = {k: float(np.mean(v)) for k, v in res.items()}
    out["recovered"] = (out["developed"] - out["coarse"]) / max(out["teacher"] - out["coarse"], 1e-6)
    print(f"EVAL {tag} {sc.name}: " + " ".join(f"{k}={v:.3f}" for k, v in out.items()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teachers", nargs="+", required=True, help="scene=ckpt_path")
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--holdout_steps", type=int, default=3000)
    ap.add_argument("--ratio", type=int, default=64)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--width", type=int, default=512)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    scenes = []
    for t in args.teachers:
        name, ckpt = t.split("=")
        scenes.append(Scene(name, f"{args.data}/{name}", ckpt, args.ratio, dev))
    train_ids = [i for i, s in enumerate(scenes) if s.name != args.holdout]

    model = Developer(levels=args.levels, width=args.width).to(dev)
    genome = Genome(len(scenes)).to(dev)
    for i, s in enumerate(scenes):
        genome.set_bounds(i, s.coarse["means"][:, :2])
    print(f"developer params {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M", flush=True)

    opt = torch.optim.Adam([{"params": model.parameters(), "lr": 3e-4},
                            {"params": genome.parameters(), "lr": 1e-2}])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1))))
    t0 = time.time()
    for step in range(args.steps):
        gi = train_ids[step % len(train_ids)]
        sc = scenes[gi]
        gt, c2w, K = sc.item(sc.train, np.random.randint(len(sc.train)), dev)
        out = render(develop(model, genome, gi, sc.coarse, args.levels), c2w, K, gt.shape[1], gt.shape[0])
        loss = photo_loss(out, gt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    results = {s.name: evaluate(model, genome, i, s, args.levels, dev, args.out, "train")
               for i, s in enumerate(scenes) if i in train_ids}

    if args.holdout:
        hi = next(i for i, s in enumerate(scenes) if s.name == args.holdout)
        sc = scenes[hi]
        results[f"{sc.name}_zero_genome"] = evaluate(model, genome, hi, sc, args.levels, dev, args.out, "zero")
        for p in model.parameters():
            p.requires_grad_(False)
        gopt = torch.optim.Adam(genome.parameters(), lr=1e-2)
        for step in range(args.holdout_steps):
            gt, c2w, K = sc.item(sc.train, np.random.randint(len(sc.train)), dev)
            out = render(develop(model, genome, hi, sc.coarse, args.levels), c2w, K, gt.shape[1], gt.shape[0])
            loss = photo_loss(out, gt)
            gopt.zero_grad(set_to_none=True)
            loss.backward()
            gopt.step()
        results[f"{sc.name}_fitted_genome"] = evaluate(model, genome, hi, sc, args.levels, dev, args.out, "holdout")

    torch.save({"developer": model.state_dict(), "genome": genome.state_dict()}, f"{args.out}/lightseed_v0.pt")
    json.dump(results, open(f"{args.out}/results.json", "w"), indent=1)
    print("RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
