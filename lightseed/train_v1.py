"""Developer v1: per-splat latent DNA + neighbour context + perceptual loss (fixes v0's blur).

Same protocol as v0 (64x-coarsened teachers, shared developer, held-out generalization), with:
  - each coarse splat owns a learnable 32-d latent (auto-decoded per scene), inherited and edited by children
  - the splitter sees its 8 nearest coarse neighbours (siblings at deeper levels)
  - loss adds LPIPS(VGG) on a random crop, so uncertain detail is no longer averaged into blur
"""
import argparse
import json
import math
import os
import time

import imageio.v2 as imageio
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .developer import DeveloperV1
from .train_v0 import Scene, photo_loss, psnr, render


def develop(model, sp, lat, nbr, levels):
    for lv in range(levels):
        sp, lat, nbr = model(sp, lat, lv, nbr)
    return sp


def crop(a, b, size=512):
    h, w = a.shape[:2]
    y, x = np.random.randint(0, h - size + 1), np.random.randint(0, w - size + 1)
    return a[y:y + size, x:x + size], b[y:y + size, x:x + size]


@torch.no_grad()
def evaluate(model, sc, lat, levels, lpips, dev, out_dir, tag):
    res = {"coarse": [], "developed": [], "teacher": [], "lpips_dev": [], "lpips_teacher": []}
    dsp = develop(model, sc.coarse, lat, sc.nbr, levels)
    os.makedirs(f"{out_dir}/{sc.name}/renders", exist_ok=True)
    for i in range(len(sc.test)):
        gt, c2w, K = sc.item(sc.test, i, dev)
        h, w = gt.shape[:2]
        for k, sp in (("coarse", sc.coarse), ("developed", dsp), ("teacher", sc.teacher)):
            r = render(sp, c2w, K, w, h).clamp(0, 1)
            res[k].append(psnr(r, gt))
            if k != "coarse":
                res["lpips_dev" if k == "developed" else "lpips_teacher"].append(
                    lpips(r.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).item())
            if k == "developed" and i in (0, 8, 16):
                imageio.imwrite(f"{out_dir}/{sc.name}/renders/val_step{tag}_{i:04d}.png",
                                (torch.cat([gt, r], 1).cpu().numpy() * 255).astype(np.uint8))
    out = {k: float(np.mean(v)) for k, v in res.items()}
    out["recovered"] = (out["developed"] - out["coarse"]) / max(out["teacher"] - out["coarse"], 1e-6)
    out["n_splats"] = len(dsp["means"])
    print(f"EVAL {tag} {sc.name}: " + " ".join(f"{k}={v:.3f}" for k, v in out.items()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--holdout_steps", type=int, default=5000)
    ap.add_argument("--ratio", type=int, default=64)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--lpips_w", type=float, default=0.1)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    model = DeveloperV1(levels=args.levels).to(dev)
    scenes, lats = [], []
    for t in args.teachers:
        name, ckpt = t.split("=")
        sc = Scene(name, f"{args.data}/{name}", ckpt, args.ratio, dev)
        sc.nbr = model.knn(sc.coarse["means"], model.n_nbr)
        scenes.append(sc)
        lats.append(torch.nn.Parameter(torch.zeros(len(sc.coarse["means"]), model.latent, device=dev)))
    train_ids = [i for i, s in enumerate(scenes) if s.name != args.holdout]
    print(f"developer params {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M | "
          f"latent numbers per scene {lats[0].numel() / 1e6:.2f}M", flush=True)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)

    opt = torch.optim.Adam([{"params": model.parameters(), "lr": 3e-4},
                            {"params": [lats[i] for i in train_ids], "lr": 3e-3}])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1))))
    t0 = time.time()

    def step_loss(i):
        sc = scenes[i]
        gt, c2w, K = sc.item(sc.train, np.random.randint(len(sc.train)), dev)
        out = render(develop(model, sc.coarse, lats[i], sc.nbr, args.levels), c2w, K, gt.shape[1], gt.shape[0])
        loss = photo_loss(out, gt)
        if args.lpips_w > 0:
            a, b = crop(out.clamp(0, 1), gt)
            loss = loss + args.lpips_w * lpips(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None])
        return loss

    for step in range(args.steps):
        loss = step_loss(train_ids[step % len(train_ids)])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    results = {s.name: evaluate(model, s, lats[i], args.levels, lpips, dev, args.out, "train")
               for i, s in enumerate(scenes) if i in train_ids}
    if args.holdout:
        hi = next(i for i, s in enumerate(scenes) if s.name == args.holdout)
        for p in model.parameters():
            p.requires_grad_(False)
        lopt = torch.optim.Adam([lats[hi]], lr=3e-3)
        for _ in range(args.holdout_steps):
            loss = step_loss(hi)
            lopt.zero_grad(set_to_none=True)
            loss.backward()
            lopt.step()
        results[f"{args.holdout}_fitted"] = evaluate(model, scenes[hi], lats[hi], args.levels, lpips, dev, args.out, "holdout")

    torch.save({"developer": model.state_dict(), "latents": [l.detach().cpu() for l in lats]}, f"{args.out}/lightseed_v1.pt")
    json.dump(results, open(f"{args.out}/results.json", "w"), indent=1)
    print("RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
