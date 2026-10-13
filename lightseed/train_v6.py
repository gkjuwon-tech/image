"""Developer v6: v4's regressor draws the skeleton (mean children), a flow samples only the residual detail.

v4 keeps structure (best zero-shot) but is blurry; the v5 preview sampled sharp lines but outlier splats turned the
scene to haze. Here the frozen v4 developer+encoder gives the mean children m; a flow learns the distribution of
(teacher - m) in normalized feature space, started from world-anchored noise. Residuals are small and clamped, so a
bad sample perturbs a splat instead of creating a screen-covering one.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .developer import DeveloperV1
from .morton import group8, merge_groups, to_parent_frame
from .train_v0 import Scene
from .train_v3 import evaluate
from .train_v4 import GroupEncoder, child_features
from .train_v5 import D, FlowDeveloper, features_to_splats, parent_features, world_noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--v4", required=True, help="lightseed_v4.pt (frozen mean model)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--sample_steps", type=int, default=24)
    ap.add_argument("--residual_scale", type=float, default=1.0, help="0 = v4 mean only")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.v4, map_location=dev)
    mean_model = DeveloperV1(latent=64, levels=1, k=8).to(dev).eval()
    mean_model.load_state_dict(ck["developer"])
    enc = GroupEncoder(latent=64).to(dev).eval()
    enc.load_state_dict(ck["encoder"])
    flow = FlowDeveloper(latent=64).to(dev)
    # the flow also sees the mean children it is correcting
    flow.inp = torch.nn.Linear(2 * 8 * D, flow.inp.out_features).to(dev)

    scenes = []
    with torch.no_grad():
        for t in args.teachers:
            name, ckpt = t.split("=")
            sc = Scene(name, f"{args.data}/{name}", ckpt, 64, dev)
            sc.teacher = group8(sc.teacher)
            sc.coarse = merge_groups(sc.teacher)
            x = child_features(to_parent_frame(sc.teacher, sc.coarse))
            sc.nbr = DeveloperV1.knn(sc.coarse["means"], 8)
            sc.lat = torch.cat([enc(x[i:i + 65536]) for i in range(0, len(x), 65536)])
            kids, _, _ = mean_model(sc.coarse, sc.lat, 0, sc.nbr)
            sc.m = child_features(to_parent_frame(kids, sc.coarse)).flatten(1)
            sc.r = x.flatten(1) - sc.m
            sc.teacher = {k: v.cpu() for k, v in sc.teacher.items()}
            del kids, x
            torch.cuda.empty_cache()
            print(f"[{name}] {len(sc.m)} groups, residual std {sc.r.std():.4f}", flush=True)
            scenes.append(sc)
    scenes_list = scenes
    train_ids = [i for i, s in enumerate(scenes_list) if s.name != args.holdout]
    allr = torch.cat([scenes_list[i].r for i in train_ids])
    rs = allr.std(0).clamp_min(1e-4)
    allm = torch.cat([scenes_list[i].m for i in train_ids])
    mm, ms = allm.mean(0), allm.std(0).clamp_min(1e-3)
    del allr, allm

    def cond(sc, idx):
        parent = {k: v[idx] for k, v in sc.coarse.items()}
        nb = sc.nbr[idx]
        size = parent["scales"].exp().max(-1).values
        rel = ((sc.coarse["means"][nb] - parent["means"][:, None]) / size[:, None, None]).clamp(-8, 8) / 8
        ctx = flow.nbr(torch.cat([rel, sc.coarse["sh0"][nb][:, :, 0], sc.lat[nb]], -1)).max(1).values
        return flow.cond(torch.cat([parent_features(parent), sc.lat[idx], ctx], -1)), parent, size

    opt = torch.optim.AdamW(flow.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1))))
    t0 = time.time()
    for step in range(args.steps):
        sc = scenes_list[train_ids[step % len(train_ids)]]
        idx = torch.randint(len(sc.m), (args.batch,), device=dev)
        c, _, _ = cond(sc, idx)
        mn = (sc.m[idx] - mm) / ms
        x1 = sc.r[idx] / rs
        z = torch.randn_like(x1)
        t = torch.rand(len(idx), device=dev)
        xt = t[:, None] * x1 + (1 - t[:, None]) * z
        loss = F.mse_loss(flow(torch.cat([xt, mn], -1), t, c), x1 - z)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"step {step} flow loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def grow(sc):
        G = len(sc.m)
        out = []
        for i in range(0, G, 65536):
            idx = torch.arange(i, min(i + 65536, G), device=dev)
            c, parent, size = cond(sc, idx)
            mn = (sc.m[idx] - mm) / ms
            x = world_noise(parent["means"], size, 8 * D)
            n = args.sample_steps
            for s in range(n):
                t = torch.full((len(x),), s / n, device=dev)
                x = x + flow(torch.cat([x, mn], -1), t, c) / n
            out.append(sc.m[idx] + args.residual_scale * x.clamp(-3, 3) * rs)
        return features_to_splats(torch.cat(out).view(G, 8, D), sc.coarse)

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)
    results = {}
    for i, sc in enumerate(scenes_list):
        if sc.name not in ("bicycle", "garden"):
            continue
        tag = "train" if i in train_ids else "holdout"
        results[sc.name if tag == "train" else f"{sc.name}_zeroshot"] = evaluate(grow, sc, lpips, dev, args.out, tag)
        torch.cuda.empty_cache()
    torch.save({"flow": flow.state_dict(), "rs": rs, "mm": mm, "ms": ms}, f"{args.out}/lightseed_v6.pt")
    json.dump(results, open(f"{args.out}/results.json", "w"), indent=1)
    print("RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
