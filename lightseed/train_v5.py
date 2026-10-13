"""Developer v5: a generative developer (conditional flow matching) instead of a regressor.

v0-v4 all stalled at ~50% fine detail: every one of them regressed a single answer, and fine texture is high-entropy,
so the best single answer is the average of the plausible ones = blur. v5 learns the distribution of the 8 children
of a Z-order group given (parent, latent, neighbours) and samples one. The starting noise of each group is hashed
from the parent's world position, so the sampled detail is sharp AND identical from every camera.

Training never renders: it is flow matching on child parameters (fast). Evaluation renders against real photos.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .developer import DeveloperV1
from .gaussians import quat_to_rot
from .morton import group8, merge_groups, to_parent_frame
from .train_v0 import Scene
from .train_v3 import evaluate
from .train_v4 import GroupEncoder, child_features

D = 59  # per-child feature size (see child_features)


def world_noise(xyz, cell, dim, salt=0):
    """Gaussian noise that depends only on the 3D cell (and salt): same parent -> same noise from any camera."""
    k = torch.floor(xyz / cell[:, None]).long()
    h = (k[:, 0] * 73856093) ^ (k[:, 1] * 19349663) ^ (k[:, 2] * 83492791) ^ salt
    h = (h.remainder(2 ** 31)).double()
    i = torch.arange(dim, device=xyz.device, dtype=torch.float64)
    u1 = torch.remainder(torch.sin(h[:, None] * 12.9898 + i * 78.233) * 43758.5453, 1.0).clamp(1e-7, 1 - 1e-7)
    u2 = torch.remainder(torch.sin(h[:, None] * 39.3467 + i * 11.135) * 24634.6345, 1.0)
    return (torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * math.pi * u2)).float()


def parent_features(p):
    return torch.cat([p["scales"], F.normalize(p["quats"], dim=-1), torch.sigmoid(p["opacities"])[:, None], p["sh0"][:, 0]], -1)


class FlowDeveloper(nn.Module):
    """v(x_t, t | parent, latent, neighbours) over the 8 children's features (8 x 59 = 472 numbers)."""

    def __init__(self, latent=64, width=1024, n_nbr=8):
        super().__init__()
        self.n_nbr = n_nbr
        self.nbr = nn.Sequential(nn.Linear(3 + 3 + latent, 128), nn.SiLU(), nn.Linear(128, 128))
        d_c = 11 + latent + 128
        self.cond = nn.Sequential(nn.Linear(d_c, width), nn.SiLU())
        self.temb = nn.Sequential(nn.Linear(64, width), nn.SiLU())
        self.inp = nn.Linear(8 * D, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                                   nn.Linear(width, width)) for _ in range(4)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 8 * D))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, x, t, c):
        f = torch.arange(32, device=x.device)
        ang = t[:, None] * 1000 * torch.exp(-math.log(10000) * f / 32)
        h = self.inp(x) + c + self.temb(torch.cat([ang.sin(), ang.cos()], -1))
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


def features_to_splats(x, parent):
    """Inverse of child_features: [G, 8, 59] (unnormalized) -> splat dict in world space."""
    G = len(parent["means"])
    pos, dls, q, op, sh0, shN = x.split([3, 3, 4, 1, 3, 45], -1)
    R = quat_to_rot(F.normalize(parent["quats"], dim=-1))
    ps = parent["scales"].exp()
    means = parent["means"][:, None] + torch.einsum("gij,gkj->gki", R, pos * 8 * ps[:, None])
    return {
        "means": means.reshape(-1, 3),
        "scales": (parent["scales"][:, None] + dls * 6).reshape(-1, 3),
        "quats": F.normalize(q, dim=-1).reshape(-1, 4),
        "opacities": torch.logit(op.clamp(1e-4, 1 - 1e-4)).reshape(-1),
        "sh0": sh0.reshape(-1, 1, 3),
        "shN": shN.reshape(-1, 15, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--sample_steps", type=int, default=24)
    ap.add_argument("--eval_scenes", nargs="*", default=None, help="evaluate only these scenes (quick preview)")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    flow = FlowDeveloper(latent=args.latent).to(dev)
    enc = GroupEncoder(latent=args.latent).to(dev)
    scenes = []
    for t in args.teachers:
        name, ckpt = t.split("=")
        sc = Scene(name, f"{args.data}/{name}", ckpt, 64, dev)
        sc.teacher = group8(sc.teacher)
        sc.coarse = merge_groups(sc.teacher)
        sc.x = child_features(to_parent_frame(sc.teacher, sc.coarse))  # [G, 8, 59] on GPU
        sc.nbr = DeveloperV1.knn(sc.coarse["means"], flow.n_nbr)
        sc.teacher = {k: v.cpu() for k, v in sc.teacher.items()}
        scenes.append(sc)
        torch.cuda.empty_cache()
        print(f"[{name}] {len(sc.coarse['means'])} groups", flush=True)
    train_ids = [i for i, s in enumerate(scenes) if s.name != args.holdout]
    allx = torch.cat([scenes[i].x.flatten(1) for i in train_ids])
    mu, sd = allx.mean(0), allx.std(0).clamp_min(1e-3)
    del allx
    print(f"flow params {sum(p.numel() for p in flow.parameters()) / 1e6:.2f}M, encoder "
          f"{sum(p.numel() for p in enc.parameters()) / 1e6:.2f}M", flush=True)

    params = list(flow.parameters()) + list(enc.parameters())
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1))))
    t0 = time.time()
    for step in range(args.steps):
        sc = scenes[train_ids[step % len(train_ids)]]
        idx = torch.randint(len(sc.coarse["means"]), (args.batch,), device=dev)
        x1 = (sc.x[idx].flatten(1) - mu) / sd
        lat = enc(sc.x[idx])
        # neighbours need latents too: encode them (cheap MLP) for this batch's neighbour indices
        nb = sc.nbr[idx]
        lat_nb = enc(sc.x[nb.flatten()]).view(len(idx), -1, args.latent)
        parent = {k: v[idx] for k, v in sc.coarse.items()}
        size = parent["scales"].exp().max(-1).values
        rel = ((sc.coarse["means"][nb] - parent["means"][:, None]) / size[:, None, None]).clamp(-8, 8) / 8
        ctx = flow.nbr(torch.cat([rel, sc.coarse["sh0"][nb][:, :, 0], lat_nb], -1)).max(1).values
        c = flow.cond(torch.cat([parent_features(parent), lat, ctx], -1))
        z = torch.randn_like(x1)
        t = torch.rand(len(idx), device=dev)
        xt = t[:, None] * x1 + (1 - t[:, None]) * z
        loss = F.mse_loss(flow(xt, t, c), x1 - z)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"step {step} flow loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def grow(sc):
        G = len(sc.coarse["means"])
        lat = torch.cat([enc(sc.x[i:i + 65536]) for i in range(0, G, 65536)])
        out = []
        for i in range(0, G, 65536):
            sl = slice(i, min(i + 65536, G))
            parent = {k: v[sl] for k, v in sc.coarse.items()}
            nb = sc.nbr[sl]
            size = parent["scales"].exp().max(-1).values
            rel = ((sc.coarse["means"][nb] - parent["means"][:, None]) / size[:, None, None]).clamp(-8, 8) / 8
            ctx = flow.nbr(torch.cat([rel, sc.coarse["sh0"][nb][:, :, 0], lat[nb]], -1)).max(1).values
            c = flow.cond(torch.cat([parent_features(parent), lat[sl], ctx], -1))
            x = world_noise(parent["means"], size, 8 * D)  # world-anchored starting noise
            n = args.sample_steps
            for s in range(n):
                t = torch.full((len(x),), s / n, device=dev)
                x = x + flow(x, t, c) / n
            out.append(x.clamp(-3, 3) * sd + mu)  # drop sampling outliers (huge/flat splats -> haze)
        return features_to_splats(torch.cat(out).view(G, 8, D), sc.coarse)

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)
    results = {}
    for i, sc in enumerate(scenes):
        if args.eval_scenes and sc.name not in args.eval_scenes:
            continue
        tag = "train" if i in train_ids else "holdout"
        results[sc.name if tag == "train" else f"{sc.name}_zeroshot"] = evaluate(grow, sc, lpips, dev, args.out, tag)
        torch.cuda.empty_cache()
    torch.save({"flow": flow.state_dict(), "encoder": enc.state_dict(), "mu": mu, "sd": sd}, f"{args.out}/lightseed_v5.pt")
    json.dump(results, open(f"{args.out}/results.json", "w"), indent=1)
    print("RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
