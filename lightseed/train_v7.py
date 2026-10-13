"""Developer v7: window-transformer restorer.

v4 grows every Z-order group of 8 on its own (plus a max-pooled glimpse of 8 neighbours). Thin structures such as
bench slats and branches span dozens of groups, so independent groups draw them inconsistently and they melt.
v7 lets 64 consecutive Z-order groups (512 splats of one local region) attend to each other before each group emits
its 8 children, and doubles the latent to 128. Losses as v4: slot-wise child loss + zoomed teacher views + photos.
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
from torch.utils.checkpoint import checkpoint
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .morton import group8, merge_groups, to_parent_frame
from .train_v0 import Scene, photo_loss, render
from .train_v1 import crop
from .train_v3 import evaluate, novel_camera
from .train_v4 import GroupEncoder, child_features
from .train_v5 import D, features_to_splats, parent_features

WIN = 64


class WindowRestorer(nn.Module):
    def __init__(self, latent=128, d=256, layers=4, heads=8, k=8):
        super().__init__()
        self.k = k
        self.inp = nn.Linear(11 + 3 + 1 + latent, d)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True,
                                                                norm_first=True, activation="gelu")
                                     for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, k * D))
        # start from a sane split: children on the parent's cube corners (+-0.5 sigma), half the parent's size
        nn.init.zeros_(self.head[1].weight)
        b = torch.zeros(k, D)
        corners = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32)[:k]
        b[:, 0:3] = torch.atanh(corners * 0.5 / 8)
        b[:, 3:6] = math.atanh((-math.log(k) / 3 / 6 + 1) / 0.55 - 1)  # each child ~ 1/k of the parent volume
        self.head[1].bias.data = b.flatten()

    def forward(self, parent, lat):
        G = len(lat)
        pad = (-G) % WIN
        pos = parent["means"]
        x = torch.cat([parent_features(parent), lat], -1)
        if pad:
            x = torch.cat([x, x[-1:].expand(pad, -1)])
            pos = torch.cat([pos, pos[-1:].expand(pad, -1)])
        x, pos = x.view(-1, WIN, x.shape[-1]), pos.view(-1, WIN, 3)
        c = pos.mean(1, keepdim=True)
        ext = (pos - c).abs().amax((1, 2), keepdim=True).clamp_min(1e-6)
        size = parent_features(parent)[:, :3].exp().amax(-1)
        if pad:
            size = torch.cat([size, size[-1:].expand(pad)])
        rel_size = (size.view(-1, WIN, 1) / ext).clamp(max=1)
        h = self.inp(torch.cat([x[..., :11], (pos - c) / ext, rel_size, x[..., 11:]], -1))
        for layer in self.layers:
            h = checkpoint(layer, h, use_reentrant=False)
        out = self.head(h).view(-1, WIN, self.k, D).flatten(0, 1)[:G]
        pos_, dls, q, op, sh0, shN = out.split([3, 3, 4, 1, 3, 45], -1)
        # dls (normalized by 6) bounded to [-1, 0.1]: a child is never far larger than its parent (that blew up
        # the rasterizer's tile lists in the first v7 run)
        return torch.cat([torch.tanh(pos_), (torch.tanh(dls) + 1) * 0.55 - 1, F.normalize(q + torch.tensor([1.0, 0, 0, 0], device=q.device), dim=-1),
                          torch.sigmoid(op), sh0, shN], -1)


def feat_loss(pred, target):
    w = torch.tensor([1.0] * 3 + [0.5] * 3 + [0.2] * 4 + [1.0] + [1.0] * 3 + [0.3] * 45, device=pred.device)
    q = (1 - (pred[..., 6:10] * target[..., 6:10]).sum(-1).abs()).mean()
    rest = ((pred - target).abs() * w).mean()
    return rest + 0.2 * q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--k", type=int, default=8, help="children per parent (compression ratio of this level)")
    ap.add_argument("--slot_w", type=float, default=4.0)
    ap.add_argument("--photo_every", type=int, default=4)
    ap.add_argument("--lpips_w", type=float, default=0.1)
    ap.add_argument("--lpips_start", type=int, default=2000)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)

    model = WindowRestorer(latent=args.latent, k=args.k).to(dev)
    enc = GroupEncoder(latent=args.latent, k=args.k).to(dev)
    scenes = []
    for t in args.teachers:
        name, ckpt = t.split("=")
        sc = Scene(name, f"{args.data}/{name}", ckpt, 64, dev)
        sc.teacher = group8(sc.teacher, args.k)
        sc.coarse = merge_groups(sc.teacher, args.k)
        sc.x = child_features(to_parent_frame(sc.teacher, sc.coarse, args.k))
        sc.teacher = {k: v.cpu().pin_memory() for k, v in sc.teacher.items()}
        scenes.append(sc)
        torch.cuda.empty_cache()
        print(f"[{name}] {len(sc.x)} groups in {math.ceil(len(sc.x) / WIN)} windows", flush=True)
    train_ids = [i for i, s in enumerate(scenes) if s.name != args.holdout]
    params = list(model.parameters()) + list(enc.parameters())
    print(f"params {sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)
    opt = torch.optim.AdamW(params, lr=3e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1))))

    def grow_feats(sc):
        lat = checkpoint(enc, sc.x, use_reentrant=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(sc.coarse, lat)
        return out.float()

    def grow(sc):
        with torch.no_grad():
            return features_to_splats(grow_feats(sc), sc.coarse)

    def perceptual(out, ref, step):
        loss = photo_loss(out, ref)
        if args.lpips_w > 0 and step >= args.lpips_start:
            a, b = crop(out.clamp(0, 1), ref, min(512, out.shape[0], out.shape[1]))
            loss = loss + args.lpips_w * lpips(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None])
        return loss

    t0 = time.time()
    for step in range(args.steps):
        sc = scenes[train_ids[step % len(train_ids)]]
        feats = grow_feats(sc)
        ls = feat_loss(feats, sc.x)
        kids = features_to_splats(feats, sc.coarse)
        loss = args.slot_w * ls
        teacher = {k: v.to(dev, non_blocking=True) for k, v in sc.teacher.items()}
        ia, ib = np.random.randint(len(sc.train), size=2)
        gt_a, ca, K = sc.item(sc.train, ia, dev)
        _, cb, _ = sc.item(sc.train, ib, dev)
        c2w, K2, _ = novel_camera(ca, cb, K, gt_a.shape[1], gt_a.shape[0])
        with torch.no_grad():
            ref = render(teacher, c2w, K2, 512, 512).clamp(0, 1)
        del teacher
        loss = loss + perceptual(render(kids, c2w, K2, 512, 512), ref, step)
        if step % args.photo_every == 0:
            gt, cp, Kp = sc.item(sc.train, np.random.randint(len(sc.train)), dev)
            loss = loss + perceptual(render(kids, cp, Kp, gt.shape[1], gt.shape[0]), gt, step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f} slot {ls.item():.4f} {time.time() - t0:.0f}s", flush=True)

    results = {}
    for i, sc in enumerate(scenes):
        tag = "train" if i in train_ids else "holdout"
        results[sc.name if tag == "train" else f"{sc.name}_zeroshot"] = evaluate(grow, sc, lpips, dev, args.out, tag)
        torch.cuda.empty_cache()
    torch.save({"restorer": model.state_dict(), "encoder": enc.state_dict()}, f"{args.out}/lightseed_v7.pt")
    json.dump(results, open(f"{args.out}/results.json", "w"), indent=1)
    print("RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
