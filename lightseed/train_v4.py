"""Developer v4: Z-order groups of exactly 8 -> slot-wise supervision.

Why: v0-v3 all stalled at ~40-50% fine detail. Uniform voxels hold anywhere from 1 to hundreds of splats, but the
developer always grows a fixed 7-8 children, so dense grass/foliage is starved -> blur. Z-order grouping gives every
parent exactly 8 children (dense regions automatically get more, smaller parents), and since the 8 teacher splats
are ordered, each predicted child is graded directly against its own target (position, size, rotation, opacity,
colour), plus teacher renders from new zoomed cameras and real photos.
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

from .developer import DeveloperV1
from .morton import group8, merge_groups, to_parent_frame
from .train_v0 import Scene, photo_loss, render
from .train_v1 import crop
from .train_v3 import evaluate, novel_camera


def child_features(rel):
    return torch.cat([rel["pos"].clamp(-8, 8) / 8, rel["dls"].clamp(-6, 6) / 6, rel["quat"],
                      torch.sigmoid(rel["op"])[..., None], rel["sh0"], rel["shN"]], -1)


class GroupEncoder(nn.Module):
    """Ordered 8 children (in the parent's frame) -> latent."""

    def __init__(self, latent=64, d=59, width=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8 * d, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, latent))

    def forward(self, feats):
        return self.net(feats.flatten(1))


def slot_loss(pred, target):
    return ((pred["pos"] - target["pos"]).clamp(-8, 8).abs().mean()
            + 0.5 * (pred["dls"] - target["dls"]).clamp(-6, 6).abs().mean()
            + 0.2 * (1 - (pred["quat"] * target["quat"]).sum(-1).abs()).mean()
            + (torch.sigmoid(pred["op"]) - torch.sigmoid(target["op"])).abs().mean()
            + (pred["sh0"] - target["sh0"]).abs().mean()
            + 0.3 * (pred["shN"] - target["shN"]).abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--slot_w", type=float, default=1.0)
    ap.add_argument("--photo_every", type=int, default=4)
    ap.add_argument("--lpips_w", type=float, default=0.1)
    ap.add_argument("--lpips_start", type=int, default=2000)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)

    model = DeveloperV1(latent=args.latent, levels=1, k=8).to(dev)
    enc = GroupEncoder(latent=args.latent).to(dev)
    scenes = []
    for t in args.teachers:
        name, ckpt = t.split("=")
        sc = Scene(name, f"{args.data}/{name}", ckpt, 64, dev)  # ratio unused below; Scene loads photos + teacher
        sc.teacher = group8(sc.teacher)
        sc.coarse = merge_groups(sc.teacher)
        sc.target_feats = child_features(to_parent_frame(sc.teacher, sc.coarse)).cpu().pin_memory()
        sc.target = {k: v.cpu().pin_memory() for k, v in to_parent_frame(sc.teacher, sc.coarse).items()}
        sc.nbr = model.knn(sc.coarse["means"], model.n_nbr)
        sc.teacher = {k: v.cpu().pin_memory() for k, v in sc.teacher.items()}
        print(f"[{name}] {len(sc.coarse['means'])} parents x 8", flush=True)
        scenes.append(sc)
        torch.cuda.empty_cache()
    train_ids = [i for i, s in enumerate(scenes) if s.name != args.holdout]
    print(f"params {sum(p.numel() for m in (model, enc) for p in m.parameters()) / 1e6:.2f}M | latent {args.latent}", flush=True)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)
    params = list(model.parameters()) + list(enc.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1))))

    def grow_with_rel(sc):
        lat = checkpoint(enc, sc.target_feats.to(dev, non_blocking=True), use_reentrant=False)
        kids, _, _ = checkpoint(model, sc.coarse, lat, 0, sc.nbr, use_reentrant=False)
        return kids, to_parent_frame(kids, sc.coarse)

    def grow(sc):
        return grow_with_rel(sc)[0]

    def perceptual(out, ref, step):
        loss = photo_loss(out, ref)
        if args.lpips_w > 0 and step >= args.lpips_start:
            a, b = crop(out.clamp(0, 1), ref, min(512, out.shape[0], out.shape[1]))
            loss = loss + args.lpips_w * lpips(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None])
        return loss

    t0 = time.time()
    for step in range(args.steps):
        sc = scenes[train_ids[step % len(train_ids)]]
        kids, rel = grow_with_rel(sc)
        target = {k: v.to(dev, non_blocking=True) for k, v in sc.target.items()}
        ls = slot_loss(rel, target)
        loss = args.slot_w * ls
        teacher = {k: v.to(dev, non_blocking=True) for k, v in sc.teacher.items()}
        ia, ib = np.random.randint(len(sc.train), size=2)
        gt_a, ca, K = sc.item(sc.train, ia, dev)
        _, cb, _ = sc.item(sc.train, ib, dev)
        c2w, K2, z = novel_camera(ca, cb, K, gt_a.shape[1], gt_a.shape[0])
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
    with torch.no_grad():
        for i, sc in enumerate(scenes):
            tag = "train" if i in train_ids else "holdout"
            results[sc.name if tag == "train" else f"{sc.name}_zeroshot"] = evaluate(grow, sc, lpips, dev, args.out, tag)
            torch.cuda.empty_cache()
    torch.save({"developer": model.state_dict(), "encoder": enc.state_dict()}, f"{args.out}/lightseed_v4.pt")
    json.dump(results, open(f"{args.out}/results.json", "w"), indent=1)
    print("RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
