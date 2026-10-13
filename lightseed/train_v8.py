"""Developer v8: v7 window restorer, fine-tuned with a patch discriminator on zoomed renders.

v5/v6 showed that randomness in splat-parameter space becomes speckle, not texture. Here sharpness is learned on the
screen: a PatchGAN (spectral norm, hinge loss) compares the restorer's render with the teacher's render from the same
new zoomed camera, so the only way to fool it is real-looking fine texture. Starts from the v7 checkpoint.
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
from torch.nn.utils import spectral_norm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .morton import group8, merge_groups, to_parent_frame
from .train_v0 import Scene, photo_loss, render
from .train_v1 import crop
from .train_v3 import evaluate, novel_camera
from .train_v4 import GroupEncoder, child_features
from .train_v5 import features_to_splats
from .train_v7 import WindowRestorer, feat_loss
from torch.utils.checkpoint import checkpoint


class PatchD(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        def blk(i, o, s):
            return [spectral_norm(nn.Conv2d(i, o, 4, s, 1)), nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*blk(3, ch, 2), *blk(ch, ch * 2, 2), *blk(ch * 2, ch * 4, 2), *blk(ch * 4, ch * 4, 1),
                                 spectral_norm(nn.Conv2d(ch * 4, 1, 4, 1, 1)))

    def forward(self, x):  # x: [B,3,H,W] in [0,1]
        return self.net(x * 2 - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--init", required=True, help="lightseed_v7.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--slot_w", type=float, default=2.0)
    ap.add_argument("--adv_w", type=float, default=0.05)
    ap.add_argument("--adv_start", type=int, default=300)
    ap.add_argument("--lpips_w", type=float, default=0.2)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.init, map_location=dev)
    model = WindowRestorer(latent=128).to(dev)
    model.load_state_dict(ck["restorer"])
    enc = GroupEncoder(latent=128).to(dev)
    enc.load_state_dict(ck["encoder"])
    D = PatchD().to(dev)

    scenes = []
    for t in args.teachers:
        name, ckpt = t.split("=")
        sc = Scene(name, f"{args.data}/{name}", ckpt, 64, dev)
        sc.teacher = group8(sc.teacher)
        sc.coarse = merge_groups(sc.teacher)
        sc.x = child_features(to_parent_frame(sc.teacher, sc.coarse))
        sc.teacher = {k: v.cpu().pin_memory() for k, v in sc.teacher.items()}
        scenes.append(sc)
        torch.cuda.empty_cache()
    train_ids = [i for i, s in enumerate(scenes) if s.name != args.holdout]
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)
    gparams = list(model.parameters()) + list(enc.parameters())
    gopt = torch.optim.AdamW(gparams, lr=1e-4, betas=(0.5, 0.9))
    dopt = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.5, 0.9))
    print(f"G {sum(p.numel() for p in gparams) / 1e6:.2f}M  D {sum(p.numel() for p in D.parameters()) / 1e6:.2f}M", flush=True)

    def grow_feats(sc):
        lat = checkpoint(enc, sc.x, use_reentrant=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(sc.coarse, lat)
        return out.float()

    def grow(sc):
        with torch.no_grad():
            return features_to_splats(grow_feats(sc), sc.coarse)

    t0 = time.time()
    for step in range(args.steps):
        sc = scenes[train_ids[step % len(train_ids)]]
        feats = grow_feats(sc)
        kids = features_to_splats(feats, sc.coarse)
        teacher = {k: v.to(dev, non_blocking=True) for k, v in sc.teacher.items()}
        ia, ib = np.random.randint(len(sc.train), size=2)
        gt_a, ca, K = sc.item(sc.train, ia, dev)
        _, cb, _ = sc.item(sc.train, ib, dev)
        c2w, K2, _ = novel_camera(ca, cb, K, gt_a.shape[1], gt_a.shape[0])
        with torch.no_grad():
            ref = render(teacher, c2w, K2, 512, 512).clamp(0, 1)
        del teacher
        fake = render(kids, c2w, K2, 512, 512).clamp(0, 1)
        fa, ra = fake.permute(2, 0, 1)[None], ref.permute(2, 0, 1)[None]

        # discriminator: teacher render = real, restorer render = fake (hinge)
        if step >= args.adv_start // 2:
            d_loss = F.relu(1 - D(ra)).mean() + F.relu(1 + D(fa.detach())).mean()
            dopt.zero_grad(set_to_none=True)
            d_loss.backward()
            dopt.step()
        else:
            d_loss = torch.zeros(())

        a, b = crop(fake, ref, 384)
        g_loss = args.slot_w * feat_loss(feats, sc.x) + photo_loss(fake, ref) \
            + args.lpips_w * lpips(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None])
        if step % 4 == 0:
            gt, cp, Kp = sc.item(sc.train, np.random.randint(len(sc.train)), dev)
            g_loss = g_loss + photo_loss(render(kids, cp, Kp, gt.shape[1], gt.shape[0]), gt)
        if step >= args.adv_start:
            g_loss = g_loss - args.adv_w * D(fa).mean()
        gopt.zero_grad(set_to_none=True)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gparams, 1.0)
        gopt.step()
        if step % 500 == 0:
            print(f"step {step} G {g_loss.item():.4f} D {d_loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    results = {}
    for i, sc in enumerate(scenes):
        tag = "train" if i in train_ids else "holdout"
        results[sc.name if tag == "train" else f"{sc.name}_zeroshot"] = evaluate(grow, sc, lpips, dev, args.out, tag)
        torch.cuda.empty_cache()
    torch.save({"restorer": model.state_dict(), "encoder": enc.state_dict(), "D": D.state_dict()}, f"{args.out}/lightseed_v8.pt")
    json.dump(results, open(f"{args.out}/results.json", "w"), indent=1)
    print("RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
