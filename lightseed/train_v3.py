"""Developer v3 = v2 + two method changes.

(B) Representative coarse level: instead of averaging each voxel into one blurry blob, keep its strongest real
    splat (opacity x area) untouched; the developer only has to place the other 7 siblings around it.
(A) Direct supervision from the teacher: besides real photos, every step renders the teacher and the student from
    a new camera (interpolated between two training cameras, randomly zoomed 1x/2x/4x into a 512px window), and a
    per-voxel set loss compares the grown siblings with the teacher's actual splats in that voxel.
"""
import argparse
import json
import math
import os
import time

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .developer import DeveloperV1, SplatEncoder
from .gaussians import quat_to_rot, rot_to_quat
from .train_v0 import Scene, photo_loss, psnr, render
from .train_v1 import crop

MAX_MEMBERS = 16


def representatives(teacher, voxel):
    """Per voxel: index of its strongest splat, voxel id of every splat, and padded member lists."""
    inv = torch.unique(torch.floor(teacher["means"] / voxel).long(), dim=0, return_inverse=True)[1]
    V = int(inv.max()) + 1
    w = torch.sigmoid(teacher["opacities"]) * teacher["scales"].exp().prod(-1) ** (2 / 3)
    best = torch.full((V,), -1.0, device=w.device).index_reduce_(0, inv, w, "amax", include_self=True)
    idx = torch.arange(len(w), device=w.device)
    cand = torch.where(w >= best[inv], idx, torch.full_like(idx, len(w)))
    rep = torch.full((V,), len(w), device=w.device).index_reduce_(0, inv, cand, "amin", include_self=True)
    # padded member table [V, MAX_MEMBERS] for the set loss (largest members first, rest dropped)
    by_w = torch.argsort(-w)
    order = by_w[torch.argsort(inv[by_w], stable=True)]  # grouped by voxel, strongest first
    inv_s = inv[order]
    start = torch.zeros(V + 1, dtype=torch.long, device=w.device)
    start[1:] = torch.cumsum(torch.bincount(inv_s, minlength=V), 0)
    rank = torch.arange(len(order), device=w.device) - start[inv_s]
    keep = rank < MAX_MEMBERS
    members = torch.full((V, MAX_MEMBERS), -1, dtype=torch.long, device=w.device)
    members[inv_s[keep], rank[keep]] = order[keep]
    return rep, inv, members


def novel_camera(c2w_a, c2w_b, K, W, H, size=512):
    """Camera between two training cameras, zoomed 1x/2x/4x into a size x size window."""
    t = torch.rand(()).item()
    qa, qb = rot_to_quat(c2w_a[None, :3, :3])[0], rot_to_quat(c2w_b[None, :3, :3])[0]
    if (qa * qb).sum() < 0:
        qb = -qb
    q = F.normalize((1 - t) * qa + t * qb, dim=0)
    c2w = torch.eye(4, device=K.device)
    c2w[:3, :3] = quat_to_rot(q[None])[0]
    c2w[:3, 3] = (1 - t) * c2w_a[:3, 3] + t * c2w_b[:3, 3]
    z = float(np.random.choice([1, 2, 4]))
    u0, v0 = np.random.uniform(0.2, 0.8) * W, np.random.uniform(0.2, 0.8) * H
    K2 = K.clone()
    K2[0, 0], K2[1, 1] = K[0, 0] * z, K[1, 1] * z
    K2[0, 2], K2[1, 2] = z * (K[0, 2] - u0) + size / 2, z * (K[1, 2] - v0) + size / 2
    return c2w, K2, z


def set_loss(children, teacher, coarse, members, k):
    """Chamfer between grown siblings (+parent) and the teacher's splats of the same voxel, in the parent frame."""
    V = len(coarse["means"])
    R = quat_to_rot(F.normalize(coarse["quats"], dim=-1))
    s = coarse["scales"].exp()

    def local(x):
        return torch.einsum("vji,vmj->vmi", R, x - coarse["means"][:, None]) / s[:, None].clamp_min(1e-6)

    def feat(sp, sel):
        return torch.cat([sp["sh0"][sel][..., 0, :], sp["scales"][sel] / 3, torch.sigmoid(sp["opacities"][sel])[..., None]], -1)

    pm = children["means"].view(V, k, 3)
    pf = torch.cat([children["sh0"][:, 0], children["scales"] / 3, torch.sigmoid(children["opacities"])[:, None]], -1).view(V, k, -1)
    valid = members >= 0
    mi = members.clamp_min(0)
    tm, tf = teacher["means"][mi], feat(teacher, mi)
    a, b = local(pm).clamp(-8, 8), local(tm).clamp(-8, 8)
    d = torch.cdist(a, b) + (~valid[:, None, :]) * 1e4  # [V,k,M]
    p2t = d.min(2)
    t2p = d.min(1)
    lp = p2t.values.mean() + (t2p.values * valid).sum() / valid.sum()
    nn_f = torch.gather(tf, 1, p2t.indices[..., None].expand(-1, -1, tf.shape[-1]))
    lf = (pf - nn_f).abs().mean()
    return lp + lf


@torch.no_grad()
def evaluate(grow, sc, lpips, dev, out_dir, tag):
    res = {"coarse": [], "developed": [], "teacher": [], "lpips_dev": [], "lpips_teacher": []}
    dsp = grow(sc)
    os.makedirs(f"{out_dir}/{sc.name}/renders", exist_ok=True)
    teacher = {k: v.to(dev) for k, v in sc.teacher.items()}
    for i in range(len(sc.test)):
        gt, c2w, K = sc.item(sc.test, i, dev)
        h, w = gt.shape[:2]
        for k, sp in (("coarse", sc.coarse), ("developed", dsp), ("teacher", teacher)):
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
    print(f"EVAL {tag} {sc.name}: " + " ".join(f"{k}={v:.3f}" for k, v in out.items()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--ratio", type=int, default=8)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--set_w", type=float, default=0.1)
    ap.add_argument("--photo_every", type=int, default=4)
    ap.add_argument("--lpips_w", type=float, default=0.1)
    ap.add_argument("--lpips_start", type=int, default=2000)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)
    K_NEW = args.ratio - 1

    model = DeveloperV1(latent=args.latent, levels=1, k=K_NEW).to(dev)
    enc = SplatEncoder(latent=args.latent).to(dev)
    scenes = []
    for t in args.teachers:
        name, ckpt = t.split("=")
        sc = Scene(name, f"{args.data}/{name}", ckpt, args.ratio, dev)
        rep, sc.inv, sc.members = representatives(sc.teacher, sc.voxel)
        sc.coarse = {k: v[rep].clone() for k, v in sc.teacher.items()}  # real splats, not averages
        sc.nbr = model.knn(sc.coarse["means"], model.n_nbr)
        sc.teacher = {k: v.cpu().pin_memory() for k, v in sc.teacher.items()}
        scenes.append(sc)
        torch.cuda.empty_cache()
    train_ids = [i for i, s in enumerate(scenes) if s.name != args.holdout]
    print(f"params {sum(p.numel() for m in (model, enc) for p in m.parameters()) / 1e6:.2f}M | "
          f"coarse splats {len(scenes[0].coarse['means'])} (real representatives) | new children per splat {K_NEW}", flush=True)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)
    params = list(model.parameters()) + list(enc.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1))))

    def grow_full(sc, teacher):
        lat = checkpoint(enc, teacher, sc.coarse, sc.inv, use_reentrant=False)
        kids, _, _ = checkpoint(model, sc.coarse, lat, 0, sc.nbr, use_reentrant=False)
        full = {k: torch.cat([sc.coarse[k], kids[k]]) for k in kids}
        return full, kids

    def grow(sc):
        teacher = {k: v.to(dev) for k, v in sc.teacher.items()}
        return grow_full(sc, teacher)[0]

    def perceptual(out, ref, step):
        loss = photo_loss(out, ref)
        if args.lpips_w > 0 and step >= args.lpips_start:
            a, b = crop(out.clamp(0, 1), ref, min(512, out.shape[0], out.shape[1]))
            loss = loss + args.lpips_w * lpips(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None])
        return loss

    t0 = time.time()
    for step in range(args.steps):
        sc = scenes[train_ids[step % len(train_ids)]]
        teacher = {k: v.to(dev, non_blocking=True) for k, v in sc.teacher.items()}
        full, kids = grow_full(sc, teacher)
        # (A1) teacher as an unlimited source of views: new camera between two training cameras, zoomed
        ia, ib = np.random.randint(len(sc.train), size=2)
        _, ca, K = sc.item(sc.train, ia, dev)
        gt_a, cb, _ = sc.item(sc.train, ib, dev)
        c2w, K2, z = novel_camera(ca, cb, K, gt_a.shape[1], gt_a.shape[0])
        with torch.no_grad():
            ref = render(teacher, c2w, K2, 512, 512).clamp(0, 1)
        loss = perceptual(render(full, c2w, K2, 512, 512), ref, step)
        # (A2) splat-level set loss against the teacher's splats in each voxel
        if args.set_w > 0:
            loss = loss + args.set_w * set_loss(kids, teacher, sc.coarse, sc.members, K_NEW)
        # real photos keep it honest
        if step % args.photo_every == 0:
            gt, c2w_p, Kp = sc.item(sc.train, np.random.randint(len(sc.train)), dev)
            loss = loss + perceptual(render(full, c2w_p, Kp, gt.shape[1], gt.shape[0]), gt, step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        del teacher
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f} zoom {z:.0f}x {time.time() - t0:.0f}s", flush=True)

    results = {}
    with torch.no_grad():
        for i, sc in enumerate(scenes):
            tag = "train" if i in train_ids else "holdout"
            results[sc.name if tag == "train" else f"{sc.name}_zeroshot"] = evaluate(grow, sc, lpips, dev, args.out, tag)
            torch.cuda.empty_cache()
    torch.save({"developer": model.state_dict(), "encoder": enc.state_dict()}, f"{args.out}/lightseed_v3.pt")
    json.dump(results, open(f"{args.out}/results.json", "w"), indent=1)
    print("RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
