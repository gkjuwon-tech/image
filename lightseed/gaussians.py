"""Gaussian splat helpers. Splats follow gsplat's simple_trainer layout:
means [N,3], scales [N,3] (log), quats [N,4] (wxyz, unnormalized), opacities [N] (logit), sh0 [N,1,3], shN [N,K,3]."""
import torch
import torch.nn.functional as F


def quat_to_rot(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(q.shape[:-1] + (3, 3))


def rot_to_quat(R):
    """Rotation matrices [N,3,3] -> unit quaternions [N,4] (wxyz)."""
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cands = torch.stack([
        torch.stack([1 + t, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], -1),
        torch.stack([m[:, 2, 1] - m[:, 1, 2], 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0]], -1),
        torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 0, 1] + m[:, 1, 0], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], m[:, 1, 2] + m[:, 2, 1]], -1),
        torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1], 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], -1),
    ], 1)  # [N,4 candidates,4]
    best = cands.norm(dim=-1).argmax(1)
    q = cands[torch.arange(len(m)), best]
    return F.normalize(q, dim=-1)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], -1)


def load_ckpt(path, device="cuda"):
    sd = torch.load(path, map_location=device)["splats"]
    return {k: v.detach().float() for k, v in sd.items()}


def merge_voxels(sp, voxel):
    """Moment-matching merge of all splats that fall in the same voxel (one level of coarsening)."""
    means, ls, q, op = sp["means"], sp["scales"], sp["quats"], sp["opacities"]
    alpha = torch.sigmoid(op)
    s = ls.exp()
    w = alpha * (s.prod(-1) ** (2 / 3)) + 1e-12  # opacity x projected-area proxy
    key = torch.floor(means / voxel).long()
    _, inv = torch.unique(key, dim=0, return_inverse=True)
    M = int(inv.max()) + 1

    def seg_sum(x):
        out = torch.zeros((M,) + x.shape[1:], device=x.device, dtype=x.dtype)
        return out.index_add_(0, inv, x)

    W = seg_sum(w)
    mu = seg_sum(w[:, None] * means) / W[:, None]
    R = quat_to_rot(q)
    cov = R @ torch.diag_embed(s * s) @ R.transpose(1, 2)
    second = cov + means[:, :, None] * means[:, None, :]
    C = seg_sum(w[:, None, None] * second) / W[:, None, None] - mu[:, :, None] * mu[:, None, :]
    C = 0.5 * (C + C.transpose(1, 2)) + torch.eye(3, device=C.device) * 1e-10
    evals, evecs = torch.linalg.eigh(C)
    evecs = evecs * torch.sign(torch.linalg.det(evecs))[:, None, None]  # proper rotation
    out = {
        "means": mu,
        "scales": evals.clamp_min(1e-12).sqrt().log(),
        "quats": rot_to_quat(evecs),
        # union opacity of the merged group, kept below 1
        "opacities": torch.logit((1 - torch.exp(seg_sum(torch.log1p(-alpha.clamp(max=0.999))))).clamp(1e-4, 0.99)),
    }
    for k in ("sh0", "shN"):
        out[k] = seg_sum(w[:, None, None] * sp[k]) / W[:, None, None]
    return out


def coarsen_to(sp, target, iters=20):
    """Binary-search a voxel size so the merged scene has about `target` splats."""
    ext = (sp["means"].quantile(0.99, 0) - sp["means"].quantile(0.01, 0)).max().item()
    lo, hi = ext * 1e-5, ext * 0.5
    best = None
    for _ in range(iters):
        v = (lo * hi) ** 0.5
        n = len(torch.unique(torch.floor(sp["means"] / v).long(), dim=0))
        best = v
        if abs(n - target) / target < 0.03:
            break
        if n > target:
            lo = v
        else:
            hi = v
    return merge_voxels(sp, best), best
