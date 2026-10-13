"""Z-order (Morton) grouping: line splats up along a space-filling curve and cut every 8, so every group has exactly
8 members. Dense regions get many small groups, sparse regions few large ones (density-adaptive by construction)."""
import torch

from .gaussians import quat_to_rot, rot_to_quat
import torch.nn.functional as F


def _spread(v):
    v = v & 0x1FFFFF
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


def morton_order(xyz):
    lo, hi = xyz.min(0).values, xyz.max(0).values
    q = ((xyz - lo) / (hi - lo).clamp_min(1e-9) * (2 ** 21 - 1)).long()
    code = _spread(q[:, 0]) | (_spread(q[:, 1]) << 1) | (_spread(q[:, 2]) << 2)
    return torch.argsort(code)


def group8(sp, k=8):
    """Sort splats in Z-order and drop the tail so the count divides by k. Returns the reordered splats [G*k]."""
    order = morton_order(sp["means"])
    n = (len(order) // k) * k
    return {key: v[order[:n]] for key, v in sp.items()}


def merge_groups(sp, k=8):
    """Moment-matching merge of each consecutive group of k splats (input already grouped by group8)."""
    G = len(sp["means"]) // k
    alpha = torch.sigmoid(sp["opacities"]).view(G, k)
    s = sp["scales"].exp()
    w = (alpha * (s.prod(-1) ** (2 / 3)).view(G, k)) + 1e-12
    wn = w / w.sum(1, keepdim=True)
    m = sp["means"].view(G, k, 3)
    mu = (wn[..., None] * m).sum(1)
    R = quat_to_rot(sp["quats"])
    cov = (R @ torch.diag_embed(s * s) @ R.transpose(1, 2)).view(G, k, 3, 3)
    d = m - mu[:, None]
    C = (wn[..., None, None] * (cov + d[..., :, None] * d[..., None, :])).sum(1)
    C = 0.5 * (C + C.transpose(1, 2)) + torch.eye(3, device=C.device) * 1e-10
    evals, evecs = torch.linalg.eigh(C)
    evecs = evecs * torch.sign(torch.linalg.det(evecs))[:, None, None]
    out = {
        "means": mu,
        "scales": evals.clamp_min(1e-12).sqrt().log(),
        "quats": rot_to_quat(evecs),
        "opacities": torch.logit((1 - torch.exp(torch.log1p(-alpha.clamp(max=0.999)).sum(1))).clamp(1e-4, 0.99)),
    }
    for key in ("sh0", "shN"):
        x = sp[key].view(G, k, *sp[key].shape[1:])
        out[key] = (wn[..., None, None] * x).sum(1)
    return out


def to_parent_frame(children, parent, k=8):
    """Child params expressed relative to their parent: offsets in the parent's frame / parent size, etc."""
    G = len(parent["means"])
    R = quat_to_rot(F.normalize(parent["quats"], dim=-1))
    ps = parent["scales"].exp()
    m = children["means"].view(G, k, 3) - parent["means"][:, None]
    return {
        "pos": torch.einsum("gji,gkj->gki", R, m) / ps[:, None].clamp_min(1e-9),
        "dls": children["scales"].view(G, k, 3) - parent["scales"][:, None],
        "quat": F.normalize(children["quats"], dim=-1).view(G, k, 4),
        "op": children["opacities"].view(G, k),
        "sh0": children["sh0"].view(G, k, 3),
        "shN": children["shN"].view(G, k, -1),
    }
