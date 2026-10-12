"""Developer: one small shared network that splits a parent splat into K children, applied at every level.
Randomness comes from world-anchored noise (hash of the parent's 3D cell), so detail is identical from any view."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussians import quat_mul, quat_to_rot

# fixed child layout at init: cube corners at +-0.5 sigma in the parent's frame (an octree split)
CORNERS = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32) * 0.5


class WorldNoise(nn.Module):
    """Deterministic noise tied to world position: same (cell, level) -> same vector, from any camera."""

    def __init__(self, dim=16, table=1 << 20, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("table", torch.randn(table, dim, generator=g))

    def forward(self, xyz, cell, level):
        k = torch.floor(xyz / cell[:, None]).long()
        h = (k[:, 0] * 73856093) ^ (k[:, 1] * 19349663) ^ (k[:, 2] * 83492791) ^ (level * 2654435761)
        return self.table[h.remainder(len(self.table))]


class Genome(nn.Module):
    """Per-scene top-down feature grid [S, C, G, G], sampled bilinearly at each splat's (x, y)."""

    def __init__(self, n_scenes, ch=16, res=32):
        super().__init__()
        self.grid = nn.Parameter(torch.zeros(n_scenes, ch, res, res))
        self.register_buffer("lo", torch.zeros(n_scenes, 2))
        self.register_buffer("hi", torch.ones(n_scenes, 2))

    def set_bounds(self, i, xy):
        self.lo[i] = xy.quantile(0.01, 0)
        self.hi[i] = xy.quantile(0.99, 0)

    def forward(self, i, xyz):
        uv = (xyz[:, :2] - self.lo[i]) / (self.hi[i] - self.lo[i]) * 2 - 1
        f = F.grid_sample(self.grid[i:i + 1], uv.clamp(-1, 1)[None, :, None], align_corners=False)
        return f[0, :, :, 0].T


class Developer(nn.Module):
    def __init__(self, genome_ch=16, noise_dim=16, levels=4, width=512, k=8, sh_rest=15):
        super().__init__()
        self.k, self.sh_rest = k, sh_rest
        self.noise = WorldNoise(noise_dim)
        self.level_emb = nn.Embedding(levels, 16)
        d_in = 3 + 4 + 1 + 3 + genome_ch + noise_dim + 16
        self.per_child = 3 + 3 + 4 + 1 + 3 + sh_rest * 3
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, k * self.per_child),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, sp, genome_feat, level):
        """sp: parent splats (dict). Returns K children per parent as a new splat dict."""
        n, k = len(sp["means"]), self.k
        ls = sp["scales"]
        q = F.normalize(sp["quats"], dim=-1)
        cell = ls.exp().max(-1).values  # hash cell ~ parent size, so the noise is stable per parent
        z = self.noise(sp["means"], cell, level)
        lv = self.level_emb(torch.full((n,), level, device=ls.device, dtype=torch.long))
        x = torch.cat([ls, q, sp["opacities"][:, None], sp["sh0"][:, 0], genome_feat, z, lv], -1)
        o = self.net(x).view(n, k, self.per_child)
        off, dls, dq, dop, dsh0, dshN = o.split([3, 3, 4, 1, 3, self.sh_rest * 3], -1)

        local = CORNERS.to(o.device)[None] + off  # [n,k,3] in units of parent sigma
        R = quat_to_rot(q)
        means = sp["means"][:, None] + torch.einsum("nij,nkj->nki", R, local * ls.exp()[:, None])
        scales = ls[:, None] - math.log(2) + dls
        dq = dq + torch.tensor([1.0, 0, 0, 0], device=o.device)
        quats = quat_mul(q[:, None].expand(n, k, 4), F.normalize(dq, dim=-1))
        return {
            "means": means.reshape(-1, 3),
            "scales": scales.reshape(-1, 3),
            "quats": quats.reshape(-1, 4),
            "opacities": (sp["opacities"][:, None] + dop[..., 0]).reshape(-1),
            "sh0": (sp["sh0"][:, None] + dsh0[:, :, None]).reshape(-1, 1, 3),
            "shN": (sp["shN"][:, None] + dshN.view(n, k, self.sh_rest, 3)).reshape(-1, self.sh_rest, 3),
        }


class DeveloperV1(nn.Module):
    """v1: every splat carries a latent "DNA" that it passes (modified) to its children, and the splitter
    looks at its k nearest neighbours. Fixes v0's two bottlenecks: too little information, no context."""

    def __init__(self, latent=32, noise_dim=16, levels=4, width=512, k=8, sh_rest=15, n_nbr=8):
        super().__init__()
        self.k, self.sh_rest, self.latent, self.n_nbr = k, sh_rest, latent, n_nbr
        self.noise = WorldNoise(noise_dim)
        self.level_emb = nn.Embedding(levels, 16)
        d_self = 3 + 4 + 1 + 3 + latent
        self.nbr = nn.Sequential(nn.Linear(3 + 3 + latent, 128), nn.SiLU(), nn.Linear(128, 128))
        self.per_child = 3 + 3 + 4 + 1 + 3 + sh_rest * 3 + latent
        self.norm = nn.LayerNorm(d_self + 128 + noise_dim + 16)
        self.net = nn.Sequential(
            nn.Linear(d_self + 128 + noise_dim + 16, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, k * self.per_child),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def knn(xyz, n, chunk=8192):
        out = []
        for i in range(0, len(xyz), chunk):
            d = torch.cdist(xyz[i:i + chunk], xyz)
            out.append(d.topk(n + 1, largest=False).indices[:, 1:])
        return torch.cat(out)

    def forward(self, sp, lat, level, nbr_idx=None):
        n, k = len(sp["means"]), self.k
        ls = sp["scales"]
        q = F.normalize(sp["quats"], dim=-1)
        size = ls.exp().max(-1).values
        if nbr_idx is None:
            nbr_idx = self.knn(sp["means"].detach(), self.n_nbr)
        rel = ((sp["means"][nbr_idx] - sp["means"][:, None]) / size[:, None, None]).clamp(-8, 8) / 8
        nf = torch.cat([rel, sp["sh0"][nbr_idx][:, :, 0], lat[nbr_idx]], -1)
        ctx = self.nbr(nf).max(1).values
        z = self.noise(sp["means"], size, level)
        lv = self.level_emb(torch.full((n,), level, device=ls.device, dtype=torch.long))
        x = torch.cat([ls, q, sp["opacities"][:, None], sp["sh0"][:, 0], lat, ctx, z, lv], -1)
        o = self.net(self.norm(x)).view(n, k, self.per_child)
        off, dls, dq, dop, dsh0, dshN, dlat = o.split([3, 3, 4, 1, 3, self.sh_rest * 3, self.latent], -1)
        # bounded edits: a child can't fly away, explode or vanish in one step (v1 collapsed without this)
        off, dls, dop = 1.5 * torch.tanh(off), 2.0 * torch.tanh(dls), 4.0 * torch.tanh(dop)

        local = CORNERS.to(o.device)[None] + off
        R = quat_to_rot(q)
        means = sp["means"][:, None] + torch.einsum("nij,nkj->nki", R, local * ls.exp()[:, None])
        dq = dq + torch.tensor([1.0, 0, 0, 0], device=o.device)
        quats = quat_mul(q[:, None].expand(n, k, 4), F.normalize(dq, dim=-1))
        children = {
            "means": means.reshape(-1, 3),
            "scales": (ls[:, None] - math.log(2) + dls).reshape(-1, 3),
            "quats": quats.reshape(-1, 4),
            "opacities": (sp["opacities"][:, None] + dop[..., 0]).reshape(-1),
            "sh0": (sp["sh0"][:, None] + dsh0[:, :, None]).reshape(-1, 1, 3),
            "shN": (sp["shN"][:, None] + dshN.view(n, k, self.sh_rest, 3)).reshape(-1, self.sh_rest, 3),
        }
        child_lat = (lat[:, None] + dlat).reshape(-1, self.latent)
        # children's neighbours: their siblings (cheap, no kNN over millions of points)
        sib = torch.arange(n * k, device=o.device).view(n, k)
        child_nbr = sib[:, None, :].expand(n, k, k).reshape(n * k, k)
        return children, child_lat, child_nbr
