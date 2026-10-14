"""Independent scorer: render a gsplat checkpoint on a chosen split and compute PSNR / SSIM / LPIPS exactly the way
gsplat's simple_trainer does (torchmetrics, data_range 1, LPIPS alex normalize=True), so train-view and held-out
scores come from the same code.
usage: python infra/eval_ckpt.py <data_dir> <factor> <ckpt.pt> <split: train|val>"""
import os
import sys

import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

sys.path.insert(0, os.environ.get("GSPLAT_EXAMPLES", "/workspace/gsplat/examples"))
from datasets.colmap import Dataset, Parser  # noqa: E402
from gsplat import rasterization  # noqa: E402

data_dir, factor, ckpt, split = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
dev = "cuda"
parser = Parser(data_dir=data_dir, factor=factor, normalize=True, test_every=8)
ds = Dataset(parser, split=split)
sp = torch.load(ckpt, map_location=dev)["splats"]
psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)
res = {"psnr": [], "ssim": [], "lpips": []}
with torch.no_grad():
    for i in range(len(ds)):
        d = ds[i]
        gt = d["image"].to(dev)[None] / 255.0
        c2w, K = d["camtoworld"].to(dev)[None], d["K"].to(dev)[None]
        h, w = gt.shape[1:3]
        img, _, _ = rasterization(sp["means"], sp["quats"], torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"]),
                                  torch.cat([sp["sh0"], sp["shN"]], 1), torch.linalg.inv(c2w), K, w, h, sh_degree=3)
        p, g = img.clamp(0, 1).permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)
        res["psnr"].append(psnr(p, g).item())
        res["ssim"].append(ssim(p, g).item())
        res["lpips"].append(lpips(p, g).item())
print(f"SCORE split={split} n={len(ds)} psnr={np.mean(res['psnr']):.3f} ssim={np.mean(res['ssim']):.4f} "
      f"lpips={np.mean(res['lpips']):.4f} ckpt={os.path.basename(os.path.dirname(os.path.dirname(ckpt)))}/{os.path.basename(ckpt)}")
