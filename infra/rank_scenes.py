"""Rank DL3DV scenes by how much their thumbnail looks like our target domain (outdoor park, natural light).
usage: python infra/rank_scenes.py <thumb_dir> <out_csv>"""
import csv
import glob
import os
import sys

import open_clip
import torch
from PIL import Image

POS = ["a photo of an outdoor park with trees and grass in natural daylight",
       "a photo of a garden with plants, a path and trees outdoors",
       "a photo of a quiet residential park with pine trees, rocks and a paved walkway",
       "a photo of a city park next to apartment buildings on a cloudy day"]
NEG = ["a photo of an indoor room", "a photo of the inside of a building", "a photo taken at night",
       "a photo of a store interior", "a blurry dark photo", "a photo of a parking garage"]

model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
tok = open_clip.get_tokenizer("ViT-L-14")
model = model.cuda().eval()
with torch.no_grad():
    t = model.encode_text(tok(POS + NEG).cuda())
    t = t / t.norm(dim=-1, keepdim=True)

files = sorted(glob.glob(f"{sys.argv[1]}/*.jpg"))
rows = []
with torch.no_grad():
    for i in range(0, len(files), 128):
        batch = files[i:i + 128]
        x = torch.stack([pre(Image.open(f).convert("RGB")) for f in batch]).cuda()
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        p = (100 * f @ t.T).softmax(-1)
        for fn, pr in zip(batch, p):
            rows.append((os.path.splitext(os.path.basename(fn))[0], float(pr[:len(POS)].sum())))

rows.sort(key=lambda r: -r[1])
with open(sys.argv[2], "w") as fh:
    w = csv.writer(fh)
    w.writerow(["hash", "park_score"])
    w.writerows(rows)
print("scored", len(rows), "| >0.9:", sum(r[1] > 0.9 for r in rows), "| >0.7:", sum(r[1] > 0.7 for r in rows))
