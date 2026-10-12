"""'AI-look meter': color/texture statistics that separate real photos from generated ones.
usage: python infra/realness.py name=path.png[:left|:right] ...  (left/right = half of a [GT|render] sheet)"""
import sys
import numpy as np
from PIL import Image


def load(spec):
    path, _, half = spec.partition(":")
    im = Image.open(path).convert("RGB")
    if half:
        w = im.width // 2
        im = im.crop((0, 0, w, im.height)) if half == "left" else im.crop((w, 0, im.width, im.height))
    return im


def stats(im):
    hsv = np.asarray(im.convert("HSV")).astype(float) / 255
    a = np.asarray(im).astype(float)
    s, h, v = hsv[..., 1], hsv[..., 0], hsv[..., 2]
    green = (h > 0.15) & (h < 0.45) & (s > 0.15)
    lum = a.mean(-1)
    sh = a[lum < np.percentile(lum, 10)].mean(0)
    hp = lum - (np.roll(lum, 1, 0) + np.roll(lum, -1, 0) + np.roll(lum, 1, 1) + np.roll(lum, -1, 1)) / 4
    return dict(sat=s.mean(), sat90=np.percentile(s, 90), gray=(s < 0.12).mean() * 100,
                green_hue_std=h[green].std() * 360, shadow_b_minus_g=sh[2] - sh[1], highpass=hp.std())


for arg in sys.argv[1:]:
    name, _, spec = arg.partition("=")
    print(f"REAL {name:10s} " + " ".join(f"{k}={v:.2f}" for k, v in stats(load(spec)).items()))
