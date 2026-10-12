"""Build real-vs-light comparison sheets from gsplat val renders (each PNG is [GT | render]).
usage: python infra/compare.py <renders_dir> <step> <out_dir> [n_views]"""
import glob, os, sys
import numpy as np
from PIL import Image

rd, step, out = sys.argv[1], sys.argv[2], sys.argv[3]
nv = int(sys.argv[4]) if len(sys.argv) > 4 else 3
os.makedirs(out, exist_ok=True)
files = sorted(glob.glob(f"{rd}/val_step{step}_*.png"))
for p in files[:: max(1, len(files) // nv)][:nv]:
    im = np.asarray(Image.open(p).convert("RGB"))
    w = im.shape[1] // 2
    gt, rn = im[:, :w], im[:, w:]
    err = np.abs(gt.astype(np.float32) - rn.astype(np.float32)).mean(-1)
    # zoom into the 256px window with the highest local error and the most textured one
    k = 256
    best, tex = None, None
    for y in range(0, gt.shape[0] - k, k // 2):
        for x in range(0, gt.shape[1] - k, k // 2):
            e = err[y:y+k, x:x+k].mean()
            t = gt[y:y+k, x:x+k].std()
            if best is None or e > best[0]: best = (e, y, x)
            if tex is None or t > tex[0]: tex = (t, y, x)
    rows = []
    for _, y, x in (best, tex):
        a = Image.fromarray(gt[y:y+k, x:x+k]).resize((k*2, k*2), Image.NEAREST)
        b = Image.fromarray(rn[y:y+k, x:x+k]).resize((k*2, k*2), Image.NEAREST)
        row = Image.new("RGB", (k*4 + 8, k*2), "white"); row.paste(a, (0, 0)); row.paste(b, (k*2 + 8, 0))
        rows.append(row)
    full = Image.new("RGB", (w*2 + 8, gt.shape[0]), "white")
    full.paste(Image.fromarray(gt), (0, 0)); full.paste(Image.fromarray(rn), (w + 8, 0))
    full.thumbnail((2400, 2400))
    stem = os.path.basename(os.path.normpath(out)) + "_" + os.path.splitext(os.path.basename(p))[0]  # unique per run
    full.save(f"{out}/{stem}_full.jpg", quality=90)
    sheet = Image.new("RGB", (rows[0].width, rows[0].height * 2 + 8), "white")
    sheet.paste(rows[0], (0, 0)); sheet.paste(rows[1], (0, rows[0].height + 8))
    sheet.save(f"{out}/{stem}_zoom.jpg", quality=92)
    print(stem, "worst-err", round(float(best[0]), 2), "mean-err", round(float(err.mean()), 2))
