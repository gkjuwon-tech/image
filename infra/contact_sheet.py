"""Grid of the top-ranked scene thumbnails. usage: python infra/contact_sheet.py <thumb_dir> <ranked_csv> <out_jpg> [n]"""
import csv
import sys

from PIL import Image, ImageDraw

d, rc, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 48
rows = list(csv.DictReader(open(rc)))[:n]
W, H, C = 320, 180, 8
sheet = Image.new("RGB", (W * C, H * ((len(rows) + C - 1) // C)), "black")
for i, r in enumerate(rows):
    im = Image.open(f"{d}/{r['hash']}.jpg").convert("RGB").resize((W, H))
    ImageDraw.Draw(im).text((4, 4), f"{i} {r['hash'][:6]} {float(r['park_score']):.2f}", fill="yellow")
    sheet.paste(im, ((i % C) * W, (i // C) * H))
sheet.save(out, quality=85)
