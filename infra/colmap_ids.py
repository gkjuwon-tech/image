"""point_triangulator matches images by IMAGE_ID: rewrite images.txt so its ids follow the database.
usage: python infra/colmap_ids.py <db.db> <model_dir>"""
import sqlite3
import sys

db, model = sys.argv[1], sys.argv[2]
ids = dict(sqlite3.connect(db).execute("SELECT name, image_id FROM images").fetchall())
cam = sqlite3.connect(db).execute("SELECT camera_id FROM cameras").fetchone()[0]
lines = [l for l in open(f"{model}/images.txt").read().splitlines() if l.strip()]
with open(f"{model}/images.txt", "w") as f:
    for l in lines:
        p = l.split()
        f.write(" ".join([str(ids[p[9]])] + p[1:8] + [str(cam), p[9]]) + "\n\n")
cl = open(f"{model}/cameras.txt").read().split()
open(f"{model}/cameras.txt", "w").write(" ".join([str(cam)] + cl[1:]) + "\n")
