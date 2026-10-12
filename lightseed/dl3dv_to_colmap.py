"""Turn an unzipped DL3DV scene (nerfstudio transforms.json + images_4/) into a COLMAP text model with known poses,
ready for `colmap point_triangulator` (which adds the 3D points gsplat needs for init and normalization).

usage: python -m lightseed.dl3dv_to_colmap <scene_dir>
  reads  <scene_dir>/transforms.json, <scene_dir>/images_4/*.png
  writes <scene_dir>/images -> images_4 symlink, <scene_dir>/sparse_known/{cameras,images,points3D}.txt
"""
import json
import os
import sys

import numpy as np


def rot_to_qvec(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def main(scene):
    t = json.load(open(f"{scene}/transforms.json"))
    first = sorted(os.listdir(f"{scene}/images_4"))[0]
    from PIL import Image
    w, h = Image.open(f"{scene}/images_4/{first}").size
    sx, sy = w / t["w"], h / t["h"]
    os.makedirs(f"{scene}/sparse_known", exist_ok=True)
    if not os.path.exists(f"{scene}/images"):
        os.symlink("images_4", f"{scene}/images")
    with open(f"{scene}/sparse_known/cameras.txt", "w") as f:
        f.write(f"1 OPENCV {w} {h} {t['fl_x'] * sx} {t['fl_y'] * sy} {t['cx'] * sx} {t['cy'] * sy} "
                f"{t['k1']} {t['k2']} {t['p1']} {t['p2']}\n")
    # nerfstudio stores c2w in OpenGL camera axes, in a world rotated by applied_transform (3x4)
    A = np.eye(4)
    A[:3] = np.array(t["applied_transform"])
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    with open(f"{scene}/sparse_known/images.txt", "w") as f:
        for i, fr in enumerate(sorted(t["frames"], key=lambda fr: fr["file_path"]), 1):
            c2w = np.linalg.inv(A) @ np.array(fr["transform_matrix"]) @ flip  # back to COLMAP world, OpenCV axes
            w2c = np.linalg.inv(c2w)
            q, tv = rot_to_qvec(w2c[:3, :3]), w2c[:3, 3]
            name = os.path.basename(fr["file_path"])
            f.write(f"{i} {' '.join(map(str, q))} {' '.join(map(str, tv))} 1 {name}\n\n")
    open(f"{scene}/sparse_known/points3D.txt", "w").close()
    print(f"{scene}: {len(t['frames'])} frames, {w}x{h}")


if __name__ == "__main__":
    main(sys.argv[1])
