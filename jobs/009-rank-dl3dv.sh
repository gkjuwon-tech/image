# semantic filter: CLIP-rank all ~10k DL3DV thumbnails for "outdoor park, natural light" (no token needed)
set -e
. jobs/lib.inc
wait_for 008-probe-dl3dv-meta
pip install -q open_clip_torch >/dev/null 2>&1
mkdir -p $W/dl3dv
python infra/rank_scenes.py $W/dl3dv-meta/visualize/imgs $W/dl3dv/ranked.csv
head -41 $W/dl3dv/ranked.csv
python infra/contact_sheet.py $W/dl3dv-meta/visualize/imgs $W/dl3dv/ranked.csv $W/dl3dv/top48.jpg 48
python infra/emit.py $W/dl3dv/top48.jpg
