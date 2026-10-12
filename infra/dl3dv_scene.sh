#!/usr/bin/env bash
# Day 2 factory for one DL3DV scene: download (960P) -> COLMAP model with known poses -> triangulate -> teacher.
# needs HF_TOKEN in the environment (register it as a Vast account env var; never commit it).
# usage: bash infra/dl3dv_scene.sh <hash> <batch e.g. 2K>
set -e
export QT_QPA_PLATFORM=offscreen  # headless COLMAP
H=$1; B=$2; W=/workspace; D=$W/data/dl3dv/$H
mkdir -p $W/data/dl3dv && cd $W/data/dl3dv
if [ ! -e $D/images_4 ]; then
  curl -fsSL -H "Authorization: Bearer $HF_TOKEN" -o $H.zip \
    "https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P/resolve/main/$B/$H.zip"
  unzip -q -o $H.zip && rm $H.zip
fi
cd $W/repo && python -m lightseed.dl3dv_to_colmap $D
cd $D && rm -rf db.db sparse && mkdir -p sparse/0
colmap feature_extractor --database_path db.db --image_path images --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV --ImageReader.camera_params "$(cut -d' ' -f5- sparse_known/cameras.txt)" \
  --SiftExtraction.use_gpu 0 > colmap.log 2>&1
colmap sequential_matcher --database_path db.db --SiftMatching.use_gpu 0 >> colmap.log 2>&1
python $W/repo/infra/colmap_ids.py db.db sparse_known
colmap point_triangulator --database_path db.db --image_path images --input_path sparse_known \
  --output_path sparse/0 >> colmap.log 2>&1
colmap model_converter --input_path sparse/0 --output_path sparse/0 --output_type TXT >> colmap.log 2>&1
echo "$H points: $(grep -vc '^#' sparse/0/points3D.txt)"
