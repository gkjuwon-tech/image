# Day 2 factory (v2: prep failures now stop that scene): top CLIP-ranked DL3DV parks (960P) -> COLMAP points -> teacher reconstruction, one by one
. jobs/lib.inc
echo "HF_TOKEN set=$([ -n "$HF_TOKEN" ] && echo yes || echo no)"
[ -n "$HF_TOKEN" ] || exit 1
wait_for 003-diag-token; wait_for 001-setup
cd $W && [ -d dl3dv-meta ] || git clone -q --depth 1 https://github.com/DL3DV-10K/Dataset.git dl3dv-meta
cd $W/repo
for H in $(cat jobs-factory/scenes.txt); do
  [ -e $W/results/park_${H:0:8}/stats/val_step29999.json ] && continue
  B=$(grep "^$H," $W/dl3dv-meta/cache/DL3DV-valid.csv | cut -d, -f3)
  t0=$(date +%s)
  bash infra/dl3dv_scene.sh $H $B > $W/logs/prep-${H:0:8}.log 2>&1 || { echo "PARK ${H:0:8} prep failed: $(tail -2 $W/logs/prep-${H:0:8}.log)"; continue; }
  tail -1 $W/logs/prep-${H:0:8}.log
  train park-${H:0:8} --data_dir $W/data/dl3dv/$H --data_factor 1 --result_dir $W/results/park_${H:0:8} \
    --antialiased --pose_opt --strategy.cap-max 3000000 --eval_steps 30000 --save_steps 30000 \
    || { echo "PARK ${H:0:8} train failed: $(tail -3 $W/logs/park-${H:0:8}-train.log)"; continue; }
  echo "PARK ${H:0:8} batch=$B $(( $(date +%s) - t0 ))s $(cat $W/results/park_${H:0:8}/stats/val_step29999.json)"
done
