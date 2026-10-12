# Day 3 prep: teacher reconstructions of the other Mip-NeRF 360 outdoor scenes (checkpoints kept)
set -e
. jobs/lib.inc
wait_for 015-pose-opt; gpu_lock
for s in garden stump flowers treehill; do
  train 016-$s --data_dir $W/data/mip360/$s --data_factor 4 --result_dir $W/results/teacher_$s \
    --antialiased --strategy.cap-max 3000000 --eval_steps 30000 --save_steps 30000
  echo "METRICS teacher_$s $(cat $W/results/teacher_$s/stats/val_step29999.json)"
done
ls -la $W/results/teacher_*/ckpts/
