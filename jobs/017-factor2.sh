# Day 1 / A: train at 2x resolution (images_2) so the model sees the grass detail the 4x-downsampled photos lose
set -e
. jobs/lib.inc
wait_for 016-teachers; gpu_lock
train 017 --data_dir $W/data/mip360/bicycle --data_factor 2 --result_dir $W/results/bicycle_f2 \
  --antialiased --strategy.cap-max 6000000 --steps_scaler 2 --eval_steps 30000 --save_steps 30000
report bicycle_f2 $W/results/bicycle_f2 59999
