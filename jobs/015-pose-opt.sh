# Day 1 / B: refine camera poses during training (sub-pixel pose error smears fine detail)
set -e
. jobs/lib.inc
wait_for 014-hq-nobilagrid; gpu_lock
train 015 --data_dir $W/data/mip360/bicycle --data_factor 4 --result_dir $W/results/bicycle_pose \
  --antialiased --strategy.cap-max 6000000 --steps_scaler 2 --pose_opt --eval_steps 30000 --save_steps 30000
report bicycle_pose $W/results/bicycle_pose 59999
