# quality push on bicycle: 6M gaussians, 2x schedule, antialiasing, bilateral grid (per-photo exposure/WB absorption)
set -e
. jobs/lib.inc
wait_for 003-baseline-bicycle
cd $W/gsplat/examples
python simple_trainer.py mcmc --disable_viewer --data_dir $W/data/mip360/bicycle --data_factor 4 \
  --result_dir $W/results/bicycle_hq --antialiased --strategy.cap-max 6000000 --steps_scaler 2 \
  --use_bilateral_grid --eval_steps 30000 --save_steps 30000 2>&1 | grep -v -E 'it/s|%\|' | tail -30
cat $W/results/bicycle_hq/stats/val_step59999.json
cd $W/repo
python infra/compare.py $W/results/bicycle_hq/renders 59999 $W/cmp/bicycle_hq 3
python infra/emit.py $W/cmp/bicycle_hq/*.jpg
