# baseline: gsplat MCMC on bicycle at the standard benchmark resolution (images_4)
set -e
. jobs/lib.inc
wait_for 001-setup; wait_for 002-data-mip360
cd $W/gsplat/examples
python simple_trainer.py mcmc --disable_viewer --data_dir $W/data/mip360/bicycle --data_factor 4 \
  --result_dir $W/results/bicycle_baseline --antialiased --strategy.cap-max 3000000 \
  --eval_steps 7000 30000 --save_steps 30000 2>&1 | grep -v -E 'it/s|%\|' | tail -40
cat $W/results/bicycle_baseline/stats/val_step29999.json
