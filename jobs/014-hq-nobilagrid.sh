# ablation: same as 011 but WITHOUT the bilateral grid (011 learned per-photo color fixes that held-out views can't use)
set -e
. jobs/lib.inc
cd $W/gsplat/examples
python simple_trainer.py mcmc --disable_viewer --data_dir $W/data/mip360/bicycle --data_factor 4 \
  --result_dir $W/results/bicycle_hq2 --antialiased --strategy.cap-max 6000000 --steps_scaler 2 \
  --eval_steps 30000 --save_steps 30000 > $W/logs/014-train.log 2>&1
cd $W/repo
python infra/compare.py $W/results/bicycle_hq2/renders 59999 $W/cmp/bicycle_hq2 3
python infra/emit.py $W/cmp/bicycle_hq2/val_step59999_0016_zoom.jpg $W/cmp/bicycle_hq2/val_step59999_0000_zoom.jpg
R=$W/results
python infra/realness.py gt=$R/bicycle_hq2/renders/val_step59999_0016.png:left hq2=$R/bicycle_hq2/renders/val_step59999_0016.png:right
echo "HQ2 val_step59999: $(cat $R/bicycle_hq2/stats/val_step59999.json)"
