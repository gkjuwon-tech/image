# images first, numbers last (so the numbers stay inside the log window)
. jobs/lib.inc
python infra/emit.py $W/cmp/bicycle_hq/val_step59999_0008_zoom.jpg $W/cmp/bicycle_hq/val_step59999_0000_zoom.jpg
R=$W/results
python infra/realness.py gt=$R/bicycle_hq/renders/val_step59999_0016.png:left \
  baseline=$R/bicycle_baseline/renders/val_step29999_0016.png:right \
  hq=$R/bicycle_hq/renders/val_step59999_0016.png:right
grep -E 'worst-err' $W/logs/011-hq-bicycle.log
for s in 29999; do echo "BASELINE val_step$s: $(cat $R/bicycle_baseline/stats/val_step$s.json)"; done
for s in 29999 59999; do echo "HQ val_step$s: $(cat $R/bicycle_hq/stats/val_step$s.json)"; done
cat $R/bicycle_hq/stats/train_step59999_rank0.json
