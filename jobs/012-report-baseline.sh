# re-print baseline metrics (they scrolled out of the log window) and re-send the missing sheet
. jobs/lib.inc
python infra/emit.py $W/cmp/bicycle_baseline/val_step29999_0008_zoom.jpg $W/cmp/bicycle_baseline/val_step29999_0008_full.jpg $W/cmp/bicycle_baseline/val_step29999_0000_full.jpg $W/cmp/bicycle_baseline/val_step29999_0000_zoom.jpg
ls $W/cmp/bicycle_baseline
for s in 6999 29999; do echo "val_step$s: $(cat $W/results/bicycle_baseline/stats/val_step$s.json)"; done
cat $W/results/bicycle_baseline/stats/train_step29999_rank0.json 2>/dev/null
grep -E 'worst-err' $W/logs/007-compare-bicycle.log
