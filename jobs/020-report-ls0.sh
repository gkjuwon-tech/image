# re-run the developer v0 report (018's emit step used the old file name)
. jobs/lib.inc
T=$W/results/lightseed_v0
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/$s/renders $tag $W/cmp/ls0_$s 3 > /dev/null
  python infra/emit.py $W/cmp/ls0_$s/ls0_${s}_val_step${tag}_0016_zoom.jpg
  python infra/realness.py gt=$T/$s/renders/val_step${tag}_0016.png:left ls0_$s=$T/$s/renders/val_step${tag}_0016.png:right
done
grep -E 'teacher|developer params|EVAL|step (0|5000|10000|14500) ' $W/logs/018-train.log
