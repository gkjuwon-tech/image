# curriculum check: one split only (8x coarsening, 1 level). If the developer can't recover most of an 8x
# gap, depth isn't the problem; if it can, we grow depth step by step.
set -e
. jobs/lib.inc
wait_for 025-lightseed-v1_1; gpu_lock
T=$W/results
python -m lightseed.train_v1 --data $W/data/mip360 --holdout bicycle --out $T/lightseed_c8 --steps 8000 --holdout_steps 1500 \
  --ratio 8 --levels 1 \
  --teachers bicycle=$T/bicycle_baseline/ckpts/ckpt_29999_rank0.pt garden=$T/teacher_garden/ckpts/ckpt_29999_rank0.pt \
  stump=$T/teacher_stump/ckpts/ckpt_29999_rank0.pt flowers=$T/teacher_flowers/ckpts/ckpt_29999_rank0.pt \
  treehill=$T/teacher_treehill/ckpts/ckpt_29999_rank0.pt > $W/logs/026-train.log 2>&1 || { tail -30 $W/logs/026-train.log; exit 1; }
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/lightseed_c8/$s/renders $tag $W/cmp/c8_$s 3 > /dev/null
  python infra/emit.py $W/cmp/c8_$s/c8_${s}_val_step${tag}_0016_zoom.jpg
  python infra/realness.py gt=$T/lightseed_c8/$s/renders/val_step${tag}_0016.png:left c8_$s=$T/lightseed_c8/$s/renders/val_step${tag}_0016.png:right
done
grep -E 'step|EVAL' $W/logs/026-train.log | awk 'NR%8==1 || /EVAL/'
