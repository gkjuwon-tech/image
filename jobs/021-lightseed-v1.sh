# Developer v1: per-splat latent DNA + neighbour context + LPIPS. Same protocol as 018 (bicycle held out).
set -e
. jobs/lib.inc
wait_for 016-teachers; gpu_lock
T=$W/results
python -m lightseed.train_v1 --data $W/data/mip360 --holdout bicycle --out $T/lightseed_v1 \
  --teachers bicycle=$T/bicycle_baseline/ckpts/ckpt_29999_rank0.pt garden=$T/teacher_garden/ckpts/ckpt_29999_rank0.pt \
  stump=$T/teacher_stump/ckpts/ckpt_29999_rank0.pt flowers=$T/teacher_flowers/ckpts/ckpt_29999_rank0.pt \
  treehill=$T/teacher_treehill/ckpts/ckpt_29999_rank0.pt > $W/logs/021-train.log 2>&1 || { tail -40 $W/logs/021-train.log; exit 1; }
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/lightseed_v1/$s/renders $tag $W/cmp/ls1_$s 3 > /dev/null
  python infra/emit.py $W/cmp/ls1_$s/ls1_${s}_val_step${tag}_0016_zoom.jpg
  python infra/realness.py gt=$T/lightseed_v1/$s/renders/val_step${tag}_0016.png:left ls1_$s=$T/lightseed_v1/$s/renders/val_step${tag}_0016.png:right
done
grep -E 'developer params|EVAL|step (0|10000|20000|29000) ' $W/logs/021-train.log
