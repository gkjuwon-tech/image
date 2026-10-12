# Day 3 preview: Developer v0. One shared 0.8M-param network restores 64x-coarsened scenes (4 train scenes),
# then generalization test on held-out bicycle (freeze the network, fit only a genome).
set -e
. jobs/lib.inc
wait_for 016-teachers; gpu_lock
T=$W/results
python -m lightseed.train_v0 --data $W/data/mip360 --holdout bicycle --out $T/lightseed_v0 \
  --teachers bicycle=$T/bicycle_baseline/ckpts/ckpt_29999_rank0.pt garden=$T/teacher_garden/ckpts/ckpt_29999_rank0.pt \
  stump=$T/teacher_stump/ckpts/ckpt_29999_rank0.pt flowers=$T/teacher_flowers/ckpts/ckpt_29999_rank0.pt \
  treehill=$T/teacher_treehill/ckpts/ckpt_29999_rank0.pt > $W/logs/018-train.log 2>&1 || { tail -40 $W/logs/018-train.log; exit 1; }
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/lightseed_v0/$s/renders $tag $W/cmp/ls0_$s 3 > /dev/null
  python infra/emit.py $W/cmp/ls0_$s/val_step${tag}_0016_zoom.jpg
done
grep -E 'teacher|developer params|EVAL' $W/logs/018-train.log
