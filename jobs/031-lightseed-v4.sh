# v4: Z-order groups of exactly 8 + slot-wise supervision (+ teacher zoom views + photos). Fixes fixed-branching starvation in dense regions.
. jobs/lib.inc
true
sleep 20
gpu_lock
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$W/results
python -m lightseed.train_v4 --data $W/data/mip360 --holdout bicycle --out $T/lightseed_v4 --steps 8000 \
  --teachers bicycle=$T/bicycle_baseline/ckpts/ckpt_29999_rank0.pt garden=$T/teacher_garden/ckpts/ckpt_29999_rank0.pt \
  stump=$T/teacher_stump/ckpts/ckpt_29999_rank0.pt flowers=$T/teacher_flowers/ckpts/ckpt_29999_rank0.pt \
  treehill=$T/teacher_treehill/ckpts/ckpt_29999_rank0.pt > $W/logs/031-train.log 2>&1 || { tail -30 $W/logs/031-train.log; exit 1; }
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/lightseed_v4/$s/renders $tag $W/cmp/v4_$s 3 > /dev/null
  python infra/emit.py $W/cmp/v4_$s/v4_${s}_val_step${tag}_0016_zoom.jpg
  python infra/realness.py gt=$T/lightseed_v4/$s/renders/val_step${tag}_0016.png:left v4_$s=$T/lightseed_v4/$s/renders/val_step${tag}_0016.png:right
done
grep -E 'params|step|EVAL' $W/logs/031-train.log | awk 'NR<=2 || NR%4==1 || /EVAL/'
