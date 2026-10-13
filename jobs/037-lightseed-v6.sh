# v6: frozen v4 regressor = skeleton (mean children); flow samples only the residual detail (world-anchored noise).
. jobs/lib.inc
gpu_lock
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$W/results
python -m lightseed.train_v6 --data $W/data/mip360 --holdout bicycle --out $T/lightseed_v6 --steps 30000 \
  --v4 $T/lightseed_v4/lightseed_v4.pt \
  --teachers bicycle=$T/bicycle_baseline/ckpts/ckpt_29999_rank0.pt garden=$T/teacher_garden/ckpts/ckpt_29999_rank0.pt \
  stump=$T/teacher_stump/ckpts/ckpt_29999_rank0.pt flowers=$T/teacher_flowers/ckpts/ckpt_29999_rank0.pt \
  treehill=$T/teacher_treehill/ckpts/ckpt_29999_rank0.pt > $W/logs/037-train.log 2>&1 || { tail -30 $W/logs/037-train.log; exit 1; }
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/lightseed_v6/$s/renders $tag $W/cmp/v6_$s 3 > /dev/null
  python infra/emit.py $W/cmp/v6_$s/v6_${s}_val_step${tag}_0016_zoom.jpg
  python infra/realness.py gt=$T/lightseed_v6/$s/renders/val_step${tag}_0016.png:left v6_$s=$T/lightseed_v6/$s/renders/val_step${tag}_0016.png:right
done
grep -E 'groups|params|step|EVAL' $W/logs/037-train.log | awk 'NR<=6 || NR%4==1 || /EVAL/'
