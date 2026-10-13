# v5 full run (30k steps), after the quick preview.
. jobs/lib.inc
true
sleep 20
wait_for 035-lightseed-v5-preview4k; gpu_lock
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$W/results
python -m lightseed.train_v5 --data $W/data/mip360 --holdout bicycle --out $T/lightseed_v5 --steps 30000 \
  --teachers bicycle=$T/bicycle_baseline/ckpts/ckpt_29999_rank0.pt garden=$T/teacher_garden/ckpts/ckpt_29999_rank0.pt \
  stump=$T/teacher_stump/ckpts/ckpt_29999_rank0.pt flowers=$T/teacher_flowers/ckpts/ckpt_29999_rank0.pt \
  treehill=$T/teacher_treehill/ckpts/ckpt_29999_rank0.pt > $W/logs/034-train.log 2>&1 || { tail -30 $W/logs/034-train.log; exit 1; }
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/lightseed_v5/$s/renders $tag $W/cmp/v5_$s 3 > /dev/null
  python infra/emit.py $W/cmp/v5_$s/v5_${s}_val_step${tag}_0016_zoom.jpg
  python infra/realness.py gt=$T/lightseed_v5/$s/renders/val_step${tag}_0016.png:left v5_$s=$T/lightseed_v5/$s/renders/val_step${tag}_0016.png:right
done
grep -E 'params|step|EVAL' $W/logs/034-train.log | awk 'NR<=2 || NR%4==1 || /EVAL/'
