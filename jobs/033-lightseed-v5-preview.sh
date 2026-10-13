# v5 quick preview: 8k steps, eval garden+bicycle only (user needs results fast); the full 30k run follows in 034.
. jobs/lib.inc
pkill -f "lightseed.train_v5" || true
sleep 20
gpu_lock
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$W/results
python -m lightseed.train_v5 --data $W/data/mip360 --holdout bicycle --out $T/lightseed_v5q --steps 8000 --eval_scenes bicycle garden \
  --teachers bicycle=$T/bicycle_baseline/ckpts/ckpt_29999_rank0.pt garden=$T/teacher_garden/ckpts/ckpt_29999_rank0.pt \
  stump=$T/teacher_stump/ckpts/ckpt_29999_rank0.pt flowers=$T/teacher_flowers/ckpts/ckpt_29999_rank0.pt \
  treehill=$T/teacher_treehill/ckpts/ckpt_29999_rank0.pt > $W/logs/033-train.log 2>&1 || { tail -30 $W/logs/033-train.log; exit 1; }
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/lightseed_v5q/$s/renders $tag $W/cmp/v5q_$s 3 > /dev/null
  python infra/emit.py $W/cmp/v5q_$s/v5q_${s}_val_step${tag}_0016_zoom.jpg
  python infra/realness.py gt=$T/lightseed_v5q/$s/renders/val_step${tag}_0016.png:left v5q_$s=$T/lightseed_v5q/$s/renders/val_step${tag}_0016.png:right
done
grep -E 'params|step|EVAL' $W/logs/033-train.log | awk 'NR<=2 || NR%4==1 || /EVAL/'
