# Plan A: the last level only restores 2x (k=2) with the v7 window restorer. If this lands close to the teacher,
# generation moves to a cascade of genome painters on the coarser levels. Cancels the queued v8 (GAN) run.
. jobs/lib.inc
pkill -f "jobs/041-lightseed-v8.sh" || true
until grep -q "EXIT\|EVAL holdout" $W/logs/040-train.log 2>/dev/null || [ -e $W/results/lightseed_v7/results.json ] || ! pgrep -f "lightseed.train_v7" > /dev/null; do sleep 30; done
gpu_lock
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$W/results
python -m lightseed.train_v7 --k 2 --data $W/data/mip360 --holdout bicycle --out $T/restorer_k2 --steps 12000 \
  --teachers bicycle=$T/bicycle_baseline/ckpts/ckpt_29999_rank0.pt garden=$T/teacher_garden/ckpts/ckpt_29999_rank0.pt \
  stump=$T/teacher_stump/ckpts/ckpt_29999_rank0.pt flowers=$T/teacher_flowers/ckpts/ckpt_29999_rank0.pt \
  treehill=$T/teacher_treehill/ckpts/ckpt_29999_rank0.pt > $W/logs/042-train.log 2>&1 || { tail -30 $W/logs/042-train.log; exit 1; }
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/restorer_k2/$s/renders $tag $W/cmp/k2_$s 3 > /dev/null
  python infra/emit.py $W/cmp/k2_$s/k2_${s}_val_step${tag}_0016_zoom.jpg
  python infra/realness.py gt=$T/restorer_k2/$s/renders/val_step${tag}_0016.png:left k2_$s=$T/restorer_k2/$s/renders/val_step${tag}_0016.png:right
done
grep -E 'groups|params|step|EVAL' $W/logs/042-train.log | awk 'NR<=7 || NR%4==1 || /EVAL/'
