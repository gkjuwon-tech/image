# v1 collapsed in the first 1k steps (loss 0.33 -> 0.58: splats vanished). Kill it and run a short v1.1
# with bounded edits, input LayerNorm, grad clipping and LPIPS warm-up, logging opacity/scale every 250 steps.
. jobs/lib.inc
pkill -f "lightseed.train_v1" || true
sleep 20
wait_for 016-teachers; gpu_lock
T=$W/results
python -m lightseed.train_v1 --data $W/data/mip360 --holdout bicycle --out $T/lightseed_v1_1 --steps 8000 --holdout_steps 1500 \
  --teachers bicycle=$T/bicycle_baseline/ckpts/ckpt_29999_rank0.pt garden=$T/teacher_garden/ckpts/ckpt_29999_rank0.pt \
  stump=$T/teacher_stump/ckpts/ckpt_29999_rank0.pt flowers=$T/teacher_flowers/ckpts/ckpt_29999_rank0.pt \
  treehill=$T/teacher_treehill/ckpts/ckpt_29999_rank0.pt > $W/logs/025-train.log 2>&1 || { tail -30 $W/logs/025-train.log; exit 1; }
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/lightseed_v1_1/$s/renders $tag $W/cmp/ls11_$s 3 > /dev/null
  python infra/emit.py $W/cmp/ls11_$s/ls11_${s}_val_step${tag}_0016_zoom.jpg
  python infra/realness.py gt=$T/lightseed_v1_1/$s/renders/val_step${tag}_0016.png:left ls11_$s=$T/lightseed_v1_1/$s/renders/val_step${tag}_0016.png:right
done
grep -E 'step|EVAL' $W/logs/025-train.log | awk 'NR%4==1 || /EVAL/'
