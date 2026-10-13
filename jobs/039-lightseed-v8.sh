# v8: v7 restorer fine-tuned with a patch discriminator on zoomed renders (screen-space sharpness).
. jobs/lib.inc
until [ -e $W/results/lightseed_v7/lightseed_v7.pt ] || grep -q "Error\|error" $W/logs/038-train.log 2>/dev/null; do sleep 60; done
[ -e $W/results/lightseed_v7/lightseed_v7.pt ] || { echo "v7 failed, no init"; exit 1; }
gpu_lock
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$W/results
python -m lightseed.train_v8 --data $W/data/mip360 --holdout bicycle --init $T/lightseed_v7/lightseed_v7.pt --out $T/lightseed_v8 --steps 6000 \
  --teachers bicycle=$T/bicycle_baseline/ckpts/ckpt_29999_rank0.pt garden=$T/teacher_garden/ckpts/ckpt_29999_rank0.pt \
  stump=$T/teacher_stump/ckpts/ckpt_29999_rank0.pt flowers=$T/teacher_flowers/ckpts/ckpt_29999_rank0.pt \
  treehill=$T/teacher_treehill/ckpts/ckpt_29999_rank0.pt > $W/logs/039-train.log 2>&1 || { tail -30 $W/logs/039-train.log; exit 1; }
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/lightseed_v8/$s/renders $tag $W/cmp/v8_$s 3 > /dev/null
  python infra/emit.py $W/cmp/v8_$s/v8_${s}_val_step${tag}_0016_zoom.jpg
  python infra/realness.py gt=$T/lightseed_v8/$s/renders/val_step${tag}_0016.png:left v8_$s=$T/lightseed_v8/$s/renders/val_step${tag}_0016.png:right
done
grep -E 'params|G |EVAL' $W/logs/039-train.log | awk 'NR<=2 || NR%3==1 || /EVAL/'
