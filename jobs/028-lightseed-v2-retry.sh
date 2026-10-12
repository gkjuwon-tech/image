# v2 retry (027 ran out of GPU memory): splat autoencoder (set encoder writes each voxel's latent from the teacher splats) + developer, 8x single split.
# bicycle held out and evaluated zero-shot (encoder only, no fitting).
set -e
. jobs/lib.inc
gpu_lock
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$W/results
python -m lightseed.train_v2 --data $W/data/mip360 --holdout bicycle --out $T/lightseed_v2 --steps 8000 \
  --teachers bicycle=$T/bicycle_baseline/ckpts/ckpt_29999_rank0.pt garden=$T/teacher_garden/ckpts/ckpt_29999_rank0.pt \
  stump=$T/teacher_stump/ckpts/ckpt_29999_rank0.pt flowers=$T/teacher_flowers/ckpts/ckpt_29999_rank0.pt \
  treehill=$T/teacher_treehill/ckpts/ckpt_29999_rank0.pt > $W/logs/028-train.log 2>&1 || { tail -30 $W/logs/028-train.log; exit 1; }
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/lightseed_v2/$s/renders $tag $W/cmp/v2_$s 3 > /dev/null
  python infra/emit.py $W/cmp/v2_$s/v2_${s}_val_step${tag}_0016_zoom.jpg
  python infra/realness.py gt=$T/lightseed_v2/$s/renders/val_step${tag}_0016.png:left v2_$s=$T/lightseed_v2/$s/renders/val_step${tag}_0016.png:right
done
grep -E 'params|step|EVAL' $W/logs/028-train.log | awk 'NR<=2 || NR%4==1 || /EVAL/'
