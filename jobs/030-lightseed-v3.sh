# v3 = (B) real representative splats as the coarse level + (A) direct teacher supervision (novel/zoomed teacher
# views + per-voxel set loss). Replaces the 029 capacity test (stopped to give v3 the GPU).
. jobs/lib.inc
pkill -f "lightseed.train_v2" || true
sleep 20
gpu_lock
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$W/results
python -m lightseed.train_v3 --data $W/data/mip360 --holdout bicycle --out $T/lightseed_v3 --steps 8000 \
  --teachers bicycle=$T/bicycle_baseline/ckpts/ckpt_29999_rank0.pt garden=$T/teacher_garden/ckpts/ckpt_29999_rank0.pt \
  stump=$T/teacher_stump/ckpts/ckpt_29999_rank0.pt flowers=$T/teacher_flowers/ckpts/ckpt_29999_rank0.pt \
  treehill=$T/teacher_treehill/ckpts/ckpt_29999_rank0.pt > $W/logs/030-train.log 2>&1 || { tail -30 $W/logs/030-train.log; exit 1; }
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/lightseed_v3/$s/renders $tag $W/cmp/v3_$s 3 > /dev/null
  python infra/emit.py $W/cmp/v3_$s/v3_${s}_val_step${tag}_0016_zoom.jpg
  python infra/realness.py gt=$T/lightseed_v3/$s/renders/val_step${tag}_0016.png:left v3_$s=$T/lightseed_v3/$s/renders/val_step${tag}_0016.png:right
done
grep -E 'params|step|EVAL' $W/logs/030-train.log | awk 'NR<=2 || NR%4==1 || /EVAL/'
