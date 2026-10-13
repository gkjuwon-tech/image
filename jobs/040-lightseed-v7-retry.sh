# v7 retry (first run OOM: random head made children ~400x their parent): window-transformer restorer (64 consecutive Z-order groups attend to each other), latent 128.
# Also stops the full v5 run (034): the preview already showed its failure mode; v6/v7 get the GPU instead.
. jobs/lib.inc
true
sleep 10
gpu_lock
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$W/results
python -m lightseed.train_v7 --data $W/data/mip360 --holdout bicycle --out $T/lightseed_v7 --steps 12000 \
  --teachers bicycle=$T/bicycle_baseline/ckpts/ckpt_29999_rank0.pt garden=$T/teacher_garden/ckpts/ckpt_29999_rank0.pt \
  stump=$T/teacher_stump/ckpts/ckpt_29999_rank0.pt flowers=$T/teacher_flowers/ckpts/ckpt_29999_rank0.pt \
  treehill=$T/teacher_treehill/ckpts/ckpt_29999_rank0.pt > $W/logs/040-train.log 2>&1 || { tail -30 $W/logs/040-train.log; exit 1; }
for s in bicycle garden; do
  tag=train; [ $s = bicycle ] && tag=holdout
  python infra/compare.py $T/lightseed_v7/$s/renders $tag $W/cmp/v7_$s 3 > /dev/null
  python infra/emit.py $W/cmp/v7_$s/v7_${s}_val_step${tag}_0016_zoom.jpg
  python infra/realness.py gt=$T/lightseed_v7/$s/renders/val_step${tag}_0016.png:left v7_$s=$T/lightseed_v7/$s/renders/val_step${tag}_0016.png:right
done
grep -E 'groups|params|step|EVAL' $W/logs/040-train.log | awk 'NR<=7 || NR%4==1 || /EVAL/'
