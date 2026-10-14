# Audit of the claimed garden 29.97 dB / LPIPS 0.024 (gsplat default strategy, factor 4, eval at step 6999).
# A) exact repro: default strategy, max 7000 steps. B) same, 30000 steps. Each checkpoint is scored by OUR scorer on
# held-out views (every 8th, the standard split) AND on training views.
set -e
. jobs/lib.inc
wait_for 001-setup; wait_for 002-data
G=$W/data/mip360/garden
cd $W/gsplat/examples
python simple_trainer.py default --disable_viewer --data_dir $G --data_factor 4 --result_dir $W/audit/g7k \
  --max_steps 7000 --eval_steps 7000 --save_steps 7000 > $W/logs/audit-g7k.log 2>&1
grep -E "PSNR:" $W/logs/audit-g7k.log | tail -2
python simple_trainer.py default --disable_viewer --data_dir $G --data_factor 4 --result_dir $W/audit/g30k \
  --max_steps 30000 --eval_steps 7000 30000 --save_steps 7000 30000 > $W/logs/audit-g30k.log 2>&1
grep -E "PSNR:" $W/logs/audit-g30k.log | tail -3
cd $W/repo
for c in $W/audit/g7k/ckpts/ckpt_6999_rank0.pt $W/audit/g30k/ckpts/ckpt_29999_rank0.pt; do
  for s in val train; do python infra/eval_ckpt.py $G 4 $c $s 2>/dev/null | grep SCORE; done
done
