# shared helpers for jobs
W=/workspace
export PIP_ROOT_USER_ACTION=ignore
wait_for() { until [ -e $W/done/$1.ok ]; do sleep 30; done; }
# serialize GPU jobs: hold an exclusive lock for the rest of the job
gpu_lock() { exec 9>$W/gpu.lock; flock 9; echo "[gpu] acquired $(date -u +%H:%M)"; }
# train with gsplat's simple_trainer (mcmc) and keep the full log in $W/logs/<name>-train.log
train() { local name=$1; shift; (cd $W/gsplat/examples && python simple_trainer.py mcmc --disable_viewer "$@" > $W/logs/$name-train.log 2>&1); }
# images first, numbers last so the numbers stay inside the log window
report() {
  local name=$1 dir=$2 step=$3
  python infra/compare.py $dir/renders $step $W/cmp/$name 3 > $W/cmp/$name.txt
  python infra/emit.py $W/cmp/$name/val_step${step}_0016_zoom.jpg
  python infra/realness.py gt=$dir/renders/val_step${step}_0016.png:left $name=$dir/renders/val_step${step}_0016.png:right
  cat $W/cmp/$name.txt
  echo "METRICS $name $(cat $dir/stats/val_step$step.json)"
}
