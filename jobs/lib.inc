# shared helpers for jobs
W=/workspace
export PIP_ROOT_USER_ACTION=ignore
wait_for() { until [ -e $W/done/$1.ok ]; do sleep 30; done; }
