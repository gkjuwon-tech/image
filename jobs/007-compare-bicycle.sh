# pull back real-vs-light comparison sheets for the bicycle baseline
set -e
. jobs/lib.inc
wait_for 003-baseline-bicycle
pip install -q pillow numpy >/dev/null 2>&1 || true
python infra/compare.py $W/results/bicycle_baseline/renders 29999 $W/cmp/bicycle_baseline 3
python infra/emit.py $W/cmp/bicycle_baseline/*.jpg
