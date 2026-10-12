# safety net: 015's report step uses the old image name and may fail; once its training stats exist,
# mark 015 done (so 016 starts) and re-run its report with the new naming
. jobs/lib.inc
until [ -e $W/results/bicycle_pose/stats/val_step59999.json ]; do sleep 60; done
sleep 180
touch $W/done/015-pose-opt.ok
report bicycle_pose $W/results/bicycle_pose 59999
