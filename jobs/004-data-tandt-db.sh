# Tanks&Temples (truck, train) + Deep Blending, the other standard 3DGS benchmark
set -e
. jobs/lib.inc
wait_for 002-data-mip360
mkdir -p $W/data && cd $W/data
aria2c -q -x 16 -s 16 https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip -q -o tandt_db.zip && rm tandt_db.zip
ls tandt db; du -sh tandt db
