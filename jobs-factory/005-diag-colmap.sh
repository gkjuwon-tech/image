# why did the first park's prep fail? show the prep log and COLMAP's own log
D=/workspace/data/dl3dv/7a2b714f55f198ec4493fb9f1be282ea91c002cc6ced688a2c3761c77bdcf4b6
tail -5 /workspace/logs/prep-7a2b714f.log
grep -v -E '^\s*$' $D/colmap.log | grep -i -E 'error|fail|abort|what|check|usage|unrecognised|unknown|option|elapsed|images' | head -20
tail -8 $D/colmap.log
colmap -h 2>&1 | head -3
ls $D $D/sparse_known | head -20; head -2 $D/sparse_known/cameras.txt
