# Mip-NeRF 360 (9 scenes) + extra outdoor scenes (flowers, treehill)
set -e
. jobs/lib.inc
mkdir -p $W/data/mip360 && cd $W/data/mip360
for f in 360_v2.zip 360_extra_scenes.zip; do
  [ -e $f.ok ] && continue
  aria2c -q -x 16 -s 16 https://storage.googleapis.com/gresearch/refraw360/$f
  unzip -q -o $f && rm $f && touch $f.ok
done
ls; du -sh .
