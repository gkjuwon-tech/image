# retry of 002: install its own downloader, then mark 002 done so 003 unblocks
set -e
. jobs/lib.inc
command -v aria2c || (apt-get install -y -qq aria2 unzip >/dev/null 2>&1 || (sleep 120; apt-get install -y -qq aria2 unzip >/dev/null))
mkdir -p $W/data/mip360 && cd $W/data/mip360
for f in 360_v2.zip 360_extra_scenes.zip; do
  [ -e $f.ok ] && continue
  aria2c -q -x 16 -s 16 https://storage.googleapis.com/gresearch/refraw360/$f
  unzip -q -o $f && rm $f && touch $f.ok
done
ls; du -sh .
touch $W/done/002-data-mip360.ok
