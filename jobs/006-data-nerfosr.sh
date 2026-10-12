# NeRF-OSR: outdoor scenes under multiple lighting conditions + env maps (relighting)
set -e
. jobs/lib.inc
wait_for 002-data-mip360
mkdir -p $W/data/nerfosr && cd $W/data/nerfosr
S=https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk
curl -sSI "$S/download" | grep -i -E 'content-(length|type|disposition)' || true
aria2c -q -x 8 -s 8 -o nerfosr.zip "$S/download"
ls -la; (unzip -q -o nerfosr.zip && rm nerfosr.zip) || file nerfosr.zip
du -sh .; find . -maxdepth 3 -type d | head -40
