# Mip-NeRF 360 (garden is the scene under audit)
set -e
. jobs/lib.inc
command -v aria2c || (apt-get install -y -qq aria2 unzip >/dev/null 2>&1 || (sleep 120; apt-get install -y -qq aria2 unzip >/dev/null))
mkdir -p $W/data/mip360 && cd $W/data/mip360
aria2c -q -x 16 -s 16 https://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip -q -o 360_v2.zip garden/* && rm 360_v2.zip
ls garden; ls garden/images | wc -l
