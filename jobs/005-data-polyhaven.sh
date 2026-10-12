# Poly Haven outdoor HDRIs (CC0, commercial-safe): 4k .hdr + metadata, for sky/lighting models
set -e
. jobs/lib.inc
wait_for 002-data-mip360
mkdir -p $W/data/polyhaven && cd $W/data/polyhaven
curl -fsSL "https://api.polyhaven.com/assets?type=hdris&categories=outdoor" -o assets.json
python - <<'PY'
import json, urllib.request
ids = json.load(open("assets.json"))
with open("urls.txt", "w") as f:
    for i in ids:
        u = json.load(urllib.request.urlopen(f"https://api.polyhaven.com/files/{i}"))["hdri"]["4k"]["hdr"]["url"]
        f.write(u + "\n  out=" + i + "_4k.hdr\n")
print(len(ids), "hdris")
PY
aria2c -q -x 4 -j 16 -i urls.txt --auto-file-renaming=false --allow-overwrite=true
ls *.hdr | wc -l; du -sh .
