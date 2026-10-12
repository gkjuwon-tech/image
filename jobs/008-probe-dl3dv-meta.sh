# check the instance secret exists (for encrypted HF token delivery) and inspect DL3DV metadata
. jobs/lib.inc
for v in CONTAINER_API_KEY VAST_CONTAINERLABEL CONTAINER_ID; do
  val=$(eval echo \$$v); echo "$v set=$([ -n "$val" ] && echo yes || echo no) sha=$(printf '%s' "$val" | sha256sum | cut -c1-12)"
done
cd $W && rm -rf dl3dv-meta && git clone -q --depth 1 https://github.com/DL3DV-10K/Dataset.git dl3dv-meta
cd dl3dv-meta && find . -path ./.git -prune -o -type f -print | head -50
for f in $(find . -name '*.csv' -o -name '*.json' | grep -v .git | head -10); do echo "== $f $(wc -l < $f) lines"; head -3 $f | cut -c1-400; done
