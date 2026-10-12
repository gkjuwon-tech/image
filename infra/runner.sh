#!/usr/bin/env bash
# Pull-based job runner for the GPU box. No inbound ports: it polls this repo's branch
# and runs every new jobs/*.sh once, streaming output to stdout (read via the Vast logs API).
REPO=https://github.com/gkjuwon-tech/image
BRANCH=${BRANCH:-ccr-4af9c5c9-nu34xh}
W=/workspace
mkdir -p $W/done $W/logs
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git curl >/dev/null)
[ -d $W/repo ] || git clone -q -b $BRANCH $REPO $W/repo
last=""
beat=0
while true; do
  sha=$(git ls-remote $REPO refs/heads/$BRANCH 2>/dev/null | cut -f1)
  if [ -n "$sha" ] && [ "$sha" != "$last" ]; then
    git -C $W/repo fetch -q origin $BRANCH && git -C $W/repo reset -q --hard FETCH_HEAD
    last=$sha
    echo "[runner] at ${sha:0:7}"
    for j in $(ls $W/repo/jobs/*.sh 2>/dev/null | sort); do
      n=$(basename $j .sh)
      [ -e $W/done/$n ] && continue
      touch $W/done/$n
      echo "[runner] start $n"
      ( cd $W/repo && bash $j 2>&1 | tee $W/logs/$n.log | sed -u "s/^/[$n] /"; rc=${PIPESTATUS[0]}; echo "[$n] EXIT $rc"; [ $rc -eq 0 ] && touch $W/done/$n.ok ) &
    done
  fi
  beat=$((beat+1))
  if [ $((beat % 10)) -eq 0 ]; then
    echo "[beat] $(date -u +%H:%M) gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ') disk=$(df -h $W | awk 'NR==2{print $3"/"$2}')"
  fi
  sleep 45
done
