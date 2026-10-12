#!/usr/bin/env bash
# usage: scripts/vlogs.sh <instance_id> [tail_lines] [grep_pattern]
K=$(cat ~/.config/vastai/vast_api_key)
U=$(curl -sS -m 30 -X PUT -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
  -d "{\"tail\":\"${2:-200}\"}" https://console.vast.ai/api/v0/instances/request_logs/$1/ \
  | python3 -c "import sys,json;print(json.load(sys.stdin).get('result_url',''))")
for i in $(seq 15); do
  out=$(curl -sS -m 30 -f "$U" 2>/dev/null) && break
  sleep 2
done
if [ -n "$3" ]; then echo "$out" | grep -E "$3"; else echo "$out"; fi
