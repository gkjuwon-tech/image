#!/usr/bin/env bash
K=$(cat ~/.config/vastai/vast_api_key)
curl -sS -m 30 -H "Authorization: Bearer $K" "https://console.vast.ai/api/v0/instances/" | python3 -c "
import sys,json
for i in json.load(sys.stdin).get('instances',[]):
    print(i['id'], i.get('gpu_name'), i.get('actual_status'), i.get('status_msg','')[:80], 'dph', round(i.get('dph_total',0),3))"
curl -sS -m 30 -H "Authorization: Bearer $K" https://console.vast.ai/api/v0/users/current/ | python3 -c "import sys,json;print('credit', round(json.load(sys.stdin)['credit'],2))"
