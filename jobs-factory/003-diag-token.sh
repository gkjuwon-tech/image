# why did HF return 401? check the token's shape (never print it) and ask HF who it belongs to
v="$HF_TOKEN"
echo "len=${#v} prefix_ok=$([[ $v == hf_* ]] && echo yes || echo no) has_space=$([[ $v =~ [[:space:]] ]] && echo yes || echo no) has_quote=$([[ $v == *\"* || $v == *\'* ]] && echo yes || echo no)"
code=$(curl -s -o /tmp/who.json -w '%{http_code}' -H "Authorization: Bearer $v" https://huggingface.co/api/whoami-v2)
echo "whoami http=$code role=$(python -c "import json;d=json.load(open('/tmp/who.json'));print(d.get('auth',{}).get('accessToken',{}).get('role'), d.get('error'))" 2>/dev/null)"
for r in DL3DV-ALL-960P DL3DV-ALL-4K; do
  echo "$r: $(curl -s -o /dev/null -w '%{http_code}' -L -r 0-9 -H "Authorization: Bearer $v" https://huggingface.co/datasets/DL3DV/$r/resolve/main/1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f.zip)"
done
