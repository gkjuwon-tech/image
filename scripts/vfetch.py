"""Reassemble files emitted by infra/emit.py from the Vast logs.
usage: python scripts/vfetch.py <instance_id> <out_dir> [tail_lines]"""
import base64, collections, os, re, subprocess, sys
iid, out = sys.argv[1], sys.argv[2]
tail = sys.argv[3] if len(sys.argv) > 3 else "20000"
log = subprocess.run(["bash", os.path.join(os.path.dirname(__file__), "vlogs.sh"), iid, tail],
                     capture_output=True, text=True).stdout
parts = collections.defaultdict(dict)
for m in re.finditer(r"@@F (\S+) (\d+) (\d+) (\S+)", log):
    name, i, n, chunk = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    parts[(name, n)][i] = chunk  # key by chunk count so older emissions of the same file don't mix in
os.makedirs(out, exist_ok=True)
for (name, n), ch in sorted(parts.items(), key=lambda kv: kv[0][1]):  # largest emission written last wins
    if len(ch) == n:
        open(os.path.join(out, name), "wb").write(base64.b64decode("".join(ch[i] for i in range(n))))
        print("ok", name)
    else:
        print("incomplete", name, len(ch), "/", n)
