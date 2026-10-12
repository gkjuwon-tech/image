"""Reassemble files emitted by infra/emit.py from the Vast logs.
usage: python scripts/vfetch.py <instance_id> <out_dir> [tail_lines]"""
import base64, collections, os, re, subprocess, sys
iid, out = sys.argv[1], sys.argv[2]
tail = sys.argv[3] if len(sys.argv) > 3 else "20000"
log = subprocess.run(["bash", os.path.join(os.path.dirname(__file__), "vlogs.sh"), iid, tail],
                     capture_output=True, text=True).stdout
parts = collections.defaultdict(dict)
total = {}
for m in re.finditer(r"@@F (\S+) (\d+) (\d+) (\S+)", log):
    name, i, n, chunk = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    parts[name][i] = chunk
    total[name] = n
os.makedirs(out, exist_ok=True)
for name, ch in parts.items():
    if len(ch) == total[name]:
        open(os.path.join(out, name), "wb").write(base64.b64decode("".join(ch[i] for i in range(total[name]))))
        print("ok", name)
    else:
        print("incomplete", name, len(ch), "/", total[name])
