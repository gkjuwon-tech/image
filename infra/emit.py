"""Print files to stdout as base64 chunks so they can be pulled back through the Vast logs API.
usage: python infra/emit.py <file> [<file> ...]   (reassemble locally with scripts/vfetch.py)"""
import base64, os, sys
CH = 400  # Vast logs truncate lines at ~500 chars
for p in sys.argv[1:]:
    b = base64.b64encode(open(p, "rb").read()).decode()
    n = (len(b) + CH - 1) // CH
    name = os.path.basename(p)
    for i in range(n):
        print(f"@@F {name} {i} {n} {b[i*CH:(i+1)*CH]}", flush=True)
