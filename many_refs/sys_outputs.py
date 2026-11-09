"""The WMT11 English->Czech system outputs for the sentences of an explorer data file, for the paper's extrinsic
evaluation (how many MT outputs does a reference set cover?). Writes explorer/sys_outputs.json: {sid: [{system, text}]}.
  ../env_vllm/bin/python sys_outputs.py explorer/data_paper_v5.json
"""
import json, sys
from pathlib import Path

D = Path("wmt11_sys/wmt11-data/plain/system-outputs/newstest2011/en-cz")
sids = sorted({s["sid"] for f in sys.argv[1:] for s in json.load(open(f))["sentences"]}, key=int)
out = {}
for f in sorted(D.iterdir()):
    system = f.name.split("en-cz.")[-1]
    lines = f.read_text(encoding="utf-8").split("\n")
    for sid in sids:
        out.setdefault(sid, []).append(dict(system=system, text=lines[int(sid) - 1].strip(), contrastive=system.endswith("-contrastive")))
json.dump(out, open("explorer/sys_outputs.json", "w"), ensure_ascii=False, indent=0)
print({k: len(v) for k, v in out.items()})
