"""Merge prefix_compare.py results (runs/<tag>/prefix_<sid>.json) and the human baseline into an explorer data file.
  ../env_vllm/bin/python merge_prefix.py runs/paper_v5 explorer/data_paper_v5.json"""
import json, sys
from pathlib import Path
run, out = Path(sys.argv[1]), sys.argv[2]
d = json.load(open(out))
for s in d["sentences"]:
    pf = run / f"prefix_{s['sid']}.json"
    if pf.exists():
        p = json.load(open(pf)); s["prefix"] = dict(union=p["union"], models=p["models"])
    hb = Path(f"counts/human_baseline_{s['sid']}.json")
    if hb.exists():
        h = json.load(open(hb)); s["human_baseline"] = dict(sizes=h["sizes"], **{k: h["cmp"][k] for k in ["node_recall", "path_match", "branch_spearman"]})
    print(s["sid"], "prefix" in s, "human_baseline" in s)
json.dump(d, open(out, "w"), ensure_ascii=False, separators=(",", ":"))
