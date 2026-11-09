"""Re-encode the browser grammars of an explorer data file with a shared filler table (see build_explorer2.compile_bubbles),
for data files built before that change.   ../env_vllm/bin/python compress_engine.py explorer/data_paper_v6.json [out.json]"""
import json, sys
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else src
d = json.load(open(src))
n0 = n1 = 0
for s in d["sentences"]:
    for x in s["sources"]:
        e = x.get("engine")
        if not e or "ftab" in e:
            continue
        n0 += len(json.dumps(e, separators=(",", ":")))
        ftab, fidx = [], {}
        for b in e["b"]:
            its = []
            for it in b[1]:
                if isinstance(it, list):
                    fl = tuple(it)
                    if fl not in fidx:
                        fidx[fl] = len(ftab); ftab.append(list(fl))
                    its.append(-(fidx[fl] + 1))
                else:
                    its.append(it)
            b[1] = its
        e["ftab"] = ftab
        n1 += len(json.dumps(e, separators=(",", ":")))
json.dump(d, open(out, "w"), ensure_ascii=False, separators=(",", ":"))
print(f"engines {n0 // 1000} KB -> {n1 // 1000} KB; file written to {out}")
