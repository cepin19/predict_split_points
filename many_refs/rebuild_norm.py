"""Rebuild every automaton in explorer/data.json with the current token normalisation (build_explorer2.toks):
human set and per-annotator parts, each LLM translator's distinct set (determinize.py), and the union."""
import json, time
from collections import defaultdict
from bubbles import Bubbles
from build_explorer2 import dawg, toks
from dawg_build import build_dawg2  # bottom-up minimal automaton; determinize.Det exploded on kimi's sentence-70 set
from to_distinct import union

import sys
PATH = sys.argv[1] if len(sys.argv) > 1 else "explorer/data.json"
d = json.load(open(PATH))
for s in d["sentences"]:
    vocab, ids = [], {}

    def tid(w):
        if w not in ids:
            ids[w] = len(vocab)
            vocab.append(w)
        return ids[w]
    by, allseq = defaultdict(set), set()
    for l in open(f"by_sent/{s['sid']}.tsv", encoding="utf-8"):
        _, who, t = l.rstrip("\n").split("\t")
        seq = tuple(tid(w) for w in toks(t))
        allseq.add(seq)
        for a in who.split(", "):
            by[a].add(seq)
    h = s["sources"][0]
    F, E, C = dawg(sorted(allseq))
    h.update(size=C[0], dawg=dict(final=F, edges=E, count=C))
    h["parts"] = {}
    for a, seqs in sorted(by.items()):
        F, E, C = dawg(sorted(seqs))
        h["parts"][a] = dict(final=F, edges=E, count=C)
    print(s["sid"], "human", h["size"], flush=True)
    llm = []
    for src in s["sources"]:
        if src["kind"] != "llm" or src.get("failed"):
            continue
        t0 = time.time()
        before = src["size"]
        try:
            F, E, C = build_dawg2(Bubbles(s.get(src.get("src_lang") or "en", s["en"]), src["clauses"]), lambda a: [tid(w) for w in toks(a)])
        except (RuntimeError, MemoryError) as e:  # should not happen any more; no exact automaton, left out of the union
            print(s["sid"], src["id"], src["label"], before, "-> normalisation failed:", e, f"{time.time() - t0:.0f}s; left out of the union", flush=True)
            src["norm_failed"] = str(e)
            src.pop("dawg", None)
            continue
        src.update(size=C[0], dawg=dict(final=F, edges=E, count=C))
        src.pop("norm_failed", None)
        llm.append(src)
        print(s["sid"], src["id"], src["label"], before, "->", C[0], f"{time.time() - t0:.0f}s", flush=True)
    u = next(x for x in s["sources"] if x["id"] == "union")
    F, E, C = union([x["dawg"] for x in llm])
    before = u["size"]
    u.update(size=C[0], dawg=dict(final=F, edges=E, count=C), sum_of_sets=sum(x["size"] for x in llm))
    print(s["sid"], "union", before, "->", C[0], flush=True)
    s["vocab"] = vocab
json.dump(d, open(PATH, "w"), ensure_ascii=False, separators=(",", ":"))
print("done")
