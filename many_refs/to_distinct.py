"""Replace the LLM bubble engines in explorer/data.json by exact automata of DISTINCT translations
(determinize.py), and build the union of the LLM translators' sets as one automaton (no double counting)."""
import json, sys, time
from bubbles import Bubbles
from build_explorer2 import toks
from determinize import Det, minimize

path = sys.argv[1] if len(sys.argv) > 1 else "explorer/data.json"
d = json.load(open(path)) if __name__ == "__main__" else {}


def union(dawgs):
    """deterministic union of several DAWGs (product construction), minimised"""
    start = tuple(0 for _ in dawgs)
    ids, final, edges, todo = {start: 0}, [False], [None], [start]
    while todo:
        S = todo.pop()
        sid = ids[S]
        final[sid] = any(s >= 0 and dawgs[i]["final"][s] for i, s in enumerate(S))
        nxt = {}
        for i, s in enumerate(S):
            if s < 0:
                continue
            for x, t in dawgs[i]["edges"][s]:
                nxt.setdefault(x, [-1] * len(dawgs))[i] = t
        e = {}
        for x, T in nxt.items():
            T = tuple(T)
            if T not in ids:
                ids[T] = len(ids)
                final.append(False)
                edges.append(None)
                todo.append(T)
            e[x] = ids[T]
        edges[sid] = e
    return minimize(final, edges)


def _main():
  pass
for s in (d["sentences"] if __name__ == "__main__" else []):
    ids = {w: i for i, w in enumerate(s["vocab"])}

    def tid(a):
        out = []
        for w in toks(a):
            if w not in ids:
                ids[w] = len(s["vocab"])
                s["vocab"].append(w)
            out.append(ids[w])
        return out
    llm = []
    for src in s["sources"]:
        if src["kind"] != "llm" or src.get("failed"):
            continue
        t0 = time.time()
        F, E, C = Det(Bubbles(s["en"], src["clauses"]), tid).build()
        src["derivations"] = src.get("derivations", src["size"])
        src["size"] = C[0]
        src["dawg"] = dict(final=F, edges=E, count=C)
        src.pop("engine", None)
        llm.append(src)
        print(s["sid"], src["id"], src["label"], src["derivations"], "derivations ->", C[0], "distinct,", len(F), "states", f"{time.time() - t0:.1f}s", flush=True)
    u = next(x for x in s["sources"] if x["id"] == "union")
    F, E, C = union([x["dawg"] for x in llm])
    u.update(size=C[0], dawg=dict(final=F, edges=E, count=C), sum_of_sets=sum(x["size"] for x in llm))
    print(s["sid"], "union", C[0], "distinct (sum of sets", u["sum_of_sets"], "),", len(F), "states", flush=True)
if __name__ == "__main__": json.dump(d, open(path, "w"), ensure_ascii=False, separators=(",", ":"))
