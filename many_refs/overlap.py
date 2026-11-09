"""Overlap between translation sets, exactly, by intersecting their automata (no enumeration):
  - every pair of LLM translators (the paper's annotators overlapped by 4.4%: inter-annotator agreement of the
    sets as |A & B| / |A u B|, reported here the same way, plus |A & B| / |A| per direction);
  - the human annotators of the sentence against each other (the paper's number, for our three sentences);
  - every LLM translator and the union against the human set.
Writes the results into the explorer data file (s["overlap"]) and prints a table.
  ../env_vllm/bin/python overlap.py explorer/data_paper_v5.json
"""
import json, sys
from determinize import minimize
from to_distinct import union


def intersect(dawgs):
    """Deterministic intersection of several DAWGs (product construction), minimised; returns (F, E, C)."""
    start = tuple(0 for _ in dawgs)
    ids, final, edges, todo = {start: 0}, [False], [None], [start]
    while todo:
        S = todo.pop()
        sid = ids[S]
        final[sid] = all(dawgs[i]["final"][s] for i, s in enumerate(S))
        maps = [dict(dawgs[i]["edges"][s]) for i, s in enumerate(S)]
        common = set(maps[0])
        for m in maps[1:]:
            common &= set(m)
        e = {}
        for x in sorted(common):
            T = tuple(m[x] for m in maps)
            if T not in ids:
                ids[T] = len(ids)
                final.append(False)
                edges.append(None)
                todo.append(T)
            e[x] = ids[T]
        edges[sid] = e
    return minimize(final, edges)


def size(d):
    return d["count"][0]


def pair(a, b):
    i = intersect([a, b])[2][0]
    u = union([a, b])[2][0]
    return dict(inter=i, union=u, jaccard=i / u if u else None, a_in_b=i / size(a) if size(a) else None, b_in_a=i / size(b) if size(b) else None)


if __name__ == "__main__":
    path = sys.argv[1]
    d = json.load(open(path))
    for s in d["sentences"]:
        srcs = {x["id"]: x for x in s["sources"] if x.get("dawg")}
        llm = [i for i in srcs if srcs[i]["kind"] == "llm"]
        out = dict(sizes={i: size(srcs[i]["dawg"]) for i in srcs}, llm_pairs={}, vs_human={})
        for n, a in enumerate(llm):
            for b in llm[n + 1:]:
                out["llm_pairs"][f"{a}|{b}"] = pair(srcs[a]["dawg"], srcs[b]["dawg"])
        for i in llm + (["union"] if "union" in srcs else []):
            out["vs_human"][i] = pair(srcs[i]["dawg"], srcs["human"]["dawg"])
        parts = srcs["human"].get("parts") or {}
        if len(parts) >= 2:
            names = sorted(parts, key=lambda k: -parts[k]["count"][0])[:2]
            out["human_pair"] = dict(a=names[0], b=names[1], sizes={k: parts[k]["count"][0] for k in names}, **pair(parts[names[0]], parts[names[1]]))
        s["overlap"] = out
        lab = lambda i: srcs[i].get("label", i) + (f" ({srcs[i]['src_lang']})" if srcs[i].get("src_lang", "en") != "en" else "")
        print(f"\n=== sentence {s['sid']}: sizes " + ", ".join(f"{lab(i)} {out['sizes'][i]:,}" for i in out["sizes"]))
        for k, v in out["llm_pairs"].items():
            a, b = k.split("|")
            print(f"  {lab(a):28s} & {lab(b):28s} shared {v['inter']:>9,}  jaccard {100 * (v['jaccard'] or 0):5.1f}%  of A {100 * (v['a_in_b'] or 0):5.1f}%  of B {100 * (v['b_in_a'] or 0):5.1f}%")
        for i, v in out["vs_human"].items():
            print(f"  {lab(i):28s} & human                        shared {v['inter']:>9,}  jaccard {100 * (v['jaccard'] or 0):5.1f}%  of LLM set {100 * (v['a_in_b'] or 0):5.1f}%  of human set {100 * (v['b_in_a'] or 0):5.1f}%")
        if "human_pair" in out:
            h = out["human_pair"]
            print(f"  human {h['a']} ({h['sizes'][h['a']]:,}) & {h['b']} ({h['sizes'][h['b']]:,}): shared {h['inter']:,}  jaccard {100 * (h['jaccard'] or 0):.1f}%  of {h['a']} {100 * (h['a_in_b'] or 0):.1f}%  of {h['b']} {100 * (h['b_in_a'] or 0):.1f}%")
    json.dump(d, open(path, "w"), ensure_ascii=False, separators=(",", ":"))
