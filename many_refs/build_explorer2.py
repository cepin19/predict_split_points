"""Exact, unpruned data for the explorer.

  - LLM sets are shipped as their bubbles, compiled for the browser engine (explorer/engine.js): per
    bubble alternative its coverage vector and its items (atom token ids, or a slot = list of filler
    alternatives). The browser counts and expands derivations itself, for any prefix, without ever
    listing the set.
  - Human sets (released only as strings) are compressed into a minimal acyclic word automaton (DAWG)
    with the number of translations below every state: exact and small.
Tokens: the annotators' own tokenisation, quotation marks unified; one id space per sentence.

  ../env_vllm/bin/python build_explorer2.py runs/paper_v3 --redo runs/paper_v3_redo:2 --out explorer/data.json
"""
import argparse, json, random, re
from collections import Counter
from pathlib import Path

from bubbles import Bubbles
from calib import norm
from detok import detok


def toks(s):
    """Tokens with punctuation normalised, so that typographic variants are one possibility:
    all double quotes (including guillemets) -> ", single quotes -> ', dashes -> -, "..." -> …;
    punctuation fused to other tokens is split off; "19 , 66" -> "19,66"."""
    s = re.sub(r"[„“”‟\"«»‹›]", '"', s)
    s = re.sub(r"[‚‘’`']", "'", s)
    s = re.sub(r"[–—−‐]", "-", s)
    s = s.replace("...", "…")
    s = re.sub(r"(\d)\s*,\s*(\d)", r"\1,\2", s)
    return re.findall(r"\d+(?:[.,]\d+)*|\w+(?:['-]\w+)*|[^\w\s]", s)


class Vocab:
    def __init__(self):
        self.ids, self.words = {}, []

    def __call__(self, w):
        if w not in self.ids:
            self.ids[w] = len(self.words)
            self.words.append(w)
        return self.ids[w]


def dawg(seqs):
    """Minimal acyclic DFA of sorted, distinct int sequences (Daciuk et al. 2000).
    Returns (final flags, edges per state as [[label, target], ...], count of words below each state)."""
    edges, final = [{}], [False]
    register = {}
    unchecked = []  # (parent, label, child)
    prev = ()

    def sig(s):
        return (final[s], tuple(sorted(edges[s].items())))

    def minimize(down_to):
        while len(unchecked) > down_to:
            parent, label, child = unchecked.pop()
            k = sig(child)
            if k in register:
                edges[parent][label] = register[k]
            else:
                register[k] = child

    for w in seqs:
        cp = 0
        while cp < len(w) and cp < len(prev) and w[cp] == prev[cp]:
            cp += 1
        minimize(cp)
        node = unchecked[-1][2] if unchecked else 0
        for x in w[cp:]:
            edges.append({})
            final.append(False)
            nxt = len(edges) - 1
            edges[node][x] = nxt
            unchecked.append((node, x, nxt))
            node = nxt
        final[node] = True
        prev = w
    minimize(0)
    # keep reachable states, renumber, count words
    order, seen, stack = [], {0: 0}, [0]
    while stack:
        s = stack.pop()
        order.append(s)
        for t in edges[s].values():
            if t not in seen:
                seen[t] = len(seen)
                stack.append(t)
    new_edges = [None] * len(seen)
    new_final = [0] * len(seen)
    for s in order:
        new_edges[seen[s]] = sorted([x, seen[t]] for x, t in edges[s].items())
        new_final[seen[s]] = int(final[s])
    count = [None] * len(seen)

    def cnt(s):  # iterative post-order
        st = [(s, False)]
        while st:
            v, done = st.pop()
            if count[v] is not None:
                continue
            if done:
                count[v] = new_final[v] + sum(count[t] for _, t in new_edges[v])
            else:
                st.append((v, True))
                st += [(t, False) for _, t in new_edges[v] if count[t] is None]
        return count[s]
    cnt(0)
    return new_final, new_edges, count


def compile_bubbles(B, vocab):
    """Bubbles -> browser form. Atoms with spaces become several tokens; empty atoms vanish.
    A slot is a negative reference -(j + 1) into ftab, the table of distinct filler lists (slots with the same word and
    conditions share one list; the raw lists made the v6 data 23 MB)."""
    out, ftab, fidx = [], [], {}
    for k, (vec, conds, items, where) in enumerate(B.b):
        its = []
        for it in items:
            if it[0] == "atom":
                its += [vocab(t) for t in toks(it[1])]
            else:
                fl = tuple(B.fillers(it))
                if fl not in fidx:
                    fidx[fl] = len(ftab)
                    ftab.append(list(fl))
                its.append(-(fidx[fl] + 1))
        out.append([list(vec), its])
    drop = [B.droppable[w] for w in B.vocab]
    return dict(full=list(B.full), drop=drop, b=out, ftab=ftab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--redo", nargs="*", default=[])
    ap.add_argument("--samples", type=int, default=80)
    ap.add_argument("--out", default="explorer/data.json")
    ap.add_argument("--old", default=None, help="previous data.json to take trajectories/stats from (optional)")
    a = ap.parse_args()
    run = Path(a.run)
    ev = {(r["sid"], r["k"]): r for r in json.load(open(run / "eval_ann.json"))} if (run / "eval_ann.json").exists() else {}
    redo = {int(r.rsplit(":", 1)[1]) - 1: r.rsplit(":", 1)[0] for r in a.redo}
    data = dict(run=a.run, sentences=[])
    rargs = json.load(open(run / "args.json")) if (run / "args.json").exists() else {}  # run settings (v6+: other_langs, skeletons, think_map)
    for f in sorted((p for p in run.glob("[0-9]*.json") if p.stem.isdigit()), key=lambda p: int(p.stem)):
        rec = json.load(open(f))
        sid = rec["sid"]
        anns = rec["annotations"]
        for k, d in redo.items():
            p = Path(d) / f"{sid}.json"
            if p.exists():
                anns[k] = json.load(open(p))["annotations"][0]
        vocab = Vocab()
        extra = json.load(open(f"inputs/{sid}.json")) if Path(f"inputs/{sid}.json").exists() else {}
        sent = dict(sid=sid, en=rec["en"], de=rec["de"], es=rec["es"], fr=rec["fr"], czech=[], sources=[],
                    other_langs=rargs.get("other_langs") is not False,  # older runs showed the German/Spanish/French versions
                    skeletons_used=bool(rargs.get("skeletons")) or any(x.get("skeletons") for x in anns))
        sent["czech"].append(dict(kind="O", label="Official WMT11 reference", text=extra.get("O", "")))
        for g in extra.get("G", []):
            sent["czech"].append(dict(kind="G", label="Translated from German", by=g["by"], text=g["text"]))
        for p in extra.get("P", []):
            sent["czech"].append(dict(kind="P", label=f"Post-edited MT ({p['system']})", by=p["by"], mt=p["mt"], text=p["post_edit"]))

        lines = [l.rstrip("\n").split("\t") for l in open(f"by_sent/{sid}.tsv", encoding="utf-8")]
        hseqs = sorted({tuple(vocab(t) for t in toks(s)) for _, _, s in lines})
        fin, edges, count = dawg(hseqs)
        hset = {" ".join(toks(s)) for _, _, s in lines}
        hs = sorted({norm(detok(s)) for _, _, s in lines})
        ann_sizes = Counter(w for _, who, _ in lines for w in who.split(", "))
        sent["sources"].append(dict(id="human", kind="human", label="Human annotators", model="Bojar et al. 2013",
                                    size=count[0], annotators=dict(ann_sizes), dawg=dict(final=fin, edges=edges, count=count),
                                    samples=random.Random(0).sample(hs, min(a.samples, len(hs)))))
        print(sid, "human", count[0], "strings ->", len(fin), "states", flush=True)
        for k, ann in enumerate(anns):
            m = ann.get("model") or "?"
            src = dict(id=f"L{k + 1}", kind="llm", model=m, label=m.split("/")[-1], tool_runs=ann.get("tool_runs"),
                       src_lang=ann.get("src_lang", "en"), skeletons=ann.get("skeletons"), skeletons_covered=ann.get("final_skeletons"),
                       think=(rargs.get("think_map") or {}).get(m, rargs.get("think", "medium")))
            traj = []
            for t in ann.get("turns", []):
                if "clauses" in t:
                    rep = t["tool"]
                    txt = rep.get("text", "")
                    status = ("syntax" if txt.startswith("SYNTAX") else "tool-error" if txt.startswith("TOOL ERROR")
                              else "errors" if "ERRORS:" in txt else "ok")
                    traj.append(dict(o=t["clauses"].count("option("), n=rep.get("n") or 0, s=status, sk=t.get("skeletons_covered")))
                elif t.get("refused_done"):
                    traj.append(dict(refused=True))
                elif t.get("done"):
                    traj.append(dict(done=True))
            src["trajectory"] = traj
            if ann.get("error") or not ann.get("final"):
                src.update(failed=True, size=0)
                sent["sources"].append(src)
                continue
            q = ann.get("qc") or {}
            clauses = q["clauses"] if q.get("applied") and q.get("clauses") else ann["final"]["clauses"]
            src["qc_summary"] = q.get("summary"); src["qc_rounds"] = q.get("rounds")
            B = Bubbles(rec[ann.get("src_lang", "en")], clauses)
            n = B.count()
            smp = [t for t, _ in B.sample(3000, seed=k, tries=30000)]
            in_h = sum(" ".join(toks(s)) in hset for s in smp) / max(1, len(smp))
            canon = sorted({norm(detok(s)) for s in smp})
            e = ev.get((sid, k + 1), {})
            src.update(size=n, engine=compile_bubbles(B, vocab), clauses=clauses,
                       removed=q.get("removed", []) if q.get("applied") else [],
                       reviewer_before=(e.get("final") or {}).get("reviewer"), reviewer_after=(e.get("qc") or {}).get("reviewer"),
                       in_human=in_h, samples=random.Random(k).sample(canon, min(a.samples, len(canon))))
            sent["sources"].append(src)
            print(sid, src["label"], n, "derivations,", len(B.b), "alternatives", flush=True)
        llm = [s for s in sent["sources"] if s["kind"] == "llm" and not s.get("failed")]
        sent["sources"].insert(1, dict(id="union", kind="union", label="All LLM translators", model="union",
                                       members=[s["id"] for s in llm], size=sum(s["size"] for s in llm),
                                       samples=random.Random(1).sample([x for s in llm for x in s["samples"]],
                                                                       min(a.samples, sum(len(s["samples"]) for s in llm)))))
        sent["vocab"] = vocab.words
        pf = run / f"prefix_{sid}.json"
        if pf.exists():
            p = json.load(open(pf))
            sent["prefix"] = dict(union=p["union"], models=p["models"])
        hb = Path(f"counts/human_baseline_{sid}.json")
        if hb.exists():
            h = json.load(open(hb))
            sent["human_baseline"] = dict(sizes=h["sizes"], **{k: h["cmp"][k] for k in ["node_recall", "path_match", "branch_spearman"]})
        data["sentences"].append(sent)
    Path(a.out).parent.mkdir(exist_ok=True)
    json.dump(data, open(a.out, "w"), ensure_ascii=False, separators=(",", ":"))
    print("wrote", a.out, Path(a.out).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
