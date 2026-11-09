"""Data for the translation-space explorer (explorer/index.html): per sentence and per source (human set,
each LLM translator, the LLM union) a weighted tree of continuations, pruned to a node budget, plus
random samples, statistics, the annotation trajectories and the bubble files.

  ../env_vllm/bin/python build_explorer.py runs/paper_v3 --redo runs/paper_v3_redo:2 --out explorer/data.json
Trees are over tokenised Czech (quotation marks unified). Sets larger than --cap derivations are
represented by --cap random derivations, each weighted so that the total equals the set size (counts in
such trees are estimates, marked "est").
"""
import argparse, heapq, json, random, re
from collections import Counter
from itertools import groupby
from pathlib import Path

from bubbles import Bubbles
from calib import norm
from detok import detok

END = "⏎"
DEPTH = 16


def toks(s):
    s = re.sub(r"[„“”‟\"]", '"', s)
    s = re.sub(r"[‚‘’']", "'", s)
    return s.split()


class Tree:
    def __init__(self, weighted):  # Counter: tuple of tokens (with END if the translation ended) -> weight
        self.keys = sorted(weighted)
        self.cum = [0.0]
        for k in self.keys:
            self.cum.append(self.cum[-1] + weighted[k])

    def w(self, lo, hi):
        return self.cum[hi] - self.cum[lo]

    def children(self, t, lo, hi):
        out, i = [], lo
        for x, grp in groupby(range(lo, hi), key=lambda j: self.keys[j][t] if len(self.keys[j]) > t else None):
            n = len(list(grp))
            if x is not None:
                out.append((x, i, i + n))
            i += n
        return out

    def build(self, budget=2500, max_children=14):
        """Best-first expansion: always expand the heaviest unexpanded node, until `budget` nodes."""
        root = ["", self.w(0, len(self.keys)), []]
        heap, count, tie = [(-root[1], 0, root, 0, 0, len(self.keys))], 1, 1
        while heap and count < budget:
            _, _, node, t, lo, hi = heapq.heappop(heap)
            if t >= DEPTH:
                continue
            ch = sorted(self.children(t, lo, hi), key=lambda c: -self.w(c[1], c[2]))
            keep, rest = ch[:max_children], ch[max_children:]
            for x, a, b in keep:
                c = [x, round(self.w(a, b), 3), []]
                node[2].append(c)
                count += 1
                if x != END:
                    heapq.heappush(heap, (-c[1], tie, c, t + 1, a, b))
                    tie += 1
            if rest:
                node[2].append([f"+{len(rest)}", round(sum(self.w(a, b) for _, a, b in rest), 3), []])
                count += 1
        return root


def compact(node):  # [word, n, children] -> drop empty child lists
    w, n, ch = node
    return [w, n, [compact(c) for c in ch]] if ch else [w, n]


def strings_to_counter(strs, weight=1.0):
    c = Counter()
    for s in strs:
        t = toks(s)
        c[tuple(t[:DEPTH]) + ((END,) if len(t) <= DEPTH else ())] += weight
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--redo", nargs="*", default=[])
    ap.add_argument("--cap", type=int, default=200_000)
    ap.add_argument("--budget", type=int, default=2500)
    ap.add_argument("--samples", type=int, default=80)
    ap.add_argument("--out", default="explorer/data.json")
    a = ap.parse_args()
    run = Path(a.run)
    ev = {(r["sid"], r["k"]): r for r in json.load(open(run / "eval_ann.json"))} if (run / "eval_ann.json").exists() else {}
    redo = {int(r.rsplit(":", 1)[1]) - 1: r.rsplit(":", 1)[0] for r in a.redo}
    data = dict(run=a.run, sentences=[])
    for f in sorted(run.glob("[0-9]*.json"), key=lambda p: int(p.stem)):
        rec = json.load(open(f))
        sid = rec["sid"]
        anns = rec["annotations"]
        for k, d in redo.items():
            p = Path(d) / f"{sid}.json"
            if p.exists():
                anns[k] = json.load(open(p))["annotations"][0]
        extra = json.load(open(f"inputs/{sid}.json")) if Path(f"inputs/{sid}.json").exists() else {}
        sent = dict(sid=sid, en=rec["en"], de=rec["de"], es=rec["es"], fr=rec["fr"], czech=[], sources=[])
        sent["czech"].append(dict(kind="O", label="Official WMT11 reference", text=extra.get("O", rec["cs"][0][1] if rec["cs"] else "")))
        for g in extra.get("G", []):
            sent["czech"].append(dict(kind="G", label="Translated from German", by=g["by"], text=g["text"]))
        for p in extra.get("P", []):
            sent["czech"].append(dict(kind="P", label=f"Post-edited MT ({p['system']})", by=p["by"], mt=p["mt"], text=p["post_edit"]))

        # human
        human_lines = [l.rstrip("\n").split("\t") for l in open(f"by_sent/{sid}.tsv", encoding="utf-8")]
        hs = sorted({norm(detok(s)) for _, _, s in human_lines})
        htok = {" ".join(toks(s)) for _, _, s in human_lines}
        ann_sizes = Counter(w for _, who, _ in human_lines for w in who.split(", "))
        sent["sources"].append(dict(id="human", label="Human annotators", model="Bojar et al. 2013", kind="human",
                                    size=len(hs), est=False, annotators=dict(ann_sizes),
                                    tree=compact(Tree(strings_to_counter(htok)).build(a.budget)),
                                    samples=random.Random(0).sample(hs, min(a.samples, len(hs)))))
        union = Counter()
        for k, ann in enumerate(anns):
            m = ann.get("model") or "?"
            src = dict(id=f"L{k + 1}", kind="llm", model=m, label=m.split("/")[-1], tool_runs=ann.get("tool_runs"))
            traj = []
            for t in ann.get("turns", []):
                if "clauses" in t:
                    rep = t["tool"]
                    txt = rep.get("text", "")
                    status = ("syntax" if txt.startswith("SYNTAX") else "tool-error" if txt.startswith("TOOL ERROR")
                              else "errors" if "ERRORS:" in txt else "ok")
                    traj.append(dict(o=t["clauses"].count("option("), n=rep.get("n") or 0, d=rep.get("distinct"), s=status))
                elif t.get("refused_done"):
                    traj.append(dict(refused=True))
                elif t.get("done"):
                    traj.append(dict(done=True))
            src["trajectory"] = traj
            if ann.get("error") or not ann.get("final"):
                src.update(failed=True, size=0, tree=["", 0], samples=[])
                sent["sources"].append(src)
                continue
            q = ann.get("qc") or {}
            clauses = q["clauses"] if q.get("applied") else ann["final"]["clauses"]
            B = Bubbles(rec[ann.get("src_lang", "en")], clauses)
            n = B.count()
            if n <= a.cap:
                strs, est = list(B.translations()), False
                weight = 1.0
            else:
                strs = [t for t, _ in B.sample(a.cap, seed=k, tries=a.cap * 5)]
                est, weight = True, n / max(1, len(strs))
            cnt = strings_to_counter(strs, weight)
            union.update(cnt)
            canon = sorted({norm(detok(s)) for s in strs})
            in_h = sum(1 for s in strs if " ".join(toks(s)) in htok) / max(1, len(strs))
            e = ev.get((sid, k + 1), {})
            src.update(size=n, distinct=None if est else len(canon), est=est, clauses=clauses[:120000],
                       removed=q.get("removed", []) if q.get("applied") else [],
                       reviewer_before=(e.get("final") or {}).get("reviewer"), reviewer_after=(e.get("qc") or {}).get("reviewer"),
                       in_human=in_h, tree=compact(Tree(cnt).build(a.budget)),
                       samples=random.Random(k).sample(canon, min(a.samples, len(canon))))
            sent["sources"].append(src)
            print(sid, src["label"], n, "est" if est else "", flush=True)
        if union:
            us = [s for s in sent["sources"] if s["kind"] == "llm" and s.get("samples")]
            pool = [x for s in us for x in s["samples"]]
            sent["sources"].insert(1, dict(id="union", kind="union", label="All LLM translators", model="union",
                                           size=round(sum(union.values())), est=any(s.get("est") for s in us),
                                           tree=compact(Tree(union).build(a.budget + 1000)),
                                           samples=random.Random(1).sample(pool, min(a.samples, len(pool)))))
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
