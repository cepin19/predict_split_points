"""Per-translator evaluation of an annotate_paper.py run, computed from the bubbles (no huge datafiles).

For each translator and each version (final = before QC, qc = after QC):
  size       number of derivations (and distinct translations when <= --max-enum)
  reviewer   share of --k random translations the reviewer accepts (random derivations: each slot filled
             by a randomly chosen filler, so not exactly uniform over the set)
  in_human   share of the translations that are in the human set (exact when enumerable, else on 20k samples)
  overlap    number of translations shared with the human set (only when enumerable)
  ../env_vllm/bin/python eval_ann.py runs/paper_v3 [--redo runs/paper_v3_redo:2]
"""
import argparse, json, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bubbles import Bubbles
from calib import norm
from detok import detok
from llm_client import connect
from qc import judge

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--redo", nargs="*", default=[], help="RUNDIR:K - take annotator 1 of RUNDIR as annotator K (1-based)")
ap.add_argument("--k", type=int, default=60)
ap.add_argument("--max-enum", type=int, default=3_000_000)
ap.add_argument("--reviewer", default="openai/gpt-6-luna")
a = ap.parse_args()

client, model = connect(a.reviewer, "openrouter")


def human_set(sid):
    return {norm(detok(l.rstrip("\n").split("\t")[2])) for l in open(f"by_sent/{sid}.tsv", encoding="utf-8")}


def evaluate(src, en, clauses, human, seed=0):
    B = Bubbles(src, clauses)
    n = B.count()
    row = dict(derivations=n)
    if n == 0:
        return row
    if n <= a.max_enum:
        tr = {norm(detok(t)) for t in B.translations()}
        row.update(distinct=len(tr), overlap=len(tr & human), in_human=len(tr & human) / len(tr))
    else:
        s = [norm(detok(t)) for t, _ in B.sample(20000, seed=seed, tries=200000)]
        row.update(in_human_sampled=sum(x in human for x in s) / max(1, len(s)))
    samp = sorted({detok(t) for t, _ in B.sample(a.k * 3, seed=seed + 1, tries=a.k * 60)})
    samp = random.Random(seed).sample(samp, min(a.k, len(samp)))
    v = judge(client, model, en, samp)
    ok = [x["ok"] for x in v if x["ok"] is not None]
    row["reviewer"] = sum(ok) / len(ok) if ok else None
    row["rejected"] = [(t, x.get("wrong"), x.get("reason")) for t, x in zip(samp, v) if x["ok"] is False][:8]
    return row


redo = {}
for r in a.redo:
    d, k = r.rsplit(":", 1)
    redo[int(k) - 1] = d

jobs, results = [], []
for f in sorted((p for p in Path(a.run).glob("[0-9]*.json") if p.stem.isdigit()), key=lambda p: int(p.stem)):
    rec = json.load(open(f))
    sid = rec["sid"]
    anns = rec["annotations"]
    for k, d in redo.items():
        p = Path(d) / f"{sid}.json"
        if p.exists():
            anns[k] = json.load(open(p))["annotations"][0]
    for k, ann in enumerate(anns):
        jobs.append((sid, k, rec, ann))

humans = {}


def one(job):
    sid, k, rec, ann = job
    if sid not in humans:
        humans[sid] = human_set(sid)
    out = dict(sid=sid, k=k + 1, model=ann.get("model"), tool_runs=ann.get("tool_runs"), error=bool(ann.get("error")))
    src = rec[ann.get("src_lang", "en")]
    if ann.get("final"):
        out["final"] = evaluate(src, rec["en"], ann["final"]["clauses"], humans[sid])
        q = ann.get("qc") or {}
        if q.get("applied"):
            out["qc"] = evaluate(src, rec["en"], q["clauses"], humans[sid])
            out["qc_removed"] = len(q["removed"])
        else:
            out["qc"], out["qc_removed"] = out["final"], 0
    return out


for sid in sorted({j[0] for j in jobs}, key=int):  # load human sets one sentence at a time
    humans.clear()
    humans[sid] = human_set(sid)
    with ThreadPoolExecutor(10) as ex:
        results += list(ex.map(one, [j for j in jobs if j[0] == sid]))

json.dump(results, open(Path(a.run) / "eval_ann.json", "w"), ensure_ascii=False, indent=1)
f = lambda x, d=2: "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
print("sid\tk\tmodel\truns\tderivations\tdistinct\toverlap\tin_human\treviewer(before QC)\tQC removed\treviewer(after QC)\tderivations after QC")
for r in results:
    fi, q = r.get("final") or {}, r.get("qc") or {}
    ih = fi.get("in_human", fi.get("in_human_sampled"))
    print(f"{r['sid']}\t{r['k']}\t{(r['model'] or '').split('/')[-1]}\t{r['tool_runs']}\t{fi.get('derivations', 'error' if r['error'] else '-')}\t"
          f"{f(fi.get('distinct'))}\t{f(fi.get('overlap'))}\t{f(ih, 4)}\t{f(fi.get('reviewer'))}\t{r.get('qc_removed', '-')}\t"
          f"{f(q.get('reviewer'))}\t{q.get('derivations', '-')}")
