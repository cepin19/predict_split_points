"""Evaluate an annotate_paper.py run against the human many-references set (evaluation only:
nothing here feeds back into the annotation).

Per sentence: size of each LLM translator's set and of their union; inter-annotator overlap as in
the paper (|A n B| / |A u B| on tokenised strings; the paper's humans: 4.4% on average); human set
size; overlap between the LLM union and the human set (detokenised, quotes normalised); and,
with --judge, the reviewer's acceptance rate on a random sample of the LLM union next to its
acceptance rate on human references (runs/judge_human_v2.json).
  ../env_vllm/bin/python eval_paper.py runs/paper_v1 --judge --provider openrouter --annotator openai/gpt-6-luna --think medium
"""
import argparse, json, random
from collections import defaultdict
from pathlib import Path

from calib import norm
from detok import detok

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--judge", action="store_true")
ap.add_argument("--datafile", default="datafile.tsv")
ap.add_argument("--k", type=int, default=60)
for k in ["--provider", "--val-provider", "--annotator", "--validator"]:
    ap.add_argument(k, default=None)
ap.add_argument("--think", nargs="?", const=True, default=False)
ap.add_argument("--val-batch", type=int, default=20)
a = ap.parse_args()

llm = defaultdict(lambda: defaultdict(set))
for l in open(Path(a.run) / a.datafile, encoding="utf-8"):
    sid, _, who, s = l.rstrip("\n").split("\t")
    for w in who.split(", "):
        llm[sid][w].add(s)

human_acc = {}
hj = Path("runs/judge_human_v2.json")
if hj.exists():
    human_acc = {r["sid"]: r["human_accepted"] for r in json.load(open(hj))}

rows = []
for sid in sorted(llm, key=int):
    sets = llm[sid]
    union = set().union(*sets.values())
    names = sorted(sets)
    iaa = None
    if len(names) >= 2:  # mean pairwise Jaccard
        from itertools import combinations
        pr = [len(sets[x] & sets[y]) / len(sets[x] | sets[y]) for x, y in combinations(names, 2)]
        iaa = sum(pr) / len(pr)
    human = {norm(detok(l.rstrip("\n").split("\t")[2])) for l in open(f"by_sent/{sid}.tsv", encoding="utf-8")}
    u_norm = {norm(detok(s)) for s in union}
    ov = len(u_norm & human)
    row = dict(sid=sid, sizes={n: len(sets[n]) for n in names}, union=len(u_norm), iaa=iaa, human=len(human),
               overlap=ov, recall=ov / len(human), in_human=ov / len(u_norm), human_reviewer_acc=human_acc.get(sid))
    if a.judge:
        from pilot import Pilot
        en = dict(l.rstrip("\n").split("\t", 1) for l in open("many-czech-references/en.txt", encoding="utf-8"))[sid]
        p = Pilot(a)
        p.en = en
        samp = random.Random(0).sample(sorted(u_norm), min(a.k, len(u_norm)))
        v = p.judge(samp)
        ok = [x["ok"] for x in v if x["ok"] is not None]
        row["llm_reviewer_acc"] = sum(ok) / len(ok) if ok else None
        row["rejected"] = [(t, x["reason"]) for t, x in zip(samp, v) if x["ok"] is False][:10]
    rows.append(row)

json.dump(rows, open(Path(a.run) / ("eval.json" if a.datafile == "datafile.tsv" else f"eval_{a.datafile}.json"), "w"), ensure_ascii=False, indent=1)
f = lambda x, d=2: "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
print("sid\tLLM sets\tunion\tIAA\thuman\toverlap\trecall\tin_human\treviewer(LLM)\treviewer(human)")
for r in rows:
    print(f"{r['sid']}\t{'+'.join(map(str, r['sizes'].values()))}\t{r['union']}\t{f(r['iaa'], 3)}\t{r['human']}\t"
          f"{r['overlap']}\t{f(r['recall'], 4)}\t{f(r['in_human'], 3)}\t{f(r.get('llm_reviewer_acc'))}\t{f(r['human_reviewer_acc'])}")
