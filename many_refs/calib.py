"""Compare LLM grammars with the human many-references sets on the same sentences.

  python calib.py runs/<tag>      (run made with pilot.py --input calib.jsonl)
For each sentence and round: |LLM| distinct outputs, |human|, overlap, recall of the human
set, share of LLM outputs inside the human set, and the reviewer's precision estimate.
"""
import json, re, sys
from pathlib import Path

from detok import detok
from grammar import Grammar


def norm(s):
    s = re.sub(r"[„“”\"]", '"', s)
    s = re.sub(r"[‚‘’']", "'", s)
    return re.sub(r"\s+", " ", s).strip()


def compare(sid, name, c, human):
    g = Grammar(c["grammar"])
    p = c.get("precision")
    if g.count() > 5_000_000:
        return f"{sid}\t{name}\t>{g.count()}\t{len(human)}\t-\t-\t-\t{p}"
    llm = {norm(t) for t, _ in g.expand()}
    ov = len(llm & human)
    return (f"{sid}\t{name}\t{len(llm)}\t{len(human)}\t{ov}\t{ov / len(human):.4f}\t{ov / len(llm):.4f}\t"
            f"{'-' if p is None else round(p, 2)}")


def main(run):
    print("sid\tgrammar\tllm\thuman\toverlap\trecall\tllm_in_human\treviewer_prec")
    for f in sorted(Path(run).glob("[0-9]*.json")):
        rec = json.load(open(f))
        sid = rec["sid"]
        human = {norm(detok(l.rstrip("\n").split("\t")[2])) for l in open(f"by_sent/{sid}.tsv", encoding="utf-8")}
        if "annotations" not in rec:  # older single-annotation runs
            items = [("initial", rec["rounds"][0][0]), (rec["final"]["how"], rec["final"])] if "final" in rec else []
        else:
            items = [(f"ann{i + 1}", a["final"]) for i, a in enumerate(rec["annotations"]) if "final" in a]
            if rec.get("union"):
                items.append(("union", rec["union"]))
        if not items:
            print(f"{sid}\tfailed")
        for name, c in items:
            print(compare(sid, name, c, human))


if __name__ == "__main__":
    main(sys.argv[1])
