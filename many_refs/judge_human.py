"""Calibrate a reviewer against the paper's tolerance: judge, per sentence, with the QC reviewer prompt (qc.PROMPT)
  - up to --k random human references (the paper's annotators accepted them: should be accepted), and
  - corrupted human references (a duplicated word or a deleted content word: should be rejected).
A reviewer stricter than the annotators would delete valid translations in QC; aim at >= 95% human acceptance
while still rejecting the corruptions.
  ../env_vllm/bin/python judge_human.py --reviewer anthropic/claude-haiku-5.5 --think medium \\
      --input manyrefs5.jsonl calib.jsonl --out runs/judge_human_haiku.json
"""
import argparse, json, random, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from calib import norm
from detok import detok
from llm_client import connect
from qc import judge

ap = argparse.ArgumentParser()
ap.add_argument("--reviewer", default="anthropic/claude-haiku-5.5")
ap.add_argument("--provider", default="openrouter")
ap.add_argument("--think", default="medium")
ap.add_argument("--k", type=int, default=40)
ap.add_argument("--n-bad", type=int, default=20)
ap.add_argument("--input", nargs="+", default=["manyrefs5.jsonl", "calib.jsonl"])
ap.add_argument("--out", default="runs/judge_human.json")
a = ap.parse_args()
client, model = connect(a.reviewer, a.provider)


def corrupt(s, rng):
    w = s.split()
    if rng.random() < 0.5:  # duplicate a word
        i = rng.randrange(len(w))
        return " ".join(w[:i + 1] + [w[i]] + w[i + 1:]), "dup"
    content = [i for i, x in enumerate(w) if len(re.sub(r"\W", "", x)) > 3]
    i = rng.choice(content or range(len(w)))
    return " ".join(w[:i] + w[i + 1:]), "del"


def one(item):
    rng = random.Random(0)
    refs = sorted({norm(detok(l.rstrip("\n").split("\t")[2])) for l in open(f"by_sent/{item['sid']}.tsv", encoding="utf-8")})
    good = rng.sample(refs, min(a.k, len(refs)))
    bad = [corrupt(rng.choice(refs), rng) for _ in range(a.n_bad)]
    bad = [(t, k) for t, k in bad if norm(t) not in set(refs)]
    cands = [(t, "human") for t in good] + bad
    rng.shuffle(cands)
    v = judge(client, model, item["en"], [t for t, _ in cands], think=a.think)
    rows = [dict(text=t, kind=k, ok=x["ok"], wrong=x.get("wrong", ""), reason=x["reason"]) for (t, k), x in zip(cands, v)]
    acc = lambda kinds: (lambda r: sum(x["ok"] for x in r) / len(r) if r else None)(
        [x for x in rows if x["kind"] in kinds and x["ok"] is not None])
    return dict(sid=item["sid"], en=item["en"], human_accepted=acc({"human"}), corrupt_accepted=acc({"dup", "del"}),
                dup_accepted=acc({"dup"}), del_accepted=acc({"del"}), rows=rows)


items = [json.loads(l) for f in a.input for l in open(f)]
with ThreadPoolExecutor(10) as ex:
    res = list(ex.map(one, items))
json.dump(dict(reviewer=a.reviewer, think=a.think, results=res), open(a.out, "w"), ensure_ascii=False, indent=1)
f = lambda x: "-" if x is None else f"{x:.2f}"
print(f"reviewer {a.reviewer} ({a.think})\nsid\thuman_acc\tcorrupt_acc\tdup_acc\tdel_acc")
for r in res:
    print(f"{r['sid']}\t{f(r['human_accepted'])}\t{f(r['corrupt_accepted'])}\t{f(r['dup_accepted'])}\t{f(r['del_accepted'])}")
m = lambda k: sum(r[k] for r in res if r[k] is not None) / sum(r[k] is not None for r in res)
print(f"mean\t{m('human_accepted'):.3f}\t{m('corrupt_accepted'):.3f}\t{m('dup_accepted'):.3f}\t{m('del_accepted'):.3f}")
rej = [x for r in res for x in r["rows"] if x["kind"] == "human" and x["ok"] is False]
print(f"\n{len(rej)} human references rejected; most common reasons:")
for reason, c in Counter(re.sub(r"\W+", " ", (x["reason"] or "").lower())[:60] for x in rej).most_common(12):
    print(f"  {c}x {reason}")
for x in rej[:15]:
    print(f"  - {x['text']}\n      wrong: {x['wrong']} | {x['reason']}")
