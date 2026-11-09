"""Prefix-level comparison of LLM bubble sets with the human many-references set, in the probed
model's tokens (gemma-2-2b-it), for one sentence.

LLM side: each translator's final bubbles (after QC when available) expanded in full, or - above
--cap derivations - --cap random derivations (each slot filled by a random filler; so deep prefixes
of very large sets are under-represented and their counts are lower bounds).
Human side: all distinct human translations. Both sides are detokenised and quotation marks unified.

Reported, for the union of the translators and for each model:
  node_recall[t]     share of distinct human prefixes of length t that also occur in the LLM set
  node_precision[t]  share of distinct LLM prefixes of length t that occur in the human set
  path_match         along --k random human translations: mean share of the translation (in tokens)
                     whose prefix still occurs in the LLM set; and the share fully contained
  branch_spearman    along the same paths, at every prefix present in both sets: Spearman correlation
                     of the number of distinct next tokens (human vs LLM), and of log #completions
  ../env/bin/python prefix_compare.py runs/paper_v3 7 --redo runs/paper_v3_redo:2 --exclude gemini
"""
import argparse, bisect, json, math, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from bubbles import Bubbles
from count_prefixes import children, prefix_range
from calib import norm
from detok import detok

DEPTHS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def canon(s):
    """Detokenise and unify quotation marks (annotators differ: „...“ vs "...")."""
    return norm(detok(s))


def tokenize(tok, strs):
    strs, ids = sorted(strs), set()
    for i in range(0, len(strs), 20000):
        ids.update(map(tuple, tok(strs[i:i + 20000], add_special_tokens=False)["input_ids"]))
    return ids


def load_human(sid, tok):
    return sorted(tokenize(tok, {canon(l.rstrip("\n").split("\t")[2]) for l in open(f"by_sent/{sid}.tsv", encoding="utf-8")}))


def has_prefix(S, p):
    i = bisect.bisect_left(S, p)
    return i < len(S) and S[i][:len(p)] == p


def prefixes(S, t):
    return {s[:t] for s in S if len(s) >= t}


def compare(H, L, paths):
    out = dict(size=len(L), node_recall={}, node_precision={})
    for t in DEPTHS:
        ph, pl = prefixes(H, t), prefixes(L, t)
        if ph:
            out["node_recall"][t] = len(ph & pl) / len(ph)
        if pl:
            out["node_precision"][t] = len(ph & pl) / len(pl)
        del ph, pl
    match, full, bh, bl, nh, nl = [], 0, [], [], [], []
    for seq in paths:
        m = 0
        for t in range(len(seq) + 1):
            p = seq[:t]
            if not has_prefix(L, p):
                break
            m = t
            hl, hh = prefix_range(H, p)
            ll, lh = prefix_range(L, p)
            bh.append(len(children(H, p, hl, hh)))
            bl.append(len(children(L, p, ll, lh)))
            nh.append(math.log(hh - hl))
            nl.append(math.log(lh - ll))
        match.append(m / len(seq))
        i = bisect.bisect_left(L, seq)
        full += i < len(L) and L[i] == seq
    out["path_match"] = float(np.mean(match))
    out["path_full"] = full / len(paths)
    out["branch_points"] = len(bh)
    out["branch_spearman"] = float(spearmanr(bh, bl).statistic) if len(set(bh)) > 1 and len(set(bl)) > 1 else None
    out["logn_spearman"] = float(spearmanr(nh, nl).statistic) if len(set(nh)) > 1 and len(set(nl)) > 1 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("sid")
    ap.add_argument("--redo", nargs="*", default=[], help="RUNDIR:K - annotator 1 of RUNDIR replaces annotator K")
    ap.add_argument("--exclude", nargs="*", default=[], help="substrings of model names to leave out")
    ap.add_argument("--cap", type=int, default=1_000_000)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--tokenizer", default="google/gemma-2-2b-it")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rec = json.load(open(Path(a.run) / f"{a.sid}.json"))
    anns = rec["annotations"]
    for r in a.redo:
        d, k = r.rsplit(":", 1)
        p = Path(d) / f"{a.sid}.json"
        if p.exists():
            anns[int(k) - 1] = json.load(open(p))["annotations"][0]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    per_model = defaultdict(set)
    info = []
    for k, ann in enumerate(anns):
        m = ann.get("model") or ""
        if not ann.get("final") or any(x in m for x in a.exclude):
            continue
        clauses = (ann.get("qc") or {}).get("clauses") if (ann.get("qc") or {}).get("applied") else None
        clauses = clauses or ann["final"]["clauses"]
        B = Bubbles(rec[ann.get("src_lang", "en")], clauses)
        n = B.count()
        if n <= a.cap:
            strs, how = B.translations(), "all"
        else:
            strs, how = {t for t, _ in B.sample(a.cap, seed=k, tries=a.cap * 5)}, "sampled"
        ids = tokenize(tok, {canon(s) for s in strs})
        per_model[m.split("/")[-1]] |= ids
        info.append(dict(k=k + 1, model=m, derivations=n, used=len(ids), how=how))
        print(info[-1], file=sys.stderr, flush=True)

    H = load_human(a.sid, tok)
    paths = random.Random(0).sample(H, min(a.k, len(H)))
    res = dict(sid=a.sid, run=a.run, human=len(H), translators=info)
    union = sorted(set().union(*per_model.values()))
    res["union"] = compare(H, union, paths)
    del union
    res["models"] = {m: compare(H, sorted(s), paths) for m, s in per_model.items()}
    out = a.out or str(Path(a.run) / f"prefix_{a.sid}.json")
    json.dump(res, open(out, "w"), indent=1)
    u = res["union"]
    print(json.dumps(dict(sid=a.sid, human=len(H), llm=u["size"], node_recall=u["node_recall"],
                          path_match=round(u["path_match"], 3), branch_spearman=u["branch_spearman"],
                          logn_spearman=u["logn_spearman"])))


if __name__ == "__main__":
    main()
