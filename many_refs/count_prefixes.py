"""Per-prefix completion counts for the many-references test set.

For one sentence: detokenize all reference strings, dedupe, tokenize with the
probed model's tokenizer, sort, and for K uniformly sampled references compute at
every prefix length t:
  n[t]          number of distinct valid translations starting with the prefix
  n_child[t]    number of distinct next tokens (EOS counts as one) = local branching
  h_child[t]    entropy (bits) of the next-token distribution, weighting each child
                by its number of completions
  n_next[t]     completions after taking the reference's own next token (n[t+1])
Outputs counts/<sid>.npz (all sorted token sequences) and counts/<sid>.paths.jsonl.
"""
import argparse, bisect, json, math, os, random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from detok import detok

BIG = 1 << 30
EOS = -1


def prefix_range(S, prefix):
    return bisect.bisect_left(S, prefix), bisect.bisect_left(S, prefix + (BIG,))


def children(S, prefix, lo, hi):
    """Counts of next tokens among S[lo:hi] (all share `prefix`)."""
    t, out, i = len(prefix), {}, lo
    if i < hi and len(S[i]) == t:  # the prefix itself is a complete translation
        out[EOS] = 1
        i += 1
    while i < hi:
        x = S[i][t]
        j = bisect.bisect_left(S, prefix + (x, BIG), i, hi)
        out[x] = j - i
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("--tokenizer", default="google/gemma-2-2b-it")
    ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--out", default="counts")
    a = ap.parse_args()

    raw = [l.rstrip("\n").split("\t")[2] for l in open(f"by_sent/{a.sid}.tsv", encoding="utf-8")]
    strs = sorted(set(detok(s) for s in raw))
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = tok(strs, add_special_tokens=False)["input_ids"]
    S = sorted(set(map(tuple, ids)))

    Path(a.out).mkdir(exist_ok=True)
    lens = np.array([len(s) for s in S], dtype=np.int32)
    np.savez_compressed(f"{a.out}/{a.sid}.npz", flat=np.fromiter((x for s in S for x in s), dtype=np.int32),
                        offsets=np.concatenate([[0], np.cumsum(lens)]))

    rng = random.Random(int(a.sid))
    picks = sorted(rng.sample(range(len(S)), min(a.k, len(S))))
    with open(f"{a.out}/{a.sid}.paths.jsonl", "w") as f:
        for idx in picks:
            seq = S[idx]
            rec = dict(sid=a.sid, idx=idx, ids=list(seq), text=tok.decode(seq), n=[], n_child=[], h_child=[], n_next=[])
            for t in range(len(seq) + 1):
                p = seq[:t]
                lo, hi = prefix_range(S, p)
                ch = children(S, p, lo, hi)
                tot = hi - lo
                rec["n"].append(tot)
                rec["n_child"].append(len(ch))
                rec["h_child"].append(-sum(c / tot * math.log2(c / tot) for c in ch.values()))
                rec["n_next"].append(ch[seq[t]] if t < len(seq) else 1)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summ = dict(sid=a.sid, lines=len(raw), distinct_strings=len(strs), distinct_token_seqs=len(S),
                len_median=int(np.median(lens)), len_min=int(lens.min()), len_max=int(lens.max()))
    json.dump(summ, open(f"{a.out}/{a.sid}.summary.json", "w"))
    print(summ)


if __name__ == "__main__":
    main()
