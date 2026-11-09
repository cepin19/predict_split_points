"""Reference for prefix_compare.py: the same measures between the two human annotators of a sentence
(the larger set plays "human", the other one plays "LLM"), in gemma-2-2b-it tokens.
  ../env/bin/python human_prefix_baseline.py 22
"""
import json, random, sys
from collections import defaultdict

from transformers import AutoTokenizer

from prefix_compare import canon, compare, tokenize

sid = sys.argv[1]
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
by = defaultdict(set)
for l in open(f"by_sent/{sid}.tsv", encoding="utf-8"):
    _, who, s = l.rstrip("\n").split("\t")
    for w in who.split(", "):
        by[w].add(canon(s))
sets = {}
for w, strs in by.items():
    sets[w] = sorted(tokenize(tok, strs))
big, small = sorted(sets, key=lambda w: -len(sets[w]))[:2]
H, L = sets[big], sets[small]
res = dict(sid=sid, human=big, other=small, sizes={big: len(H), small: len(L)},
           cmp=compare(H, L, random.Random(0).sample(H, min(1000, len(H)))))
json.dump(res, open(f"counts/human_baseline_{sid}.json", "w"), indent=1)
c = res["cmp"]
print(json.dumps(dict(sid=sid, sizes=res["sizes"], node_recall=c["node_recall"], path_match=round(c["path_match"], 3),
                      branch_spearman=c["branch_spearman"], logn_spearman=c["logn_spearman"])))
