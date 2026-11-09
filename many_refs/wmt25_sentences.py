"""Split WMT25 general-track en->cs source documents into sentences.

Writes wmt25_sents.jsonl (all usable sentences) and pilot.jsonl (a stratified
sample for the annotation pilot).
"""
import json, random, re, sys
from collections import defaultdict

SRC = "/lnet/work/people/jon/wmt26/arab/wmt25.jsonl"
SPLIT = re.compile(r'(?<=[.!?…])["”’)]?\s+(?=["“‘(]?[A-Z0-9])')
PILOT = dict(news=8, literary=6, social=3, dialogue=3)  # speech = ASR transcripts, left out of the pilot


def sentences(text):
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        start = 0
        for m in SPLIT.finditer(para):
            yield para[start:m.end()].strip()
            start = m.end()
        yield para[start:].strip()


def usable(s):
    w = s.split()
    return (8 <= len(w) <= 40 and s[0].isupper() and re.search(r'[.!?…]["”’)]?$', s)
            and not re.search(r"https?://|www\.|@\w|#\w", s) and sum(c.isalpha() for c in s) > 0.7 * len(s.replace(" ", "")))


def main():
    out, seen = [], set()
    for line in open(SRC, encoding="utf-8"):
        d = json.loads(line)
        if (d["src_lang"], d["tgt_lang"], d["collection_id"]) != ("en", "cs_CZ", "general"):
            continue
        for i, s in enumerate(sentences(d["src_text"])):
            s = re.sub(r"^[A-Z]: ", "", s)  # dialogue speaker tags
            if usable(s) and s not in seen:
                seen.add(s)
                out.append(dict(id=f"{d['doc_id']}#{i}", domain=d["domain"], doc_id=d["doc_id"], en=s))
    with open("wmt25_sents.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_dom = defaultdict(list)
    for r in out:
        if 10 <= len(r["en"].split()) <= 30:
            by_dom[r["domain"]].append(r)
    rng = random.Random(25)
    pilot = []
    for dom, k in PILOT.items():
        docs = defaultdict(list)
        for r in by_dom[dom]:
            docs[r["doc_id"]].append(r)
        order = list(docs)
        rng.shuffle(order)
        # spread over documents: round-robin one sentence per document
        while len([p for p in pilot if p["domain"] == dom]) < k and any(docs.values()):
            for doc in order:
                if docs[doc] and len([p for p in pilot if p["domain"] == dom]) < k:
                    pilot.append(docs[doc].pop(rng.randrange(len(docs[doc]))))
    with open("pilot.jsonl", "w") as f:
        for r in pilot:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cnt = defaultdict(int)
    for r in out:
        cnt[r["domain"]] += 1
    print(len(out), "sentences", dict(cnt), "| pilot", len(pilot), file=sys.stderr)


if __name__ == "__main__":
    main()
