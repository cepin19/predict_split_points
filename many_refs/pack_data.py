"""Pack an explorer data file for publishing (the artifact allows 16 MB per file).

  python pack_data.py explorer/data_<tag>.json [--out explorer/pub]

Writes <out>/data_<tag>.json, an index whose sentences are stubs {"sid", "file"}, and one file per sentence
<out>/data_<tag>_<sid>.json.  In the per-sentence files every automaton ("dawg") is stored compactly:
  final: a string of 0/1 characters, one per state;
  edges: one string - for each state, the number of edges, then per edge (sorted by label) the label delta
         to the previous edge of the state (the label itself for the first) and the absolute target state;
         every number is a little-endian base-32 varint written in URL-safe base-64 characters (5 data bits
         per character, the 6th bit = "more chunks follow").
engine.js (DawgSource) decodes both forms, so the working file (plain arrays) and the packed copy behave alike.
The working file explorer/data_<tag>.json is not modified; re-run after every change to it.
"""
import argparse, json
from pathlib import Path

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def varint(n, out):
    assert n >= 0
    while True:
        chunk = n & 31
        n >>= 5
        if n:
            out.append(ALPHABET[chunk | 32])
        else:
            out.append(ALPHABET[chunk])
            return


def pack_dawg(g):
    out = []
    for edges in g["edges"]:
        edges = sorted(edges)
        varint(len(edges), out)
        prev = 0
        for label, target in edges:
            varint(label - prev, out)
            varint(target, out)
            prev = label
    return dict(final="".join("1" if f else "0" for f in g["final"]), edges="".join(out), count=g["count"])


def unpack_dawg(g):  # reference decoder for the tests; engine.js has the JavaScript one
    if not isinstance(g["edges"], str):
        return g
    pos, s, n = 0, g["edges"], 0
    val = {c: i for i, c in enumerate(ALPHABET)}

    def read():
        nonlocal pos
        x, shift = 0, 0
        while True:
            v = val[s[pos]]
            pos += 1
            x |= (v & 31) << shift
            shift += 5
            if v < 32:
                return x

    edges = []
    while pos < len(s):
        k, prev, e = read(), 0, []
        for _ in range(k):
            prev += read()
            e.append([prev, read()])
        edges.append(e)
    return dict(final=[1 if c == "1" else 0 for c in g["final"]], edges=edges, count=g["count"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="explorer/pub")
    a = ap.parse_args()
    src = Path(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = json.load(open(src))
    dump = lambda x, p: p.write_text(json.dumps(x, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    stubs = []
    for sent in d["sentences"]:
        for s in sent.get("sources", []):
            if s.get("dawg"):
                g = s["dawg"]
                before = len(json.dumps(g, separators=(",", ":")))
                s["dawg"] = pack_dawg(g)
                back = unpack_dawg(s["dawg"])
                assert back["edges"] == [sorted(e) for e in g["edges"]] and back["final"] == [1 if f else 0 for f in g["final"]]
                after = len(json.dumps(s["dawg"], separators=(",", ":")))
                print(f"  {sent['sid']} {s.get('label')}: dawg {before / 1e6:.2f} -> {after / 1e6:.2f} MB")
        name = f"{src.stem}_{sent['sid']}.json"
        dump(sent, out / name)
        stubs.append(dict(sid=sent["sid"], file=name))
        print(f"{out / name}: {(out / name).stat().st_size / 1e6:.2f} MB")
    dump(dict(d, sentences=stubs), out / src.name)
    print(f"{out / src.name}: index with {len(stubs)} sentences")


if __name__ == "__main__":
    main()
