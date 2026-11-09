"""Bubble annotation as in Bojar et al. (2013), Sec. 2.2-2.3: a re-implementation of their Prolog tool.

The annotator writes clauses
    option(SrcWordsCovered, ConditionsMet, OutputAtomsAndSlots).
e.g. (the paper's example)
    option([the, city, council], [], [městská, rada, [approved, fem]]).
    option([approved], [fem], [schválila, [regulation, acc]]).
    option([a, new, regulation], [acc], [nový, předpis]).
A slot (a list inside the output) names at least one source word that the bubble filling it must
cover, plus optional conditions the filling bubble must meet. A translation is a tree of bubbles
whose covered source words add up to exactly the tokens of the source sentence.

Points the paper leaves open, decided here:
  - Slot elements that are tokens of the source sentence are source words; the rest are conditions.
  - Source words are matched case-insensitively (the paper's pre-processor allowed any case).
  - Any bubble can be the root of a translation; a source token may be covered only once (a token
    occurring twice in the sentence may be covered twice).
  - Punctuation tokens are tokens of the source and must be covered too.
  - or(c1, c2, ...) in a slot's conditions is met if any of c1, c2, ... is met.
  - drop([w, ...]). declares source tokens that may stay uncovered ("simply drop a token").
  - Output is the sequence of atoms, space-separated (tokenised, like the released data).
"""
import random, re, sys
from collections import Counter
from functools import lru_cache

PUNCT_ONLY = re.compile(r"^\W+$")


def tokenize_source(s):
    """Tokenise like the released data: words (incl. inner apostrophes/hyphens), numbers, punctuation."""
    return re.findall(r"\d+(?:[.,]\d+)*|\w+(?:['’-]\w+)*|[^\w\s]", s)


# ---------------------------------------------------------------- parsing
class ParseError(Exception):
    pass


# Pre-processor (the original tool had one for uppercase and Czech letters). Besides standard quoted
# atoms we accept: a lone " or ' as an atom, apostrophes inside words (it's), ,, for the comma atom,
# bare ( and ), and atoms separated by spaces instead of commas ([ještě dodává]). A quoted atom is a
# single token: it cannot contain brackets or a comma followed by a space (except the comma atom ', ').
TOK = re.compile(r"""\s*(?:(%[^\n]*)|('(?:\s*,\s*|(?:[^'\\,\[\]\n]|,(?!\s)|\\.)*)'|"(?:\s*,\s*|(?:[^"\\,\[\]\n]|,(?!\s)|\\.)*)")|(\[|\]|\(|\)|,)|((?:[^\s\[\](),'"%]|(?<=\w)'(?=\w))+|["']))""")


def _lex(text):
    pos, out = 0, []
    while pos < len(text):
        m = TOK.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == "":
                break
            raise ParseError(f"cannot read near: {text[pos:pos + 40]!r}")
        pos = m.end()
        comment, quoted, sym, bare = m.groups()
        if comment:
            continue
        if quoted:
            out.append(("atom", re.sub(r"\\(.)", r"\1", quoted[1:-1].strip() or quoted[1:-1])))
        elif sym:
            out.append(("sym", sym))
        elif bare:
            out.append(("atom", bare))
    return out


def _parse_term(toks, i):
    kind, v = toks[i]
    if kind == "sym" and v == "[":
        items, i = [], i + 1
        if toks[i] == ("sym", "]"):
            return items, i + 1
        while True:
            if toks[i] == ("sym", ","):  # empty element: ",," is the comma atom
                t, i = ",", i + 1
            elif toks[i] in (("sym", "("), ("sym", ")")):
                t, i = toks[i][1], i + 1
            else:
                t, i = _parse_term(toks, i)
            items.append(t)
            if toks[i] == ("sym", ","):
                i += 1
            elif toks[i] == ("sym", "]"):
                return items, i + 1
            elif toks[i][0] == "atom" or toks[i] in (("sym", "["), ("sym", "("), ("sym", ")")):
                continue  # missing comma: accept juxtaposed atoms
            else:
                raise ParseError(f"expected ',' or ']' but found {toks[i][1]!r} after {str(t)[:40]!r} "
                                 "(atoms must be separated by commas: [jeden, říká, to], not [jeden říká to])")
    if kind == "atom":
        if v in ("or", "option", "drop") and i + 1 < len(toks) and toks[i + 1] == ("sym", "("):  # compound: or(a, b), option(...), drop(...)
            args, i = [], i + 2
            while True:
                t, i = _parse_term(toks, i)
                args.append(t)
                if toks[i] == ("sym", ","):
                    i += 1
                elif toks[i] == ("sym", ")"):
                    return (v, args if v in ("option", "drop") else tuple(args)), i + 1
                else:
                    raise ParseError(f"expected ',' or ')' in {v}(...) but found {toks[i][1]!r}")
        return v, i + 1
    raise ParseError(f"unexpected {v!r}")


def parse(text):
    """Return (options, drops); options = list of (src list, cond list, out list)."""
    text = re.sub(r"```\w*", "", text)
    toks = _lex(text)
    options, drops, i = [], [], 0
    while i < len(toks):
        if toks[i] == ("atom", "."):
            i += 1
            continue
        try:
            t, i = _parse_term(toks, i)
        except IndexError:
            raise ParseError("unexpected end of input (unbalanced brackets?)")
        if isinstance(t, tuple) and t[0] == "option" and len(t[1]) == 3 and all(isinstance(x, list) for x in t[1]):
            options.append(tuple(t[1]))
        elif isinstance(t, tuple) and t[0] == "drop" and len(t[1]) == 1 and isinstance(t[1][0], list):
            drops += t[1][0]
        else:
            raise ParseError(f"expected option([...], [...], [...]) or drop([...]), found {str(t)[:80]}")
        if i < len(toks) and toks[i] == ("atom", "."):
            i += 1
    if not options:
        raise ParseError("no option(...) clauses found")
    return options, drops


# ---------------------------------------------------------------- the tool
class Bubbles:
    def __init__(self, source, text):
        self.source = source
        self.src_toks = [t.lower() for t in tokenize_source(source)]
        self.vocab = sorted(set(self.src_toks))
        self.vix = {w: i for i, w in enumerate(self.vocab)}
        self.full = tuple(Counter(self.src_toks)[w] for w in self.vocab)
        options, drops = parse(text)
        self.errors = []
        self.droppable = Counter()
        for w in drops:
            if not isinstance(w, str) or w.lower() not in self.vix:
                self.errors.append(f"drop: {w!r} is not a token of the source sentence")
            else:  # a dropped token may be left untranslated wherever it occurs (both quotation marks, every comma ...)
                self.droppable[w.lower()] = self.full[self.vix[w.lower()]]
        self.options, self.drops = options, drops
        self.b = []  # (cover vector, conds frozenset, out items, text)
        self.orig = []  # index into self.options of each entry of self.b
        for k, (src, conds, out) in enumerate(options):
            where = f"option #{k + 1} {self._show(src, conds, out)}"
            if not src:
                self.errors.append(f"{where}: covers no source word")
                continue
            bad = [w for w in src if not isinstance(w, str) or w.lower() not in self.vix]
            if bad:
                self.errors.append(f"{where}: not tokens of the source sentence: {bad}")
                continue
            cov = Counter(w.lower() for w in src)
            vec = tuple(cov[w] for w in self.vocab)
            if any(a > b for a, b in zip(vec, self.full)):
                self.errors.append(f"{where}: covers a word more often than it occurs in the source")
                continue
            if not all(isinstance(c, str) for c in conds):
                self.errors.append(f"{where}: conditions met must be plain atoms")
                continue
            items = []
            for x in out:
                if isinstance(x, str):
                    items.append(("atom", x))
                elif isinstance(x, list):
                    words = [y.lower() for y in x if isinstance(y, str) and y.lower() in self.vix]
                    cs = [y for y in x if not (isinstance(y, str) and y.lower() in self.vix)]
                    if not words:
                        self.errors.append(f"{where}: slot {self._term(x)} names no source word")
                    bad = [c for c in cs if not (isinstance(c, str) or (isinstance(c, tuple) and c[0] == "or"
                                                                          and all(isinstance(y, str) for y in c[1])))]
                    if bad:
                        self.errors.append(f"{where}: slot {self._term(x)}: a condition must be an atom or or(atom, ...), "
                                           f"not {self._term(bad[0])}")
                        cs = [c for c in cs if c not in bad]
                    items.append(("slot", tuple(words), tuple(cs)))
                elif isinstance(x, tuple) and x[0] == "or":
                    self.errors.append(f"{where}: or(...) is allowed among a slot's conditions, not as an output atom")
                else:
                    self.errors.append(f"{where}: unknown output element {self._term(x)}")
            self.b.append((vec, frozenset(conds), tuple(items), where))
            self.orig.append(k)
        self._fillers = {}
        self.notes = []
        for k, (vec, _, items, where) in enumerate(self.b):
            for it in items:
                if it[0] != "slot" or not it[1]:
                    continue
                fs = self.fillers(it)
                slot = list(it[1]) + [self._term(c) for c in it[2]]
                if not fs:
                    self.errors.append(f"{where}: no option can fill slot {slot}")
                elif all(any(a + b > c for a, b, c in zip(vec, self.b[f][0], self.full)) for f in fs):
                    others = [f for f in fs if f != k] or fs
                    clash = sorted({w for f in others for w, a, b, c in zip(self.vocab, vec, self.b[f][0], self.full)
                                    if a + b > c})
                    self.errors.append(f"{where}: slot {slot} can only be filled by options that cover {clash}, which "
                                       "this option already covers itself. An option lists only the source words it "
                                       "translates itself, not the words translated by the options in its slots.")

    # helpers
    @staticmethod
    def _term(x):
        if isinstance(x, list):
            return "[" + ", ".join(Bubbles._term(y) for y in x) + "]"
        if isinstance(x, tuple):
            return f"{x[0]}(" + ", ".join(Bubbles._term(y) for y in x[1]) + ")"
        return str(x)

    def _show(self, src, conds, out):
        return f"option({self._term(src)}, {self._term(conds)}, {self._term(out)})"

    @staticmethod
    def _meets(have, cond):
        if isinstance(cond, tuple) and cond[0] == "or":
            return any(Bubbles._meets(have, c) for c in cond[1])
        return cond in have

    def fillers(self, slot):
        key = slot[1:]
        if key not in self._fillers:
            words, conds = slot[1], slot[2]
            need = Counter(words)
            self._fillers[key] = [
                k for k, (vec, have, _, _) in enumerate(self.b)
                if all(vec[self.vix[w]] >= n for w, n in need.items()) and all(self._meets(have, c) for c in conds)]
        return self._fillers[key]

    # counting: memo on (bubble, remaining capacity)
    def _count(self, k, rem):
        key = (k, rem)
        if key in self._memo:
            return self._memo[key]
        vec, _, items, _ = self.b[k]
        if any(a > r for a, r in zip(vec, rem)):
            self._memo[key] = Counter()
            return self._memo[key]
        states = Counter({vec: 1})  # used so far -> number of ways
        for it in items:
            if it[0] != "slot":
                continue
            new = Counter()
            for used, c in states.items():
                left = tuple(r - u for r, u in zip(rem, used))
                for f in self.fillers(it):
                    for u2, c2 in self._count(f, left).items():
                        new[tuple(a + b for a, b in zip(used, u2))] += c * c2
            states = new
            if not states:
                break
        self._memo[key] = states
        return states

    def _complete(self, used):
        return all(f - u <= self.droppable[w] and u <= f for w, f, u in zip(self.vocab, self.full, used))

    def count(self):
        """Number of complete derivations (translations, possibly with duplicates)."""
        self._memo = {}
        return sum(c for k in range(len(self.b)) for used, c in self._count(k, self.full).items() if self._complete(used))

    def _gen(self, k, rem):
        vec, _, items, _ = self.b[k]
        if any(a > r for a, r in zip(vec, rem)):
            return

        def rec(i, used):
            if i == len(items):
                yield [], used
                return
            it = items[i]
            if it[0] == "atom":
                for rest, u in rec(i + 1, used):
                    yield [it[1]] + rest, u
                return
            left = tuple(r - u for r, u in zip(rem, used))
            for f in self.fillers(it):
                if not self._count(f, left):
                    continue
                for words, u2 in self._gen(f, left):
                    for rest, u in rec(i + 1, tuple(a + b for a, b in zip(used, u2))):
                        yield words + rest, u

        yield from rec(0, vec)

    def _gen_ids(self, k, rem):
        """Like _gen, but also yields the indices (into self.b) of the options used in the derivation."""
        vec, _, items, _ = self.b[k]
        if any(a > r for a, r in zip(vec, rem)):
            return

        def rec(i, used):
            if i == len(items):
                yield [], used, ()
                return
            it = items[i]
            if it[0] == "atom":
                for rest, u, ids in rec(i + 1, used):
                    yield [it[1]] + rest, u, ids
                return
            left = tuple(r - u for r, u in zip(rem, used))
            for f in self.fillers(it):
                if not self._count(f, left):
                    continue
                for words, u2, ids2 in self._gen_ids(f, left):
                    for rest, u, ids in rec(i + 1, tuple(a + b for a, b in zip(used, u2))):
                        yield words + rest, u, ids2 + ids

        for words, u, ids in rec(0, vec):
            yield words, u, (k,) + ids

    def identical(self):
        """[(k, j)] option number k (1-based, as in the file) is identical to the earlier option j."""
        seen, out = {}, []
        for i, (vec, conds, items, _) in enumerate(self.b):
            key = (vec, conds, items)
            if key in seen:
                out.append((self.orig[i] + 1, self.orig[seen[key]] + 1))
            else:
                seen[key] = i
        return out

    def redundant(self, limit=200000):
        """Options that add nothing: every translation they take part in is also produced without them.
        Returns (redundant, unused) with 1-based option numbers; redundant = [(k, j)] where j is an option of
        another derivation of the same translation. None when the set exceeds `limit` derivations."""
        self._memo = {}
        prod, total = {}, 0
        for k in range(len(self.b)):
            if not any(self._complete(u) for u in self._count(k, self.full)):
                continue
            for words, used, ids in self._gen_ids(k, self.full):
                if self._complete(used):
                    total += 1
                    if total > limit:
                        return None
                    prod.setdefault(" ".join(w for w in words if w), []).append(frozenset(ids))
        essential, involved = set(), set()
        for ds in prod.values():
            essential |= frozenset.intersection(*ds)
            involved |= frozenset.union(*ds)
        red = []
        for k in range(len(self.b)):
            if k in essential or k not in involved:
                continue
            partner = None
            for ds in prod.values():
                mine = [d for d in ds if k in d]
                if mine:
                    others = [d for d in ds if k not in d]
                    if others:
                        partner = next(iter(sorted(others[0] - mine[0])), None)
                        break
            red.append((self.orig[k] + 1, self.orig[partner] + 1 if partner is not None else None))
        unused = [self.orig[k] + 1 for k in range(len(self.b)) if k not in involved]
        return red, unused

    def translations(self, limit=None):
        """Distinct complete translations (tokenised strings)."""
        self._memo = {}
        out = set()
        for k in range(len(self.b)):
            if not any(self._complete(u) for u in self._count(k, self.full)):
                continue
            for words, used in self._gen(k, self.full):
                if self._complete(used):
                    out.add(" ".join(w for w in words if w))
                    if limit and len(out) >= limit:
                        return out
        return out


def _q(x):
    if isinstance(x, list):
        return "[" + ", ".join(_q(y) for y in x) + "]"
    if isinstance(x, tuple):
        return f"{x[0]}(" + ", ".join(_q(y) for y in x[1]) + ")"
    return "'" + str(x).replace("\\", "\\\\").replace("'", "\\'") + "'"


def to_text(options, drops=()):
    """Clauses back to text (all atoms quoted)."""
    out = [f"option({_q(src)}, {_q(conds)}, {_q(o)})." for src, conds, o in options]
    if drops:
        out.append(f"drop({_q(list(drops))}).")
    return "\n".join(out)


def _sample_methods():
    def _rand(self, k, rem, rng, prefer):
        vec, _, items, _ = self.b[k]
        if any(a > r for a, r in zip(vec, rem)):
            return None
        words, own, used, der = [], [], vec, []
        for it in items:
            if it[0] == "atom":
                words.append(it[1])
                own.append(it[1])
                continue
            left = tuple(r - u for r, u in zip(rem, used))
            fs = [f for f in self.fillers(it) if self._count(f, left)]
            if not fs:
                return None
            f = prefer if prefer in fs and rng.random() < 0.7 else rng.choice(fs)
            r = _rand(self, f, left, rng, prefer)
            if r is None:
                return None
            words += r[0]
            used = tuple(a + b for a, b in zip(used, r[1]))
            der += r[2]
        return words, used, [(k, own)] + der

    def sample(self, n, seed=0, prefer=None, tries=None):
        """Random complete derivations: [(tokenised string, [(bubble index, its own atoms), ...])]."""
        rng = random.Random(seed)
        self._memo = {}
        roots = [k for k in range(len(self.b)) if any(self._complete(u) for u in self._count(k, self.full))]
        out = []
        for _ in range(tries or n * 20):
            if len(out) >= n or not roots:
                break
            k = prefer if prefer in roots and rng.random() < 0.5 else rng.choice(roots)
            r = _rand(self, k, self.full, rng, prefer)
            if r and self._complete(r[1]):
                out.append((" ".join(w for w in r[0] if w), r[2]))
        return out
    return sample


Bubbles.sample = _sample_methods()


def structure_lines(b, n):
    """How factorised is the clause file? A flat list of whole sentences grows by one translation per option."""
    opts = len(b.b)
    if not opts:
        return [], 0
    slots = sum(1 for _, _, items, _ in b.b for it in items if it[0] == "slot")
    lines = [f"Structure: {opts} options, {slots} slots, {n / opts:.1f} translations per option."]
    if opts >= 10 and (slots == 0 or n < 2 * opts):
        lines.append("NOTE: you are listing whole sentences as separate options, so the set grows by one translation per "
                     "option. Factorise instead: keep a few sentence frames with slots and put the interchangeable words and "
                     "phrases into small bubbles that fill them, so that every structure combines with every lexical choice "
                     "(a few hundred clauses then yield hundreds of thousands of translations, as the human annotators' did).")
    return lines, slots


def redundancy_lines(b, n, enumerable):
    """Options that add no translation: identical clauses (always checked) and, for sets that can be enumerated,
    options whose every translation is also produced without them, and options taking part in no translation."""
    lines, red = [], []
    ident = b.identical()
    if ident:
        red += ident
        lines.append("Options identical to an earlier option: " + ", ".join(f"#{k} = #{j}" for k, j in ident[:20])
                     + (" ..." if len(ident) > 20 else ""))
    if enumerable and n > 0:
        r = b.redundant()
        if r is not None:
            more, unused = r
            dup = {k for k, _ in ident} | {j for _, j in ident}
            more = [(k, j) for k, j in more if k not in dup]  # identical pairs are already listed
            if more:
                red += more
                lines.append("Redundant options (every translation they produce is also produced without them): "
                             + ", ".join(f"#{k}" + (f" (see #{j})" if j else "") for k, j in more[:20]) + (" ..." if len(more) > 20 else ""))
            if unused:
                lines.append("Options taking part in no complete translation: " + ", ".join(f"#{k}" for k in unused[:20])
                             + (" ..." if len(unused) > 20 else ""))
    return lines, red


def tool_report(source, text, prev=None, show_all=300, show_sample=100, seed=0, max_enum=3_000_000):
    """What the annotator sees after each edit (the Prolog tool / web interface output)."""
    try:
        b = Bubbles(source, text)
    except ParseError as e:
        return dict(ok=False, text=f"SYNTAX ERROR: {e}"), None
    lines = []
    if b.errors:
        lines.append("ERRORS:")
        lines += [f"  {e}" for e in b.errors[:30]]
    n = b.count()
    lines.append(f"Source tokens to cover: {' '.join(t.lower() for t in tokenize_source(source))}"
                 + (f"   (may be dropped: {', '.join(repr(w) for w in b.vocab if b.droppable[w])})" if any(b.droppable.values()) else ""))
    lines.append(f"Number of complete translations (derivations): {n}")
    if n == 0:
        lines.append("No combination of options covers every source token exactly once.")
        avail = [sum(v[i] for v, _, _, _ in b.b) for i in range(len(b.vocab))]
        miss = [f"{w!r} (occurs {f}x)" for w, f, a in zip(b.vocab, b.full, avail) if a == 0 and b.droppable[w] < f]
        if miss:
            lines.append("Source tokens not covered by any option: " + ", ".join(miss))
        else:
            best = max((u for k in range(len(b.b)) for u in b._count(k, b.full)), key=sum, default=None)
            if best:
                left = [f"{w!r}" + (f" x{f - u}" if f - u > 1 else "") for w, f, u in zip(b.vocab, b.full, best) if f - u > 0]
                lines.append("Every token is covered by some option, but no single combination covers them all. "
                             "The most complete combination leaves uncovered: " + ", ".join(left))
        return dict(ok=False, n=0, text="\n".join(lines)), set()
    st, slots = structure_lines(b, n)
    if n > max_enum:
        rl, red = redundancy_lines(b, n, False)
        lines += st + rl
        samp = sorted({t for t, _ in b.sample(show_sample, seed=seed, tries=show_sample * 50)})
        lines.append(f"Too many to list in full (limit {max_enum}). Random {len(samp)} translations:")
        lines += [f"  {t}" for t in samp]
        return dict(ok=not b.errors, n=n, slots=slots, redundant=red, redundancy="\n".join(rl), text="\n".join(lines)), None
    tr = b.translations()
    lines.append(f"Distinct translations: {len(tr)}" + (f" ({n - len(tr)} of the derivations are duplicates)" if n > len(tr) else ""))
    rl, red = redundancy_lines(b, n, True)
    lines += st + rl
    rng = random.Random(seed)
    if len(tr) <= show_all:
        lines.append("All translations:")
        lines += [f"  {t}" for t in sorted(tr)]
    elif prev is not None:
        added, removed = sorted(tr - prev), sorted(prev - tr)
        lines.append(f"Differences against the previous state: {len(added)} added, {len(removed)} removed.")
        for name, xs in [("added", added), ("removed", removed)]:
            if xs:
                lines.append(f"Random {min(show_sample, len(xs))} {name}:")
                lines += [f"  {t}" for t in sorted(rng.sample(xs, min(show_sample, len(xs))))]
    else:
        lines.append(f"Random {show_sample} translations:")
        lines += [f"  {t}" for t in sorted(rng.sample(sorted(tr), show_sample))]
    return dict(ok=not b.errors, n=n, distinct=len(tr), slots=slots, redundant=red, redundancy="\n".join(rl), text="\n".join(lines)), tr


if __name__ == "__main__":
    # python bubbles.py "English source" clauses.pl
    rep, _ = tool_report(sys.argv[1], open(sys.argv[2]).read())
    print(rep["text"])
