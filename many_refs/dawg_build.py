"""Minimal acyclic automaton (DAWG) of the distinct translations of a bubble set, built bottom-up with hash-consed
states so that every intermediate automaton is already minimal.

determinize.Det (subset construction, minimised only at the end) explodes on sets with many alignment-ambiguous
options: kimi-k3.1-pro's sentence-70 set in run paper_v6 needed > 3,000,000 subset states (213 GB) although its
minimal automaton has a few hundred states. Here a state is identified by (final flag, sorted transitions to
canonical children), unions are memoised product constructions on canonical states, and a bubble's automaton is
built with its continuation attached (a map: coverage used by the bubble -> continuation state), so the language
of every sub-result is exactly a set of complete-translation suffixes.
Output format as determinize.Det.build(): (final flags, edges [[label, target], ...] per state, count of strings
below each state); state 0 is the start.
  build_dawg(B, tok_ids) -> (F, E, C)
"""
import sys

sys.setrecursionlimit(20000)


def _fits(a, rem):
    return all(x <= r for x, r in zip(a, rem))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


class Builder:
    def __init__(self, B, tok_ids):
        self.B = B
        self.full = tuple(B.full)
        self.vec = [tuple(b[0]) for b in B.b]
        self.items = []
        for vec, conds, items, where in B.b:
            its = []
            for it in items:
                if it[0] == "atom":
                    its.append(("a", tuple(tok_ids(it[1]))))
                else:
                    its.append(("s", tuple(B.fillers(it))))
            self.items.append(its)
        self.nodes = [(True, ())]  # canonical states: id -> (final, ((label, child), ...)); 0 accepts only the empty string
        self.index = {(True, ()): 0}
        self.FINAL = 0
        self.m_A, self.m_union = {}, {}
        B._memo = {}

    def node(self, final, trans):
        key = (final, trans)
        i = self.index.get(key)
        if i is None:
            i = len(self.nodes)
            self.nodes.append(key)
            self.index[key] = i
        return i

    def union(self, a, b):
        if a is None or a == b:
            return b
        if b is None:
            return a
        key = (a, b) if a < b else (b, a)
        r = self.m_union.get(key)
        if r is not None:
            return r
        fa, ta = self.nodes[a]
        fb, tb = self.nodes[b]
        da, db = dict(ta), dict(tb)
        trans = []
        for t in sorted(set(da) | set(db)):
            x, y = da.get(t), db.get(t)
            trans.append((t, self.union(x, y) if x is not None and y is not None else (x if x is not None else y)))
        r = self.node(fa or fb, tuple(trans))
        self.m_union[key] = r
        return r

    def chain(self, toks, child):
        for t in reversed(toks):
            child = self.node(False, ((t, child),))
        return child

    def A(self, k, i, used, rem, cont):
        """Automaton of the strings produced from item i of bubble k on, given the coverage `used` so far (bubble k's own
        words included) within the budget `rem`, followed by the continuation cont[total coverage of bubble k]."""
        key = (k, i, used, rem, cont)
        r = self.m_A.get(key, -1)
        if r != -1:
            return r
        items = self.items[k]
        if i == len(items):
            r = dict(cont).get(used)
        else:
            kind, x = items[i]
            if kind == "a":
                child = self.A(k, i + 1, used, rem, cont)
                r = None if child is None else self.chain(x, child)
            else:
                left = _sub(rem, used)
                r = None
                for f in x:
                    if not _fits(self.vec[f], left):
                        continue
                    cf = []
                    for u2 in self.B._count(f, left):  # coverage vectors the filler can end with (its own words included)
                        after = self.A(k, i + 1, _add(used, u2), rem, cont)
                        if after is not None:
                            cf.append((u2, after))
                    if cf:
                        r = self.union(r, self.A(f, 0, self.vec[f], left, tuple(sorted(cf))))
        self.m_A[key] = r
        return r

    def build(self):
        root = None
        for k in range(len(self.vec)):
            if not _fits(self.vec[k], self.full):
                continue
            cont = tuple(sorted((u, self.FINAL) for u in self.B._count(k, self.full) if self.B._complete(u)))
            if cont:
                root = self.union(root, self.A(k, 0, self.vec[k], self.full, cont))
        return self.export(root)

    def export(self, root):
        if root is None:
            return [0], [[]], [0]
        ids, order = {root: 0}, [root]
        for s in order:  # breadth-first numbering, start = 0
            for _, c in self.nodes[s][1]:
                if c not in ids:
                    ids[c] = len(ids)
                    order.append(c)
        F = [int(bool(self.nodes[s][0])) for s in order]  # final flag (bool) or coverage labels (set)
        E = [sorted([t, ids[c]] for t, c in self.nodes[s][1]) for s in order]
        Cn, stack = {}, [(root, False)]  # counts in post-order (children first)
        while stack:
            s, ready = stack.pop()
            if s in Cn:
                continue
            if ready:
                Cn[s] = int(bool(self.nodes[s][0])) + sum(Cn[c] for _, c in self.nodes[s][1])
            else:
                stack.append((s, True))
                stack.extend((c, False) for _, c in self.nodes[s][1] if c not in Cn)
        return F, E, [Cn[s] for s in order]


def build_dawg(B, tok_ids):
    return Builder(B, tok_ids).build()


class Builder2(Builder):
    """Same result, context-free memoisation: R(k, i, used, rem) is the automaton of the suffixes of bubble k from item i
    on, with final states labelled by the coverage the bubble ends with; a slot attaches the filler's automaton to the
    continuation by replacing those labels (attach), so a filler's automaton is built once per (filler, budget)
    rather than once per parent context."""

    def __init__(self, B, tok_ids):
        super().__init__(B, tok_ids)
        self.nodes, self.index = [], {}
        self.m_R, self.m_att, self.m_fin = {}, {}, {}
        self.ACC = frozenset([()])  # label of an accepting state in the finished automaton

    def chain(self, toks, child):
        for t in reversed(toks):
            child = self.node(frozenset(), ((t, child),))
        return child

    def union(self, a, b):  # labels are sets here: the union keeps every coverage label
        if a is None or a == b:
            return b
        if b is None:
            return a
        key = (a, b) if a < b else (b, a)
        r = self.m_union.get(key)
        if r is not None:
            return r
        la, ta = self.nodes[a]
        lb, tb = self.nodes[b]
        da, db = dict(ta), dict(tb)
        trans = []
        for t in sorted(set(da) | set(db)):
            x, y = da.get(t), db.get(t)
            trans.append((t, self.union(x, y) if x is not None and y is not None else (x if x is not None else y)))
        r = self.node(la | lb, tuple(trans))
        self.m_union[key] = r
        return r

    def R(self, k, i, used, rem):
        key = (k, i, used, rem)
        r = self.m_R.get(key, -1)
        if r != -1:
            return r
        items = self.items[k]
        if i == len(items):
            r = self.node(frozenset([used]), ())
        else:
            kind, x = items[i]
            if kind == "a":
                child = self.R(k, i + 1, used, rem)
                r = None if child is None else self.chain(x, child)
            else:
                left = _sub(rem, used)
                r = None
                for f in x:
                    if not _fits(self.vec[f], left):
                        continue
                    s = self.R(f, 0, self.vec[f], left)
                    if s is not None:
                        r = self.union(r, self.attach(s, k, i + 1, used, rem))
        self.m_R[key] = r
        return r

    def attach(self, s, k, i, used, rem):
        """Replace every final label u2 of automaton s by the continuation R(k, i, used + u2, rem)."""
        key = (s, k, i, used, rem)
        r = self.m_att.get(key, -1)
        if r != -1:
            return r
        labels, trans = self.nodes[s]
        r = None
        for u2 in labels:
            r = self.union(r, self.R(k, i, _add(used, u2), rem))
        tr = []
        for t, child in trans:
            a = self.attach(child, k, i, used, rem)
            if a is not None:
                tr.append((t, a))
        if tr:
            r = self.union(r, self.node(frozenset(), tuple(tr)))
        self.m_att[key] = r
        return r

    def finish(self, s):
        """Keep the paths whose coverage is complete (dropped words allowed); labels become the accepting mark."""
        key = s
        r = self.m_fin.get(key, -1)
        if r != -1:
            return r
        labels, trans = self.nodes[s]
        tr = []
        for t, c in trans:
            a = self.finish(c)
            if a is not None:
                tr.append((t, a))
        fin = any(self.B._complete(u) for u in labels)
        r = self.node(self.ACC if fin else frozenset(), tuple(tr)) if fin or tr else None
        self.m_fin[key] = r
        return r

    def build(self):
        root = None
        for k in range(len(self.vec)):
            if not _fits(self.vec[k], self.full):
                continue
            s = self.R(k, 0, self.vec[k], self.full)
            if s is not None:
                root = self.union(root, self.finish(s))
        return self.export(root)


def build_dawg2(B, tok_ids):
    return Builder2(B, tok_ids).build()
