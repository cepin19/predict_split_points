"""Exact set of DISTINCT translations of a bubble file, as a minimal acyclic word automaton (DAWG),
built without listing the translations.

Derivation configurations (stack of partially realised bubbles + covered source tokens) form a
nondeterministic automaton over output tokens; subset construction makes it deterministic, so every
distinct token sequence is one path, however many derivations produce it (also after token
normalisation, e.g. „ “ and " unified). The result is minimised and returned in the same format as
build_explorer2.dawg: (final flags, edges [[token, target], ...], number of distinct strings below each state).
"""
from collections import Counter


class Det:
    def __init__(self, B, tok_ids):
        """B: bubbles.Bubbles; tok_ids(atom string) -> list of token ids (normalised, possibly several or none)."""
        self.B = B
        self.full = B.full
        self.V = len(B.full)
        self.vec = [b[0] for b in B.b]
        self.items = []
        for vec, conds, items, where in B.b:
            its = []
            for it in items:
                if it[0] == "atom":
                    its += [("a", t) for t in tok_ids(it[1])]
                else:
                    its.append(("s", tuple(B.fillers(it))))
            self.items.append(its)
        self.m_rest, self.m_full, self.m_viable, self.m_close = {}, {}, {}, {}

    def fits(self, a, rem):
        return all(x <= r for x, r in zip(a, rem))

    def sub(self, a, b):
        return tuple(x - y for x, y in zip(a, b))

    def add(self, a, b):
        return tuple(x + y for x, y in zip(a, b))

    def complete(self, u):
        return all(f - x <= self.B.droppable[w] and x <= f for w, f, x in zip(self.B.vocab, self.full, u))

    def rest(self, k, i, rem):
        """set of extra coverage vectors achievable finishing bubble k from item i within rem"""
        key = (k, i, rem)
        if key in self.m_rest:
            return self.m_rest[key]
        states = {tuple([0] * self.V)}
        for kind, x in self.items[k][i:]:
            if kind != "s":
                continue
            nw = set()
            for u in states:
                left = self.sub(rem, u)
                for f in x:
                    if self.fits(self.vec[f], left):
                        for u2 in self.full_of(f, left):
                            nw.add(self.add(u, u2))
            states = nw
            if not states:
                break
        self.m_rest[key] = frozenset(states)
        return self.m_rest[key]

    def full_of(self, f, rem):
        key = (f, rem)
        if key not in self.m_full:
            r = set()
            if self.fits(self.vec[f], rem):
                r = {self.add(u, self.vec[f]) for u in self.rest(f, 0, self.sub(rem, self.vec[f]))}
            self.m_full[key] = frozenset(r)
        return self.m_full[key]

    def viable(self, frames, used):
        """can this configuration (top frame at item i, lower frames waiting in the slot at i) complete?"""
        key = (frames, used)
        if key in self.m_viable:
            return self.m_viable[key]
        states = {used}
        for j in range(len(frames) - 1, -1, -1):
            k, i = frames[j]
            start = i if j == len(frames) - 1 else i + 1
            states = {self.add(u, u2) for u in states for u2 in self.rest(k, start, self.sub(self.full, u))}
            if not states:
                break
        r = any(self.complete(u) for u in states)
        self.m_viable[key] = r
        return r

    def close(self, frames, used):
        """configurations at the next emitted token (or END = frames ())"""
        key = (frames, used)
        if key in self.m_close:
            return self.m_close[key]
        out = set()
        stack = [(frames, used)]
        while stack:
            fr, u = stack.pop()
            if not fr:
                if self.complete(u):
                    out.add(((), u))
                continue
            k, i = fr[-1]
            its = self.items[k]
            if i >= len(its):
                fr = fr[:-1]
                if fr:
                    pk, pi = fr[-1]
                    fr = fr[:-1] + ((pk, pi + 1),)
                stack.append((fr, u))
                continue
            kind, x = its[i]
            if kind == "a":
                if self.viable(fr, u):
                    out.add((fr, u))
                continue
            rem = self.sub(self.full, u)
            for f in x:
                if self.fits(self.vec[f], rem) and self.full_of(f, rem):
                    stack.append((fr + ((f, 0),), self.add(u, self.vec[f])))
        r = frozenset(out)
        self.m_close[key] = r
        return r

    def build(self, max_states=3_000_000):
        start = set()
        for k in range(len(self.vec)):
            if self.fits(self.vec[k], self.full) and self.full_of(k, self.full):
                start |= self.close(((k, 0),), tuple(self.vec[k]))
        start = frozenset(start)
        ids, edges, final = {start: 0}, [None], [False]
        todo = [start]
        while todo:
            S = todo.pop()
            sid = ids[S]
            nxt, fin = {}, False
            for fr, u in S:
                if not fr:
                    fin = True
                    continue
                k, i = fr[-1]
                tok = self.items[k][i][1]
                nxt.setdefault(tok, set()).update(self.close(fr[:-1] + ((k, i + 1),), u))
            e = {}
            for tok, T in nxt.items():
                T = frozenset(T)
                if T not in ids:
                    if len(ids) >= max_states:
                        raise RuntimeError(f"more than {max_states} states")
                    ids[T] = len(ids)
                    edges.append(None)
                    final.append(False)
                    todo.append(T)
                e[tok] = ids[T]
            edges[sid] = e
            final[sid] = fin
        return minimize(final, edges)


def minimize(final, edges):
    """Merge equivalent states of an acyclic DFA bottom-up; return (final, edges, counts) renumbered."""
    n = len(final)
    order, seen = [], [False] * n  # post-order (children first)
    for s0 in range(n):
        if seen[s0]:
            continue
        stack = [(s0, iter(edges[s0].values()))]
        seen[s0] = True
        while stack:
            s, it = stack[-1]
            for t in it:
                if not seen[t]:
                    seen[t] = True
                    stack.append((t, iter(edges[t].values())))
                    break
            else:
                stack.pop()
                order.append(s)
    rep, register = {}, {}
    for s in order:
        sig = (final[s], tuple(sorted((x, rep[t]) for x, t in edges[s].items())))
        rep[s] = register.setdefault(sig, s)
    keep = sorted(set(rep.values()), key=lambda s: s != rep[0])
    new = {s: i for i, s in enumerate([rep[0]] + [s for s in keep if s != rep[0]])}
    F = [0] * len(new)
    E = [None] * len(new)
    for s, i in new.items():
        F[i] = int(final[s])
        E[i] = sorted([x, new[rep[t]]] for x, t in edges[s].items())
    C = [None] * len(new)
    for s in order:
        if rep[s] == s:
            i = new[s]
            C[i] = F[i] + sum(C[t] for _, t in E[i])
    return F, E, C


def contains(B, tok_ids, seqs):
    """Membership test without enumeration: for each token-id sequence in seqs, does the bubble set B
    produce it? (tok_ids as in Det.) Returns a list of booleans."""
    D = Det(B, tok_ids)
    start = set()
    for k in range(len(D.vec)):
        if D.fits(D.vec[k], D.full) and D.full_of(k, D.full):
            start |= D.close(((k, 0),), tuple(D.vec[k]))
    out = []
    for seq in seqs:
        S = start
        for tok in seq:
            nxt = set()
            for fr, u in S:
                if fr and D.items[fr[-1][0]][fr[-1][1]][1] == tok:
                    k, i = fr[-1]
                    nxt |= D.close(fr[:-1] + ((k, i + 1),), u)
            S = nxt
            if not S:
                break
        out.append(any(not fr for fr, _ in S))
    return out
