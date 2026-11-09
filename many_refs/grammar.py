"""Translation grammars ("bubbles"): a JSON format, a checker, a counter and an expander.

Format (what the annotator LLM writes):
{
  "start": "S",
  "rules": {
    "S":    [ {"seq": [{"ref": "SUBJ", "f": {"g": "?g"}}, {"ref": "VERB", "f": {"g": "?g"}}, "novou vyhlášku", "."]} ],
    "SUBJ": [ {"f": {"g": "f"}, "seq": ["Městská rada"]},
              {"f": {"g": "n"}, "seq": ["Zastupitelstvo města"]} ],
    "VERB": [ {"f": {"g": "f"}, "seq": ["schválila"]},
              {"f": {"g": "n"}, "seq": ["schválilo"]} ]
  }
}
- A rule is a non-empty list of alternatives. An alternative is a string (shorthand for
  {"seq": [string]}) or {"seq": [...], "f": {feature: value}}.
- Sequence items: a string (literal text, may be "" for "nothing"),
  {"ref": NAME, "f": {...}} (expand rule NAME), {"opt": [items]} (optional),
  {"alt": [[items], [items], ...]} (inline choice).
- Features: values are constants ("f", "sg") or variables ("?g"), scoped to one
  alternative. {"ref": X, "f": {"g": "?g"}} only allows X's alternatives whose g is
  compatible, and binds ?g to the value the chosen alternative has. An alternative's own
  "f" may use variables to pass a feature up ({"f": {"g": "?g"}, "seq": [{"ref": "N",
  "f": {"g": "?g"}}]}). An alternative without a feature key accepts any value for it.
- A rule may instead be {"src": "source words it covers", "alts": [...]}; the checker
  then reports source words no rule covers.
- {"f": F, "alts": [ALT, ...]} inside a rule is shorthand for several alternatives sharing F.
- Rules must not be recursive. The first letter of each output is capitalised.
"""
import argparse, itertools, json, math, random, re, sys
from collections import Counter, defaultdict

PUNCT_BEFORE = re.compile(r"\s+([,.:;!?…“)\]%])")
PUNCT_AFTER = re.compile(r"([„(\[])\s+")


def join(pieces):
    s = " ".join(p for p in pieces if p)
    s = PUNCT_AFTER.sub(r"\1", PUNCT_BEFORE.sub(r"\1", s))
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r",(\s*,)+", ",", s)  # two adjacent rules both adding a comma
    m = re.search(r"\w", s)  # sentence-initial capital
    return s[:m.start()] + s[m.start()].upper() + s[m.start() + 1:] if m else s


def is_var(v):
    return isinstance(v, str) and v.startswith("?")


class GrammarError(Exception):
    pass


class Grammar:
    def __init__(self, g):
        self.errors, self.warnings = [], []
        if not isinstance(g, dict) or "rules" not in g:
            raise GrammarError("top level must be an object with 'rules' (and 'start')")
        self.start = g.get("start", "S")
        self.rules, self.src = {}, {}
        for name, alts in g["rules"].items():
            if isinstance(alts, dict):  # {"src": "words covered", "alts": [...]}
                self.src[name] = alts.get("src", "")
                alts = alts.get("alts")
            if not isinstance(alts, list) or not alts:
                self.errors.append(f"rule {name}: must be a non-empty list of alternatives")
                continue
            flat = []  # {"f": F, "alts": [a, b]} inside a rule = several alternatives sharing F
            for a in alts:
                if isinstance(a, dict) and "alts" in a and "seq" not in a and isinstance(a["alts"], list):
                    for b in a["alts"]:
                        b = {"seq": [b]} if isinstance(b, str) else dict(b)
                        b["f"] = {**(a.get("f") or {}), **(b.get("f") or {})}
                        flat.append(b)
                else:
                    flat.append(a)
            self.rules[name] = [self._alt(name, i, a) for i, a in enumerate(flat)]
        if self.start not in self.rules:
            self.errors.append(f"start rule {self.start!r} is not defined")
        self._check_refs()
        if self.errors:
            raise GrammarError("; ".join(self.errors))
        self._memo = {}

    # ---------- normalisation & static checks ----------
    def _alt(self, name, i, a):
        where = f"{name}[{i}]"
        if isinstance(a, str):
            return {"f": {}, "seq": [("lit", a)], "id": (name, i)}
        if not isinstance(a, dict) or "seq" not in a:
            self.errors.append(f"{where}: alternative must be a string or an object with 'seq'")
            return {"f": {}, "seq": [], "id": (name, i)}
        f = a.get("f", {}) or {}
        if not isinstance(f, dict) or not all(isinstance(v, str) for v in f.values()):
            self.errors.append(f"{where}: 'f' must map feature names to strings")
            f = {}
        return {"f": f, "seq": self._seq(where, a["seq"]), "id": (name, i)}

    def _seq(self, where, items):
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list):
            self.errors.append(f"{where}: 'seq' must be a list")
            return []
        out = []
        for it in items:
            if isinstance(it, str):
                out.append(("lit", it))
            elif isinstance(it, dict) and "ref" in it:
                f = it.get("f", {}) or {}
                if not isinstance(f, dict):
                    self.errors.append(f"{where}: 'f' of ref {it['ref']} must be an object")
                    f = {}
                out.append(("ref", it["ref"], f))
            elif isinstance(it, dict) and "opt" in it:
                out.append(("opt", self._seq(where, it["opt"])))
            elif isinstance(it, dict) and "alt" in it and isinstance(it["alt"], list) and it["alt"]:
                out.append(("alt", [self._seq(where, s) for s in it["alt"]]))
            else:
                self.errors.append(f"{where}: unknown item {json.dumps(it, ensure_ascii=False)[:80]}")
        return out

    def _items(self, seq):
        for it in seq:
            yield it
            if it[0] == "opt":
                yield from self._items(it[1])
            elif it[0] == "alt":
                for s in it[1]:
                    yield from self._items(s)

    def _check_refs(self):
        deps = defaultdict(set)
        for name, alts in self.rules.items():
            for a in alts:
                bound_somewhere = {v for v in a["f"].values() if is_var(v)}
                ref_vars = Counter()
                for it in self._items(a["seq"]):
                    if it[0] == "ref":
                        if it[1] not in self.rules:
                            self.errors.append(f"{name}: reference to undefined rule {it[1]!r}")
                        deps[name].add(it[1])
                        ref_vars.update(v for v in it[2].values() if is_var(v))
                for v in bound_somewhere - set(ref_vars):
                    self.warnings.append(f"{name}: variable {v} in 'f' is never bound by a ref")
                for v, c in ref_vars.items():
                    if c == 1 and v not in bound_somewhere:
                        self.warnings.append(f"{name}: variable {v} is used only once (no agreement enforced)")
        # recursion
        state = {}

        def visit(n, path):
            state[n] = 1
            for m in deps.get(n, ()):
                if state.get(m) == 1:
                    self.errors.append("recursive rules: " + " -> ".join(path + [m]))
                elif m in self.rules and not state.get(m):
                    visit(m, path + [m])
            state[n] = 2

        for n in self.rules:
            if not state.get(n):
                visit(n, [n])
        reach, todo = {self.start}, [self.start]
        while todo:
            for m in deps.get(todo.pop(), ()):
                if m not in reach:
                    reach.add(m)
                    todo.append(m)
        for n in self.rules:
            if n not in reach:
                self.warnings.append(f"rule {n} is never used")

    # ---------- unification helpers ----------
    @staticmethod
    def _init_env(alt_f, cons):
        env = {}
        for k, c in cons.items():
            v = alt_f.get(k)
            if v is None:
                continue
            if is_var(v):
                if env.get(v, c) != c:
                    return None
                env[v] = c
            elif v != c:
                return None
        return env

    @staticmethod
    def _cons(ref_f, env):
        out = {}
        for k, v in ref_f.items():
            if is_var(v):
                if v in env:
                    out[k] = env[v]
            else:
                out[k] = v
        return out

    @staticmethod
    def _bind(env, ref_f, feats):
        new = None
        for k, v in ref_f.items():
            if is_var(v) and v not in env and k in feats:
                new = new or dict(env)
                new[v] = feats[k]
        return new or env

    @staticmethod
    def _resolve(alt_f, env):
        out = {}
        for k, v in alt_f.items():
            if is_var(v):
                if v in env:
                    out[k] = env[v]
            else:
                out[k] = v
        return out

    # ---------- counting (no enumeration) ----------
    def dist(self, name, cons=None):
        """{frozen features of the expansion: number of derivations}"""
        cons = cons or {}
        key = (name, frozenset(cons.items()))
        if key not in self._memo:
            out = Counter()
            for a in self.rules[name]:
                env = self._init_env(a["f"], cons)
                if env is None:
                    continue
                for env2, c in self._dist_seq(a["seq"], {frozenset(env.items()): 1}).items():
                    out[frozenset(self._resolve(a["f"], dict(env2)).items())] += c
            self._memo[key] = out
        return self._memo[key]

    def _dist_seq(self, seq, states):
        for it in seq:
            new = Counter()
            if it[0] == "lit":
                continue
            if it[0] == "ref":
                for fenv, c in states.items():
                    env = dict(fenv)
                    for feats, c2 in self.dist(it[1], self._cons(it[2], env)).items():
                        new[frozenset(self._bind(env, it[2], dict(feats)).items())] += c * c2
            elif it[0] == "opt":
                new.update(states)
                new.update(self._dist_seq(it[1], states))
            elif it[0] == "alt":
                for s in it[1]:
                    new.update(self._dist_seq(s, states))
            states = new
        return states

    def count(self):
        return sum(self.dist(self.start).values())

    # ---------- enumeration ----------
    def expand(self, name=None, cons=None, deriv=False):
        """Yield (text, derivation) for every derivation; derivation = tuple of alternative ids."""
        for pieces, _, d in self._gen_rule(name or self.start, cons or {}):
            yield join(pieces), (d if deriv else None)

    def _gen_rule(self, name, cons):
        for a in self.rules[name]:
            env = self._init_env(a["f"], cons)
            if env is None:
                continue
            for pieces, env2, d in self._gen_seq(a["seq"], 0, env):
                yield pieces, self._resolve(a["f"], env2), (a["id"],) + d

    def _gen_seq(self, seq, i, env):
        if i == len(seq):
            yield [], env, ()
            return
        it = seq[i]
        if it[0] == "lit":
            heads = [([it[1]], env, ())]
        elif it[0] == "ref":
            heads = ((p, self._bind(env, it[2], f), d) for p, f, d in self._gen_rule(it[1], self._cons(it[2], env)))
        elif it[0] == "opt":
            heads = itertools.chain([([], env, ())], self._gen_seq(it[1], 0, env))
        else:
            heads = itertools.chain.from_iterable(self._gen_seq(s, 0, env) for s in it[1])
        for p, env2, d in heads:
            for p2, env3, d2 in self._gen_seq(seq, i + 1, env2):
                yield p + p2, env3, d + d2

    def sample(self, k, seed=0, deriv=False, max_enum=5_000_000):
        """k distinct strings drawn uniformly from the distinct expansions."""
        n = self.count()
        if n > max_enum:
            raise GrammarError(f"{n} derivations; too many to enumerate (limit {max_enum})")
        seen = {}
        for text, d in self.expand(deriv=deriv):
            seen.setdefault(text, d)
        rng = random.Random(seed)
        keys = rng.sample(sorted(seen), min(k, len(seen)))
        return [(t, seen[t]) for t in keys]

    # ---------- serialisation (normalised form) ----------
    @staticmethod
    def _seq_json(seq):
        out = []
        for it in seq:
            if it[0] == "lit":
                out.append(it[1])
            elif it[0] == "ref":
                out.append({"ref": it[1], "f": it[2]} if it[2] else {"ref": it[1]})
            elif it[0] == "opt":
                out.append({"opt": Grammar._seq_json(it[1])})
            else:
                out.append({"alt": [Grammar._seq_json(x) for x in it[1]]})
        return out

    def to_json(self, drop=()):
        """The grammar as JSON, without the alternatives whose ids are in `drop`."""
        rules = {}
        for name, alts in self.rules.items():
            keep = [{"f": a["f"], "seq": self._seq_json(a["seq"])} if a["f"] else {"seq": self._seq_json(a["seq"])}
                    for a in alts if a["id"] not in drop]
            r = {"alts": keep}
            if self.src.get(name):
                r["src"] = self.src[name]
            rules[name] = r
        return {"start": self.start, "rules": rules}

    def alt_label(self, aid):
        name, i = aid
        a = self.rules[name][i]
        return f"{name}[{i}]: " + " ".join(t[1] if t[0] == "lit" else ("<" + t[1] + ">" if t[0] == "ref" else "(...)")
                                           for t in a["seq"])


STOP = set("a an the of to and or in on at for by with is are was were be been it its this that "
           "these those as from but not".split())


def uncovered(g, source):
    words = {w.lower() for w in re.findall(r"[\w'’-]+", source)} - STOP
    covered = {w.lower() for s in g.src.values() for w in re.findall(r"[\w'’-]+", s)}
    return sorted(words - covered)


def report(g, source=None, enum_limit=2_000_000):
    """Static check plus counts; returns a dict suitable for feeding back to the annotator."""
    out = dict(ok=True, warnings=list(g.warnings), derivations=g.count())
    if source:  # every source word must be translated by some rule (as in the paper's Prolog tool)
        miss = uncovered(g, source)
        if not g.src:
            out["ok"] = False
            out["error"] = "no rule has a 'src' field: list the English words each rule translates"
        elif miss:
            out["ok"] = False
            out["error"] = ("source words not covered by any rule's 'src' (translate them, or add them to the src "
                            "of the rule that already translates them): " + ", ".join(miss))
    out["log10_derivations"] = round(math.log10(out["derivations"]), 2) if out["derivations"] else None
    if out["derivations"] == 0:
        out["ok"] = False
        out["error"] = "the grammar produces no output (feature constraints never satisfiable?)"
        return out
    elif out["derivations"] <= enum_limit:
        texts = Counter(t for t, _ in g.expand())
        out["distinct"] = len(texts)
        out["ambiguous_share"] = round(1 - len(texts) / out["derivations"], 3)
        if out["ambiguous_share"] > 0.2:
            out["warnings"] = out["warnings"] + [f"{out['ambiguous_share']:.0%} of derivations duplicate another "
                                                 "(overlapping alternatives, e.g. an empty option plus an empty alternative)"]
    # dead alternatives: never usable under any constraint reachable from start
    used = set()
    if out["derivations"] and out["derivations"] <= enum_limit:
        for _, d in g.expand(deriv=True):
            used.update(d)
        dead = [g.alt_label(a["id"]) for alts in g.rules.values() for a in alts if a["id"] not in used]
        if dead:
            out["warnings"] = out["warnings"] + ["alternatives never used (feature clash?): " + "; ".join(dead[:10])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "expand", "sample"])
    ap.add_argument("file")
    ap.add_argument("-k", type=int, default=20)
    ap.add_argument("--source", help="English source, for the coverage check")
    a = ap.parse_args()
    data = json.load(open(a.file, encoding="utf-8"))
    data_src = data.get("en")
    data = data.get("grammar", data)
    try:
        g = Grammar(data)
    except GrammarError as e:
        print(json.dumps(dict(ok=False, error=str(e)), ensure_ascii=False))
        sys.exit(1)
    if a.cmd == "check":
        print(json.dumps(report(g, a.source or data_src), ensure_ascii=False, indent=1))
    elif a.cmd == "expand":
        for t, _ in g.expand():
            print(t)
    else:
        for t, _ in g.sample(a.k):
            print(t)


if __name__ == "__main__":
    main()


def union(grammars, prefixes=None):
    """One grammar producing the union of the outputs of several grammars (rules renamed apart)."""
    rules, start_alts = {}, []
    for k, gj in enumerate(grammars):
        pre = (prefixes or [f"A{i + 1}_" for i in range(len(grammars))])[k]
        g = Grammar(gj)
        ren = lambda n: pre + n

        def seq(items):
            out = []
            for it in items:
                if isinstance(it, dict) and "ref" in it:
                    out.append({**it, "ref": ren(it["ref"])})
                elif isinstance(it, dict) and "opt" in it:
                    out.append({"opt": seq(it["opt"])})
                elif isinstance(it, dict) and "alt" in it:
                    out.append({"alt": [seq(x) for x in it["alt"]]})
                else:
                    out.append(it)
            return out

        for name, r in g.to_json()["rules"].items():
            rules[ren(name)] = dict(r, alts=[dict(a, seq=seq(a["seq"])) for a in r["alts"]])
        start_alts.append({"seq": [{"ref": ren(g.start)}]})
    rules["UNION"] = {"alts": start_alts}
    return {"start": "UNION", "rules": rules}
