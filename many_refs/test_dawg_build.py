"""dawg_build.build_dawg2 must accept exactly the distinct normalised translations of a bubble set: checked against
enumeration and against determinize.Det on small sets.   ../env_vllm/bin/python test_dawg_build.py"""
from bubbles import Bubbles
from build_explorer2 import toks
from determinize import Det
from dawg_build import build_dawg, build_dawg2
from test_bubbles import PAPER, SRC
from to_distinct import union

QUOTES_SRC = 'He said : " Yes . "'
QUOTES = """
option([he, said, ':'], [], [řekl, ':', ['"', open], [yes], ['.'], ['"', close]]).
option([he, said, ':'], [], [prohlásil, ':', ['"', open], [yes], ['.'], ['"', close]]).
option(['"'], [open], ['„']).
option(['"'], [open], ['"']).
option(['"'], [close], ['“']).
option(['"'], [close], ['"']).
option([yes], [], [ano]).
option([yes], [], [jo]).
option(['.'], [], ['.']).
"""
DROP_SRC = "The council approved it ."
DROP = """option([the, council], [], [rada, [approved, or(fem, any)]]).
option([approved, '.'], [fem], [ji, schválila, '.']).
option([approved], [any], [schválila, [it]]).
option([it], [], [to]).
drop([it])."""


def walk(F, E, s=0, pre=()):
    if F[s]:
        yield pre
    for t, c in E[s]:
        yield from walk(F, E, c, pre + (t,))


def check(src, clauses, expect_distinct=None):
    B = Bubbles(src, clauses)
    assert not B.errors, B.errors
    ids = {}
    tid = lambda a: [ids.setdefault(w, len(ids)) for w in toks(a)]
    want = {tuple(ids.setdefault(w, len(ids)) for w in toks(t)) for t in B.translations()}
    assert want, "fixture yields no translation"
    F, E, C = build_dawg2(B, tid)
    assert set(walk(F, E)) == want, (sorted(walk(F, E)), sorted(want))
    assert C[0] == len(want)
    F1, E1, C1 = build_dawg(B, tid)
    F2, E2, C2 = Det(B, tid).build()
    assert C1[0] == C2[0] == C[0] and len(F1) == len(F2) == len(F), "same minimal size as the subset construction"
    Fu, Eu, Cu = union([dict(final=F, edges=E), dict(final=F2, edges=E2)])
    assert Cu[0] == C[0], "same language as the subset construction"
    if expect_distinct is not None:
        assert C[0] == expect_distinct, C[0]
    print(f"OK {src!r}: {B.count()} derivations, {C[0]} distinct normalised translations, {len(F)} states")


def test_paper_example():
    check(SRC, PAPER, 4)  # schválila + 2 acc, souhlasila s + 2 ins


def test_quote_variants_merge():
    check(QUOTES_SRC, QUOTES, 4)  # 2 verbs x 2 yes x 2 x 2 quote styles = 16 derivations, 4 distinct after normalisation


def test_drop_and_conditions():
    check(DROP_SRC, DROP, 1)


if __name__ == "__main__":
    test_paper_example()
    test_quote_variants_merge()
    test_drop_and_conditions()
