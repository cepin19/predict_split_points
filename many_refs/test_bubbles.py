"""Tests for bubbles.py: python3 test_bubbles.py"""
from bubbles import Bubbles, tool_report, parse, ParseError

PAPER = """
% option(+SrcWordsCovered, +ConditionsMet, +OutputAtomsAndSlots).
option([the, city, council], [], [městská, rada, [approved, fem]]).
option([approved], [fem], [schválila, [regulation, acc]]).
option([approved], [fem], [souhlasila, s, [regulation, ins]]).
option([a, new, regulation], [acc], [nový, předpis]).
option([a, new, regulation], [ins], [novým, předpisem]).
option([a, new, regulation], [acc], [novou, směrnici]).
option([a, new, regulation], [ins], [novou, směrnicí]).
"""
SRC = "The city council approved a new regulation"


def test_paper_example():
    b = Bubbles(SRC, PAPER)
    assert not b.errors, b.errors
    assert b.count() == 4
    assert b.translations() == {"městská rada schválila nový předpis", "městská rada schválila novou směrnici",
                                "městská rada souhlasila s novým předpisem", "městská rada souhlasila s novou směrnicí"}


def test_coverage_required():
    # without the "a new regulation" bubbles nothing covers those words -> no translation
    b = Bubbles(SRC, "\n".join(l for l in PAPER.splitlines() if "new" not in l))
    assert b.count() == 0


def test_drop_and_or_and_punct():
    src = "The council approved it ."
    txt = """option([the, council], [], [rada, [approved, or(fem, any)]]).
    option([approved, '.'], [fem], [ji, schválila, '.']).
    option([approved], [any], [schválila, [it]]).
    option([it], [], [to]).
    drop([it])."""
    b = Bubbles(src, txt)
    assert not b.errors, b.errors
    # "rada ji schválila ." covers the,council,approved,. and drops "it"; "rada schválila to" leaves "." uncovered
    assert b.translations() == {"rada ji schválila ."}, b.translations()


def test_errors():
    b = Bubbles(SRC, "option([the, city, council, mayor], [], [rada]).\noption([approved],[],[x, [regulation, acc]]).")
    assert any("not tokens" in e for e in b.errors)
    assert any("no option can fill" in e for e in b.errors)
    try:
        parse("option([a, b], [], [c, [d]")
        assert False
    except ParseError:
        pass
    rep, _ = tool_report(SRC, PAPER)
    assert "All translations" in rep["text"] and rep["n"] == 4


def test_duplicate_source_tokens():
    src = "the cat saw the dog"
    txt = """option([cat], [], [kočka, [saw]]).
    option([saw], [], [viděla, [dog]]).
    option([dog], [], [psa]).
    option([the], [], []).
    option([the, cat], [], [ta, kočka, [saw]])."""
    b = Bubbles(src, txt)
    # each "the" must be covered; the bare option([the]) can't be used (nobody slots it), so only "ta kočka ..." covers
    # one "the" and the other stays uncovered -> no complete translation; with drop([the]) the first works too
    assert b.count() == 0
    b = Bubbles(src, txt + "\ndrop([the, the]).")
    assert b.translations() == {"kočka viděla psa", "ta kočka viděla psa"}, b.translations()




def test_drop_covers_every_occurrence():
    """drop(['"']) must allow both quotation marks of the source to stay untranslated."""
    src = '"Who is Voldemort?"'
    text = """drop(['"']).
option([who, is], [], ['Kdo', je, [voldemort], [?]]).
option([voldemort], [], ['Voldemort']).
option([?], [], ['?']).
option(['"', who, is, voldemort, '?', '"'], [], ['„', 'Kdo', je, 'Voldemort', '?', '“'])."""
    b = Bubbles(src, text)
    assert not b.errors, b.errors
    tr = b.translations()
    assert "Kdo je Voldemort ?" in tr and "„ Kdo je Voldemort ? “" in tr, tr
    assert b.count() == 2
    rep, _ = tool_report(src, text)
    assert "not covered" not in rep["text"] and rep["n"] == 2, rep["text"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
