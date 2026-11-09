"""Tests for grammar.py: python -m pytest -q test_grammar.py (or just run it)."""
import json
from grammar import Grammar, GrammarError, report


def g(rules, start="S"):
    return Grammar({"start": start, "rules": rules})


def raises(rules):
    try:
        g(rules)
    except GrammarError as e:
        return str(e)
    raise AssertionError("no error")


def test_errors():
    assert "undefined" in raises({"S": [{"seq": [{"ref": "X"}]}]})
    assert "recursive" in raises({"S": [{"seq": [{"ref": "A"}]}], "A": [{"seq": ["a", {"ref": "S"}]}]})
    assert "unknown item" in raises({"S": [{"seq": [42]}]})
    assert "start" in raises({"A": ["a"]})


def test_agreement_and_counts():
    G = g({"S": [{"seq": [{"ref": "N", "f": {"g": "?g"}}, {"ref": "V", "f": {"g": "?g"}}]}],
           "N": [{"f": {"g": "f"}, "seq": ["žena"]}, {"f": {"g": "m"}, "seq": ["muž"]}, "dítě"],
           "V": [{"f": {"g": "f"}, "seq": ["přišla"]}, {"f": {"g": "m"}, "seq": ["přišel"]}]})
    out = sorted(t for t, _ in G.expand())
    # "dítě" has no g: ?g stays unbound, both verbs allowed (annotator's responsibility)
    assert out == ["Dítě přišel", "Dítě přišla", "Muž přišel", "Žena přišla"], out
    assert G.count() == 4


def test_pass_up_and_opt_alt():
    G = g({"S": [{"seq": [{"ref": "NP", "f": {"g": "?g"}}, {"ref": "V", "f": {"g": "?g"}}, "."]}],
           "NP": [{"f": {"g": "?x"}, "seq": [{"opt": [{"ref": "A", "f": {"g": "?x"}}]}, {"ref": "N", "f": {"g": "?x"}}]}],
           "A": [{"f": {"g": "f"}, "seq": [{"alt": [["malá"], ["mladá"]]}]}, {"f": {"g": "m"}, "seq": ["malý"]}],
           "N": [{"f": {"g": "f"}, "seq": ["žena"]}, {"f": {"g": "m"}, "seq": ["muž"]}],
           "V": [{"f": {"g": "f"}, "seq": ["přišla"]}, {"f": {"g": "m"}, "seq": ["přišel"]}]})
    out = sorted(t for t, _ in G.expand())
    assert out == ["Malá žena přišla.", "Malý muž přišel.", "Mladá žena přišla.", "Muž přišel.", "Žena přišla."], out
    assert G.count() == 5


def test_report_dead_and_ambiguous():
    G = g({"S": [{"seq": [{"ref": "N", "f": {"g": "f"}}]}], "N": [{"f": {"g": "m"}, "seq": ["muž"]}, {"f": {"g": "f"}, "seq": ["žena"]}]})
    r = report(G)
    assert any("never used" in w for w in r["warnings"]), r
    G = g({"S": [{"seq": [{"opt": [""]}, "a"]}]})
    assert report(G)["ambiguous_share"] == 0.5


def test_examples():
    for e in ["ex1", "ex2"]:
        d = json.load(open(f"examples/{e}.json"))
        r = report(Grammar(d["grammar"]), d["en"])
        assert r["ok"] and not r["warnings"] and r["ambiguous_share"] == 0, r


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
