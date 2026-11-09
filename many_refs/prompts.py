"""Prompts for the LLM annotation workflow (annotate -> check -> validate -> repair)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
FORMAT = """\
## Grammar format (JSON)
{"start": "S", "rules": {NAME: {"src": "source words covered", "alts": [ALTERNATIVE, ...]}, ...}}

ALTERNATIVE is a string, or {"seq": [ITEM, ...], "f": {FEATURE: VALUE, ...}}.
ITEM is one of:
  "text"                               literal Czech text (several words allowed)
  {"ref": NAME, "f": {FEATURE: VALUE}} a slot filled by rule NAME, restricted to its alternatives with matching features
  {"opt": [ITEM, ...]}                 optional part
  {"alt": [[ITEM, ...], [ITEM, ...]]}  inline choice between short sequences
Feature values are constants ("f", "pl", "acc", ...) or variables ("?g"). A variable is shared by
all slots of one alternative, so {"ref": "SUBJ", "f": {"g": "?g"}} ... {"ref": "VERB", "f": {"g": "?g"}}
makes subject and verb agree. An alternative may pass a feature up: {"f": {"g": "?x"}, "seq": [{"ref": "N", "f": {"g": "?x"}}]}.
An alternative without a feature accepts any value of it. Rules must not be recursive.
{"f": {...}, "alts": ["a", "b"]} inside a rule is shorthand for several alternatives sharing the same features.
Punctuation is written as separate items; spaces before , . : ; ! ? are removed automatically. Every
complete expansion must end with the sentence's final punctuation. The first letter of the sentence is
capitalised automatically, so a common word that may start the sentence can be written in lower case;
names and other words that are always capitalised keep their capital letters everywhere.
"""

GUIDE = """\
You are an expert English-to-Czech translator building a compact network that encodes ALL acceptable
Czech translations of one English sentence, as in the "bubble" annotation of Bojar et al. (2013).

Work like this:
1. Start from the main verb. Give its translation alternatives and slots for its arguments and adjuncts.
   Each argument gets its own rule ("bubble") with all its translations. Recurse into clauses.
2. Use features for every dependency between choices: subject-verb agreement (gender, number,
   person), the case a verb or preposition requires, adjective-noun agreement, aspect, register
   (formal "vy" vs informal "ty"), and anything else that makes some combinations wrong.
3. Cover the real variety a group of professional translators would produce: synonyms, different
   constructions (active/passive, verbal/nominal, clause/phrase), all acceptable word orders
   (Czech word order is free but not arbitrary: respect clitic placement and topic-focus),
   optional words that do not change the meaning, alternative punctuation where valid.
4. Every single expansion must be a grammatical, fluent Czech sentence with the same meaning as the
   source: nothing added, nothing lost. If some combination would be wrong, block it with features or
   split the alternatives; do not include it. Precision matters more than size.
5. Keep names, numbers and units correct; follow Czech typography („quotes“, decimal comma).
6. Fill "src" of each rule with the English words it translates; together the rules must cover the
   whole source sentence.
Aim for thousands to millions of expansions for a typical sentence; do not pad with near-duplicates.
Use features only where they block wrong combinations; a feature with the same value everywhere is useless.
"""


def _examples():
    out = []
    for p in sorted((HERE / "examples").glob("ex*.json")):
        d = json.loads(p.read_text())
        out.append(f"English: {d['en']}\nGrammar:\n{json.dumps(d['grammar'], ensure_ascii=False)}")
    return "\n\n".join(out)


def annotate(en, seeds=(), paraphrases=()):
    s = GUIDE + "\n" + FORMAT + "\n## Examples\n\n" + _examples() + "\n\n## Your sentence\n\nEnglish: " + en + "\n"
    if seeds:
        s += "\nExisting Czech translations, for inspiration only (they may contain errors):\n"
        s += "".join(f"- {t}\n" for t in seeds)
    if paraphrases:
        s += ("\nStructurally different translations collected earlier. Your grammar should produce every one of them "
              "that is correct, and generalise them: combine their choices wherever the combination is still correct.\n")
        s += "".join(f"- {t}\n" for t in paraphrases)
    return s + "\nReply with the grammar JSON only."


def repair(en, grammar, report, bad=(), missing=()):
    s = GUIDE + "\n" + FORMAT + "\n## Examples\n\n" + _examples() + f"\n\n## Your sentence\n\nEnglish: {en}\n\nYour current grammar:\n"
    s += json.dumps(grammar, ensure_ascii=False) + "\n"
    if report.get("error"):
        s += ("\nTHE GRAMMAR IS INVALID. The checker says:\n" + report["error"] +
              "\nFix exactly these problems first (e.g. define every rule you reference, use the format above).\n")
    elif report.get("warnings"):
        s += "\nChecker warnings:\n" + "".join(f"- {w}\n" for w in report["warnings"])
    if bad:
        s += "\nA reviewer rejected these expansions of your grammar (with reasons and the alternatives used):\n"
        for b in bad:
            s += f"- {b['text']}\n  reason: {b['reason']}\n  alternatives: {'; '.join(b['alts'])}\n"
        s += "\nFix the grammar so that none of these (and no similar ones) can be produced, without losing correct translations.\n"
    if missing:
        s += ("\nThese translations collected earlier are not produced by your grammar. Where they are correct, "
              "extend the grammar to produce them (and their correct combinations with existing choices):\n")
        s += "".join(f"- {t}\n" for t in missing)
    return s + "\nReply with the complete corrected grammar JSON only."


def validate(en, candidates):
    s = ("You are reviewing Czech translations of an English sentence from a news article. For each numbered "
         "candidate, decide whether a professional translator could reasonably have produced it.\n\n"
         "Meaning equivalence is NOT strict (as in Bojar et al. 2013): accept slight divergence if it can be "
         "reasonably expected that a human translator would pick either alternative. Accept idiomatic and "
         "restructured renderings, changed word order, synonyms, small shifts of emphasis or modality, making "
         "implicit things explicit or the reverse, and choices that would fit the article's context.\n"
         "Reject only real errors: ungrammatical or misspelled Czech (agreement, case, word forms, clitic order), "
         "clearly unnatural Czech, wrong punctuation that a proofreader would fix, wrong names or numbers, "
         "content that is missing, added or changed beyond slight divergence.\n\n"
         f"English: {en}\n\nCandidates:\n")
    s += "".join(f"{i}. {c}\n" for i, c in enumerate(candidates))
    return s + ('\nReply with a JSON list, one object per candidate, in order: '
                '[{"i": 0, "ok": true}, {"i": 1, "ok": false, "reason": "short reason"}, ...]')


def paraphrase(en, seeds=(), n=20):
    s = (f"Give {n} Czech translations of the English sentence below that differ from each other as much as "
         "possible: different sentence structures (e.g. existential vs. possessive, clause vs. phrase, active vs. "
         "passive), word orders and word choices. Every translation must be fully correct, fluent and faithful: "
         "nothing added, nothing lost.\n\nEnglish: " + en + "\n")
    if seeds:
        s += "\nSome translations to go beyond (they may contain errors):\n" + "".join(f"- {t}\n" for t in seeds)
    return s + '\nReply with a JSON list of strings only.'


def translate(en):
    return f"Translate the following English sentence into Czech. Reply with the translation only.\n\n{en}"
