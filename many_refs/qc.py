"""Quality control by repair (an addition to the paper's procedure).

qc_repair(): for one LLM translator's final clauses
  1. sample complete translations, recording which option (#k) produced which words; judge ~3 per option
     with a REVIEWER model (different from the evaluation reviewer); rejections quote the wrong words;
  2. hand the rejected translations, the reviewer's reasons and the options involved back to the
     TRANSLATOR model and ask for an ```edit block that fixes them - by conditions or small changes, not by
     deleting whole structures; an edit is accepted if the set still yields translations and does not
     shrink by more than half; up to --rounds rounds;
  3. fallback: an option still blamed afterwards in >= 2 rejected translations with different partner options
     (so the option itself, not a combination, is at fault) and in >= half of its judged uses is deleted.
The old deletion-only QC (blame by word overlap) removed correct reporting-clause alternatives for errors that
lay in the combination with another bubble; this version asks for the paper's own remedy, a condition.
"""
import json, random, re
from concurrent.futures import ThreadPoolExecutor

from bubbles import Bubbles, ParseError, parse, to_text, tool_report
from detok import detok
from llm_client import chat, connect

PROMPT = """You are reviewing Czech translations of an English sentence from a news article. The translations come from a
collection built like Bojar et al. (2013), where professional translators listed every rendering they would accept.
For each numbered candidate, decide whether such a translator could have listed it.

Be as tolerant as those translators were. Meaning equivalence is NOT strict: slight divergence is accepted if it can
be reasonably expected that a human translator would pick either alternative. Accept, without hesitation:
  - restructuring, changed word order, direct speech rendered as indirect speech, split or merged clauses;
  - synonyms and near-synonyms, idiomatic and colloquial or regional expressions, a slightly different verb of saying
    or of emotion, a shift of emphasis, register or modality, a different rendering of a contrast or a demonstrative
    (e.g. "tohle ... tohle", "to ... ono", "to ... tamto" all for "this ... that");
  - small omissions or additions of particles, connectives and intensifiers (prý, však, jen, totiž, zase, pak, ale,
    hned, přitom), making implicit things explicit or the reverse, a minor word left out or added when the
    information of the sentence is still there;
  - punctuation choices that are debatable rather than wrong (a comma before "a"/"ale" between clauses, a colon
    instead of a comma, the position of quotation marks).
Reject ONLY clear errors that any Czech proofreader would mark:
  - a non-word or misspelling, a wrong word form (case, gender, number agreement, verb form), wrong government of a
    verb or preposition, a doubled conjunction ("a ale"), unnatural clitic order;
  - broken punctuation (an unclosed quotation, a stray space before a comma, doubled marks);
  - a wrong name or number, a flipped negation, or content missing, added or changed so much that the sentence
    reports something else than the English one.

English: {en}

Candidates:
{cands}
Reply with a JSON list, one object per candidate, in order. For a rejected candidate give "wrong": the exact
wrong word(s) copied from the candidate (for missing content: the words next to the gap), and a short reason:
[{{"i": 0, "ok": true}}, {{"i": 1, "ok": false, "wrong": "...", "reason": "..."}}, ...]"""

REPAIR = """You wrote the bubbles below (Bojar et al. 2013 "option" clauses) for the Czech translations of the sentence:
  {en}
A reviewer rejected some of the translations your bubbles produce. For each rejected translation you see the
reviewer's reason and the options (#k) it was built from.

Fix the bubbles so that these wrong translations are no longer produced while keeping as many correct
translations as possible:
  - if the error lies in the COMBINATION of two options (punctuation before a reporting clause, case, gender,
    number agreement), add CONDITIONS: let the alternatives declare what they are (e.g. [comma_close] vs
    [stop_close], [fem] vs [masc]) and let the slots require the compatible one;
  - if one alternative is wrong Czech in itself, correct it or delete it;
  - do not delete whole structures or many alternatives because of one combination error.
Reply with ONE ```edit block, lines:   + option(...).   = 7 option(...).   - 7     (add / replace #7 / delete #7).

Your current clauses:
```prolog
{clauses}
```

Rejected translations:
{rejected}
"""


def parse_json(s):
    m = re.search(r"\[.*\]", s or "", re.S)
    return json.loads(m.group(0)) if m else None


def judge(client, model, en, texts, batch=20, think="medium", workers=8):
    def one(b):
        for _ in range(3):
            try:
                v = parse_json(chat(client, model, PROMPT.format(en=en, cands="".join(f"{j}. {t}\n" for j, t in enumerate(b))),
                                    think=think, max_tokens=65536))
                if isinstance(v, list) and len(v) == len(b):
                    return [dict(ok=bool(x.get("ok")), wrong=x.get("wrong", ""), reason=x.get("reason", "")) for x in v]
            except (ValueError, TypeError):
                pass
        return [dict(ok=None, wrong="", reason="unparsable")] * len(b)
    with ThreadPoolExecutor(workers) as ex:
        return [v for vs in ex.map(one, [texts[i:i + batch] for i in range(0, len(texts), batch)]) for v in vs]


def sample_per_option(B, per_option=3, seed=0):
    """Random complete translations covering every option at least per_option times when possible.
    Returns dict tokenised text -> derivation [(bubble index, own atoms), ...]."""
    rng = random.Random(seed)
    samples = {}
    for t, d in B.sample(3000, seed=seed, tries=30000):
        samples.setdefault(t, d)
    uses = {k: [t for t, d in samples.items() if any(x[0] == k for x in d)] for k in range(len(B.b))}
    for k in range(len(B.b)):
        if len(uses[k]) < per_option:
            for t, d in B.sample(per_option * 3, seed=seed + k + 1, prefer=k, tries=600):
                if any(x[0] == k for x in d):
                    samples.setdefault(t, d)
                    if t not in uses[k]:
                        uses[k].append(t)
    pick = set()
    for k in range(len(B.b)):
        pick.update(rng.sample(uses[k], min(per_option, len(uses[k]))))
    return {t: samples[t] for t in sorted(pick)}


def review(B, en, rclient, rmodel, think, seed):
    picked = sample_per_option(B, seed=seed)
    texts = list(picked)
    verdicts = judge(rclient, rmodel, en, [detok(t) for t in texts], think=think)
    rows = []
    for t, v in zip(texts, verdicts):
        rows.append(dict(text=detok(t), tok=t, ok=v["ok"], wrong=v.get("wrong") or "", reason=v.get("reason") or "",
                         options=sorted({B.orig[k] for k, _ in picked[t]})))
    return rows


def qc_repair(source, en, clauses, translator, reviewer="anthropic/claude-haiku-5.5", provider="openrouter", think="medium",
              rounds=3, verbose=False, tag=""):
    from annotate_paper import parse_edit, show_clauses
    from build_explorer2 import toks
    from determinize import contains
    rclient, rmodel = connect(reviewer, provider)
    tclient, tmodel = connect(translator, provider)
    opts, drops = parse(clauses)
    cur = show_clauses(opts, drops)
    n0 = Bubbles(source, cur).count()
    log, all_rows = [], []
    rows = review(Bubbles(source, cur), en, rclient, rmodel, think, seed=0)
    judged0 = sum(r["ok"] is not None for r in rows)
    rej0 = sum(r["ok"] is False for r in rows)
    for rnd in range(rounds):
        rejected = [r for r in rows if r["ok"] is False]
        all_rows += rows
        if not rejected:
            break
        B = Bubbles(source, cur)
        listing = "\n".join(f"{i + 1}. {r['text']}\n   reviewer: {r['reason']} (wrong: {r['wrong']})\n   built from options {', '.join('#' + str(k + 1) for k in r['options'])}"
                            for i, r in enumerate(rejected[:40]))
        prompt = REPAIR.format(en=en, clauses=cur, rejected=listing)
        accepted, note = False, ""
        for attempt in range(2):
            reply = chat(tclient, tmodel, prompt + ("\n\n" + note if note else ""), think=think, max_tokens=120000) or ""
            ed = re.findall(r"```edit\s*\n(.*?)```", reply, re.S)
            try:
                if not ed:
                    raise ParseError("no ```edit block in the reply")
                new_opts, new_drops = opts, drops
                for e in ed:
                    new_opts, new_drops = parse_edit(e, new_opts, new_drops)
                new = show_clauses(new_opts, new_drops)
                rep, _ = tool_report(source, new, None)
                n = rep.get("n") or 0
                if n == 0:
                    raise ParseError("the edited bubbles yield no complete translation:\n" + rep["text"][:1500])
                nb = Bubbles(source, cur).count()
                if n < 0.5 * nb:
                    raise ParseError(f"the edit shrinks the set from {nb} to {n} translations; fix the combinations with conditions instead of deleting")
            except ParseError as e:
                note = f"Your previous edit was not accepted: {e}\nTry again."
                continue
            # which rejected translations are gone?
            vocab = {}
            tid = lambda s: [vocab.setdefault(w, len(vocab)) for w in toks(s)]
            still = contains(Bubbles(source, new), tid, [tid(r["tok"]) for r in rejected])
            fixed = sum(not x for x in still)
            opts, drops, cur = new_opts, new_drops, new
            log.append(dict(round=rnd + 1, rejected=len(rejected), fixed=fixed, n_before=nb, n_after=n, edit="\n".join(ed)[:4000]))
            accepted = True
            if verbose:
                print(f"  [QC {tag}] round {rnd + 1}: {len(rejected)} rejected, edit removed {fixed}, size {nb} -> {n}", flush=True)
            break
        if not accepted:
            log.append(dict(round=rnd + 1, rejected=len(rejected), fixed=0, failed=note[:500]))
            break
        rows = review(Bubbles(source, cur), en, rclient, rmodel, think, seed=rnd + 1)
    else:
        all_rows += rows
    # fallback deletion: options at fault by themselves
    B = Bubbles(source, cur)
    final_rows = rows
    rejected = [r for r in final_rows if r["ok"] is False]
    blame, partners, used = {}, {}, {}
    for r in final_rows:
        for k in r["options"]:
            used[k] = used.get(k, 0) + 1
    for r in rejected:
        wrong = {w.lower() for w in re.findall(r"\w+|[^\w\s]", r["wrong"])}
        if len(wrong) > 0.5 * len(r["tok"].split()):
            continue
        for k in r["options"]:
            own = {a.lower() for a in re.findall(r"\w+|[^\w\s]", to_text([B.options[k]]).split(", [")[-1] if False else "")}
            # blame by overlap with the option's output atoms
            atoms = {str(x).lower() for x in B.options[k][2] if isinstance(x, str)}
            if wrong & atoms:
                blame[k] = blame.get(k, 0) + 1
                partners.setdefault(k, set()).add(frozenset(r["options"]) - {k})
    remove = sorted(k for k in blame if blame[k] >= 2 and len(partners[k]) >= 2 and blame[k] >= 0.5 * used.get(k, 1))
    removed = []
    if remove:
        keep = [o for i, o in enumerate(B.options) if i not in set(remove)]
        new = show_clauses(keep, B.drops)
        if keep and Bubbles(source, new).count() > 0:
            removed = [to_text([B.options[i]]) for i in remove]
            cur = new
    n1 = Bubbles(source, cur).count()
    rej1 = sum(r["ok"] is False for r in final_rows)
    judged1 = sum(r["ok"] is not None for r in final_rows)
    summary = f"rejected {rej0}/{judged0} -> {rej1}/{judged1}, {len(log)} repair rounds, {len(removed)} deleted, size {n0} -> {n1}"
    return dict(clauses=cur, applied=cur != clauses, reviewer=rmodel, rounds=log, removed=removed, rows=all_rows,
                judged=judged0, rejected=rej0, judged_final=judged1, rejected_final=rej1, n_before=n0, n_after=n1, summary=summary)
