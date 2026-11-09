"""The annotation procedure of Bojar et al. (2013), with LLMs in place of the human translators.

Per sentence, each LLM translator (its own conversation, independent of the others like the paper's annotators):
  - gets exactly the paper's inputs: the English source sentence and the Czech translations the annotators
    had - the official WMT11 reference (O), three translations from the German version (G) and post-edited
    MT outputs (P); G and P are not public, so make_inputs.py recreates them with LLMs (inputs/<sid>.json).
    Nothing else (the German/Spanish/French versions of the sentence only with --other-langs; a translator
    working from another language only with --src-langs);
  - writes bubbles as option(...) clauses (the paper's Prolog interface) and keeps editing them: either
    the complete file, or edits (add / replace #k / delete #k), like a person editing the clause file;
    "undo" restores the previous version;
  - after every version sees the tool's output (bubbles.py, re-implementing the Prolog tool): errors,
    the number of translations, the full list (short sentences) or the differences against the previous
    state, like the web interface, plus diagnostics a human would notice by eye: how factorised the file is
    (options, slots, translations per option) and options that add nothing (identical or redundant);
  - a version that yields no translation is rejected and the last valid version stays current; an addition
    that yields no new translation is rejected too (in paper_v5 one translator spent 16 of 19 runs re-adding
    the same 18 sentences); a version that shrinks the set by more than half triggers a warning;
  - keeps refining until it replies DONE (not accepted before --min-runs tool runs) or uses up its
    budget (--max-runs tool runs; the paper's annotators had at most 2 hours per sentence).
Each turn the translator sees the instructions, its current clauses, the latest tool output and a
one-line-per-run log (like a human with the file and the tool; this also keeps long sessions within the
context window). --skeletons adds a first step not in the paper (listing structurally different full
translations first, checked by the tool after every run); it made gpt-6-luna list sentences instead of
building bubbles in paper_v5, so it is off by default. The LLM translators' sets are combined by simple union,
as the paper unites its annotators (who overlapped by 4.4%); the default is two independent sessions per
model. Optional quality control (qc.py) repairs reviewer-rejected translations by asking the translator for edits.

  ../env_vllm/bin/python annotate_paper.py --tag paper_v6 --sids 7 70 112 --min-runs 40
Writes runs/<tag>/<sid>.json (transcripts, final clauses) and runs/<tag>/datafile.tsv in the format of the
released many-references-datafile (sentence, annotators, interface, tokenised Czech string).
"""
import argparse, json, re, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bubbles import Bubbles, ParseError, parse, to_text, tokenize_source, tool_report
from llm_client import connect

NT = "wmt11_dl/dev/newstest2011"
LANGS = {"de": "German", "es": "Spanish", "fr": "French", "en": "English"}

PAPER_EXAMPLE = """% option(+SrcWordsCovered, +ConditionsMet, +OutputAtomsAndSlots).
option([the, city, council], [], [městská, rada, [approved, fem]]).
  % "The city council" can be translated as "městská rada" and then
  % it requires the translation of "approved" in feminine gender.
option([approved], [fem], [schválila, [regulation, acc]]).
option([approved], [fem], [souhlasila, s, [regulation, ins]]).
  % Different translations of "approved" require "regulation" in
  % either "acc"usative or "ins"trumental cases.
option([a, new, regulation], [acc], [nový, předpis]).
option([a, new, regulation], [ins], [novým, předpisem]).
option([a, new, regulation], [acc], [novou, směrnici]).
option([a, new, regulation], [ins], [novou, směrnicí])."""

GUIDE = """You are a professional translator taking part in an annotation project (Bojar et al., 2013,
"Scratching the Surface of Possible Translations"). The goal is to produce as many correct Czech
translations of one {lang} sentence as possible - ideally the whole space of translations a professional
translator would consider - in a compact representation that a tool expands into all of them.

The representation: bubbles. A bubble is defined by
  - the set (possibly discontinuous) of source tokens it covers;
  - the set of conditions it meets;
  - the set of translation alternatives in Czech; every alternative in the set covers the same source tokens
    and meets the same conditions.
A translation alternative consists of atoms (Czech tokens) and slots. A slot is a position to be filled by
another bubble: it names at least one source word that the filling bubble must cover and optionally further
conditions the filling bubble must meet (e.g. that its form is in the accusative). A bubble lists only the
source words it translates itself, not those translated by the bubbles in its slots: in the example below,
the first bubble covers "the city council" only, and "approved" is covered by the bubble filling its slot.
Conditions are not restricted in any way: morphological categories, but also anything else you need
(co-reference, auxiliaries in coordination, negation, style, register, punctuation agreement ...).

You write each bubble alternative as a clause of the predicate option():
  option(SrcWordsCovered, ConditionsMet, OutputAtomsAndSlots).
Example:
{example}

Further constructs: or(c1, c2) among a slot's conditions is met by a bubble meeting c1 or c2;
drop([word, ...]). declares source tokens that may be left untranslated - use it whenever a restructuring
makes a source token disappear: direct speech turned into indirect speech drops the quotation marks (and
often the colon), a colon can become a comma, a dash can vanish, etc. Dropped tokens simply need no bubble.
The tool checks the constraints (bubble-slot compatibility) and makes sure that all tokens of the source
sentence - including punctuation, unless dropped - are covered: a translation is any combination of bubbles,
starting from any bubble and filling all its slots recursively, that covers every source token exactly once.
Write source tokens in lower case as they appear in the token list below; write Czech atoms with correct
capitalisation and diacritics, one token per atom, with punctuation as separate atoms in quotes (e.g. ',' '.' '"').

Use conditions to keep combinations correct. Typical case, reported speech: the closing of the quote and the
reporting clause must agree - „..., “ říká X   but   „... . “ X říká only when the sentence ends there.
Encode this as a condition, e.g. the quote-closing bubble meets [comma_close] or [stop_close] and the
reporting-clause bubble asks for the slot [...quote..., comma_close]. Never leave a wrong combination
reachable: every translation the bubbles produce must be correct Czech.

How to work: typically one starts at the verb, defines its translations and creates slots for its arguments
(and adjuncts). Each argument typically receives its own bubble. A bubble can define alternations for a whole
noun phrase, or use slots again to separate the description of a modifier that could be reused elsewhere.
Occasionally a bubble represents a subordinated clause and the process is applied recursively. You are not
forced to annotate a full syntactic tree; it is possible to create one flat bubble for the whole sentence. The
only guiding principle in creating nested bubbles is economy: a set of alternations useful at two or more places
is a candidate for a new bubble. The decomposition need not reflect linguistic constituents: sometimes it is
practical to take punctuation as the root rather than the verb; high-level word order decisions drive the
distribution of commas and quotation marks around clauses; co-ordination could be preferred over dependency.

Aim at the whole space. Human translators differ most in STRUCTURE: where the reporting clause goes ("říká X" /
"X říká" / at the front: X říká: "..."), direct vs. indirect speech (X říká, že ...), the order of the main
parts of the sentence, verbal vs. nominal phrasing (zvýší se / dojde ke zvýšení / jde o zvýšení), active vs.
passive or reflexive, splitting or merging clauses. Within a structure they are EXHAUSTIVE: list every
synonym a translator might use at each position (typically 5-10: e.g. hned / ihned / okamžitě / vzápětí /
obratem / rychle / spěšně), every word-order variant of adverbs and clitics, and optional particles and
connectives (prý, však, jen, totiž, zase, pak, ale, a) as bubbles that may or may not be inserted. Encode
all of it as bubbles so that every structure combines with every lexical choice.

The definition of meaning equivalence is not strict: slight divergence is permitted if it can be reasonably
expected that a human translator will pick either of the alternatives.

Common mistakes: (1) an option listing a word among its own covered words and also having a slot for it -
the slot's word belongs to the filling bubble only; (2) forgetting to cover punctuation, including both
quotation marks (or forgetting to drop() them when the structure has none); (3) combinations that are
individually fine but wrong together (punctuation, case, gender) - fix them with conditions, not by deleting;
(4) adding whole sentences one by one, or options identical to existing ones: the set then grows by at most
one translation per option, and the tool rejects additions that yield no new translation.

PROCEDURE
{procedure}The tool keeps your clauses; every option has a number (#k).
Reply with ONE of:
  - an ```edit block with lines    + option(...).      (add an option; also + drop([...]).)
                                     = 7 option(...).    (replace option #7)
                                     - 7                 (delete option #7)
    A clause may continue on the following lines. This is the normal way to work: small, cheap changes.
  - a ```prolog block with the complete new file (replaces everything);
  - undo   (restore the version before your last accepted change);
  - DONE   (only when you really cannot find any more correct translations).
A version that yields no complete translation is not accepted: the tool shows the errors and your current
version stays. An addition that yields no new translation is not accepted either. If a change makes the set
much smaller, the tool warns you - use undo if that was not intended.

You have a budget of {runs} tool runs (the human annotators had at most 2 hours per sentence; more than 1 hour
was needed for a typical sentence). Use the budget to keep extending the set with further correct translations
and to fix wrong ones.
"""

PROC_PLAIN = """Your first reply: a ```prolog block with your first bubbles. Start from the official reference translation
and factorise it: the main verb with slots for its arguments and adjuncts, each argument in its own bubble, and in
every bubble all the alternatives (synonyms, word forms, word orders) you would accept at that place. Then keep
extending: further structures, further alternatives in every bubble, optional particles. """

PROC_SKEL = """Step 1 (your first reply): list 15-20 structurally different complete Czech translations of the sentence
(different word orders, direct/indirect speech, verbal/nominal, active/passive, clause splits ...), one per
line, in a ```skeletons block. The tool will tell you after every run which of them your bubbles already
produce; make sure all of them are produced eventually, and then multiply each with lexical alternatives.
Step 2 onwards: write and extend the bubbles - factorised, not as a list of sentences. """


def load_inputs(sid):
    extra = Path(f"inputs/{sid}.json")
    rec = json.load(open(extra)) if extra.exists() else None
    n = int(sid)
    line = lambda lang: open(f"{NT}.{lang}", encoding="utf-8").read().split("\n")[n - 1]
    en = dict(l.rstrip("\n").split("\t", 1) for l in open("many-czech-references/en.txt", encoding="utf-8"))[sid]
    cs = dict(l.rstrip("\n").split("\t", 1) for l in open("many-czech-references/cs.txt", encoding="utf-8"))[sid]
    assert line("en").strip() == en.strip(), (sid, line("en"), en)
    czech = [("official reference", cs)]
    if rec:
        czech += [("translated from the German version", g["text"]) for g in rec["G"]]
        czech += [("post-edited machine translation", p["post_edit"]) for p in rec["P"]]
    return dict(sid=sid, en=en, de=line("de"), es=line("es"), fr=line("fr"), cs=czech)


def first_message(inp, src_lang, runs, other_langs=False, skeletons=False):
    src = inp[src_lang]
    s = GUIDE.format(lang=LANGS[src_lang], example=PAPER_EXAMPLE, runs=runs, procedure=PROC_SKEL if skeletons else PROC_PLAIN)
    toks = [t.lower() for t in tokenize_source(src)]
    punct = [t for t in toks if not re.search(r"\w", t)]
    s += f"\nSource sentence ({LANGS[src_lang]}): {src}\nSource tokens: {' '.join(toks)}\n"
    if punct:
        s += f"(The punctuation tokens {' '.join(repr(t) for t in punct)} are tokens too: each must be covered by an option or declared in drop().)\n"
    s += "\nPre-existing human translations, for inspiration:\n"
    for l in (["en", "de", "es", "fr"] if other_langs or src_lang != "en" else []):
        if l != src_lang:
            s += f"  {LANGS[l]}: {inp[l]}\n"
    for how, t in inp["cs"]:
        s += f"  Czech ({how}): {t}\n"
    return s


# ---------------------------------------------------------------- the clause file and edits
def show_clauses(opts, drops):
    lines = [f"{to_text([o])}  % #{k + 1}" for k, o in enumerate(opts)]
    if drops:
        lines.append(to_text([], drops))
    return "\n".join(lines)


def blocks(reply, kind):
    return re.findall(r"```" + kind + r"\s*\n(.*?)```", reply, re.S)


def parse_edit(text, opts, drops):
    """Apply an edit block; returns (opts, drops) or raises ParseError."""
    opts, drops = list(opts), list(drops)
    entries, cur = [], None
    for line in text.split("\n"):
        m = re.match(r"\s*([+=-])\s*(\d+)?\s*(.*)$", line)
        if m and (m.group(1) == "+" or m.group(2)):
            cur = [m.group(1), int(m.group(2)) if m.group(2) else None, m.group(3)]
            entries.append(cur)
        elif cur is not None and line.strip():
            cur[2] += "\n" + line
        elif line.strip() and not line.strip().startswith("%"):
            raise ParseError(f"edit line not understood: {line.strip()[:60]!r} (use '+ clause', '= N clause' or '- N')")
    if not entries:
        raise ParseError("empty edit block")
    replace, delete, add, add_drops = {}, set(), [], []
    for op, k, body in entries:
        if op == "-":
            if not 1 <= k <= len(opts):
                raise ParseError(f"- {k}: there is no option #{k}")
            delete.add(k - 1)
            continue
        if not body.strip():
            raise ParseError(f"'{op} {k or ''}' without a clause")
        o, d = parse(body) if "option(" in body else ([], parse("option([x],[],[x]). " + body)[1])
        if op == "=":
            if not 1 <= k <= len(opts):
                raise ParseError(f"= {k}: there is no option #{k}")
            if len(o) != 1:
                raise ParseError(f"= {k} must be followed by exactly one option(...) clause")
            replace[k - 1] = o[0]
        else:
            add += o
            add_drops += d
    new = [replace.get(i, o) for i, o in enumerate(opts) if i not in delete] + add
    return new, drops + add_drops


# ---------------------------------------------------------------- skeletons
QUOTES = '"„“”«»'


def fix_end(line, src):
    """A skeleton must end like the source: its final punctuation is a source token the bubbles must cover."""
    end = tokenize_source(src)[-1]
    t = line.rstrip()
    if re.search(r"\w", end):
        return t
    if end in QUOTES:  # a final quotation mark may legitimately be dropped (indirect speech): leave the line alone
        return t
    core = re.sub(f"[{QUOTES}]+$", "", t).rstrip()
    return t if core.endswith(end) or re.search(r"[.?!…]$", core) else t + end  # any sentence-final mark counts


def skel_tokens(s):
    from build_explorer2 import toks
    return toks(s)


def skeleton_report(B, skeletons, vocab):
    """Which skeletons does the bubble set produce? vocab: dict word -> id (grown in place)."""
    from determinize import contains
    if not skeletons:
        return None, []
    tid = lambda a: [vocab.setdefault(w, len(vocab)) for w in skel_tokens(a)]
    try:
        ok = contains(B, tid, [tid(s) for s in skeletons])
    except Exception:
        return None, []
    missing = [i for i, x in enumerate(ok) if not x]
    return sum(ok), missing


# ---------------------------------------------------------------- the LLM call
def call(client, model, msgs, think, max_tokens):
    kw = {"extra_body": {"reasoning": {"effort": think if isinstance(think, str) else "medium"}}} if think else {}
    r = client.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens, **kw)
    ch = r.choices[0]
    native = getattr(ch, "native_finish_reason", None) or (r.model_dump()["choices"][0].get("native_finish_reason"))
    return ch.message.content or "", ch.finish_reason, native


def session_finished(rec, a):
    """True if a saved session ended by itself (DONE accepted, or the run budget used up) rather than by API errors or
    empty replies. A DONE accepted before --min-runs (after three refusals) counts as finished unless --resume-early-done."""
    turns = rec.get("turns", [])
    runs = rec.get("tool_runs") or 0
    done = any(t.get("done") and not t.get("refused_done") for t in turns)
    if runs >= a.max_runs:
        return True
    return done and (runs >= a.min_runs or not getattr(a, "resume_early_done", False))


# ---------------------------------------------------------------- one translator
def annotate(a, inp, model, src_lang, prev=None):
    """prev: a saved record of the same translator (resume: continue from its final clauses and run count)."""
    client, model = connect(model, a.provider)
    think = a.think_map.get(model, a.think)
    src = inp[src_lang]
    first = first_message(inp, src_lang, a.max_runs, a.other_langs, a.skeletons)
    turns, log = [], []
    skeletons, vocab = [], {}
    opts, drops, cur_text, cur_rep, prev_tr = [], [], None, None, None
    history = []  # accepted (opts, drops, rep_text, derivations, distinct) for undo and the no-gain check
    best = None
    failed = None  # (what, error text) of the last rejected submission
    extra_note = None
    runs = 0
    if prev and prev.get("final"):
        skeletons = prev.get("skeletons") or []
        opts, drops = parse(prev["final"]["clauses"])
        cur_text = show_clauses(opts, drops)
        rep, prev_tr = tool_report(src, cur_text, None)
        n_sk, missing = skeleton_report(Bubbles(src, cur_text), skeletons, vocab)
        cur_rep = rep["text"] + (f"\nSkeletons produced: {n_sk}/{len(skeletons)}" + (f"; not yet: {', '.join('#' + str(i + 1) for i in missing[:12])}" if missing else "") if n_sk is not None else "")
        history.append((opts, drops, cur_rep, rep.get("n"), rep.get("distinct")))
        runs = prev.get("tool_runs") or 0
        turns = [t for t in prev.get("turns", []) if "api_error" not in t]
        log = [f"run {runs}: resumed session -> {len(opts)} options, {rep.get('n')} translations"]
        best = dict(prev["final"], clauses=cur_text)
        extra_note = "Your session was interrupted and has been resumed from your last accepted version. Continue extending the set."
    for step in range(a.max_runs + 30):
        # ---- compose the single user message
        m = first
        if a.skeletons and not skeletons:
            m += "\nStart with Step 1: reply with your ```skeletons block."
        else:
            if skeletons:
                m += "\nYour skeletons:\n" + "\n".join(f"  #{i + 1} {s}" for i, s in enumerate(skeletons)) + "\n"
            if cur_text is None:
                m += "\nNo clauses yet. Reply with a ```prolog block containing your first bubbles (or an ```edit block with + lines)."
            else:
                m += f"\nYour current clauses (version accepted at run {len(history)}):\n```prolog\n{cur_text}\n```\n"
                if len(log) > 1:
                    m += "\nLog of your tool runs:\n" + "\n".join(log[-40:]) + "\n"
                m += f"\nTool output for your current clauses:\n{cur_rep}\n"
            if failed:
                m += f"\nYour last submission ({failed[0]}) was NOT accepted and your current version is unchanged. Reason:\n{failed[1]}\n"
            if extra_note:
                m += "\n" + extra_note + "\n"
            left = a.max_runs - runs
            m += (f"\n{left} tool runs left. Reply with an ```edit block, a complete ```prolog block, undo, or DONE."
                  if left else "\nThis was your last tool run. Reply DONE.")
        msgs = [{"role": "user", "content": m}]
        if runs == 3 and cur_text is not None and "example_message" not in locals():
            example_message = m
        try:
            reply, finish, native = call(client, model, msgs, think, a.max_tokens)
        except Exception as e:
            turns.append(dict(api_error=str(e)[:500]))
            if a.verbose:
                print(f"  [{inp['sid']} {model} {src_lang}] run {runs}: API error ({sum('api_error' in t for t in turns)}/6): {str(e)[:200]}", flush=True)
            if sum("api_error" in t for t in turns) > 5:
                break
            continue
        failed, extra_note = None, None
        rec = dict(reply=reply[:4000], finish=finish, native=native)
        # ---- refusals / truncation
        if not reply.strip() or finish == "content_filter":
            rec["empty"] = True
            turns.append(rec)
            if a.verbose:
                print(f"  [{inp['sid']} {model} {src_lang}] run {runs}: empty reply or content filter ({sum(t.get('empty', False) for t in turns)}/5)", flush=True)
            if sum(t.get("empty", False) for t in turns) >= 5:
                break
            continue
        if finish == "length":
            failed = ("reply", "your reply was cut off (output length limit). Send a short ```edit block instead of the whole file.")
        # ---- interpret
        up = reply.strip().upper()
        if a.skeletons and not skeletons:
            sk = blocks(reply, "skeletons")
            lines = [re.sub(r"^\s*(?:[-*\u2022#]+|\d+[.)]|#\d+)?\s*", "", l).strip() for l in (sk[0] if sk else reply).split("\n")]
            lines = [l for l in lines if len(l.split()) >= 3 and not l.startswith("```")]
            lines = [fix_end(l, src) for l in lines]
            if len(lines) < 5:
                failed = ("skeletons", "please list at least 10 complete, structurally different Czech translations, one per line, in a ```skeletons block.")
                turns.append(dict(rec, skeletons_rejected=True))
                continue
            skeletons = lines[:25]
            turns.append(dict(rec, skeletons=skeletons))
            continue
        if up.startswith("DONE") or up.endswith("DONE") and len(up) < 40:
            rec["done"] = True
            turns.append(rec)
            if runs < a.min_runs and cur_text is not None and sum(t.get("refused_done", False) for t in turns) < 3:
                turns[-1]["refused_done"] = True
                if a.verbose:
                    print(f"  [{inp['sid']} {model} {src_lang}] run {runs}: DONE refused ({sum(t.get('refused_done', False) for t in turns)}/3)", flush=True)
                n_sk, missing = (skeleton_report(Bubbles(src, cur_text), skeletons, vocab) if cur_text else (0, []))
                extra_note = (f"DONE not accepted yet: you have used only {runs} of {a.max_runs} tool runs; the human annotators typically "
                              "worked for more than an hour on one sentence. "
                              + (f"Skeletons not produced yet: {', '.join('#' + str(i + 1) for i in missing[:10])}. " if missing else "")
                              + "Add further structures, exhaustive synonyms at every position, optional particles and word-order variants.")
                continue
            if a.verbose:
                print(f"  [{inp['sid']} {model} {src_lang}] run {runs}: DONE accepted", flush=True)
            break
        if up.startswith("UNDO"):
            if len(history) >= 2:
                history.pop()
                opts, drops, cur_rep = history[-1][:3]
                cur_text = show_clauses(opts, drops)
                log.append(f"undo -> back to the version of run {len(history)}")
            else:
                failed = ("undo", "nothing to undo")
            turns.append(dict(rec, undo=True))
            continue
        ed, full = blocks(reply, "edit"), blocks(reply, "prolog") or blocks(reply, "pl")
        if not ed and not full and "option(" in reply:
            full = [reply]
        if not ed and not full:
            failed = ("reply", "no ```edit or ```prolog block found. Reply with an ```edit block, a complete ```prolog block, undo, or DONE.")
            turns.append(rec)
            continue
        what = "complete file"
        try:
            if ed:
                new_opts, new_drops = opts, drops
                for e in ed:
                    new_opts, new_drops = parse_edit(e, new_opts, new_drops)
                what = f"edit ({len(ed)} block{'s' if len(ed) > 1 else ''})"
            else:
                new_opts, new_drops = parse(full[-1])
                what = "complete file"
        except ParseError as e:
            failed = (what, f"SYNTAX ERROR: {e}")
            turns.append(dict(rec, rejected=failed[1][:1000]))
            continue
        new_text = show_clauses(new_opts, new_drops)
        try:
            rep, tr = tool_report(src, new_text, prev_tr)
        except Exception as e:
            rep, tr = dict(ok=False, n=0, text=f"TOOL ERROR: {type(e).__name__}: {e}"), None
        runs += 1
        n = rep.get("n") or 0
        rec.update(what=what, clauses=new_text, tool=rep, n=n)
        new_d = rep.get("distinct") or n
        prev_n, prev_d = (history[-1][3], history[-1][4] or history[-1][3]) if history else (None, None)
        pure_addition = (history and len(new_opts) > len(opts) and new_opts[:len(opts)] == opts and list(new_drops) == list(drops))
        if n == 0:
            failed = (what, rep["text"][:6000])
            log.append(f"run {runs}: {what} rejected - no complete translation" + (" (errors)" if "ERRORS" in rep["text"] else ""))
            turns.append(rec)
        elif pure_addition and prev_d is not None and new_d <= prev_d:
            added = len(new_opts) - len(opts)
            failed = (what, f"NO GAIN: your {what} added {added} option{'s' if added > 1 else ''} but no new translation - the set still has "
                      f"{new_d} distinct translations ({n} derivations). In the version you submitted:\n{rep.get('redundancy') or '(the new options only repeat translations that existing options already produce)'}\n"
                      "Add alternatives that are not produced yet, or factorise: put interchangeable words and phrases into slots filled by small bubbles.")
            rec["no_gain"] = True
            log.append(f"run {runs}: {what} rejected - added {added} options, no new translation")
            turns.append(rec)
        else:
            opts, drops, cur_text, cur_rep = new_opts, new_drops, new_text, rep["text"]
            if tr is not None:
                prev_tr = tr
            history.append((opts, drops, rep["text"], n, rep.get("distinct")))
            if rep.get("ok"):
                best = dict(clauses=new_text, n=n, distinct=rep.get("distinct"), run=runs)
            elif best is None or not best.get("ok_version", True):
                best = dict(clauses=new_text, n=n, distinct=rep.get("distinct"), run=runs, ok_version=False)
            B = Bubbles(src, new_text)
            n_sk, missing = skeleton_report(B, skeletons, vocab)
            sk_note = (f"Skeletons produced: {n_sk}/{len(skeletons)}" + (f"; not yet: {', '.join('#' + str(i + 1) for i in missing[:12])}" if missing else "")
                       if n_sk is not None else "")
            cur_rep = rep["text"] + ("\n" + sk_note if sk_note else "")
            history[-1] = (opts, drops, cur_rep, n, rep.get("distinct"))
            rec["skeletons_covered"] = n_sk
            status = (f"{n} translations" + (f" ({rep['distinct']} distinct)" if rep.get("distinct") and rep["distinct"] != n else "")
                      + (f", {rep['slots']} slots" if rep.get("slots") is not None else "")
                      + (", with errors" if "ERRORS:" in rep["text"] else "") + (f", skeletons {n_sk}/{len(skeletons)}" if n_sk is not None else ""))
            log.append(f"run {runs}: {what} accepted -> {len(opts)} options, {status}")
            if prev_n and n < 0.5 * prev_n:
                extra_note = (f"WARNING: your set shrank from {prev_n} to {n} translations. If this was not intended, reply undo.")
                rec["shrink_warning"] = True
            turns.append(rec)
        if a.verbose:
            print(f"  [{inp['sid']} {model} {src_lang}] {log[-1] if log else rec.get('what', '')}", flush=True)
        if runs >= a.max_runs:
            break
    if best is not None:
        best.pop("ok_version", None)
    ended = ("done" if any(t.get("done") and not t.get("refused_done") for t in turns) else
             "api_errors" if sum("api_error" in t for t in turns) > 5 else
             "empty_replies" if sum(t.get("empty", False) for t in turns) >= 5 else
             "max_runs" if runs >= a.max_runs else "step_limit")
    return dict(model=model, src_lang=src_lang, skeletons=skeletons, turns=turns, final=best, tool_runs=runs, ended=ended,
                example_message=locals().get("example_message"), first_message=first,
                final_skeletons=(skeleton_report(Bubbles(src, best["clauses"]), skeletons, vocab)[0] if best else None))


# ---------------------------------------------------------------- all translators, union, output
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sids", nargs="+", required=True)
    ap.add_argument("--annotators", nargs="+", default=["openai/gpt-6-luna", "openai/gpt-6-luna", "moonshotai/kimi-k3.1-pro",
                                                        "moonshotai/kimi-k3.1-pro", "anthropic/claude-haiku-5.5", "anthropic/claude-haiku-5.5"],
                    help="one independent session per entry (default: two per model, like the paper's independent annotators)")
    ap.add_argument("--src-langs", nargs="+", default=None, help="source language per annotator (default en, as in the paper)")
    ap.add_argument("--other-langs", action="store_true", help="also show the German/Spanish/French versions of the sentence (not in the paper)")
    ap.add_argument("--skeletons", action="store_true", help="first step: list structurally different translations (not in the paper)")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--think", nargs="?", const=True, default="medium", help="reasoning effort for all models")
    ap.add_argument("--think-for", nargs="*", default=["openai/gpt-6-luna=high"], metavar="MODEL=EFFORT",
                    help="reasoning effort per model (default: high for gpt-6-luna, whose formalism work is the most reasoning-heavy)")
    ap.add_argument("--max-runs", type=int, default=200)
    ap.add_argument("--min-runs", type=int, default=40, help="DONE is not accepted before this many tool runs")
    ap.add_argument("--max-tokens", type=int, default=120000, help="output limit per reply (all three models allow >=128k on OpenRouter; reasoning tokens count too)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--qc", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--qc-reviewer", default="anthropic/claude-haiku-5.5", help="QC reviewer (keep different from the evaluation reviewer)")
    ap.add_argument("--datafile-cap", type=int, default=3_000_000, help="larger sets are left out of datafile.tsv")
    ap.add_argument("--resume", default=None, help="runs/<tag> of an interrupted run with the same annotators: continue the sessions "
                    "that ended by API errors or empty replies; sessions that said DONE or used the run budget are kept")
    ap.add_argument("--resume-early-done", action="store_true", help="with --resume: also continue sessions that said DONE before --min-runs")
    a = ap.parse_args()
    a.think_map = dict(x.split("=", 1) for x in a.think_for)
    langs = a.src_langs or ["en"] * len(a.annotators)
    assert len(langs) == len(a.annotators), "--src-langs must have one entry per annotator"
    out = Path("runs") / a.tag
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)

    jobs = [(sid, k) for sid in a.sids for k in range(len(a.annotators))]
    inputs = {sid: load_inputs(sid) for sid in a.sids}

    def one(job):
        sid, k = job
        prev = None
        if a.resume:
            p = Path(a.resume) / f"{sid}.json"
            if p.exists():
                anns = json.load(open(p))["annotations"]
                prev = anns[k] if k < len(anns) and anns[k].get("model") == a.annotators[k] and anns[k].get("src_lang", "en") == langs[k] else None
        if prev and session_finished(prev, a):  # ended by DONE or by the run budget, not by API errors: keep as it is
            if a.verbose:
                print(f"  [{sid} {a.annotators[k]} {langs[k]}] finished in the previous run ({prev.get('tool_runs')} runs, {prev.get('ended')}): kept", flush=True)
            return job, prev
        try:
            return job, annotate(a, inputs[sid], a.annotators[k], langs[k], prev)
        except Exception:
            return job, dict(model=a.annotators[k], src_lang=langs[k], error=traceback.format_exc())

    with ThreadPoolExecutor(len(jobs)) as ex:
        res = dict(ex.map(one, jobs))
    for (sid, k), r in res.items():
        (out / "ckpt").mkdir(exist_ok=True)
        json.dump(r, open(out / "ckpt" / f"{sid}_L{k + 1}.json", "w"), ensure_ascii=False, indent=1)  # checkpoint

    if a.qc:  # quality control by repair (not in the paper), see qc.py
        from qc import qc_repair

        def one_qc(job):
            sid, k = job
            r = res[job]
            if r.get("final") and not (r.get("qc") and not r["qc"].get("error")):  # a kept session keeps its QC
                try:
                    r["qc"] = qc_repair(inputs[sid][langs[k]], inputs[sid]["en"], r["final"]["clauses"], translator=r["model"],
                                        reviewer=a.qc_reviewer, provider=a.provider, think=a.think_map.get(r["model"], a.think),
                                        verbose=a.verbose, tag=f"{sid} {r['model']}")
                except Exception:
                    r["qc"] = dict(error=traceback.format_exc())
        with ThreadPoolExecutor(len(jobs)) as ex:
            list(ex.map(one_qc, jobs))

    files = {"datafile.tsv": "qc", "datafile_noqc.tsv": "final"} if a.qc else {"datafile.tsv": "final"}
    dfs = {f: open(out / f, "w", encoding="utf-8") for f in files}
    for sid in a.sids:
        anns = [res[(sid, k)] for k in range(len(a.annotators))]
        sizes = {}
        for f, key in files.items():
            who = {}
            for k, r in enumerate(anns):
                c = (r.get(key) or {}).get("clauses")
                if c:
                    B = Bubbles(inputs[sid][langs[k]], c)
                    if B.count() > a.datafile_cap:
                        r.setdefault("not_in_datafile", []).append(f)
                        continue
                    for t in B.translations():
                        who.setdefault(t, []).append(f"L{k + 1}")
            for t, ws in who.items():
                dfs[f].write(f"{sid}\t{', '.join(['llm'] * len(ws))}\t{', '.join(ws)}\t{t}\n")
            sizes[f] = len(who)
        json.dump(dict(inputs[sid], annotations=anns, union=sizes), open(out / f"{sid}.json", "w"), ensure_ascii=False, indent=1)
        per = [(r["final"] or {}).get("n") if not r.get("error") else "error" for r in anns]
        runs = [r.get("tool_runs") for r in anns]
        sk = [f"{r.get('final_skeletons')}/{len(r.get('skeletons') or [])}" for r in anns] if a.skeletons else None
        qcd = [(r.get("qc") or {}).get("summary", "-") for r in anns]
        print(f"{sid}\tderivations {per}\ttool runs {runs}\t" + (f"skeletons {sk}\t" if sk else "") + f"union {sizes}\tQC {qcd}", flush=True)
    for f in dfs.values():
        f.close()


if __name__ == "__main__":
    main()
