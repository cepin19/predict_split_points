"""Annotation pilot: LLM-written translation grammars for WMT25 en->cs sentences.

Per sentence:
  1. seeds     a few translations from the LLM (the human annotators had up to four), then
               --n-para structurally different translations (paraphrases) to generalise
  2. annotate  the LLM writes a grammar; checker errors go back to it (up to --fix-tries)
  3. validate  a reviewer judges --n-val sampled expansions; score = precision x distinct size
               (estimated number of correct translations)
  4. improve   two candidates per round: an LLM repair (rejected expansions with the alternatives
               that produced them, plus paraphrases not yet covered) and blame-based pruning of
               alternatives with many rejections; the best-scoring grammar so far is kept
  5. final     the winner is re-judged on a fresh sample (selection inflates the estimate)
  Steps 2-5 run --annotations times independently (like the paper's several translators); the
  final grammars are united and the union is judged on a fresh sample.
Writes runs/<tag>/<n>.json per sentence (all grammars, reports and verdicts) and summary.tsv.
  ../env_vllm/bin/python pilot.py --tag gemma4_r1
"""
import argparse, json, re, sys, traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import prompts
from grammar import Grammar, GrammarError, report, union
from llm_client import chat, connect


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=0)
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


class Pilot:
    def __init__(self, a):
        self.a = a
        self.ann_client, self.ann_model = connect(a.annotator, a.provider)
        self.val_client, self.val_model = (connect(a.validator, a.val_provider or a.provider) if a.validator
                                           else (self.ann_client, self.ann_model))

    def ask(self, prompt, who="ann", think=None, **kw):
        c, m = (self.ann_client, self.ann_model) if who == "ann" else (self.val_client, self.val_model)
        think = self.a.think if think is None else think
        return chat(c, m, prompt, think=think, max_tokens=kw.pop("max_tokens", 24576 if think else 8192), **kw)

    def build(self, prompt, log):
        """Ask for a grammar until it parses and checks (checker feedback each time)."""
        g = None
        for t in range(self.a.fix_tries + 1):
            raw = self.ask(prompt, temperature=self.a.temp)
            entry, data = dict(raw=raw), {}
            log.append(entry)
            try:
                data = parse_json(raw)
                data = data.get("grammar", data)
                g = Grammar(data)
                rep = report(g, self.en)
                entry.update(grammar=data, report=rep)
                if rep["ok"] and rep["derivations"] <= self.a.max_derivations:
                    return data, g, rep
                if rep["derivations"] > self.a.max_derivations:
                    rep = dict(rep, ok=False, error=f"{rep['derivations']} expansions; keep it under {self.a.max_derivations}")
            except (ValueError, GrammarError) as e:
                rep = dict(ok=False, error=str(e)[:2000])
                entry.update(error=rep["error"])
                if not isinstance(data, dict):
                    data = {}
            prompt = prompts.repair(self.en, data, rep)
        return None, None, None

    def judge(self, texts):
        verdicts = []
        for i in range(0, len(texts), self.a.val_batch):
            batch = texts[i:i + self.a.val_batch]
            for _ in range(3):
                try:
                    v = parse_json(self.ask(prompts.validate(self.en, batch), who="val", temperature=0.0,
                                            max_tokens=16384 if self.a.think else 4096))
                    if isinstance(v, list) and len(v) == len(batch):
                        verdicts += [dict(ok=bool(x.get("ok")), reason=x.get("reason", "")) for x in v]
                        break
                except ValueError:
                    pass
            else:
                verdicts += [dict(ok=None, reason="unparsable verdict")] * len(batch)
        return verdicts

    def evaluate(self, data, g, rep, seed, paras=(), seeds=(), how=""):
        """Judge a sample; returns a candidate record with the estimated number of correct translations."""
        samp = g.sample(self.a.n_val, seed=seed, deriv=True)
        v = self.judge([t for t, _ in samp])
        rows = [dict(text=t, ids=sorted(set(d)), alts=sorted({g.alt_label(x) for x in d}), **vv)
                for (t, d), vv in zip(samp, v)]
        judged = [r for r in rows if r["ok"] is not None]
        prec = sum(r["ok"] for r in judged) / len(judged) if judged else None
        size = rep.get("distinct", rep["derivations"])
        allt = {t for t, _ in g.expand()} if rep["derivations"] <= 2_000_000 else set()
        return dict(how=how, grammar=data, g=g, report=rep, verdicts=rows, precision=prec, size=size,
                    est_correct=(prec or 0) * size, missing=[p for p in paras if p not in allt] if allt else [],
                    seeds_covered=sum(x in allt for x in seeds) if allt else None,
                    paraphrases_covered=sum(p in allt for p in paras) if allt else None)

    def prune(self, cand):
        """Drop alternatives that rejected samples went through (blame), keeping every rule non-empty."""
        g, rows = cand["g"], cand["verdicts"]
        use, rej = Counter(), Counter()
        for r in rows:
            for i in r["ids"]:
                use[tuple(i)] += 1
                rej[tuple(i)] += r["ok"] is False
        blamed = sorted((a for a in use if rej[a] >= 2 and rej[a] / use[a] >= self.a.blame_rate),
                        key=lambda a: -rej[a] / use[a])
        drop, left = set(), {n: len(alts) for n, alts in g.rules.items()}
        for a in blamed:
            if left[a[0]] > 1:
                drop.add(a)
                left[a[0]] -= 1
        if not drop:
            return None
        data = g.to_json(drop)
        g2 = Grammar(data)
        rep = report(g2, self.en)
        return (data, g2, rep) if rep["ok"] else None

    def seeds_for(self, item):
        """The paper's annotators had the official reference and up to four Czech translations."""
        seeds = [item["official_ref"]] if item.get("official_ref") else []
        llm = {self.ask(prompts.translate(self.en), think=False, temperature=0.8, max_tokens=256).strip()
               for _ in range(self.a.n_seeds - len(seeds))}
        return seeds + sorted(llm - set(seeds))

    def annotate_once(self, seeds):
        """One independent annotation: paraphrases, grammar, review/repair rounds, final estimate."""
        rec = dict(rounds=[])
        paras = []
        if self.a.n_para:
            for _ in range(3):
                try:
                    paras = [p.strip() for p in parse_json(self.ask(prompts.paraphrase(self.en, seeds, self.a.n_para),
                                                                    temperature=0.7)) if isinstance(p, str)]
                    break
                except ValueError:
                    pass
        rec["paraphrases"] = paras
        log = []
        data, g, rep = self.build(prompts.annotate(self.en, seeds, paras), log)
        rec["build_log"] = log
        if g is None:
            rec["failed"] = "no valid grammar"
            return rec
        ev = lambda d, g, r, seed, how: self.evaluate(d, g, r, seed, paras, seeds, how)
        best = ev(data, g, rep, 0, "initial")
        rec["rounds"].append([best])
        for r in range(1, self.a.rounds + 1):
            cands = []
            bad = [x for x in best["verdicts"] if x["ok"] is False][: self.a.max_bad]
            if bad or best["missing"]:
                log = []
                new = self.build(prompts.repair(self.en, best["grammar"], best["report"], bad, best["missing"]), log)
                if new[0] is not None:
                    cands.append(ev(*new, r, "llm_repair"))
            if self.a.prune:
                pr = self.prune(best)
                if pr:
                    cands.append(ev(*pr, r, "prune"))
            rec["rounds"].append(cands)
            best = max([best] + cands, key=lambda c: c["est_correct"])
        # selection favours lucky samples: re-estimate the winner on a fresh sample
        rec["final"] = ev(best["grammar"], best["g"], best["report"], 1000, "final:" + best["how"])
        for rnd in rec["rounds"]:
            for c in rnd:
                c.pop("g", None)
        return rec

    def run(self, item):
        self.en = item["en"]
        seeds = self.seeds_for(item)
        with ThreadPoolExecutor(self.a.annotations) as ex:
            anns = list(ex.map(lambda _: self.annotate_once(seeds), range(self.a.annotations)))
        rec = dict(item, seeds=seeds, annotations=anns)
        finals = [x["final"]["grammar"] for x in anns if "final" in x]
        if not finals:
            rec["failed"] = "no annotation produced a grammar"
            return rec
        for x in anns:
            if "final" in x:
                x["final"].pop("g", None)
        data = union(finals) if len(finals) > 1 else finals[0]
        g = Grammar(data)
        rep = report(g, self.en, enum_limit=self.a.max_derivations)
        paras = [p for x in anns for p in x["paraphrases"]]
        try:
            u = self.evaluate(data, g, rep, 2000, paras, seeds, f"union of {len(finals)}")
            u.pop("g")
            rec["union"] = u
        except GrammarError as e:  # too large to enumerate
            rec["union"] = dict(grammar=data, report=rep, error=str(e))
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--input", default="pilot.jsonl")
    ap.add_argument("--provider", default=None, help="openrouter / openai (default: local vLLM server)")
    ap.add_argument("--val-provider", default=None)
    ap.add_argument("--annotator", default=None)
    ap.add_argument("--validator", default=None, help="model name of a different served LLM (default: the annotator)")
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--n-para", type=int, default=20, help="structurally different translations to generalise (0 = off)")
    ap.add_argument("--think", nargs="?", const=True, default=False,
                    help="enable reasoning; for remote models optionally an effort: low/medium/high")
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--val-batch", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--fix-tries", type=int, default=3)
    ap.add_argument("--max-bad", type=int, default=15)
    ap.add_argument("--blame-rate", type=float, default=0.6, help="prune alternatives with this share of rejected uses")
    ap.add_argument("--prune", action="store_true", help="also try blame-based pruning (never won in the pilot)")
    ap.add_argument("--annotations", type=int, default=3, help="independent annotations per sentence, united")
    ap.add_argument("--max-derivations", type=int, default=5_000_000)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.input)][: a.limit]
    out = Path("runs") / a.tag
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)

    def one(ix):
        n, item = ix
        try:
            rec = Pilot(a).run(item)
        except Exception:
            rec = dict(item, failed=traceback.format_exc())
        json.dump(rec, open(out / f"{n:03d}.json", "w"), ensure_ascii=False, indent=1)
        cell = lambda x: f"{x['size']}@{'-' if x['precision'] is None else round(x['precision'], 2)}"
        anns = [cell(x["final"]) if "final" in x else "failed" for x in rec.get("annotations", [])]
        u = rec.get("union")
        u = u if u and "error" not in u else None
        line = "\t".join([str(n), item.get("sid", item["domain"]), cell(u) if u else "-",
                          f"{u['paraphrases_covered']}/{len(u['missing']) + (u['paraphrases_covered'] or 0)}" if u else "-",
                          " ".join(anns), rec.get("failed", "")[:80].replace("\n", " ")])
        print(line, flush=True)
        return line

    with ThreadPoolExecutor(a.workers) as ex:
        lines = list(ex.map(one, enumerate(items)))
    (out / "summary.tsv").write_text("n\tsid\tunion size@precision\tparaphrases covered\tannotations size@precision\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
