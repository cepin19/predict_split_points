"""Czech translations available to the annotators, as in Bojar et al. (2013), section 3:
  O  the official WMT11 reference (many-czech-references/cs.txt);
  G  three Czech translations made from the German version of the sentence (the paper's were by
     humans; here each by a different LLM that sees only the German sentence);
  P  post-edited MT outputs: WMT11 English->Czech system outputs (wmt11_sys/...), post-edited by LLMs
     (one system per LLM).
  ../env_vllm/bin/python make_inputs.py --sids 7 22 70 109 112
Writes inputs/<sid>.json: {"O": ..., "G": [...], "P": [{"system": ..., "mt": ..., "post_edit": ..., "by": ...}]}
"""
import argparse, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm_client import chat, connect

NT = "wmt11_dl/dev/newstest2011"
SYS = "wmt11_sys/wmt11-data/plain/system-outputs/newstest2011/en-cz/newstest2011.en-cz."
G_MODELS = ["google/gemini-3.5-flash-lite", "moonshotai/kimi-k3.1-pro", "anthropic/claude-haiku-5.5"]
# The paper's P: "two manual post-edits of a phrase-based MT system similar to those participating in the WMT11
# competition", each sentence post-edited by two independent annotators. Here: the output of cu-bojar (a phrase-based
# Moses system that did participate), post-edited independently by two LLMs.
P_SYSTEMS = ["cu-bojar", "cu-bojar"]
P_MODELS = ["moonshotai/kimi-k3.1-pro", "anthropic/claude-haiku-5.5"]


def line(path, n):
    return open(path, encoding="utf-8").read().split("\n")[n - 1].strip()


def ask(model, prompt):
    c, m = connect(model, "openrouter")
    for _ in range(3):
        out = (chat(c, m, prompt, think="medium", max_tokens=8000) or "").strip()
        if out:
            return out.strip('`').strip().split("\n")[0].strip()
    raise RuntimeError(f"no answer from {model}")


def g_prompt(de):
    return ("You are a professional translator. Translate the following German sentence from a news article into "
            "Czech. Reply with the Czech translation only.\n\n" + de)


def p_prompt(en, mt):
    return ("You are a professional translator post-editing machine translation. Correct the Czech machine "
            "translation of the English sentence below so that it is a correct, fluent and faithful Czech translation. "
            "Keep as much of the machine translation as possible; change only what must be changed. Reply with the "
            "post-edited Czech sentence only.\n\n"
            f"English: {en}\nMachine translation: {mt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids", nargs="+", required=True)
    ap.add_argument("--keep-g", action="store_true", help="keep the G translations of an existing inputs/<sid>.json, redo only P")
    a = ap.parse_args()
    Path("inputs").mkdir(exist_ok=True)
    cs = dict(l.rstrip("\n").split("\t", 1) for l in open("many-czech-references/cs.txt", encoding="utf-8"))
    jobs = []
    for sid in a.sids:
        n = int(sid)
        de, en = line(f"{NT}.de", n), line(f"{NT}.en", n)
        if not (a.keep_g and Path(f"inputs/{sid}.json").exists()):
            jobs += [(sid, "G", m, g_prompt(de), None) for m in G_MODELS]
        for s, m in zip(P_SYSTEMS, P_MODELS):
            mt = line(SYS + s, n)
            jobs.append((sid, "P", m, p_prompt(en, mt), (s, mt)))
    with ThreadPoolExecutor(16) as ex:
        outs = list(ex.map(lambda j: ask(j[2], j[3]), jobs))
    for sid in a.sids:
        rec = dict(O=cs[sid], G=[], P=[])
        if a.keep_g and Path(f"inputs/{sid}.json").exists():
            rec["G"] = json.load(open(f"inputs/{sid}.json"))["G"]
        for (s2, kind, m, _, extra), out in zip(jobs, outs):
            if s2 != sid:
                continue
            if kind == "G":
                rec["G"].append(dict(by=m, text=out))
            else:
                rec["P"].append(dict(system=extra[0], mt=extra[1], post_edit=out, by=m))
        json.dump(rec, open(f"inputs/{sid}.json", "w"), ensure_ascii=False, indent=1)
        print(sid, json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
