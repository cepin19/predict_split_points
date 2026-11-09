"""Add the verbatim prompts of a run to an explorer data file.
  ../env_vllm/bin/python explorer_prompts.py runs/paper_v5 explorer/data_paper_v5.json
"""
import json, sys
from pathlib import Path

from make_inputs import G_MODELS, NT, P_MODELS, P_SYSTEMS, SYS, g_prompt, line, p_prompt
from qc import PROMPT as QC_PROMPT, REPAIR

run, out = Path(sys.argv[1]), sys.argv[2]
d = json.load(open(out))
for s in d["sentences"]:
    sid = s["sid"]
    rec = json.load(open(run / f"{sid}.json"))
    anns = [a for a in rec["annotations"] if a.get("first_message")]
    a = next((x for x in anns if x.get("example_message")), anns[0] if anns else None)
    n = int(sid)
    qrows = [r for x in rec["annotations"] for r in ((x.get("qc") or {}).get("rows") or [])][:3]
    cands = "".join(f"{j}. {x['text']}\n" for j, x in enumerate(qrows)) or "0. …\n"
    rej = [r for x in rec["annotations"] for r in ((x.get("qc") or {}).get("rows") or []) if r.get("ok") is False][:3]
    listing = "\n".join(f"{i + 1}. {r['text']}\n   reviewer: {r['reason']} (wrong: {r['wrong']})\n   built from options {', '.join('#' + str(k + 1) for k in r['options'])}" for i, r in enumerate(rej)) or "1. …"
    first_by = {x["src_lang"]: x["first_message"] for x in anns}
    s["prompts"] = dict(
        first=first_by.get("en") or (a["first_message"] if a else ""),
        first_other={k: v for k, v in first_by.items() if k != "en"},
        turn=(a.get("example_message") if a else None) or "(no later turn recorded)",
        example_model=a["model"] if a else "", example_lang=a["src_lang"] if a else "",
        g=[dict(model=m, prompt=g_prompt(line(f"{NT}.de", n))) for m in G_MODELS],
        p=[dict(model=m, system=sy, prompt=p_prompt(line(f"{NT}.en", n), line(SYS + sy, n))) for sy, m in zip(P_SYSTEMS, P_MODELS)],
        qc=QC_PROMPT.format(en=s["en"], cands=cands),
        repair=REPAIR.format(en=s["en"], clauses="(the translator's current clauses, numbered #1, #2, ...)", rejected=listing),
        reviewer=next(((x.get("qc") or {}).get("reviewer") for x in rec["annotations"] if (x.get("qc") or {}).get("reviewer")), None))
json.dump(d, open(out, "w"), ensure_ascii=False, separators=(",", ":"))
print("prompts added for", [s["sid"] for s in d["sentences"]])
