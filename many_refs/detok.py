"""Detokenizer for the many-references data (space-separated Czech tokens)."""
import re

NO_SPACE_BEFORE = set(",.:;!?)]}%…“‘«") | {"''", "..."}
NO_SPACE_AFTER = set("([{„‚»")  # Czech guillemets: »text«


def detok(s: str) -> str:
    toks = s.split()
    out, quote_open = [], False
    glue_next = False
    for t in toks:
        if t == '"':  # straight quote: alternate opening / closing
            if quote_open:
                if out:
                    out[-1] += t
                else:
                    out.append(t)
            else:
                out.append(t)
                glue_next = True
            quote_open = not quote_open
            continue
        if out and (glue_next or t in NO_SPACE_BEFORE):
            out[-1] += t
        else:
            out.append(t)
        glue_next = t in NO_SPACE_AFTER
    s = " ".join(out)
    return re.sub(r"(\d) , (\d)", r"\1,\2", s)


if __name__ == "__main__":
    for x in ['" Kdo je to Voldemort ? "', 'RPG však argumentuje : „ V těchto případech je řeč o 19,66 korun . “',
              'tepelných úspor - nových oken ( viz graf ) .']:
        print(detok(x))
