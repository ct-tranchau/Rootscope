"""Species knowledge used by the exodermis decision.

The table ``data/species_exodermis.csv`` holds, for every training species,
whether an exodermis was annotated in the training sections, and for a few
species outside the training set what the literature reports. Perumalla,
Peterson and Enstone (1990, Bot J Linn Soc 103:93) found a hypodermis with
Casparian bands in 89% of 181 angiosperm species, so most species have the
layer; but the exodermis is a variable, stress- and age-dependent structure
and the literature disagrees even for pea. The table is therefore a hint:
only the training annotations decide the labels, the literature only informs
the report.
"""
import csv
import re
from pathlib import Path

_TABLE = None


def table():
    global _TABLE
    if _TABLE is None:
        p = Path(__file__).parent / "data" / "species_exodermis.csv"
        with open(p, newline="") as f:
            _TABLE = list(csv.DictReader(f))
    return _TABLE


def _norm(s):
    s = re.sub(r"[^a-z\s-]", " ", str(s).lower().replace("_", " "))
    return " ".join(s.split())


def lookup(name):
    """Match a user-supplied species name against the table.

    Accepts ``"Pisum sativum"``, ``"P. sativum"``, ``"Psativum"``, ``"pea"``-style
    strings are NOT resolved (no common names). Returns the row (dict) with an
    added ``match`` key ("species" or "genus"), or None. A genus-only match is
    returned only when every species of that genus in the table agrees."""
    if not name or not str(name).strip():
        return None
    q = _norm(name)
    rows = table()
    for r in rows:                                   # full binomial
        if _norm(r["binomial"]) == q:
            return dict(r, match="species")
    parts = q.split()
    for r in rows:                                   # abbreviated genus: "p sativum", "psativum"
        g, sp = _norm(r["binomial"]).split()[:2]
        if len(parts) == 2 and parts[1] == sp and g.startswith(parts[0][0]) and len(parts[0]) <= 2:
            return dict(r, match="species")
        if len(parts) == 1 and parts[0] == g[0] + sp:
            return dict(r, match="species")
    genus = parts[0]
    same = [r for r in rows if _norm(r["binomial"]).split()[0] == genus]
    if same:
        vals = {(r["exodermis_in_training"] or r["exodermis_literature"]) for r in same}
        if len(vals) == 1:
            return dict(same[0], match="genus", binomial=genus.capitalize() + " sp.")
    return None
