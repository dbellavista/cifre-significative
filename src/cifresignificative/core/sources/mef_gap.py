"""MEF/Dipartimento delle Finanze: "Relazione sull'economia non osservata e sull'evasione fiscale e contributiva" (2025 edition), Italy only.

Parses table I.4 (tax-revenue gap, EUR million), I.5 (propensity to gap, % of tax due) and the totals row of table I.1 (tax + social contributions,
upper estimate) from the PDF text. Values are the report's central/upper ("max") estimates for 2018-2022.
"""
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

PDF = Path("data/external/source/evasion/Relazione-evasione-fiscale-e-contributiva-2025_2310_ore1230.pdf")
URL_MIRROR = "https://www.bollettinoadapt.it/wp-content/uploads/2025/11/Relazione-evasione-fiscale-e-contributiva-2025_2310_ore1230.pdf"
YEARS = [2018, 2019, 2020, 2021, 2022]
NUM = r"-?\d[\d\.]*(?:,\d+)?%?"


def _num(tok: str) -> float:
    return float(tok.rstrip("%").replace(".", "").replace(",", ".")) if tok.endswith("%") is False else float(tok.rstrip("%").replace(",", "."))


def _table(text: str, start: str, stop: str) -> dict[str, list[float]]:
    a = text.index(start)
    b = text.find(stop, a + len(start))
    block = text[a:(b if b > 0 else len(text))]
    out = {}
    for line in block.split("\n"):
        m = re.match(rf"^(?P<label>[A-Za-zÀ-ú' \(\)/\.-]+?)\s+(?P<nums>(?:{NUM}\s+){{6,}}{NUM})\s*$", line.strip())
        if m and not m["label"].startswith(("Tipologia", "TABELLA", "Differenza")):
            vals = [_num(t) for t in m["nums"].split()]
            out[m["label"].strip()] = vals[:5]
    return out


def load() -> pd.DataFrame:
    pages = [p.extract_text() or "" for p in PdfReader(PDF).pages]
    p4 = next(t for t in pages if "TABELLA I.4" in t and "TABELLA I.5" in t)
    gap = _table(p4, "TABELLA I.4", "TABELLA I.5")
    prop = _table(p4, "TABELLA I.5", "Nota")
    rows = []
    for tax, vals in gap.items():
        rows += [(tax, y, "gap_meur", v) for y, v in zip(YEARS, vals, strict=True)]
    for tax, vals in prop.items():
        rows += [(tax, y, "propensity_pct", v) for y, v in zip(YEARS, vals, strict=True)]
    p1 = next(t for t in pages if "TABELLA I.1" in t and "contributive" in t)
    for pattern, key in [(r"Totale\s+entrate\s+contributive\s+", "Total social-contribution gap (upper estimate)"), (r"Totale\s+entrate\s+tributarie\s+e\s+contributive\s+", "Total tax and contribution gap (upper estimate)")]:
        m = re.search(pattern + rf"((?:{NUM}\s+){{13}}{NUM})", p1)
        assert m is not None, pattern
        toks = [_num(t) for t in m.group(1).split()]
        rows += [(key, y, "gap_meur", toks[2 * k + 1]) for k, y in enumerate(YEARS)]  # (min, max) pairs per year: keep max
    p2 = next(t for t in pages if "TABELLA I.2" in t)
    seg = p2[p2.index("TABELLA I.2"):]
    for pattern, key in [(r"Totale\s+entrate\s+tributarie\s+", "Totale entrate tributarie"), (r"Totale\s+entrate\s+tributarie\s+\(al\s+netto\s+delle\s+accise\s+e\s+dell.IMU\)\s+", "Totale entrate tributarie (al netto di accise e IMU)")]:
        m = re.search(pattern + rf"((?:{NUM}\s+){{13}}{NUM})", seg)
        assert m is not None, pattern
        toks = [_num(t) for t in m.group(1).split()]
        rows += [(key, y, "propensity_pct", toks[2 * k + 1]) for k, y in enumerate(YEARS)]  # upper estimate
    d = pd.DataFrame(rows, columns=["tax", "year", "measure", "value"])
    return d.assign(source="MEF, Relazione sull'economia non osservata e sull'evasione fiscale e contributiva, 2025 (tables I.1, I.4, I.5)")
