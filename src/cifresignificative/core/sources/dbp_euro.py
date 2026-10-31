"""Draft budgetary plans (DBPs) of the euro-area members, English copies on the Commission's pages (2018-2025): the net lending (EDP B.9, % of GDP) each government planned for its budget year, the second reference of the deficit-surprise placebo ranking (the first is the Commission's autumn forecast, `ec_forecasts.py`).

`data/external/dbp_urls.csv` lists the English PDF of each country and budget year (taken from the yearly index pages `draft-budgetary-plans-YYYY_en`; only the October plan, not the updates or replies; 14-16 countries a year, Spain and some other members have no English copy listed and the 2016-2017 pages have another layout). The PDFs go to `data/raw/dbp_euro/` (gitignored); the committed table `data/external/dbp_targets.csv` is the source of truth.
Parsing (`parse_plan`): the macro table of the Commission template has three B.9 rows (rest of the world, private sector, general government) whose values add up (row 1 = row 2 + row 3), which finds the general-government row whatever its label; the budget-year target is its last value. `evidence`: `identity_and_subsector_row` (also in an S.13 row), `identity_only`, `label_only` (fallback without the check; not used), or the reason a plan was not parsed (about half of the plans: new templates, fractions, wrapped labels). Checks: Italy's eight plans give the figures of `deficit_targets.csv`; against the Commission forecast of the autumn before the verified rows differ by 0.1 at the median.
`uv run python -m cifresignificative.core.sources.dbp_euro fetch|build`.
"""
import itertools
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources import deficit_targets as DT
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/dbp_euro")
URLS = Path("data/external/dbp_urls.csv")
CSV = Path("data/external/dbp_targets.csv")
ISO3 = {"AT": "AUT", "BE": "BEL", "CY": "CYP", "DE": "DEU", "EE": "EST", "EL": "GRC", "ES": "ESP", "FI": "FIN", "FR": "FRA", "HR": "HRV", "IE": "IRL", "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT", "NL": "NLD", "PT": "PRT", "SI": "SVN", "SK": "SVK"}


def path(year: int, cc: str) -> Path:
    return RAW_DIR / f"{year}_{cc}.pdf"


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for r in pd.read_csv(URLS).to_dict("records"):
        p = path(r["budget_year"], r["country"])
        if p.exists() and not force:
            continue
        time.sleep(1.5)  # the server answers 429 to quick successions
        for attempt in range(4):  # ... and sometimes ends a response early
            try:
                resp = requests.get(r["url"], headers=DT.UA, timeout=180)
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(15 * (attempt + 1))
        if not resp.content.startswith(b"%PDF"):
            raise RuntimeError(f"{r["budget_year"]} {r["country"]}: not a PDF")
        p.write_bytes(resp.content)
        record(f"dbp_{r["budget_year"]}_{r["country"]}", source="draft budgetary plan", dataset=f"DBP {r["budget_year"]} {r["country"]}", url=r["url"])
    return RAW_DIR


NUM = r"-?\d+[.,]\d+"


def _f(x: str) -> float:
    return float(x.replace(",", "."))


def parse_plan(pdf: Path) -> tuple[float, bool | None]:
    """(budget-year net lending of the general government, % of GDP; whether the same figure is also in an S.13 row of the sub-sector table; `None` for the label-only fallback, which has no arithmetic check). The macro table has three rows with the code B.9 and three % of GDP columns (t-2, t-1, t): 1. the rest of the world, 2. the private sector, 3. the general government. Labels differ between plans (and wrap), so the rows are found by the accounting identity: three B.9 rows within a few lines of each other where row 1 = row 2 + row 3 in every column (rounding 0.15); the target is the last value of row 3.
    Unicode minus signs, decimal commas and percent signs are normalised; fractions (3/4, in a few German plans) and plans without the macro table (new templates) are not parsed."""
    t = DT._text(pdf).replace("\u2013", "-").replace("\u2212", "-").replace("%", " ")
    lines = t.split("\n")
    rows = [(i, [_f(x) for x in m]) for i, l in enumerate(lines) if (m := re.search(rf"B\.?\s?9\s*(?:\(\d\))?\s+({NUM})\s+({NUM})\s+({NUM})\s*$", l)) and (m := list(m.groups()))]
    for (i1, r1), (i2, r2), (i3, r3) in zip(rows, rows[1:], rows[2:], strict=False):
        if i3 - i1 <= 12 and all(abs(a - b - c) <= 0.15 for a, b, c in zip(r1, r2, r3, strict=True)) and any(r1):
            v = r3[2]
            subs = [m.group(1) for m in re.finditer(rf"\bS\.13\s+((?:{NUM}\s*)+)$", t, re.MULTILINE)]
            return v, any(abs(_f(x) - v) <= 0.051 for row in subs for x in re.findall(NUM, row))
    for (i2, r2), (i3, r3) in itertools.pairwise(rows):  # fallback: the private-sector row followed by a general-government row (the rest-of-the-world row has no numbers on its line)
        if i3 - i2 <= 3 and re.search(r"private sector|households", lines[i2], re.IGNORECASE) and re.search(r"general government", lines[i3], re.IGNORECASE):
            subs = [m.group(1) for m in re.finditer(rf"\bS\.13\s+((?:{NUM}\s*)+)$", t, re.MULTILINE)]
            return r3[2], any(abs(_f(x) - r3[2]) <= 0.051 for row in subs for x in re.findall(NUM, row))
    for i, l in enumerate(lines):  # last resort: a macro-table row labelled net lending of the general government with three numbers and no B.9 code (no arithmetic check: kept in the table as `label_only` and not used in the analysis; against the Commission forecast of the year before two of the ten were off by more than 5 points)
        m = re.match(rf"\s*(?:\d\.\s*)?net lending\s*/\s*(?:net )?borrowing of (?:the )?general government\D*?\s({NUM})\s+({NUM})\s+({NUM})\s*$", l, re.IGNORECASE)
        if m:
            return _f(m.group(3)), None
    raise ValueError("macro table rows not found")


def build_csv(out: Path = CSV) -> pd.DataFrame:
    rows = []
    for r in pd.read_csv(URLS).to_dict("records"):
        base = {"budget_year": r["budget_year"], "country": r["country"], "iso3": ISO3[r["country"]], "url": r["url"]}
        try:
            macro, ok = parse_plan(path(r["budget_year"], r["country"]))
        except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as e:
            rows.append({**base, "target_pct_gdp": None, "evidence": f"not parsed: {e}"})
            continue
        rows.append({**base, "target_pct_gdp": macro, "evidence": "label_only" if ok is None else ("identity_and_subsector_row" if ok else "identity_only")})
    d = pd.DataFrame(rows).sort_values(["budget_year", "country"], ignore_index=True)
    d.to_csv(out, index=False)
    return d


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "fetch":
        print(fetch("--force" in sys.argv))
    else:
        d = build_csv()
        print(d.evidence.str[:12].value_counts().to_string())
