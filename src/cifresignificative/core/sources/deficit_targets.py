"""Programmatic deficit targets: the net borrowing (EDP B.9, % of GDP) the Italian government committed to for the year of each budget law, as stated when the law was proposed.

2016-2022: the draft budgetary plan (DPB) sent to the European Commission in October (the Commission's standard template). Its macro table has the general-government net lending in three columns (t-2, t-1, t) and the
last one is the target for the budget year t; the table by sub-sector (row "General government S.13", columns t-1, t, t+1, ...) must give the same figure as its second value, which is the check. 2023-2025 come from the plans as well (and are cross-checked against the "indebitamento netto programmatico" row of the budget service's dossier, `sources/budget_laws.py`); 2026 only from its dossier. The same tables give the revenue, expenditure and interest targets and the forecast growth, used to decompose the miss. Some targets were revised before the law passed (2019: 2.4 -> 2.04 after the talks with the
Commission); `target_final` records the revised figure where a source states it. PDFs go to `data/raw/dbp/` (gitignored); the committed table `data/external/deficit_targets.csv` is the source of truth.
`uv run python -m cifresignificative.core.sources.deficit_targets fetch|build`.
"""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources import budget_laws
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/dbp")
CSV = Path("data/external/deficit_targets.csv")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
EC = "https://economy-finance.ec.europa.eu"
# budget year -> (draft budgetary plan, URL, language). 2016 and 2017 are the Commission's copies (English); 2022 is the MEF's (the Commission copy of that year was not needed).
DPB = {
    2016: ("DPB 2016, submitted 15 October 2015", "https://commission.europa.eu/document/download/1b179ada-91d2-483f-9623-f018c0d40bc2_en", "en"),
    2017: ("DPB 2017, submitted October 2016 (the Commission page also lists an update of 19 October)", "https://commission.europa.eu/document/download/20413c1e-2b53-4c32-b9a6-5d27563a7bd9_en", "en"),
    2018: ("DPB 2018, submitted October 2017", f"{EC}/document/download/ef2831ee-f1b3-4046-998f-49e813192513_en?filename=2018_dbp_it_en.pdf", "en"),
    2019: ("DPB 2019, submitted 16 October 2018", f"{EC}/document/download/437ac0ea-6e7d-4501-8320-86b630c0ee98_en?filename=2019_dbp_it_en.pdf", "en"),
    2020: ("DPB 2020, submitted October 2019", f"{EC}/system/files/2019-10/2020_dbp_it_it.pdf", "it"),
    2021: ("DPB 2021, submitted October 2020", f"{EC}/system/files/2020-11/2021_dbp_it_it.pdf", "it"),
    2022: ("DPB 2022, submitted October 2021", "https://www.rgs.mef.gov.it/_Documenti/VERSIONE-I/Attivit--i/Contabilit_e_finanza_pubblica/DPB/2022/IT-DPB-2022.pdf", "it"),
    2023: ("DPB 2023, updated version of November 2022 (Meloni government; the October one was the outgoing government's)", f"{EC}/document/download/ff1cb4d2-6c5a-4f1f-8fe0-f6307188419f_it?filename=2023_dbp_it_update_it.pdf", "it"),
    2024: ("DPB 2024, submitted 16 October 2023", f"{EC}/document/download/b9e79133-9a0b-4610-808a-6fa13aa2bd75_it?filename=2024_dbp_it_it.pdf", "it"),
    2025: ("DPB 2025, submitted 16 October 2024", f"{EC}/document/download/f0d459e2-ecfd-425b-b2ca-622cebe08663_it?filename=2025_dbp_it_it.pdf", "it"),
}
# cross-check of the 2023-2026 targets against the programmatic row of the budget-law dossier. budget year -> (budget-law dossier, position of the budget year among the percentages of the "programmatic" row: that row starts one year earlier in the 2024 and 2025 dossiers)
DOSSIER_TARGETS = {2023: ("N19_DFP006", 0), 2024: ("L19_DFP11", 1), 2025: ("L19_DFP14", 1), 2026: ("L19_DFP17", 0)}
REVISED = {2019: ("N18_DFP006", r"programmatico[^.]{0,40}portato al (\d+,\d+) per cento")}  # final target stated in the dossier's text
NUM = r"(-?\d+[.,]\d+)"


def path(year: int) -> Path:
    return RAW_DIR / f"DPB{year}.pdf"


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year, (title, url, _) in DPB.items():
        if path(year).exists() and not force:
            continue
        resp = requests.get(url, headers=UA, timeout=180)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            raise RuntimeError(f"DPB {year}: not a PDF: {resp.content[:60]!r}")
        path(year).write_bytes(resp.content)
        record(f"dpb_{year}", source="draft budgetary plan", dataset=title, url=url)
    return RAW_DIR


def _text(pdf: Path) -> str:
    return subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout


def _f(token: str) -> float:
    return float(token.replace(",", "."))


def parse_dpb(pdf: Path) -> tuple[float, float]:
    """(target for the budget year from the macro table, the same figure from the S.13 sub-sector table); the two must agree."""
    t = _text(pdf)
    macro = [m for m in re.finditer(rf"(?:general government|Amministrazioni pubbliche)[^\n]*?B\.?9\s+{NUM}\s+{NUM}\s+{NUM}\s*$", t, re.IGNORECASE | re.MULTILINE)]
    sub = [m for m in re.finditer(rf"S\.13\s+{NUM}\s+{NUM}", t) if "S.1311" not in m.group()]
    if not macro or not sub:
        raise ValueError(f"{pdf.name}: net-lending rows not found")
    return _f(macro[0].group(3)), _f(sub[0].group(2))


def _level(token: str) -> float:
    """GDP level in EUR million, English (`1,573,002.2`) or Italian (`1.718.515`) number format."""
    return float(token.replace(",", "")) if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", token) else float(token.replace(".", "").replace(",", "."))


def parse_components(pdf: Path) -> dict[str, float]:
    """From the target tables of a draft budgetary plan: revenue, expenditure and interest targets of the budget year (% of GDP, the second of the two columns t-1 and t), and the forecast growth of real and nominal GDP for that year
    (third growth value after the level, in the first two rows of the macro table: real, then nominal GDP), with the nominal GDP level of year t-2 and the growth of t-1 from which the forecast level of year t follows. The target table is the second table with a `TR` row (the first one is the projection at unchanged policies)."""
    t = _text(pdf)
    tr = [m for m in re.finditer(rf"\bTR\s+{NUM}\s+{NUM}", t)]
    te = [m for m in re.finditer(rf"\bTE\s*(?:\(\d\))?\s+{NUM}\s+{NUM}", t)]
    if len(tr) < 2 or len(te) < 2:
        raise ValueError(f"{pdf.name}: target tables not found")
    interest = re.search(rf"D\.41\s+{NUM}\s+{NUM}", t[tr[1].end():])
    growth = [m for m in re.finditer(rf"B1\*g\s+([\d.,]{{6,}})\s+((?:{NUM}\s+){{3,}})", t)]
    if interest is None or len(growth) < 2:
        raise ValueError(f"{pdf.name}: interest or growth rows not found")
    def g(m: re.Match, k: int) -> float:
        return [_f(x) for x in re.findall(NUM, m.group(2))][k]
    block = t[te[1].end(): te[1].end() + 3500]  # rows 2.1-2.8 of the expenditure target table
    cat = {}
    for key, pat in (("D1", r"\bD\.1\s+"), ("P2", r"\bP\.2\s+"), ("D62_D632", r"D\.62,\s*D\.63\d?\s+"), ("D3", r"\bD\.3\s+"), ("P51G", r"\bP\.51g?\s+"), ("D9", r"\bD\.9\s+"), ("other", r"\b2\.8\.[^\n]*?\s+(?=-?\d+[.,]\d\s+-?\d+[.,]\d)")):
        m = re.search(pat + rf"{NUM}\s+{NUM}", block)
        cat[f"target_{key}"] = _f(m.group(2)) if m else float("nan")
    return {**cat, "revenue_target": _f(tr[1].group(2)), "expenditure_target": _f(te[1].group(2)), "interest_target": _f(interest.group(2)), "real_growth_forecast": g(growth[0], 2), "nominal_growth_forecast": g(growth[1], 2),
            "nominal_growth_tm1": g(growth[1], 1), "nominal_gdp_level_tm2": _level(growth[1].group(1))}


def dossier_target(doc: str, index: int) -> float:
    """The budget year's figure from the "indebitamento netto programmatico" row of a budget-law dossier (percentages, signs as printed)."""
    t = budget_laws.pages(budget_laws.path(doc))
    for page in t:
        for line in page.split("\n"):
            if re.search(r"Indebitamento netto programmatico", line):
                pct = re.findall(r"(?<![\d.,])(-?\d{1,2},\d{1,2})(?![\d.])", line)  # percentages of GDP (decimal comma, no thousands dot), with or without the % sign
                if len(pct) > index:
                    return _f(pct[index])
    raise ValueError(f"{doc}: programmatic row not found")


def build_csv(out: Path = CSV) -> pd.DataFrame:
    """One row per budget year: the net-borrowing target and, for 2016-2025 (draft budgetary plans), the revenue, expenditure and interest targets and the forecast growth of real and nominal GDP. 2023-2025 are also read from the
    dossiers' programmatic row and must agree with the plan; 2026 (no plan parsed: the new framework changed the template) comes from its dossier only."""
    rows = []
    for year, (title, url, lang) in DPB.items():
        macro, sub = parse_dpb(path(year))
        if abs(macro - sub) > 0.051:
            raise ValueError(f"DPB {year}: the macro table gives {macro} and the sub-sector table {sub}")
        if year in DOSSIER_TARGETS and abs(dossier_target(*DOSSIER_TARGETS[year]) - macro) > 0.051:
            raise ValueError(f"DPB {year}: the plan gives {macro} and the dossier {dossier_target(*DOSSIER_TARGETS[year])}")
        rows.append({"budget_year": year, "source": title, "target_pct_gdp": macro, "target_final_pct_gdp": macro, **parse_components(path(year)), "evidence": "two_tables_agree" + ("_and_dossier" if year in DOSSIER_TARGETS else "")})
    for year, (doc, idx) in DOSSIER_TARGETS.items():
        if year not in DPB:
            rows.append({"budget_year": year, "source": f"programmatic row of the dossier {doc}", "target_pct_gdp": dossier_target(doc, idx), "target_final_pct_gdp": None, "evidence": "dossier_row"})
    d = pd.DataFrame(rows)
    for year, (doc, pat) in REVISED.items():
        text = re.sub(r"\s+", " ", _text(budget_laws.path(doc)))
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            d.loc[d.budget_year == year, "target_final_pct_gdp"] = -_f(m.group(1))
    d["target_final_pct_gdp"] = d.target_final_pct_gdp.fillna(d.target_pct_gdp)
    d = d.sort_values("budget_year", ignore_index=True)
    d.to_csv(out, index=False)
    return d


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "fetch":
        print(fetch("--force" in sys.argv))
    else:
        print(build_csv().to_string(index=False))
