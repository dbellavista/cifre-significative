"""Half-year report on the controlled companies ("Report semestrale società partecipate al 30/06"), from the PDFs on the municipal transparency portal.

The Comune collects from each company a half-year income statement (sheet 1.a) and a sheet of financial position ("Situazione contabile": net financial position, PFN over EBITDA). One page per group of companies (4-5 columns), the company names in a header of two or three lines above the first row. Companies change between editions (Filiera Ortofrutticola and Unica Reti and Romagna Acque in 2024, not in 2026; Cesena Fiera only in some). Files exist for 2024 and 2026 (no report for 2025 is on the portal); the figures are the companies' own unaudited half-year accounts, euros, and the basis differs by company (a balance "al 30/6", a "situazione", Romagna Acque a "preconsuntivo"): compare a company with itself across years, not companies with each other.
The net result is printed on both sheets and the two can disagree (ATR 2026: 0 on the income statement, 71,600 on the financial sheet): both are kept (`net_result_eur`, `net_result_sheet2_eur`). The comparison page with the previous year, whose cells are broken across lines, is not parsed.
"""
import re
from pathlib import Path

import pandas as pd

from cifresignificative.core.geo import CESENA
from cifresignificative.studies.cesena.analysis.indicatori import pdf_text
from cifresignificative.studies.cesena.analysis.rendiconto import RAW_DIR

COMPANIES = [(r"ENERGIE", "Energie per la Città"), (r"FILIERA", "Filiera Ortofrutticola Romagnola"), (r"Cesena Fiera", "Cesena Fiera"), (r"UNICA RETI", "Unica Reti"), (r"ROMAGNA ACQUE", "Romagna Acque"), (r"TECHN", "Techne"), (r"\bATR\b", "ATR"), (r"SER\.IN\.AR", "Ser.In.Ar")]
INCOME = {"Ricavi delle vendite": "sales_eur", "Valore della Produzione Operativa": "operating_output_eur", "Costi esterni operativi": "external_costs_eur", "Valore Aggiunto": "value_added_eur", "Costi del personale": "personnel_cost_eur",
          "Margine Operativo Lordo": "ebitda_eur", "Ammortamenti e accantonamenti": "depreciation_eur", "Risultato Operativo": "operating_result_eur", "Risultato area accessoria": "ancillary_result_eur", "Risultato area finanziaria": "financial_result_eur",
          "Risultato Lordo": "pretax_result_eur", "Imposte": "tax_eur", "RISULTATO NETTO": "net_result_eur"}
FINANCE = {"RISULTATO NETTO": "net_result_sheet2_eur", "Posizione Finanziaria Netta": "pfn_eur", "PFN / EBITDA": "pfn_over_ebitda"}
NUMBER = re.compile(r"-?\d{1,3}(?:\.\d{3})*(?:,\d+)?|-?\d+(?:,\d+)?")
YEAR = re.compile(r"AL 30/06/(\d{4})")


def to_float(t: str) -> float:
    """`-1.450.871` (dot thousands, no decimals) or `42,374` (decimal comma)."""
    return float(t.replace(".", "").replace(",", "."))


def page_companies(head: list[str]) -> list[str]:
    """Company names left to right by the column where each first appears in the header lines."""
    found = [(min(m.start() for ln in head if (m := re.search(pat, ln))), name) for pat, name in COMPANIES if any(re.search(pat, ln) for ln in head)]
    return [name for _, name in sorted(found)]


def page_rows(page: str, labels: dict[str, str], start: str) -> dict[str, list]:
    """`company` plus label column -> one value per company, for the rows after the line containing `start`; a row whose last tokens are not all numbers (a wrapped label, another table) is skipped."""
    lines = page.splitlines()
    top = next(i for i, ln in enumerate(lines) if start in ln)
    first = next((i for i in range(top + 1, len(lines)) if any(lines[i].strip().startswith(lab) for lab in labels)), len(lines))
    names = page_companies(lines[:first])  # the header is above the marker (income) or between marker and first row (finance)
    out: dict[str, list[float]] = {}
    for ln in lines[top + 1:]:
        tail = ln.split()[-len(names):] if names else []
        label = next((col for lab, col in labels.items() if ln.strip().startswith(lab)), None)
        if label and label not in out and len(tail) == len(names) and all(NUMBER.fullmatch(t) for t in tail):
            out[label] = [to_float(t) for t in tail]
    return {"company": names, **out} if out else {}


def parse_semestrale(text: str) -> pd.DataFrame:
    """Rows (company, year, the income-statement and financial-position columns above), one per company; empty when the layout is not found."""
    year = YEAR.search(text)
    if year is None:
        return pd.DataFrame()
    parts: dict[str, pd.DataFrame] = {}
    for page in text.split("\f"):
        for key, labels, start in (("income", INCOME, "1.a Conto Economico"), ("finance", FINANCE, "Situazione Contabile")):
            if start in page and (rows := page_rows(page, labels, start)):
                parts[f"{key}{len(parts)}"] = pd.DataFrame(rows)
    income = [t for k, t in parts.items() if k.startswith("income")]
    finance = [t for k, t in parts.items() if k.startswith("finance")]
    if not income:
        return pd.DataFrame()
    out = pd.concat(income, ignore_index=True)
    if finance:
        out = out.merge(pd.concat(finance, ignore_index=True), on="company", how="left")
    out.insert(0, "year", int(year[1]))
    out.insert(0, "comune_id", CESENA)
    return out.drop_duplicates(["year", "company"]).reset_index(drop=True)


def load_semestrale(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted({*(raw_dir / "files").glob("*/*prospettocompleto*.pdf"), *(raw_dir / "files").glob("*/*REPORTSEM*.pdf")})
    parts = [t for f in files if len(t := parse_semestrale(pdf_text(f)))]
    return pd.concat(parts, ignore_index=True).sort_values(["year", "company"]).reset_index(drop=True) if parts else pd.DataFrame()
