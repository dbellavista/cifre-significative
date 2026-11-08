"""Cesena's report on the crisis risk of the controlled companies ("Rischio di crisi aziendale controllate"), one table per annual accounts year.

The Comune's half-year report ("presa d'atto") carries, for each controlled company, three yes/no alarms (operating result negative for three years, cumulated losses above 15% of equity, going-concern doubt in the auditor's report) and three ratios with the threshold in the row label: the financial-structure index (equity plus medium and long-term debt over fixed assets, alarm below 0.8), the liquidity index (current assets over current liabilities, alarm below 1) and the financial charges over turnover (alarm above 5%). The report prints an alarm only for the three yes/no rows; the ratio alarms are applied here from the thresholds in the labels. The companies are those under control, not all holdings (Cesena Fiera and Romagna Acque are in other lists).
"""
import re
from pathlib import Path

import pandas as pd

from cifresignificative.core.geo import CESENA
from cifresignificative.studies.cesena.analysis.indicatori import pdf_text
from cifresignificative.studies.cesena.analysis.rendiconto import RAW_DIR

COMPANIES = [("Energie", "Energie per la Città"), ("Ortofrutticola", "Filiera Ortofrutticola Romagnola"), (r"\bATR\b", "ATR"), ("Techne", "Techne"), ("Unica", "Unica Reti"), ("LEPIDA", "Lepida"), (r"\bacque\b", "Romagna Acque"), (r"Ser\.in\.ar", "Ser.In.Ar")]
YES_NO = {"NO": False, "SI": True, "SÌ": True}
PCT, RATIO = re.compile(r"\d+,\d+%"), re.compile(r"\d+,\d+")


def to_float(t: str) -> float:
    return float(t.rstrip("%").replace(",", "."))


def parse_risk(text: str) -> pd.DataFrame:
    """Rows (company, year, negative_operations_3y, losses_over_15pct, going_concern_doubt, structure_index, liquidity_index, financial_charges_pct) with the three computed ratio alarms; empty when the layout is not found. Columns are named from the header, left to right by position."""
    lines = text.splitlines()
    top = next((i for i, ln in enumerate(lines) if "Gestione operativa" in ln), None)
    year = re.search(r"Soglie di allarme da bilancio\s+.*?\n.*?\b(20\d\d)\b", "\n".join(lines[:6]))
    if top is None or year is None:
        return pd.DataFrame()
    head = lines[:top]
    cols = sorted((min(m.start() for ln in head if (m := re.search(pat, ln))), name) for pat, name in COMPANIES if any(re.search(pat, ln) for ln in head))
    names = [n for _, n in cols]
    flags, ratios, charges = [], None, None
    for ln in lines[top:]:
        tail = ln.split()[-len(names):]
        if len(tail) < len(names):
            continue
        if all(t in YES_NO for t in tail):
            flags.append([YES_NO[t] for t in tail])
        elif all(PCT.fullmatch(t) for t in tail):
            charges = [to_float(t) for t in tail]
        elif all(RATIO.fullmatch(t) for t in tail):
            ratios = ratios + [[to_float(t) for t in tail]] if ratios else [[to_float(t) for t in tail]]
    if len(flags) != 3 or not ratios or len(ratios) != 2 or charges is None:
        return pd.DataFrame()
    out = pd.DataFrame({"comune_id": CESENA, "company": names, "year": int(year[1]), "negative_operations_3y": flags[0], "losses_over_15pct": flags[1], "going_concern_doubt": flags[2], "structure_index": ratios[0], "liquidity_index": ratios[1], "financial_charges_pct": charges})
    out["structure_alarm"] = out["structure_index"] < 0.8
    out["liquidity_alarm"] = out["liquidity_index"] < 1
    out["charges_alarm"] = out["financial_charges_pct"] > 5
    return out


def load_risk(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    parts = [parse_risk(pdf_text(f)) for f in sorted((raw_dir / "files").glob("*/*[Rr]ischiodicrisiaziendale*.pdf"))]
    parts = [t for t in parts if len(t)]
    return pd.concat(parts, ignore_index=True).sort_values(["year", "company"]).reset_index(drop=True) if parts else pd.DataFrame()
