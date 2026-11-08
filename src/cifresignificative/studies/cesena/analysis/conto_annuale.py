"""Cesena's Conto annuale del personale (Ragioneria generale dello Stato), from the PDFs on the municipal transparency portal.

Each yearly PDF opens with a three-year summary of permanent staff (Tab. 1: heads at 31 December, full-time equivalents from the months paid, spending on fixed and accessory pay, arrears) and of the total annual labour cost (Tab. 12 + 13 + 14). The PDFs for 2022, 2023 and 2024 therefore cover 2020-2024; where two files give a year the later one wins (it carries any revision). The table is the municipality alone: the Unione's staff is not in it.
"""
import re
from pathlib import Path

import pandas as pd

from cifresignificative.core.geo import CESENA
from cifresignificative.studies.cesena.analysis.indicatori import pdf_text
from cifresignificative.studies.cesena.analysis.rendiconto import RAW_DIR

YEAR = re.compile(r"Anno\s*:\s*(\d{4})")
NUMBER = re.compile(r"-?\d[\d.]*(?:,\d+)?")


def numbers(line: str) -> list[float]:
    """Dots are always thousands separators and commas decimals in these PDFs (`922.594`, `502,27`)."""
    return [float(m.replace(".", "").replace(",", ".")) for m in NUMBER.findall(line)]


def parse_summary(text: str) -> pd.DataFrame:
    """Rows (year, heads, fte, pay_eur, arrears_eur, labour_cost_eur) for the three years of the summary; empty when the layout is not found."""
    m = YEAR.search(text)
    total = next((numbers(ln) for ln in text.splitlines() if ln.startswith("Totale ") and len(numbers(ln)) == 12), None)
    cost = next((numbers(ln.split(")", 1)[1]) for ln in text.splitlines() if "Totale costo annuo del lavoro" in ln), None)
    if m is None or total is None or cost is None or len(cost) != 3:
        return pd.DataFrame()
    last = int(m[1])
    years = [last - 2, last - 1, last]
    return pd.DataFrame({"comune_id": CESENA, "year": years, "heads": total[0:3], "fte": total[3:6], "pay_eur": total[6:9], "arrears_eur": total[9:12], "labour_cost_eur": cost})


def load_conto_annuale(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Every `*CONTOANNUALE*` PDF; the file of the latest survey year wins for a year covered twice."""
    parts = []
    for f in sorted((raw_dir / "files").glob("*/*[Cc][Oo][Nn][Tt][Oo][Aa][Nn][Nn][Uu][Aa][Ll][Ee]*.pdf")):
        t = parse_summary(pdf_text(f))
        if len(t):
            parts.append(t.assign(survey=int(t["year"].max())))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_values(["year", "survey"]).drop_duplicates("year", keep="last")
    return out.drop(columns="survey").reset_index(drop=True)


# --- detail tables of the same PDFs: flexible staff, pay by contract level, decentralised-bargaining funds ---
FLEX_ROWS = {"Personale a tempo": "fixed_term", "L.S.U.": "lsu", "Lavoratori Interinali": "agency", "Con Contratti formazione": "training", "N. contratti co": "cococo", "N. incarichi di": "consultancy", "N. contratti per prestazioni": "services"}
FUND_ROWS = {"Fondo risorse decentrate": "decentralised_fund", "Posizioni organizzative": "po", "Straordinario": "overtime", "Incarichi di Elevata": "eq", "Risorse per la retribuzione": "dirigenti", "Segretario comunale": "secretary"}
LEVEL_ROWS = {"SEGRETARI": "secretary", "DIRIGENTI E ALTE": "dirigenti_outside_org", "DIRIGENTI": "dirigenti", "CATEGORIA D": "cat_D", "CATEGORIA C": "cat_C", "CATEGORIA B": "cat_B", "CATEGORIA A": "cat_A", "FUNZIONARI ED": "area_funzionari_eq", "ISTRUTTORI": "area_istruttori", "OPERATORI ESPERTI": "area_operatori_esperti", "OPERATORI": "area_operatori", "RESTANTE": "other_staff", "Totale per Istituzione": "total"}
NUM_TOKEN = re.compile(r"-?\d[\d.]*(?:,\d+)?")


def tokens(line: str) -> tuple[str, list[float]]:
    """(text before the first numeric token, the whole-token numbers of the line; a lone `-` is a blank cell and becomes 0). `P015`, `(SI1)` and `2022)` are not numbers."""
    label, vals = [], []
    for t in line.split():
        if NUM_TOKEN.fullmatch(t) or t == "-":
            vals.append(0.0 if t == "-" else float(t.replace(".", "").replace(",", ".")))
        elif not vals:
            label.append(t)
    return " ".join(label), vals


def section(text: str, start: str, end: str) -> list[str]:
    a = text.find(start)
    b = text.find(end, a + 1) if a >= 0 else -1
    return text[a:b].splitlines() if a >= 0 and b > a else []


def survey_year(text: str) -> int | None:
    m = YEAR.search(text)
    return int(m[1]) if m else None


def match_row(label: str, rows: dict[str, str]) -> str | None:
    """First key of `rows` that starts the label (keys are ordered most specific first)."""
    return next((v for k, v in rows.items() if label.startswith(k)), None)


def parse_flexible(text: str) -> pd.DataFrame:
    """Rows (year, kind, units, cost_eur, avg_cost_eur) from `Personale Flessibile (Tab. 2 e SI1)`: n of contracts (FTE for staff), labour cost (Tab. 14) and mean per contract, for the three years."""
    last, rows = survey_year(text), []
    for ln in section(text, "Personale Flessibile (Tab. 2A)", "Personale Flessibile (Tab. 2 e SI1)"):  # heads on fixed-term contracts (T2A): the TOTALE row
        label, v = tokens(ln)
        if label == "TOTALE" and last and len(v) == 3:
            rows += [{"comune_id": CESENA, "year": last - 2 + i, "kind": "fixed_term_heads", "units": v[i], "cost_eur": float("nan"), "avg_cost_eur": float("nan")} for i in range(3)]
    for ln in section(text, "Personale Flessibile (Tab. 2 e SI1)", "Contrattazione Integrativa - Dati"):
        label, v = tokens(ln)
        kind = match_row(label, FLEX_ROWS)
        if kind and last and len(v) == 9:
            rows += [{"comune_id": CESENA, "year": last - 2 + i, "kind": kind, "units": v[i], "cost_eur": v[3 + i], "avg_cost_eur": v[6 + i]} for i in range(3)]
    return pd.DataFrame(rows)


def parse_levels(text: str) -> pd.DataFrame:
    """Rows (year, level, fte, avg_total_eur, avg_fixed_eur, avg_accessory_eur) from `Spese medie pro-capite` (pages 2-3). Levels with no FTE in a year are not in the contract classification of that year, so they are dropped; `regime` is `categorie` up to 2022 and `aree` from the CCNL 2019-21 (2023): the two are not comparable level by level."""
    last, rows = survey_year(text), []
    for ln in section(text, "Spese medie pro-capite annue in euro", "1. Le Spese medie annue"):
        label, v = tokens(ln)
        level = match_row(label, LEVEL_ROWS)
        if level and last and len(v) in (18, 15):  # 2022 file carries 18 columns, a row missing the last arrears pair 15
            for i in range(3):
                if v[i] > 0:
                    rows.append({"comune_id": CESENA, "year": last - 2 + i, "level": level, "fte": v[i], "avg_total_eur": v[3 + i], "avg_fixed_eur": v[6 + i], "avg_accessory_eur": v[9 + i]})
    out = pd.DataFrame(rows)
    return out.assign(regime=out["level"].map(lambda x: "aree" if x.startswith("area_") else "categorie" if x.startswith("cat_") else "both")) if len(out) else out


def parse_funds(text: str) -> pd.DataFrame:
    """Rows (year, fund, resources_eur, used_eur) from `Contrattazione Integrativa - Dati riepilogativi` (resources to finance the fund, then its use, three years each)."""
    last, rows = survey_year(text), []
    for ln in section(text, "Contrattazione Integrativa - Dati", "Tabella di Riconciliazione"):
        label, v = tokens(ln.replace("PERSONALE NON DIRIGENTE", "").replace("DIRIGENTI", "").replace("SEGRETARI", ""))
        fund = match_row(label, FUND_ROWS)
        if fund and last and len(v) == 6:
            rows += [{"comune_id": CESENA, "year": last - 2 + i, "fund": fund, "resources_eur": v[i], "used_eur": v[3 + i]} for i in range(3)]
    return pd.DataFrame(rows)


def load_detail(parser, key: list[str], raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Apply `parser` to every Conto annuale PDF; the latest survey wins for a year covered twice (it carries the revision)."""
    parts = []
    for f in sorted((raw_dir / "files").glob("*/*[Cc][Oo][Nn][Tt][Oo][Aa][Nn][Nn][Uu][Aa][Ll][Ee]*.pdf")):
        text = pdf_text(f)
        t = parser(text)
        if len(t):
            parts.append(t.assign(survey=survey_year(text)))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_values(["year", *key, "survey"]).drop_duplicates(["year", *key], keep="last")
    return out.drop(columns="survey").reset_index(drop=True)


def load_flexible(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return load_detail(parse_flexible, ["kind"], raw_dir)


def load_levels(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return load_detail(parse_levels, ["level"], raw_dir)


def load_funds(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return load_detail(parse_funds, ["fund"], raw_dir)
