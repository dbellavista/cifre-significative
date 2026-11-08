"""Cesena's staffing plan (programmazione del fabbisogno di personale) as published in the DUP, one vintage per file.

Two tables are read from every Documento Unico di Programmazione and its update notes: the personnel-spending limit of art. 1 c. 557 L. 296/2006 (outturn of the last closed year plus the budget years: spending of the Comune, the staff transferred to the Unione Valle Savio, and their sum) and, from the 2025-2029 DUP on, the permanent staff in service by contract area with its reference date. The same year appears in several vintages: the first one that lists it as a budget year is the plan, the one that lists it as outturn the result.
"""
import re
from pathlib import Path

import pandas as pd

from cifresignificative.core.geo import CESENA
from cifresignificative.studies.cesena.analysis.conto_annuale import tokens
from cifresignificative.studies.cesena.analysis.indicatori import pdf_text
from cifresignificative.studies.cesena.analysis.rendiconto import RAW_DIR

LIMIT_TITLE = "LIMITI SPESA DEL PERSONALE"
YEAR_TOKEN = re.compile(r"\b(20[12]\d)\b")
DATE = re.compile(r"in servizio a tempo\s+indeterminato\s+alla data del (\d\d/\d\d/\d{4})")
AREAS = ["operatori", "operatori_esperti", "istruttori", "funzionari_eq", "dirigenti"]


def parse_limits(text: str) -> pd.DataFrame:
    """Rows (year, kind, outturn, commitments_eur, municipal_eur, unione_eur, total_eur) of the first spending-limit table: `municipal_eur` is the first `Totale spesa personale` (after the exclusions), `total_eur` the second (plus staff transferred to the Unione)."""
    a = text.find(LIMIT_TITLE)
    if a < 0:
        return pd.DataFrame()
    lines = text[a:].splitlines()[:140]
    head = " ".join(lines[1:8])
    years = sorted({int(y) for y in YEAR_TOKEN.findall(head)})  # headers wrap, so the printed order can be 2024 before 2023
    n_out = len(re.findall(r"Consuntivo", head))
    totals, vals = [], {}
    for ln in lines:
        for key, name in (("impegni personale (macro 01)", "commitments_eur"), ("Personale trasferito ad Unione", "unione_eur")):
            if key in ln and name not in vals:
                vals[name] = tokens(ln.split(key, 1)[1])[1]
        if "Totale spesa personale" in ln:
            totals.append(tokens(ln.split("Totale spesa personale", 1)[1])[1])
    if not years or len(totals) < 2 or len(totals[1]) != len(years) or len(totals[0]) != len(years):
        return pd.DataFrame()
    vals["municipal_eur"], vals["total_eur"] = totals[0], totals[1]
    rows = []
    for i, y in enumerate(years):
        r = {"comune_id": CESENA, "year": y, "outturn": i < n_out}
        r |= {k: (v[i] if len(v) == len(years) else float("nan")) for k, v in vals.items()}
        rows.append(r)
    return pd.DataFrame(rows)


def parse_staff(text: str) -> dict | None:
    """Permanent staff in service from the `Personale a tempo indeterminato` total row (areas, total, posts filled, full-time, part-time) and the fixed-term total, with the reference date; None when the table is absent."""
    m = DATE.search(text)
    row = next((tokens(ln)[1] for ln in text.splitlines() if re.match(r"\s*Personale a tempo indeterminato\s+\d", ln)), None)
    term = next((tokens(ln)[1] for ln in text.splitlines() if re.match(r"\s*personale a tempo determinato\s+\d", ln)), None)
    if m is None or row is None or len(row) < 8 or abs(sum(row[:5]) - row[5]) > 0.5 or abs(row[6] + row[7] - row[5]) > 0.5:
        return None
    d, mm, y = m[1].split("/")
    return {"comune_id": CESENA, "ref_date": f"{y}-{mm}-{d}", **dict(zip(AREAS, row[:5], strict=True)), "heads": row[5], "full_time": row[6], "part_time": row[7], "fixed_term": term[4] if term and len(term) > 4 else float("nan")}


def parse_retirements(text: str) -> pd.DataFrame:
    """Rows (year, retirements, saving_eur) of the `CESSAZIONI ECONOMIA` forecast table (Sept 2025 DUP layout); TOTALE excluded."""
    a = text.find("CESSAZIONI")
    rows = []
    for ln in text[a:].splitlines()[:12] if a >= 0 else []:
        _, v = tokens(ln)
        if len(v) == 3 and 2020 <= v[0] <= 2040:
            rows.append({"comune_id": CESENA, "year": int(v[0]), "retirements": v[1], "saving_eur": v[2]})
    return pd.DataFrame(rows)


def dup_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    return sorted((f for f in (raw_dir / "files").glob("*/*[Dd][Uu][Pp]*.pdf") if not re.search(r"Estratto|Obiettivi", f.name)), key=lambda f: int(f.name.split("_")[0]))


def load_limits(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Every vintage: rows keyed by `source` (file stem), `order` (position in publication order) and `vintage` (first year of its table); a row identical to one of an earlier file is dropped (update notes often repeat the table)."""
    parts = []
    for order, f in enumerate(dup_files(raw_dir)):  # portal upload ids rise with publication
        t = parse_limits(pdf_text(f))
        if len(t):
            parts.append(t.assign(source=f.stem, vintage=int(t["year"].min()), order=order))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["year", "outturn", "commitments_eur", "municipal_eur", "unione_eur", "total_eur"], keep="first").reset_index(drop=True)


def plan_vs_outturn(limits: pd.DataFrame, column: str = "total_eur") -> pd.DataFrame:
    """Per year closed as outturn: the last published plan (latest file listing it as a budget year, which is the amended budget) against the first outturn, and the gap in percent of the plan."""
    rows = []
    for y in sorted(limits.loc[limits["outturn"], "year"].unique()):
        plan, out = limits[(limits["year"] == y) & ~limits["outturn"]], limits[(limits["year"] == y) & limits["outturn"]]
        if len(plan) and len(out):
            p, o = plan.sort_values("order").iloc[-1], out.sort_values("order").iloc[0]
            rows.append({"year": int(y), "plan_eur": p[column], "outturn_eur": o[column], "gap_pct": (o[column] / p[column] - 1) * 100, "plan_source": p["source"]})
    return pd.DataFrame(rows)


def load_staff(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    rows = [r | {"source": f.stem} for f in dup_files(raw_dir) if (r := parse_staff(pdf_text(f)))]
    return pd.DataFrame(rows).drop_duplicates(["ref_date", "heads", "fixed_term"]).sort_values("ref_date").reset_index(drop=True) if rows else pd.DataFrame()


def load_retirements(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    parts = [parse_retirements(pdf_text(f)).assign(source=f.stem) for f in dup_files(raw_dir)]
    parts = [t for t in parts if len(t)]
    return pd.concat(parts, ignore_index=True).drop_duplicates("year", keep="last").reset_index(drop=True) if parts else pd.DataFrame()  # the latest forecast wins
