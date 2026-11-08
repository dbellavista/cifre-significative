"""Accrual figures of Cesena from the Rendiconto CSVs attached to the approval acts on the municipal transparency portal (`core.sources.trasparenza`).

One city, years 2022 onwards from the Liferay portal, 2018-2021 from the old site (`core.sources.comune_archive`, same CSV names but a different layout: amounts zero-padded as `000051026819,07`, revenue by Tipologia with the Titolo totals, expenditure by macroaggregate with the three totals (competenza, FPV, cassa) as the last columns). The CSVs exist for no other municipality. Use it only as a single-city anchor for the SIOPE cash series, never for the peer ranking.
"""
import csv
import re
from pathlib import Path

import pandas as pd

from cifresignificative.core.geo import CESENA

RAW_DIR = Path("data/raw/trasparenza/cesena")
ARCHIVE_DIR = Path("data/raw/trasparenza/sititematici")
YEAR = re.compile(r"RENDICONTO\s+(\d{4})")
OLD_YEAR = re.compile(r"RENDICONTO ANNO:\s*(\d{4})")


def num(s: str) -> float:
    """Italian number: `1.234,56` (dot thousands) or `1234,56`; empty is 0."""
    s = s.strip()
    if not s:
        return 0.0
    return float(s.replace(".", "").replace(",", ".")) if "," in s and "." in s else float(s.replace(",", "."))


def read_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="latin1", newline="") as f:
        return [r + [""] * (2 - len(r)) for r in csv.reader(f, delimiter=";")]  # short title rows get empty columns


def parse_entrate(rows: list[list[str]]) -> dict[str, float]:
    """Current revenue (Titoli 1-3, summed over Tipologie): accrued and collected."""
    cur = [r for r in rows if re.match(r"[123]-", r[0]) and r[1].startswith("TIPOLOGIA")]
    return {"entrate_correnti_acc_eur": sum(num(r[2]) for r in cur), "entrate_correnti_risc_eur": sum(num(r[3]) for r in cur)}


def parse_spese(rows: list[list[str]]) -> dict[str, float]:
    """Committed expenditure from the three total columns (Impegnato, FPV, Pagato) that close every row: personnel and interest are macroaggregates of Titolo 1, repayment is the Titolo 4 total."""
    def committed(r: list[str]) -> float:
        return num(r[-3])

    return {
        "spesa_personale_imp_eur": sum(committed(r) for r in rows if r[0].startswith("1-") and r[1].startswith("MACROAGGR. 1 -")),
        "interessi_passivi_imp_eur": sum(committed(r) for r in rows if r[0].startswith("1-") and r[1].startswith("MACROAGGR. 7 ")),
        "rimborso_prestiti_imp_eur": sum(committed(r) for r in rows if r[0].startswith("TOTALE TITOLO 4")),
        "spese_correnti_imp_eur": sum(committed(r) for r in rows if r[0].startswith("TOTALE TITOLO 1")),
    }


def accrual_row(entrate: list[list[str]], spese: list[list[str]]) -> dict[str, float | int | str] | None:
    """One year, or None when the revenue file is not in the Titolo x Tipologia layout or names no year."""
    m = YEAR.search(entrate[0][0]) if entrate else None
    if m is None or not any(r[1].startswith("TIPOLOGIA") for r in entrate if len(r) > 1):
        return None
    row: dict[str, float | int | str] = {"comune_id": CESENA, "year": int(m.group(1)), **parse_entrate(entrate), **parse_spese(spese)}
    fixed = float(row["spesa_personale_imp_eur"]) + float(row["interessi_passivi_imp_eur"]) + float(row["rimborso_prestiti_imp_eur"])
    row["rigidity_accrual_pct"] = fixed / float(row["entrate_correnti_acc_eur"]) * 100
    return row


def archive_row(entrate: list[list[str]], spese: list[list[str]]) -> dict[str, float | int | str] | None:
    """One year of the old-site layout (2018-2021), same columns as `accrual_row`; None when a file names no year or lacks a Titolo 1 or Titolo 4 total. Personnel and interest are the macroaggregates 101 and 107 inside Titolo 1 (code 100 is its total), repayment is `Totale TITOLO 4`; the total columns close every expenditure row."""
    m = next((OLD_YEAR.search(r[0]) for r in entrate[:3] + spese[:3] if r and OLD_YEAR.search(r[0])), None)
    cur = [num(r[2]) for r in entrate if r[0] in ("10000", "20000", "30000") and len(r) > 2]
    start = next((i for i, r in enumerate(spese) if len(r) > 1 and r[1].strip() == "TITOLO 1"), None)
    rep = next((num(r[-3]) for r in spese if len(r) > 3 and r[1].strip().startswith("Totale TITOLO 4")), None)
    if m is None or len(cur) != 3 or start is None or rep is None:
        return None
    t1 = {r[0]: num(r[-3]) for r in spese[start + 1:] if len(r) > 3 and r[0] in ("101", "107", "100")}
    if {"101", "107", "100"} - t1.keys():
        return None
    row: dict[str, float | int | str] = {"comune_id": CESENA, "year": int(m[1]), "entrate_correnti_acc_eur": sum(cur), "entrate_correnti_risc_eur": sum(num(r[3]) for r in entrate if r[0] in ("10000", "20000", "30000")),
                                         "spesa_personale_imp_eur": t1["101"], "interessi_passivi_imp_eur": t1["107"], "rimborso_prestiti_imp_eur": rep, "spese_correnti_imp_eur": t1["100"]}
    row["rigidity_accrual_pct"] = (t1["101"] + t1["107"] + rep) / sum(cur) * 100
    return row


def load_archive_accruals(archive_dir: Path = ARCHIVE_DIR) -> pd.DataFrame:
    """Rendiconto 2018-2021 from the `Entrate_del_rendiconto_YYYY.csv` / `Spese_del_rendiconto_YYYY.csv` pairs of the old site; empty frame when none."""
    rows = []
    for e in sorted((archive_dir / "files").glob("*/Entrate_del_rendiconto_*.csv")):
        s = e.with_name(e.name.replace("Entrate", "Spese"))
        row = archive_row(read_rows(e), read_rows(s)) if s.exists() else None
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def load_accruals(raw_dir: Path = RAW_DIR, archive_dir: Path = ARCHIVE_DIR) -> pd.DataFrame:
    """Every approval act with a `*ENTRATERENDICONTO*.csv` and its `*SPESERENDICONTO*.csv` sibling (2022 onwards), plus the old-site years 2018-2021; empty frame when none."""
    rows = []
    for e in sorted((raw_dir / "files").glob("*/*ENTRATERENDICONTO*.csv")):
        s = next(iter(e.parent.glob("*SPESERENDICONTO*.csv")), None)
        row = accrual_row(read_rows(e), read_rows(s)) if s else None
        if row:
            rows.append(row)
    out = pd.concat([load_archive_accruals(archive_dir), pd.DataFrame(rows)], ignore_index=True)
    return out.drop_duplicates("year").sort_values("year").reset_index(drop=True) if len(out) else out
