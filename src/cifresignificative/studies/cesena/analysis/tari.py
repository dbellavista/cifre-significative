"""Waste-tax (TARI) revenue of Cesena from the "Scheda Tassa sui Rifiuti" of the Giunta's report to the Rendiconto, to put the accrual rigidity on a base without it.

Until 2022 the Comune levied the TARI tax (about a sixth of its current revenue); from 2023 the levy became a Hera-billed tariff (TCP) outside the budget and only the recovery of evaded TARI stays. The accrual rigidity of `rendiconto.py` therefore has a denominator that shrinks by the levy between 2022 and 2023: the ex-TARI variant removes the levy from 2022 so the years compare. The recovery of evasion is kept in every year (it is still accrued after 2022).
"""
import re
from pathlib import Path

import pandas as pd

from cifresignificative.core.geo import CESENA
from cifresignificative.studies.cesena.analysis.indicatori import pdf_text
from cifresignificative.studies.cesena.analysis.rendiconto import ARCHIVE_DIR, RAW_DIR

AMOUNT = re.compile(r"\d{1,3}(?:\.\d{3})*,\d\d")
SHEET = "Scheda Tassa sui Rifiuti"


def to_eur(s: str) -> float:
    return float(s.replace(".", "").replace(",", "."))


def parse_tari_sheet(text: str) -> pd.DataFrame:
    """Rows (year, levy_eur, evasion_recovery_eur) of the sheet (accrued sums). A year line with two amounts is levy and recovery; with one amount only the recovery, the levy column printing `///` (no levy: 0). Empty when the sheet is absent."""
    a = text.find(SHEET)
    if a < 0:
        return pd.DataFrame()
    rows = []
    for ln in text[a:].splitlines()[1:30]:
        m = re.match(r"\s*(20\d\d)\b", ln)
        amounts = [to_eur(x) for x in AMOUNT.findall(ln)]
        if m and amounts:
            rows.append({"comune_id": CESENA, "year": int(m[1]), "levy_eur": amounts[0] if len(amounts) > 1 else 0.0, "evasion_recovery_eur": amounts[-1]})
    return pd.DataFrame(rows)


def load_tari(raw_dir: Path = RAW_DIR, archive_dir: Path = ARCHIVE_DIR) -> pd.DataFrame:
    """The sheet of every Giunta report to the Rendiconto: `*C1*` PDFs of the portal (2022 onwards) after the old-site reports (2018-2021, each covering three years back); the later report wins for a year covered twice."""
    archive = sorted((archive_dir / "files").glob("*/Relazione_di_Giunta_al_Rendiconto_*.pdf"))
    portal = sorted((raw_dir / "files").glob("*/*.pdf"), key=lambda f: int(f.name.split("_")[0]))
    parts = []
    for f in archive + [f for f in portal if re.search(r"[Cc]1", f.name) and re.search(r"[Rr][Ee][Ll][Aa][Zz]", f.name)]:
        t = parse_tari_sheet(pdf_text(f))
        if len(t):
            parts.append(t)
    return pd.concat(parts).drop_duplicates("year", keep="last").sort_values("year").reset_index(drop=True) if parts else pd.DataFrame()


def add_ex_tari(acc: pd.DataFrame, tari: pd.DataFrame) -> pd.DataFrame:
    """`acc` (rendiconto accruals) with `tari_levy_eur` and `rigidity_accrual_ex_tari_pct` = fixed spending over current revenue without the levy; NaN for a year the sheet does not cover."""
    out = acc.merge(tari[["year", "levy_eur"]].rename(columns={"levy_eur": "tari_levy_eur"}), on="year", how="left")
    fixed = out["spesa_personale_imp_eur"] + out["interessi_passivi_imp_eur"] + out["rimborso_prestiti_imp_eur"]
    out["rigidity_accrual_ex_tari_pct"] = fixed / (out["entrate_correnti_acc_eur"] - out["tari_levy_eur"]) * 100
    return out
