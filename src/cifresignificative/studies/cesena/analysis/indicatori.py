"""Cesena's Piano degli indicatori e dei risultati attesi di bilancio (rendiconto), from the PDFs on the municipal transparency portal.

The synthetic indicators (D.Lgs. 118/2011 Allegato 1-bis) are printed as `code  name  definition ...  value` with the value as the last token of the first line of each row (Italian decimal comma, three decimals). Rendiconto PDFs exist for 2023 onwards; they are a single-city official check on the ratios computed from the Rendiconto CSVs, whose definitions differ (the official rigidity adds IRAP and the FPV adjustments).
"""
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from cifresignificative.core.geo import CESENA
from cifresignificative.studies.cesena.analysis.rendiconto import RAW_DIR, num

ROW = re.compile(r"\s*(\d{1,2}\.\d{1,2})\s+\S.*?\s(-?\d+,\d{3})\s*$")
NAME = re.compile(r"Rendiconto(\d{4})\.pdf$")
OLD_NAME = re.compile(r"Piano_degli_indicatori_di_bilancio_(\d{4})_-?_?[Rr]endiconto\.pdf$")
ARCHIVE_DIR = Path("data/raw/trasparenza/sititematici")


def parse_indicators(text: str) -> dict[str, float]:
    """Code -> value for every row that ends in a three-decimal number; the first occurrence wins (definitions wrap over several lines)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = ROW.match(line)
        if m and m[1] not in out:
            out[m[1]] = num(m[2])
    return out


def pdf_text(path: Path) -> str:
    if shutil.which("pdftotext") is None:
        raise RuntimeError("pdftotext (poppler) is needed to read the indicator PDFs")
    return subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True, capture_output=True, text=True).stdout


def load_indicators(raw_dir: Path = RAW_DIR, archive_dir: Path = ARCHIVE_DIR) -> pd.DataFrame:
    """Long table (comune_id, year, code, value) of the rendiconto indicator PDFs: 2023 onwards from the portal, 2020-2022 from the old site (2020 and 2021 print 35-36 of the 54 indicators); empty when there are none."""
    rows = []
    files = [(f, NAME) for f in sorted((raw_dir / "files").glob("*/*Pianoindicatorierisultatiattesi-Rendiconto*.pdf"))] + [(f, OLD_NAME) for f in sorted((archive_dir / "files").glob("*/Piano_degli_indicatori_di_bilancio_*.pdf"))]
    for f, pattern in files:
        m = pattern.search(f.name)
        if m:
            rows += [{"comune_id": CESENA, "year": int(m[1]), "code": k, "value": v} for k, v in parse_indicators(pdf_text(f)).items()]
    return pd.DataFrame(rows).drop_duplicates(["year", "code"]) if rows else pd.DataFrame(columns=["comune_id", "year", "code", "value"])
