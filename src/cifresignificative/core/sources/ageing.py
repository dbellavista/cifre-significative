"""EC/EPC 2024 Ageing Report: statistical annex (Excel, one a/b/c sheet triple per country) -> long projections table 2022-2070.

Sheet a = demography and macro assumptions, b = pension expenditure, c = health care, long-term care, education, total cost of ageing.
"""
import re
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.countries import ISO3
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

# Found on https://economy-finance.ec.europa.eu/publications/2024-ageing-report-economic-and-budgetary-projections-eu-member-states-2022-2070_en
URL = "https://economy-finance.ec.europa.eu/document/download/e248db46-f876-4e72-8821-efae678e81ea_en?filename=2024_Ageing_Report-Statistical_annex_all_countryfiches.xlsx"
RAW = Path("data/external/source/ageing_report/2024_ageing_report_statistical_annex_all_countries.xlsx")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}


def fetch(force: bool = False) -> Path:
    if RAW.exists() and not force:
        return RAW
    RAW.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, headers=UA, timeout=300)
    resp.raise_for_status()
    RAW.write_bytes(resp.content)
    record("ageing_report", source="ec-ageing-report-2024", dataset="statistical annex (country fiches)", url=URL)
    return RAW


def _parse_sheet(df: pd.DataFrame) -> list[tuple[str, str, int, float]]:
    """(section, label, year, value) rows; a section starts at a header row whose column 2 is 'Ch 22-70' / 'AVG 22-70'."""
    years, section, parent, out = None, "", "", []
    for i in range(len(df)):
        label, tag = df.iat[i, 1], df.iat[i, 2]
        if not isinstance(label, str) or not label.strip():
            continue
        label = label.strip()
        if isinstance(tag, str) and re.match(r"(Ch|AVG)\s+\d", tag):
            years = [int(y) if isinstance(y, (int, float)) and pd.notna(y) else None for y in df.iloc[i, 3:].tolist()]
            section, parent = label, ""
            continue
        vals = df.iloc[i, 3:].tolist()
        numeric = [(y, v) for y, v in zip(years or [], vals, strict=False) if y and isinstance(v, (int, float)) and pd.notna(v)]
        if not numeric:
            parent = label if label[:1] != "(" else parent  # a row without numbers names the group that follows
            continue
        name = f"{parent} - {label}" if parent and df.iat[i, 1].startswith(" ") is False and label in ("males", "females", "male", "female") else label
        out += [(section, name, y, float(v)) for y, v in numeric]
    return out


def load() -> pd.DataFrame:
    xl = pd.ExcelFile(RAW)
    rows = []
    for sheet in xl.sheet_names:
        m = re.fullmatch(r"([A-Z]{2}|EU27)([abcABC])", sheet)  # later countries use upper-case suffixes
        code = m.group(1) if m else None
        code = {"EU27": "EU27_2020"}.get(code, code)  # EA (euro area) is not in our geo list and is skipped
        if not m or code not in ISO3:
            continue
        geo, part = code, m.group(2).lower()
        for section, label, year, value in _parse_sheet(pd.read_excel(xl, sheet_name=sheet, header=None)):
            rows.append((ISO3[geo], geo, part, section, label, year, value))
    d = pd.DataFrame(rows, columns=["iso3", "geo", "sheet", "section", "variable", "year", "value"])
    return d.drop_duplicates(["iso3", "section", "variable", "year"])
