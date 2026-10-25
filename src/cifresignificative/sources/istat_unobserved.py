"""Istat, "Economia non osservata nei conti nazionali" (2025 edition, years 2011-2023), Italy only.

The workbook is a manual download from the Istat release page (the tables are not in the SDMX API). Tavola 1 gives the components of the
non-observed economy (EUR million and % of GDP); Tavola 3 the incidence of the underground economy on value added by industry.
"""
from pathlib import Path

import pandas as pd

XLSX = Path("data/external/source/istat_unobserved/Tavole-Economia-non-osservata_2025.xlsx")
URL = "https://www.istat.it/comunicato-stampa/economia-non-osservata-nei-conti-nazionali-anno-2023/"
COMPONENTS = {
    "Economia sommersa": "underground",
    "- da Sottodichiarazione": "underreporting",
    "- da Lavoro irregolare": "irregular_labour",
    "- Altro": "other_underground",
    "Attività illegali": "illegal",
    "Economia non osservata": "non_observed_total",
}


def load_components() -> pd.DataFrame:
    """Columns: component, year, meur, pct_gdp."""
    d = pd.read_excel(XLSX, sheet_name="TAVOLA 1", header=None)
    years = {c: int(v) for c, v in d.iloc[3].items() if isinstance(v, (int, float)) and pd.notna(v)}
    rows = []
    for _, r in d.iloc[6:12].iterrows():
        comp = COMPONENTS[str(r[0]).strip()]
        for c, y in years.items():
            rows.append((comp, y, float(r[c]), float(r[c + 1])))
    return pd.DataFrame(rows, columns=["component", "year", "meur", "pct_gdp"])


def load_by_industry() -> pd.DataFrame:
    """Underground economy as % of industry value added: columns industry, year, component, pct_value_added."""
    d = pd.read_excel(XLSX, sheet_name="TAVOLA 3", header=None)
    years = {c: int(v) for c, v in d.iloc[3].items() if isinstance(v, (int, float)) and pd.notna(v)}
    names = d.iloc[4]
    rows = []
    for _, r in d.iloc[5:17].iterrows():
        for c, y in years.items():
            for k in range(4):
                label = str(names[c + k]).strip()
                comp = {"Sottodichiarazione": "underreporting", "Lavoro irregolare": "irregular_labour", "Altro": "other_underground"}.get(label, "underground")
                rows.append((str(r[0]).strip(), y, comp, float(r[c + k])))
    return pd.DataFrame(rows, columns=["industry", "year", "component", "pct_value_added"])
