"""World Bank Subnational Business Ready (B-READY) in the European Union 2025: Italy (13 cities), topic and pillar scores 0-100.

Data file: 'Download Data' on https://www.worldbank.org/en/regulatoryefficiency/economy/italy (openknowledge.worldbank.org, a JavaScript app: the
file must be downloaded with a browser, `cifresignificative.core.sources.browser`). Five topics are covered (business entry, location, utility services, dispute
resolution, insolvency); the other B-READY topics were not surveyed at city level. Only each city's total score is used (the two sub-columns of the
workbook are not documented in the report).
"""
from pathlib import Path

import pandas as pd

from cifresignificative.core.sources import browser as br

XLSX = Path("data/external/source/bready/italy_subnational_data.xlsx")
URL = "https://openknowledge.worldbank.org/bitstreams/8d8c5800-1162-4b68-9908-766b086a28c1/download"
AREA = {"Ancona": "Centre", "Bari": "South", "Bologna": "North", "Cagliari": "Islands", "Florence": "Centre", "Genoa": "North", "Milan": "North", "Naples": "South",
        "Padua": "North", "Palermo": "Islands", "Reggio Calabria": "South", "Rome": "Centre", "Turin": "North"}


def fetch(force: bool = False) -> Path:
    if XLSX.exists() and not force:
        return XLSX
    XLSX.parent.mkdir(parents=True, exist_ok=True)
    with br.browser() as ctx:
        page = ctx.new_page()
        with page.expect_download(timeout=90000) as dl:
            try:
                page.goto(URL, wait_until="commit", timeout=60000)
            except Exception:  # noqa: BLE001, S110 - navigation is replaced by the download
                pass
        dl.value.save_as(str(XLSX))
    return XLSX


def load() -> pd.DataFrame:
    x = pd.ExcelFile(XLSX)
    rows = []
    for sheet in x.sheet_names:
        d = pd.read_excel(x, sheet_name=sheet, header=None)
        cities = [(i, str(v)) for i, v in enumerate(d.iloc[0]) if isinstance(v, str) and i >= 6]
        for _, r in d.iloc[2:].iterrows():
            if r.iloc[1] not in ("Topic", "Pillar"):
                continue
            level = "topic" if r.iloc[1] == "Topic" else "pillar"
            name = str(r.iloc[2]).strip().replace(" Score", "")
            for i, city in cities:
                rows.append((city, AREA.get(city, ""), sheet, level, name, float(r.iloc[i + 2])))
    return pd.DataFrame(rows, columns=["city", "area", "topic", "level", "item", "score"])
