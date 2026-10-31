"""EU VAT compliance gap (European Commission, CASE study), 2025 report tables, Table 205 (all available estimates)."""
import time
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.countries import ISO3
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

# Document link found on https://taxation-customs.ec.europa.eu/taxation/vat/fight-against-vat-fraud/vat-gap_en
# (the UUID changes with each report edition: update it when a new report is published)
URL = "https://taxation-customs.ec.europa.eu/document/download/b3045343-a25a-4fcf-97c8-b27c413b95ad_en?filename=VAT-GAP-2025-tables.xlsx"
RAW = Path("data/raw/vat_gap/VAT-GAP-2025-tables.xlsx")
SHEET = "Table 205"
FIRST_PROPER_YEAR = 2013  # earlier years are a "backcasted series" in the source (assumed to end where the main tables start)


def fetch(force: bool = False) -> Path:
    if RAW.exists() and not force:
        return RAW
    RAW.parent.mkdir(parents=True, exist_ok=True)
    for wait in (15, 60, 180, None):  # the Commission site answers 429 to CI runners: back off, honouring Retry-After
        resp = requests.get(URL, timeout=120)
        if resp.status_code != 429 or wait is None:
            break
        time.sleep(max(wait, int(resp.headers.get("Retry-After", 0) or 0)))
    resp.raise_for_status()
    RAW.write_bytes(resp.content)
    record("vat_gap", source="ec-vat-gap-2025", dataset=SHEET, url=URL)
    return RAW


def load() -> pd.DataFrame:
    raw = pd.read_excel(RAW, sheet_name=SHEET, header=None)
    years = raw.iloc[1, 1:].dropna().astype(int).tolist()
    rows = raw[raw[0].isin(ISO3)]  # keeps member states (and UK); LU, candidates and the EU average rows are not in ISO3 / flagged unreliable
    long = rows.set_index(0).iloc[:, : len(years)]
    long.columns = years
    long = long.stack().rename("value").reset_index().rename(columns={0: "geo", "level_1": "year"})
    long["value"] = pd.to_numeric(long["value"], errors="coerce") * 100  # source stores fractions
    long = long.dropna(subset=["value"])
    return long.assign(iso3=long.geo.map(ISO3), indicator="vat_compliance_gap_pct", theme="governance", source="ec-vat-gap-2025:Table205",
                       flag=lambda d: d.year.map(lambda y: "backcast" if y < FIRST_PROPER_YEAR else None))[
        ["iso3", "geo", "year", "indicator", "value", "flag", "theme", "source"]
    ]
