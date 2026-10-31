"""IMF World Economic Outlook via the DataMapper API (general-government series, incl. US/JP)."""
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.countries import EU27, ISO3, START_YEAR
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

API = "https://www.imf.org/external/datamapper/api/v1"
RAW_DIR = Path("data/raw/imf")
LAST_YEAR = 2025  # later years are projections; the WEO vintage is in the manifest

# panel name -> (WEO code, theme)
WEO = {
    "imf_gov_debt_pct_gdp": ("GGXWDG_NGDP", "fiscal"),
    "imf_gov_net_lending_pct_gdp": ("GGXCNL_NGDP", "fiscal"),
    "imf_current_account_pct_gdp": ("BCA_NGDPD", "external"),
    "imf_unemployment_rate": ("LUR", "labour"),
}
COUNTRIES = [*EU27.values(), "GBR", "USA", "JPN"]


def fetch(name: str, force: bool = False) -> Path:
    code, _ = WEO[name]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{name}.parquet"
    if out.exists() and not force:
        return out
    resp = requests.get(f"{API}/{code}", timeout=120)  # all countries; a long country path 404s
    resp.raise_for_status()
    data = resp.json()["values"][code]
    rows = [(iso, int(y), v) for iso, series in data.items() if iso in COUNTRIES for y, v in series.items() if START_YEAR <= int(y) <= LAST_YEAR]
    pd.DataFrame(rows, columns=["iso3", "year", "value"]).to_parquet(out, index=False)
    record(name, source="imf-weo", dataset=code, url=resp.url)
    return out


def load(name: str) -> pd.DataFrame:
    code, theme = WEO[name]
    df = pd.read_parquet(RAW_DIR / f"{name}.parquet")
    geo_of = {v: k for k, v in ISO3.items()}
    return df.assign(geo=df["iso3"].map(geo_of), indicator=name, flag=None, theme=theme, source=f"imf-weo:{code}")[
        ["iso3", "geo", "year", "indicator", "value", "flag", "theme", "source"]
    ]
