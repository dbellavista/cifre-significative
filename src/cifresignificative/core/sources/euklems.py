"""EU KLEMS & INTANProd (Luiss, release 2025): growth accounts by industry, 1995-2021, from the extended all-country CSV.

The raw CSVs are large (about 200-300 MB) and are not committed (data/external/source/euklems is gitignored); only the selected variables are
kept in data/processed/euklems_growth.parquet. LP1 = labour productivity per hour; Con = contribution to its growth in percentage points.
Download page: https://euklems-intanprod-llee.luiss.it/download/ (all-country CSV 'growth accounts').
"""
from pathlib import Path

import pandas as pd

from cifresignificative.core.countries import ISO3

URL = "https://www.dropbox.com/s/w0mdttie9kjlh82/growth%20accounts.csv?dl=1"
RAW = Path("data/external/source/euklems/growth-accounts-extended.csv")
VARS = ["LP1_G", "LP1ConTFP", "LP1ConLC", "LP1ConTangICT", "LP1ConTangNICT", "LP1ConIntang", "LP1TFP_I"]
INDUSTRIES = ["TOT", "MARKT", "MARKTxAG", "A", "B", "C", "D-E", "F", "G", "H", "I", "J", "K", "L68A", "M-N", "O-Q", "R-S"]


def fetch(force: bool = False) -> Path:
    import requests

    if RAW.exists() and not force:
        return RAW
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(URL, stream=True, timeout=600, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with RAW.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return RAW


def load() -> pd.DataFrame:
    d = pd.read_csv(RAW, usecols=["nace_r2_code", "geo_code", "year", "var", "value"], low_memory=False)
    d = d[d["var"].isin(VARS) & d.nace_r2_code.isin(INDUSTRIES) & d.geo_code.isin(ISO3)].dropna(subset=["value"])
    d["year"] = d["year"].astype(int)
    return d.assign(iso3=d.geo_code.map(ISO3)).rename(columns={"geo_code": "geo", "var": "variable"})[["iso3", "geo", "year", "nace_r2_code", "variable", "value"]]
