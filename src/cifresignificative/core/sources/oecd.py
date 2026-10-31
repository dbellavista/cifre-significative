"""OECD Pensions at a Glance via SDMX (sdmx.oecd.org). Time series go to the panel; 2024 snapshots stay in a side table."""
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.countries import ISO3
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

BASE = "https://sdmx.oecd.org/public/rest/data/OECD.ELS.SPD"
RAW_DIR = Path("data/raw/oecd")
FLOWS = {"pag": "DSD_PAG@DF_PAG", "prr": "DSD_PAG@DF_PRR", "dps": "DSD_PAG@DF_DPS"}

# panel name -> (flow, MEASURE, SEX or None)
SERIES = {
    "oecd_exit_age_male": ("pag", "ELMEA", "M"),  # effective labour-market exit age
    "oecd_exit_age_female": ("pag", "ELMEA", "F"),
    "oecd_pension_exp_pct_gdp": ("pag", "PEP", None),  # public pension expenditure, 1990-2021
    "oecd_oldage_to_workingage_ratio": ("pag", "OAWAR", None),
}
# 2024 cross-section (replacement rates, statutory ages, pension wealth) -> data/processed/pensions_oecd_2024.parquet
SNAPSHOT_FLOWS = ("pag", "prr", "dps")


def fetch(flow: str, force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{flow}.csv"
    if out.exists() and not force:
        return out
    url = f"{BASE},{FLOWS[flow]},/all?dimensionAtObservation=AllDimensions"
    resp = requests.get(url, headers={"Accept": "application/vnd.sdmx.data+csv"}, timeout=300)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    record(f"oecd_{flow}", source="oecd-sdmx", dataset=FLOWS[flow], url=url)
    return out


def _read(flow: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"{flow}.csv", dtype={"TIME_PERIOD": int})


def load_series() -> pd.DataFrame:
    geo_of = {v: k for k, v in ISO3.items()}
    out = []
    for name, (flow, measure, sex) in SERIES.items():
        df = _read(flow)
        df = df[(df.MEASURE == measure) & df.REF_AREA.isin(geo_of) & (df.TIME_PERIOD <= 2025)]
        if sex:
            df = df[df.SEX == sex]
        elif "_T" in set(df.SEX):
            df = df[df.SEX == "_T"]
        if df.duplicated(["REF_AREA", "TIME_PERIOD"]).any():
            dims = df.groupby(["SEX", "AGE", "OPTIONALITY", "UNIT_MEASURE"]).size()
            raise ValueError(f"{name}: ambiguous after filtering:\n{dims}")
        out.append(df.assign(iso3=df.REF_AREA, geo=df.REF_AREA.map(geo_of), year=df.TIME_PERIOD, indicator=name,
                             value=df.OBS_VALUE, flag=df.OBS_STATUS, theme="pensions", source=f"oecd:{measure}")[
            ["iso3", "geo", "year", "indicator", "value", "flag", "theme", "source"]
        ])
    return pd.concat(out, ignore_index=True)


def load_snapshot() -> pd.DataFrame:
    """All 2024 pension-design measures, keeping SEX/AGE/OPTIONALITY, for the EU27+benchmarks."""
    geo_of = {v: k for k, v in ISO3.items()}
    frames = []
    for flow in SNAPSHOT_FLOWS:
        df = _read(flow)
        df = df[df.REF_AREA.isin(geo_of) & (df.TIME_PERIOD == 2024)]
        frames.append(df.assign(flow=flow)[["flow", "REF_AREA", "MEASURE", "UNIT_MEASURE", "SEX", "AGE", "OPTIONALITY", "TIME_PERIOD", "OBS_VALUE"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates()
