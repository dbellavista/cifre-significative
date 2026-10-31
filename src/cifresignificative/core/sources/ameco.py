"""AMECO (European Commission DG ECFIN) bulk CSV: TFP, hours, compensation, output gap."""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.countries import ISO3, START_YEAR
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

URL = "https://ec.europa.eu/economy_finance/db_indicators/ameco/documents/ameco0_CSV.zip"
RAW = Path("data/raw/ameco/ameco0_CSV.zip")

# panel name -> (AMECO variable suffix, theme)
SERIES = {
    "ameco_tfp_index": ("ZVGDF", "output"),  # total factor productivity, 2020=100
    "ameco_hours_per_worker": ("NLHA", "labour"),
    "ameco_comp_per_employee_ncu": ("HWCDW", "labour"),  # national currency (1000s): use as index/growth only
    "ameco_output_gap_pct_potential": ("AVGDGP", "output"),
}
AMECO_ISO = {**{v: v for v in ISO3.values()}, "UK": "GBR", "ROM": "ROU"}  # AMECO writes the UK as "UK" and Romania as "ROM": a code missing here is dropped without a warning


def fetch(force: bool = False) -> Path:
    if RAW.exists() and not force:
        return RAW
    RAW.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, timeout=300)
    resp.raise_for_status()
    RAW.write_bytes(resp.content)
    record("ameco", source="ameco", dataset="ameco0_CSV", url=URL)
    return RAW


def _read_all() -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(RAW) as zf:
        for member in zf.namelist():
            if member.upper().endswith(".CSV"):
                frames.append(pd.read_csv(io.BytesIO(zf.read(member)), dtype=str, encoding_errors="replace"))
    return pd.concat(frames, ignore_index=True)


def load() -> pd.DataFrame:
    raw = _read_all()
    parts = raw["CODE"].str.split(".", expand=True)
    raw = raw.assign(prefix=parts[0], suffix=parts[parts.columns[-1]], p2=parts[2], p3=parts[3])
    year_cols = [c for c in raw.columns if c.isdigit() and START_YEAR <= int(c) <= 2025]
    out = []
    for name, (suffix, theme) in SERIES.items():
        sub = raw[(raw.suffix == suffix) & (raw.p2 == "0") & (raw.p3 == "0") & raw.prefix.isin(AMECO_ISO)]
        long = sub.melt(id_vars=["prefix", "CODE"], value_vars=year_cols, var_name="year", value_name="value")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["value"]).assign(year=lambda d: d["year"].astype(int))
        if long.duplicated(["prefix", "year"]).any():
            raise ValueError(f"{name}: ambiguous AMECO series for suffix {suffix}: {sorted(sub.CODE.unique())[:5]}")
        iso = long["prefix"].map(AMECO_ISO)
        geo_of = {v: k for k, v in ISO3.items()}
        out.append(long.assign(iso3=iso, geo=iso.map(geo_of), indicator=name, flag=None, theme=theme, source=f"ameco:{suffix}")[
            ["iso3", "geo", "year", "indicator", "value", "flag", "theme", "source"]
        ])
    return pd.concat(out, ignore_index=True)


# Long history (1960-2025) in national currency / native units, for analyses that need the pre-2005 period (euro entry, 1990s).
# Compare countries through indices or ratios, never through raw levels.
LONG_SERIES = {
    "gdp_real_ncu_bn": "OVGD", "gdp_nominal_ncu_bn": "UVGD", "population_th": "NPTD", "employment_th": "NETD", "hours_total_m": "NLHT",
    "gov_debt_ncu_bn": "UDGG", "unemployment_rate": "ZUTN", "gfcf_nominal_ncu_bn": "UIGT", "gov_gfcf_nominal_ncu_bn": "UIGG",
    "wage_share_adjusted_pct_gdp": "ALCD0", "gdp_deflator_2020_100": "PVGD", "comp_per_employee_nominal_th": "HWCDW",
}


def load_long() -> pd.DataFrame:
    """Long table (iso3, geo, year, variable, value) for LONG_SERIES plus derived debt and investment ratios."""
    raw = _read_all()
    parts = raw["CODE"].str.split(".", expand=True)
    raw = raw.assign(prefix=parts[0], suffix=parts[parts.columns[-1]], p2=parts[2], p3=parts[3])
    year_cols = [c for c in raw.columns if c.isdigit() and int(c) <= 2025]
    geo_of = {v: k for k, v in ISO3.items()}
    frames = []
    for name, suffix in LONG_SERIES.items():
        sub = raw[(raw.suffix == suffix) & (raw.p2 == "0") & (raw.p3 == "0") & raw.prefix.isin(AMECO_ISO)]
        if suffix in ("OVGD", "PVGD"):  # these carry other sub-chapter digits in the code
            sub = raw[(raw.suffix == suffix) & (raw.p2 == "1") & (raw.p3 == "0") & raw.prefix.isin(AMECO_ISO)]  # p2 = 1: 2020 reference prices/levels
        long = sub.melt(id_vars=["prefix"], value_vars=year_cols, var_name="year", value_name="value")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["value"]).assign(year=lambda d: d["year"].astype(int))
        if long.duplicated(["prefix", "year"]).any():
            raise ValueError(f"{name}: ambiguous AMECO series for {suffix}")
        iso = long["prefix"].map(AMECO_ISO)
        frames.append(long.assign(iso3=iso, geo=iso.map(geo_of), variable=name)[["iso3", "geo", "year", "variable", "value"]])
    out = pd.concat(frames, ignore_index=True)
    wide = out.pivot_table(index=["iso3", "geo", "year"], columns="variable", values="value")
    derived = {
        "gov_debt_pct_gdp": wide["gov_debt_ncu_bn"] / wide["gdp_nominal_ncu_bn"] * 100,
        "gfcf_pct_gdp": wide["gfcf_nominal_ncu_bn"] / wide["gdp_nominal_ncu_bn"] * 100,
        "gov_gfcf_pct_gdp": wide["gov_gfcf_nominal_ncu_bn"] / wide["gdp_nominal_ncu_bn"] * 100,
        "gdp_real_per_head_ncu": wide["gdp_real_ncu_bn"] / wide["population_th"] * 1e6,
        "gdp_real_per_hour_ncu": wide["gdp_real_ncu_bn"] / wide["hours_total_m"] * 1e3,
        "employment_per_head": wide["employment_th"] / wide["population_th"] * 100,
    }
    der = pd.concat([s.dropna().rename("value").reset_index().assign(variable=k) for k, s in derived.items()], ignore_index=True)
    return pd.concat([out, der], ignore_index=True)
