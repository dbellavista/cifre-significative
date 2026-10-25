"""World Bank WDI fetcher: used for benchmark countries (US/JP/UK) and cross-checks vs Eurostat."""
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import truststore
import wbgapi as wb

from cifresignificative.countries import ISO3, START_YEAR

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/worldbank")
MANIFEST = Path("data/raw/manifest.json")

# indicator name -> (WDI code, theme)
WDI = {
    "wb_gdp_pc_ppp_const": ("NY.GDP.PCAP.PP.KD", "output"),
    "wb_gdp_growth": ("NY.GDP.MKTP.KD.ZG", "output"),
    "wb_gfcf_pct_gdp": ("NE.GDI.FTOT.ZS", "investment"),
    "wb_gov_debt_pct_gdp": ("GC.DOD.TOTL.GD.ZS", "fiscal"),
    "wb_population": ("SP.POP.TOTL", "demography"),
    "wb_pop_65plus_pct": ("SP.POP.65UP.TO.ZS", "demography"),
    "wb_fertility_rate": ("SP.DYN.TFRT.IN", "demography"),
    "wb_unemployment_ilo": ("SL.UEM.TOTL.ZS", "labour"),
    "wb_inflation_cpi": ("FP.CPI.TOTL.ZG", "prices"),
    "wb_tertiary_enrol_gross": ("SE.TER.ENRR", "education"),
    "wb_homicide_rate": ("VC.IHR.PSRC.P5", "crime"),  # UNODC intentional homicide per 100,000; covers US/JP and the 1990s
    "wb_gini": ("SI.POV.GINI", "inequality"),  # EU countries: derived from EU-SILC, so continuous with Eurostat `gini` (2014+)
    # PISA scores (triennial; World Bank EdStats, db 12): OECD publishes no PISA SDMX flow
    "wb_pisa_math": ("LO.PISA.MAT", "education"),
    "wb_pisa_reading": ("LO.PISA.REA", "education"),
    "wb_pisa_science": ("LO.PISA.SCI", "education"),
    # Worldwide Governance Indicators (db 3), 0-100 score under the 2025 methodology (history recalculated)
    "wgi_government_effectiveness": ("GOV_WGI_GE.SC", "governance"),
    "wgi_regulatory_quality": ("GOV_WGI_RQ.SC", "governance"),
    "wgi_rule_of_law": ("GOV_WGI_RL.SC", "governance"),
    "wgi_control_of_corruption": ("GOV_WGI_CC.SC", "governance"),
}
DATABASE = {"LO.PISA.MAT": 12, "LO.PISA.REA": 12, "LO.PISA.SCI": 12}  # EdStats; WGI codes below are db 3
DATABASE.update({c: 3 for c, _ in WDI.values() if c.startswith("GOV_WGI")})
ECONOMIES = [iso for iso in ISO3.values() if iso != "EU27"] + ["EUU"]


def fetch(name: str, force: bool = False) -> Path:
    code, _ = WDI[name]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{name}.parquet"
    if out.exists() and not force:
        return out
    df = wb.data.DataFrame(code, db=DATABASE.get(code), economy=ECONOMIES, time=range(START_YEAR, datetime.now().year + 1), labels=False, columns="time")
    df = df.rename(columns=lambda c: int(str(c)[2:]) if str(c).startswith("YR") else c)
    long = df.reset_index().melt(id_vars="economy", var_name="year", value_name="value").dropna(subset=["value"])
    long.to_parquet(out, index=False)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[name] = {"source": "worldbank", "dataset": code, "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds")}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out


def load(name: str) -> pd.DataFrame:
    code, theme = WDI[name]
    df = pd.read_parquet(RAW_DIR / f"{name}.parquet")
    iso_to_geo = {v: k for k, v in ISO3.items()}
    iso_to_geo["EUU"] = "EU27_2020"
    df["iso3"] = df["economy"].replace({"EUU": "EU27"})
    df["geo"] = df["economy"].map(iso_to_geo)
    return df.assign(indicator=name, flag=None, theme=theme, source=f"worldbank:{code}")[
        ["iso3", "geo", "year", "indicator", "value", "flag", "theme", "source"]
    ]
