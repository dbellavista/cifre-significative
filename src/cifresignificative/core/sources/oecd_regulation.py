"""OECD product market regulation (PMR, flow DSD_PMR@DF_PMR, waves 2018 and 2023, 57 economies, indicators 0-6 from least to most restrictive) and employment protection
legislation (EPL, DSD_EPL@DF_EPL, annual, 1985-2025, 23 EU members; strictness 0-6) through the OECD SDMX API. PMR is a snapshot of two waves, EPL a time series: they stay in side tables
(`load_pmr`, `load_epl`), not in the annual panel. The PMR sector indicators (professions, retail, energy, transport, e-communications) are the same in both waves for most professions,
because the rules were not changed; the older PMR waves (1998-2013) are another vintage of the index and are not in this flow. EPL comes in four methodology versions: VERSION1 runs from 1985 (regular and
temporary contracts), VERSION2 adds collective dismissals from 1998, VERSION3 starts 2008 and VERSION4 2013, each with its own scoring, so a series must not be spliced across versions."""
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

BASE = "https://sdmx.oecd.org/public/rest/data"
RAW_DIR = Path("data/raw/oecd")
FLOWS = {"pmr": "OECD.ECO.GCRD,DSD_PMR@DF_PMR,1.4", "epl": "OECD.ELS.JAI,DSD_EPL@DF_EPL,1.0"}


def fetch(force: bool = False) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name, flow in FLOWS.items():
        path = RAW_DIR / f"{name}.csv"
        if force or not path.exists():
            url = f"{BASE}/{flow}/all?dimensionAtObservation=AllDimensions"
            resp = requests.get(url, headers={"Accept": "application/vnd.sdmx.data+csv"}, timeout=300)
            resp.raise_for_status()
            if not resp.text.startswith("DATAFLOW,"):
                raise RuntimeError(f"OECD {name}: not a CSV answer: {resp.text[:80]!r}")
            path.write_bytes(resp.content)
            record(f"oecd_{name}", source="oecd-sdmx", dataset=flow, url=url)
        out.append(path)
    return out


def load_pmr(path: Path = RAW_DIR / "pmr.csv") -> pd.DataFrame:
    """Long `iso3, measure, year, value` (0-6, higher = more restrictive); empty cells (aggregates the OECD does not publish for a wave) are dropped."""
    d = pd.read_csv(path, usecols=["REF_AREA", "MEASURE", "TIME_PERIOD", "OBS_VALUE"]).dropna(subset=["OBS_VALUE"])
    return d.rename(columns={"REF_AREA": "iso3", "MEASURE": "measure", "TIME_PERIOD": "year", "OBS_VALUE": "value"}).assign(source="oecd:PMR").reset_index(drop=True)


def load_epl(path: Path = RAW_DIR / "epl.csv", version: str = "VERSION1") -> pd.DataFrame:
    """Long `iso3, measure, year, value` for one methodology version (default VERSION1, 1985 on): EPL_R regular contracts, EPL_T temporary contracts, EPL_OV overall (and EPL_CD collective dismissals in the later versions)."""
    d = pd.read_csv(path, usecols=["REF_AREA", "MEASURE", "TIME_PERIOD", "OBS_VALUE", "VERSION"]).dropna(subset=["OBS_VALUE"])
    d = d[d.VERSION == version].drop(columns="VERSION")
    return d.rename(columns={"REF_AREA": "iso3", "MEASURE": "measure", "TIME_PERIOD": "year", "OBS_VALUE": "value"}).assign(source=f"oecd:EPL:{version}").reset_index(drop=True)
