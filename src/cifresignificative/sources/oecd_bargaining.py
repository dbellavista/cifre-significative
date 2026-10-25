"""OECD/AIAS collective bargaining coverage (`DSD_TUD_CBC@DF_CBC`, share of employees covered, from the ICTWSS database, annual, 22 EU members) and trade union density (`DF_TUD`, share of employees, 22 EU members)
through the OECD SDMX API (the ICTWSS site itself was not reachable from here). Italy's coverage is recorded as 100% in every year (the collective contracts' minimum pay is extended by case law): the series has no
variation for Italy and Italy is a top-coded point. Observation flags: A normal, B break (4 and 2 values), E estimated. Side tables, not in the annual panel."""
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.sources._common import record

truststore.inject_into_ssl()

BASE = "https://sdmx.oecd.org/public/rest/data/OECD.ELS.SAE,DSD_TUD_CBC@"
RAW_DIR = Path("data/raw/oecd")
FLOWS = {"cbc": "DF_CBC,1.0", "tud": "DF_TUD,1.0"}


def fetch(force: bool = False) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name, flow in FLOWS.items():
        path = RAW_DIR / f"{name}.csv"
        if force or not path.exists():
            url = f"{BASE}{flow}/all?dimensionAtObservation=AllDimensions"
            resp = requests.get(url, headers={"Accept": "application/vnd.sdmx.data+csv; labels=id"}, timeout=300)  # labels=id: plain codes (the default labels every code)
            resp.raise_for_status()
            if not resp.text.startswith("DATAFLOW,REF_AREA,"):
                raise RuntimeError(f"OECD {name}: unexpected answer: {resp.text[:80]!r}")
            path.write_bytes(resp.content)
            record(f"oecd_{name}", source="oecd-sdmx", dataset=flow, url=url)
        out.append(path)
    return out


def load(measure: str, path: Path | None = None) -> pd.DataFrame:
    """Long `iso3, year, value, flag` for `cbc` (coverage) or `tud` (union density), per cent of employees."""
    d = pd.read_csv(path or RAW_DIR / f"{measure}.csv", usecols=["REF_AREA", "TIME_PERIOD", "OBS_VALUE", "OBS_STATUS"]).dropna(subset=["OBS_VALUE"])
    return d.rename(columns={"REF_AREA": "iso3", "TIME_PERIOD": "year", "OBS_VALUE": "value", "OBS_STATUS": "flag"}).assign(source=f"oecd:{measure.upper()}").reset_index(drop=True)
