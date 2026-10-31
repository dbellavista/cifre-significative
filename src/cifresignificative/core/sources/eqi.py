"""European Quality of Government Index (EQI, University of Gothenburg QoG Institute), regional file: survey-based perceptions of quality, impartiality and corruption of public services.

Waves 2010, 2013, 2017, 2021 and 2024 (210 regions per wave: 186 at NUTS2, 19 at NUTS1 in Germany and Belgium, plus country rows). Only NUTS2 rows are kept, so Germany and Belgium are partly missing (they come at NUTS1);
a NUTS1 value is never copied to its NUTS2 children. Index values are standardised (mean 0 over the sample of each wave), so they rank regions but are not comparable in level across waves; `EQI_me` is the margin of error.
Perceptions of the sub-national government, not measured outcomes: descriptive associations only.
"""
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

URL = "https://www.qogdata.pol.gu.se/data/qog_eqi_long_24.csv"
RAW = Path("data/raw/eqi/qog_eqi_long_24.csv")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
# source column -> indicator name (same `nuts2_` convention as the Eurostat regional catalog)
COLUMNS = {"EQI": "nuts2_eqi", "qualityp": "nuts2_eqi_quality", "impartialityp": "nuts2_eqi_impartiality", "corruptionp": "nuts2_eqi_corruption", "EQI_me": "nuts2_eqi_margin_of_error"}


def fetch(force: bool = False) -> Path:
    if RAW.exists() and not force:
        return RAW
    RAW.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, headers=UA, timeout=120)
    resp.raise_for_status()
    if not resp.text.startswith("cname,"):
        raise RuntimeError(f"EQI: not the expected CSV: {resp.text[:80]!r}")
    RAW.write_bytes(resp.content)
    record("eqi", source="qog.gu.se", dataset="qog_eqi_long_24", url=URL)
    return RAW


def load(path: Path = RAW, nuts1: tuple[str, ...] = ()) -> pd.DataFrame:
    """Regional-table format (`geo, year, value, flag, indicator, theme, source, country`) for the NUTS2 rows, one row per region, wave and index component. `nuts1` lists countries (Germany and Belgium publish NUTS1 only)
    whose NUTS1 rows are added: they are separate rows with 3-character codes, never copied to NUTS2 children."""
    d = pd.read_csv(path, low_memory=False)
    d = d[d.nuts_level.eq(2) | (d.nuts_level.eq(1) & d.region_code.str[:2].isin(nuts1))].rename(columns={"region_code": "geo"})
    out = d.melt(id_vars=["geo", "year"], value_vars=list(COLUMNS), var_name="column", value_name="value").dropna(subset=["value"])
    return out.assign(indicator=out.pop("column").map(COLUMNS), flag=None, theme="governance", source="qog:EQI", country=out.geo.str[:2])[["geo", "year", "value", "flag", "indicator", "theme", "source", "country"]].reset_index(drop=True)
