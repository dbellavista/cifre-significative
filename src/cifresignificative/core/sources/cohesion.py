"""EU Cohesion Open Data Platform (cohesiondata.ec.europa.eu, Socrata API): ESIF 2014-2020 financial implementation by member state and fund.

Dataset 99js-gm52 'ESIF 2014-2020 Finance Implementation Details'. Planned = programmed total amount (EU + national) summed over programmes; spent =
total eligible expenditure declared at the latest reporting period (`latest_period` = 'Y'). Absorption = spent / planned (can exceed 100% when
programmes over-commit). EAFRD (rural development) and fisheries funds are excluded because their expenditure is reported in different fields.
"""
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

BASE = "https://cohesiondata.ec.europa.eu/resource/99js-gm52.json"
RAW = Path("data/raw/cohesion/esif_2014_2020_by_ms_fund.json")
FUNDS = ["ERDF", "ESF", "CF", "YEI"]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}


def fetch(force: bool = False) -> Path:
    if RAW.exists() and not force:
        return RAW
    RAW.parent.mkdir(parents=True, exist_ok=True)
    params = {"$select": "ms,fund,sum(total_amount) as planned,sum(total_eligible_expenditure) as spent", "$where": "latest_period='Y'", "$group": "ms,fund", "$limit": "5000"}
    resp = requests.get(BASE, params=params, headers=UA, timeout=120)
    resp.raise_for_status()
    RAW.write_text(resp.text)
    record("cohesion_esif", source="cohesiondata.ec.europa.eu", dataset="99js-gm52", url=resp.url)
    return RAW


def load() -> pd.DataFrame:
    d = pd.read_json(RAW)
    d = d.assign(planned=pd.to_numeric(d.planned), spent=pd.to_numeric(d.spent))
    return d.rename(columns={"ms": "geo"})
