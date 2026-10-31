"""Istat SDMX (esploradati.istat.it): irregular employment rates, Italy only (a measure of the informal economy).

Dataflow 92_507_DF_DCCN_OCCNSEC2010_2 'Tassi di irregolarita' (national accounts, latest editions). Measures: RUEP irregular persons
rate, RUFTE full-time-equivalent rate, RUJ jobs rate (%). EMPLOYMENT_STATUS code 9 is treated as the total because its rate lies between those of
codes 1 and 2 (the code list was not decoded; check against Istat before quoting).
"""
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

FLOW = "92_507_DF_DCCN_OCCNSEC2010_2"
URL = f"https://esploradati.istat.it/SDMXWS/rest/data/IT1,{FLOW},1.0/all?startPeriod=2005"
RAW = Path("data/raw/istat/irregular_work.csv")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36", "Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}


def fetch(force: bool = False) -> Path:
    if RAW.exists() and not force:
        return RAW
    RAW.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, headers=UA, timeout=300)
    resp.raise_for_status()
    RAW.write_bytes(resp.content)
    record("istat_irregular_work", source="istat-sdmx", dataset=FLOW, url=URL)
    return RAW


def load() -> pd.DataFrame:
    """year x industry irregular-work rates (latest edition per year), columns: year, industry, measure, status, value."""
    d = pd.read_csv(RAW, low_memory=False)
    d = d[d.EMPLOYMENT_STATUS == 9]
    latest = d.groupby("TIME_PERIOD").EDITION.transform("max")
    d = d[d.EDITION == latest]
    return d.rename(columns={"TIME_PERIOD": "year", "BRKDW_INDUSTRY_NACE_REV2": "industry", "DATA_TYPE_AGGR": "measure", "OBS_VALUE": "value"})[["year", "industry", "measure", "value", "EDITION"]]
