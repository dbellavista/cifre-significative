"""Istat internal migration between Italian regions (esploradati SDMX, dataflow 28_185_DF_DCIS_MIGRAZIONI_1): annual changes of residence, 2002 onwards, as an origin-destination matrix.

The matrix is published for every NUTS2 region and for macro-area and national aggregates (`ITC`, `ITD`, ..., `IT`, `ITCD`, `ITDA`, `ITFG`), which are dropped here; the region codes are translated from NUTS 2010 (ITD, ITE) to the current ones (ITH, ITI). Sex and citizenship are kept; age is only `TOTAL`, so movers' education is not available.
A region's own diagonal (moves within the region) is kept in the raw file and dropped in the derived net flows. Residence changes recorded in the registry, not people (one person can move twice); recent years are provisional (`OBS_STATUS` p).
"""
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

FLOW = "28_185_DF_DCIS_MIGRAZIONI_1"
URL = f"https://esploradati.istat.it/SDMXWS/rest/data/IT1,{FLOW},1.0/all?startPeriod=2002"
RAW = Path("data/raw/istat/internal_migration.csv")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36", "Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}
SEX = {1: "M", 2: "F", 9: "T"}
NUTS_2010 = {"ITD": "ITH", "ITE": "ITI"}  # Istat still publishes the NUTS 2010 codes of the North-East (ITD) and the Centre (ITE); Eurostat's current codes are ITH and ITI
CITIZENSHIP = {"IT": "italian", "FOR": "foreign", "TOTAL": "total"}


def fetch(force: bool = False) -> Path:
    if RAW.exists() and not force:
        return RAW
    RAW.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, headers=UA, timeout=900)
    resp.raise_for_status()
    RAW.write_bytes(resp.content)
    record("istat_internal_migration", source="istat-sdmx", dataset=FLOW, url=URL)
    return RAW


def load_flows(path: Path = RAW) -> pd.DataFrame:
    """Origin-destination flows between NUTS2 regions: `origin, destination, year, sex, citizenship, value (moves), provisional`."""
    d = pd.read_csv(path, low_memory=False, usecols=["REF_AREA", "TERRITORY_NEXT_RESID", "TIME_PERIOD", "CITIZENSHIP", "SEX", "OBS_VALUE", "OBS_STATUS"])
    d = d[d.REF_AREA.str.len().eq(4) & d.TERRITORY_NEXT_RESID.str.len().eq(4) & ~d.REF_AREA.isin(["ITCD", "ITDA", "ITFG"]) & ~d.TERRITORY_NEXT_RESID.isin(["ITCD", "ITDA", "ITFG"])]
    def nuts(codes: pd.Series) -> pd.Series:
        return codes.map(lambda c: NUTS_2010.get(c[:3], c[:3]) + c[3:])
    return pd.DataFrame({"origin": nuts(d.REF_AREA), "destination": nuts(d.TERRITORY_NEXT_RESID), "year": d.TIME_PERIOD, "sex": d.SEX.map(SEX), "citizenship": d.CITIZENSHIP.map(CITIZENSHIP), "value": d.OBS_VALUE, "provisional": d.OBS_STATUS.eq("p")}).reset_index(drop=True)


def net_flows(flows: pd.DataFrame) -> pd.DataFrame:
    """Per region and year (all sexes, all citizenships): moves in from, and out to, the other regions; `net` = in - out."""
    f = flows[flows.sex.eq("T") & flows.citizenship.eq("total") & flows.origin.ne(flows.destination)]
    inflow = f.groupby(["destination", "year"]).value.sum().rename("inflow")
    outflow = f.groupby(["origin", "year"]).value.sum().rename("outflow")
    out = pd.concat([inflow.rename_axis(["geo", "year"]), outflow.rename_axis(["geo", "year"])], axis=1).reset_index()
    return out.assign(net=out.inflow - out.outflow)
