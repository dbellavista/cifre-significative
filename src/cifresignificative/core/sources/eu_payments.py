"""EU payments by NUTS2 region (European Commission, DG REGIO: "Historic EU payments - regionalised and modelled", cohesiondata.ec.europa.eu dataset `tc55-7ysv`).

Annual payments (current EUR) of the ERDF, the Cohesion Fund, the ESF and the EAFRD/EAGGF mapped to NUTS2 regions (predominantly the NUTS 2013 codes), by programming period, calendar year of the payment (the Commission's payment cycle,
not the date of the spending on the ground; the dataset also has a modelled series of real expenditure, not used here). Interreg is not included. The 2007-2013 programme has payments from 2006 to 2022, most of them up to 2015.
Codes are translated to NUTS 2021 with `clean.nuts` (renames carried over, regions that were split left out), and the payments of a programme are summed per region: a cumulative amount per inhabitant and as a share of regional GDP over the programme years,
which is how much support the region received, not how well it was spent. Raw file in `data/raw/cohesion/` (gitignored). `uv run python -m cifresignificative.core.fetch --only eu_payments`.
"""
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.clean import nuts
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

URL = "https://cohesiondata.ec.europa.eu/resource/tc55-7ysv.csv?$limit=100000"
RAW = Path("data/raw/cohesion/historic_payments_nuts2.csv")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
COHESION = ("ERDF", "CF", "ESF")  # cohesion policy; the EAFRD/EAGGF (rural development) is added in the "all funds" variant


def fetch(force: bool = False) -> Path:
    if RAW.exists() and not force:
        return RAW
    RAW.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, headers=UA, timeout=300)
    resp.raise_for_status()
    if not resp.text.startswith('"country"'):
        raise RuntimeError(f"EU payments: not the expected CSV: {resp.text[:80]!r}")
    RAW.write_bytes(resp.content)
    record("eu_payments", source="cohesiondata.ec.europa.eu", dataset="tc55-7ysv", url=URL)
    return RAW


def load_period(period: str = "2007-2013", path: Path = RAW) -> pd.DataFrame:
    """Cumulative payments of a programming period by NUTS 2021 region (EUR): columns `ERDF`, `CF`, `ESF`, `EAFRD`, `cohesion` (ERDF + CF + ESF) and `all_funds`; a fund with no payment in a region counts as zero."""
    d = pd.read_csv(path)
    d = d[d.programming_period == period]
    s = d.groupby(["nuts2_id", "fund"]).eu_payment_annual.sum().unstack(fill_value=0)
    long = s.stack().rename("value").reset_index().rename(columns={"nuts2_id": "geo", "fund": "indicator"})
    long = long.assign(year=2010, flag=None, country=long.geo.str[:2])
    h = nuts.harmonise(long)
    out = h.pivot_table(index="geo", columns="indicator", values="value", aggfunc="sum").fillna(0)
    for f in (*COHESION, "EAFRD"):
        if f not in out:
            out[f] = 0.0
    out["cohesion"] = out[list(COHESION)].sum(axis=1)
    out["all_funds"] = out["cohesion"] + out["EAFRD"]
    return out[[*COHESION, "EAFRD", "cohesion", "all_funds"]].rename_axis("geo")
