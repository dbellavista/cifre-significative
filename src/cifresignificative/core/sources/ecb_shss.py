"""ECB Securities Holdings Statistics, sector level (dataflow SHSS, Italy as holder area): who holds domestic debt securities, quarterly, from 2021-Q1.
The holder-by-issuer dataflow `SHS` is blocked by the ECB firewall, `SHSS` is served with a full 18-position key (a short or wildcard key can trigger the firewall page).
Banks against issuer S13 (and against S11) are not published: `it_banks_gov_nfc_residual` = banks' domestic holdings (counterpart S1) minus the S12 and S1M counterparts = general government PLUS non-financial corporations, an upper bound for the government holdings (identity assumed, not confirmed)."""
import io
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

API = "https://data-api.ecb.europa.eu/service/data/SHSS"
RAW = Path("data/raw/ecb_shss/shss_it.csv")
# FREQ.ADJ.REF_AREA.CPART_AREA.REF_SECTOR.CPART_SECTOR.CONSOL.ENTRY.STO.INSTR.MATURITY.EXPEND.UNIT.CURR.VALUATION.PRICES.TRANSF.CUST
KEY = "Q.N.IT.W2.{holder}.{issuer}.N.A.LE.F3.T._Z.XDC._T.M.V.N._T"  # stock, debt securities, all maturities, market value, domestic issuers, EUR million
SERIES = {  # name -> (holder sector, issuer sector)
    "it_fin_corps_gov": ("S12", "S13"),
    "it_central_bank_gov": ("S121", "S13"),
    "it_banks_domestic_total": ("S122", "S1"),
    "it_banks_in_fin": ("S122", "S12"),
    "it_banks_in_households": ("S122", "S1M"),
}
DERIVED = "it_banks_gov_nfc_residual"


def fetch(force: bool = False) -> Path:
    if RAW.exists() and not force:
        return RAW
    frames = []
    for name, (holder, issuer) in SERIES.items():
        resp = requests.get(f"{API}/{KEY.format(holder=holder, issuer=issuer)}", headers={"Accept": "text/csv"}, timeout=120)
        resp.raise_for_status()
        if not resp.text.startswith("KEY,"):  # the firewall answers with an HTML page
            raise RuntimeError(f"SHSS {name}: not a CSV answer (firewall or wrong key): {resp.text[:80]!r}")
        frames.append(pd.read_csv(io.StringIO(resp.text), usecols=["KEY", "TIME_PERIOD", "OBS_VALUE"]).assign(series=name))
    RAW.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(RAW, index=False)
    record("ecb_shss", source="data-api.ecb.europa.eu", dataset="SHSS", url=f"{API}/<key>", keys=KEY)
    return RAW


def load(path: Path = RAW) -> pd.DataFrame:
    """Long format `series, period (quarter end), value (EUR million), source`, including the derived banks-to-government series."""
    d = pd.read_csv(path)
    w = d.pivot(index="TIME_PERIOD", columns="series", values="OBS_VALUE")
    w[DERIVED] = w["it_banks_domestic_total"] - w["it_banks_in_fin"] - w["it_banks_in_households"]
    out = w.stack().rename("value").reset_index().rename(columns={"TIME_PERIOD": "quarter"})
    out["period"] = pd.Series(pd.PeriodIndex(out["quarter"].astype(str), freq="Q")).dt.end_time.dt.normalize()
    return out.assign(source="ecb:SHSS")[["series", "period", "value", "source"]].sort_values(["series", "period"], ignore_index=True)
