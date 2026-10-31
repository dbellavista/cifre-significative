"""Istat general-government economic account by subsector (esploradati SDMX, annual, EUR million): the capital transfers paid by the general government **by receiving sector**, in every edition (vintage) Istat published.

Two dataflows, one per period of publication (and two more for the quarterly account, `95_42_DF_DCCN_FPQ_1` and `5_42_DF_DCCN_FPQ_B19_1`, built into `data/external/istat_fpq_s13.csv`, items `OOKE_D_W0` other capital expenditure payable = D.92 + D.99, `OKE_D_W0` total capital expenditure, `P51G_D_W0`, `B9_B_W0`, ...): `95_815_DF_DCCN_FPA_1` (editions from September 2024) and `5_815_DF_DCCN_FPA_B19_1` (editions from September 2019 to April 2024; the March 2023 edition is the first that books the Superbonus and the facade bonus as payable credits in full). An edition is the month of publication, `2023M3` = March 2023.
Items (`DATA_TYPE_AGGR`, as `<aggregate>_<C receivable | D payable>_<W0 domestic | W1 rest of world>[_<counterpart>]`): `D92_D_W0_S14` investment grants to households, `_SIM` to corporations, `_S15` to non-profit institutions, `D99_D_W0_*` other capital transfers (the cash grants of the pandemic decrees),
`D9_C_W0` capital transfers receivable, `B9` net lending, `D3` subsidies, ... The raw files are large (9 and 16 MB, all subsectors and all items), so the build keeps the general government (`S13`) and writes `data/external/istat_fpa_s13.csv` (committed): `flow, edition, item, year, value_meur` (labels in `data/external/istat_fpa_labels.csv`); tests and notebooks never fetch.
`uv run python -m cifresignificative.core.sources.istat_fpa fetch|build`. Descriptive accounting data: the receiving sector of a transfer is Istat's classification, the flows are accrual (the credit is booked in full in the year the works are done).
"""
import sys
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

BASE = "https://esploradati.istat.it/SDMXWS/rest"
FLOWS = {"current": "95_815_DF_DCCN_FPA_1", "b19": "5_815_DF_DCCN_FPA_B19_1"}
RAW_DIR = Path("data/raw/istat")
CSV = Path("data/external/istat_fpa_s13.csv")
LABELS_CSV = Path("data/external/istat_fpa_labels.csv")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36", "Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}
FLOWS_Q = {"current": "95_42_DF_DCCN_FPQ_1", "b19": "5_42_DF_DCCN_FPQ_B19_1"}  # the quarterly account (non-adjusted `N` and seasonally adjusted `Y` series)
KEEP_Q = ("OOKE_D_W0", "OKE_D_W0", "P51G_D_W0", "B9_B_W0", "B9P_B_W0", "OTE_D_W0", "OTR_C_W0", "D41_D_W0", "D1_D_W0", "P2_D_W0", "D62_D_W0", "D9_C_W0")  # quarterly items kept, from 2015
CSV_Q = Path("data/external/istat_fpq_s13.csv")
LABELS_Q_CSV = Path("data/external/istat_fpq_labels.csv")
FIRST_YEAR = 1995
KEEP = ("OTE_D_W0", "OTR_C_W0", "OCE_D_W0", "OCR_C_W0", "OKE_D_W0", "B9_B_W0", "B9P_B_W0", "D3_D_W0", "D39_C_W0", "D92_", "D99_", "D9_C_W0", "D91_C_W0", "D75_D_W0", "D62_D_W0", "D632_D_W0", "D1_D_W0", "P2_D_W0", "P5_D_W0", "D41_D_W0", "D5_C_W0", "D2_C_W0", "D61_C_W0")  # prefixes of the items kept


def path(name: str, quarterly: bool = False) -> Path:
    return RAW_DIR / f"{'fpq' if quarterly else 'fpa'}_{name}.csv"


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for quarterly, flows in ((False, FLOWS), (True, FLOWS_Q)):
        for name, flow in flows.items():
            if path(name, quarterly).exists() and not force:
                continue
            url = f"{BASE}/data/IT1,{flow},1.0/all?startPeriod={FIRST_YEAR}"
            resp = requests.get(url, headers=UA, timeout=900)
            resp.raise_for_status()
            path(name, quarterly).write_bytes(resp.content)
            record(f"istat_{'fpq' if quarterly else 'fpa'}_{name}", source="istat-sdmx", dataset=flow, url=url)
    return RAW_DIR


def labels(flow: str = FLOWS["current"]) -> dict[str, str]:
    """English labels of the items from the dataflow's codelist."""
    r = requests.get(f"{BASE}/dataflow/IT1/{flow}/1.0?references=descendants&detail=full", headers={"Accept": "application/vnd.sdmx.structure+json;version=1.0"}, timeout=600)
    r.raise_for_status()
    out = {}
    for cl in r.json()["data"]["codelists"]:
        if cl["id"] == "CL_TIPO_DATO_DCCN_FP":
            out |= {c["id"]: (c.get("names") or {}).get("en", c["id"]) for c in cl["codes"]}
    return out


def build_csv(out: Path = CSV, labels_out: Path = LABELS_CSV) -> pd.DataFrame:
    lab = labels()
    parts = []
    for name in FLOWS:
        d = pd.read_csv(path(name), low_memory=False, usecols=["DATA_TYPE_AGGR", "INSTITUTIONAL_SECTOR", "EDITION", "TIME_PERIOD", "OBS_VALUE"])
        d = d[d.INSTITUTIONAL_SECTOR.eq("S13") & d.DATA_TYPE_AGGR.str.startswith(KEEP)]
        parts.append(pd.DataFrame({"flow": name, "edition": d.EDITION, "item": d.DATA_TYPE_AGGR, "year": d.TIME_PERIOD, "value_meur": d.OBS_VALUE}))
    df = pd.concat(parts, ignore_index=True).sort_values(["flow", "edition", "item", "year"], ignore_index=True)
    missing = sorted(set(df.item) - set(lab))
    if missing:
        raise ValueError(f"items without a label: {missing}")
    df.to_csv(out, index=False)
    pd.DataFrame({"item": sorted(set(df.item)), "label": [lab[i] for i in sorted(set(df.item))]}).to_csv(labels_out, index=False)
    return df


def build_quarterly_csv(out: Path = CSV_Q, labels_out: Path = LABELS_Q_CSV) -> pd.DataFrame:
    """The quarterly general-government account (the items of `KEEP_Q` from 2015, both adjustments, all editions) as `flow, edition, adjustment, item, period, value_meur`."""
    lab = labels(FLOWS_Q["current"])
    parts = []
    for name in FLOWS_Q:
        d = pd.read_csv(path(name, True), low_memory=False, usecols=["DATA_TYPE_AGGR", "INSTITUTIONAL_SECTOR", "ADJUSTMENT", "EDITION", "TIME_PERIOD", "OBS_VALUE"])
        d = d[d.INSTITUTIONAL_SECTOR.eq("S13") & d.DATA_TYPE_AGGR.isin(KEEP_Q) & (d.TIME_PERIOD.str[:4].astype(int) >= 2015)]
        parts.append(pd.DataFrame({"flow": name, "edition": d.EDITION, "adjustment": d.ADJUSTMENT, "item": d.DATA_TYPE_AGGR, "period": d.TIME_PERIOD, "value_meur": d.OBS_VALUE}))
    df = pd.concat(parts, ignore_index=True).sort_values(["flow", "edition", "adjustment", "item", "period"], ignore_index=True)
    items = sorted(set(df["item"]))
    pd.DataFrame({"item": items, "label": [lab.get(i, i) for i in items]}).to_csv(labels_out, index=False)
    df.to_csv(out, index=False)
    return df


def load_quarterly(csv: Path = CSV_Q) -> pd.DataFrame:
    d = pd.read_csv(csv)
    d["published"] = pd.to_datetime(d.edition.str.replace("M", "-", regex=False) + "-01", format="%Y-%m-%d")
    return d


def load(csv: Path = CSV) -> pd.DataFrame:
    """`flow, edition, item, year, value_meur` (labels in `data/external/istat_fpa_labels.csv`); add `published` (first day of the publication month) for ordering."""
    d = pd.read_csv(csv)
    d["published"] = pd.to_datetime(d.edition.str.replace("M", "-", regex=False) + "-01", format="%Y-%m-%d")
    return d


def latest_edition(d: pd.DataFrame) -> str:
    return d.loc[d.published.idxmax(), "edition"]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "fetch":
        print(fetch("--force" in sys.argv))
    else:
        out = build_csv()
        print(len(out), "rows;", out.groupby("flow").edition.nunique().to_dict(), "editions; items", out.item.nunique())
        q = build_quarterly_csv()
        print("quarterly:", len(q), "rows;", q.groupby("flow").edition.nunique().to_dict(), "editions; items", q["item"].nunique())
