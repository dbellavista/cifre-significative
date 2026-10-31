"""EBA EU-wide transparency exercise, sovereign exposures (`tr_sov.csv`, semi-annual, 2019-12 to 2025-06): Italian banks' direct on-balance-sheet exposures to the Italian general government.
Full-database files are direct downloads (`assets/TE<year>/Full_database/<id>/tr_sov.csv`; 2020 from tools.eba.europa.eu). Scope differs from the ECB SHSS series: about 12 large Italian banks (the sample changes by round), non-derivative financial assets of all kinds (loans and securities; amortised cost included), exposures to central, regional and local government.
`Item` codes change by vintage, so rows are selected by exact `Label`. `Country` 15 = Italy (sheet `Country` of `TR_Metadata.xlsx`); `Maturity` 8 = total. Amounts are EUR million."""
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/eba_transparency")
ASSETS = "https://www.eba.europa.eu/assets"
FILES = {  # year -> url of tr_sov.csv
    2020: "https://tools.eba.europa.eu/interactive-tools/2020/Transparency_2020/tr_sov.csv",
    2021: f"{ASSETS}/TE2021/Full_database/tr_sov.csv",
    2022: f"{ASSETS}/TE2022/Full_database/tr_sov.csv",
    2023: f"{ASSETS}/TE2023/Full_database/837203/tr_sov.csv",
    2024: f"{ASSETS}/TE2024/Full_database/256109/tr_sov.csv",
    2025: f"{ASSETS}/TE2025/Full_database/883401/tr_sov.csv",
}
ITALY, TOTAL_MATURITY = "15", "8"
MEASURES = {
    "gross": "Direct exposures - On balance sheet - Total gross carrying amount of non-derivative financial assets",
    "net": "Direct exposures - On balance sheet - Total carrying amount of non-derivative financial assets (net of short positions)",
}


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year, url in FILES.items():
        out = RAW_DIR / f"{year}_tr_sov.csv"
        if out.exists() and not force:
            continue
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=600)
        resp.raise_for_status()
        out.write_bytes(resp.content)
        record(f"eba_sov_{year}", source="eba-transparency", dataset="tr_sov", url=url)
    return RAW_DIR


def _read(path: Path) -> pd.DataFrame:
    cols = ["LEI_Code", "NSA", "Period", "Label", "Country", "Maturity", "Amount"]
    try:
        return pd.read_csv(path, dtype=str, usecols=cols)
    except UnicodeDecodeError:  # the 2020 file is latin-1
        return pd.read_csv(path, dtype=str, usecols=cols, encoding="latin-1")


def load(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Italian banks' exposure to Italy, long format `measure, period (reference date), value (EUR bn), n_banks, source`. A reference date present in two files must agree (the newest file wins otherwise it raises)."""
    labels = {v: k for k, v in MEASURES.items()}
    parts = []
    for path in sorted(raw_dir.glob("*_tr_sov.csv")):
        d = _read(path)
        d = d[(d.NSA == "IT") & (d.Country == ITALY) & (d.Maturity == TOTAL_MATURITY) & d.Label.isin(labels)]
        d = d.assign(measure=d.Label.map(labels), amount=pd.to_numeric(d.Amount) / 1e3, file=path.name)
        g = d.groupby(["file", "Period", "measure"]).agg(value=("amount", "sum"), n_banks=("LEI_Code", "nunique")).reset_index()
        parts.append(g)
    g = pd.concat(parts, ignore_index=True)
    dup = g[g.duplicated(["Period", "measure"], keep=False)]
    if len(dup) and (dup.groupby(["Period", "measure"]).value.agg(lambda s: s.max() - s.min()) > 1e-6).any():
        raise ValueError("EBA files disagree on a reference date")
    g = g.drop_duplicates(["Period", "measure"], keep="last")
    g["period"] = pd.to_datetime(g.Period, format="%Y%m") + pd.offsets.MonthEnd(0)
    return g.assign(source="eba:tr_sov")[["measure", "period", "value", "n_banks", "source"]].sort_values(["measure", "period"], ignore_index=True)
