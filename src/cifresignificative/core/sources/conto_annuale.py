"""Public-sector headcount by contract (RGS Conto annuale, table OCCUPAZIONE), to weight the public Istat agreement groups of notebook 39.

The Ragioneria Generale dello Stato collects the Conto annuale from every public administration (https://contoannuale.rgs.mef.gov.it/, "Download dati"); the portal serves one zip per year and type at `ext/CSV/<year>Dati.zip` and `<year>Anagrafiche.zip`
(the page's own JavaScript builds those URLs). `OCCUPAZIONE_<year>.CSV` has one row per institution, contract, category and qualification with the heads of staff, in six columns (full time, part time up to and over 50%, men and women);
the heads of a contract are the sum of all six columns over all rows. The contract codes (`MNST` ministries, `SSNA` health service, `SCLA` schools, `RALN` regions and local authorities, ...) are listed with their area in `CompartoContratto_<year>.CSV`.
Years 2021 and 2024 (31 December, the start and the end of the wage window of notebook 39; 2021 gives 3.24 million heads over 47 contracts), including the police, armed forces, fire brigade and magistrates that the CNEL-INPS counts (private sector only) do not cover.
The raw zips go to `data/raw/conto_annuale/` (gitignored); the build writes `data/external/conto_annuale_headcount.csv` (`year, contract, area_code, area, contract_name, heads`, committed), so tests and CI never fetch. The hand-curated map from Istat groups to contract codes is
`data/external/conto_annuale_istat_map.csv`. `uv run python -m cifresignificative.core.sources.conto_annuale fetch|build`.
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

BASE = "https://contoannuale.rgs.mef.gov.it/ext/CSV/"
RAW_DIR = Path("data/raw/conto_annuale")
CSV = Path("data/external/conto_annuale_headcount.csv")
MAP = Path("data/external/conto_annuale_istat_map.csv")
YEAR = 2021
YEARS = (2021, 2024)  # 31 December of the start and of the end of the wage window of notebook 39
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}


def path(kind: str, year: int = YEAR) -> Path:
    return RAW_DIR / f"{year}{kind}.zip"


def fetch(year: int = YEAR, force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for kind in ("Dati", "Anagrafiche"):
        if path(kind, year).exists() and not force:
            continue
        resp = requests.get(f"{BASE}{year}{kind}.zip", headers=UA, timeout=600)
        resp.raise_for_status()
        if not resp.content.startswith(b"PK"):
            raise RuntimeError(f"{year}{kind}: not a zip: {resp.content[:60]!r}")
        path(kind, year).write_bytes(resp.content)
        record(f"conto_annuale_{year}_{kind.lower()}", source="RGS Conto annuale, download dati", url=f"{BASE}{year}{kind}.zip")
    return RAW_DIR


def _year_table(year: int) -> pd.DataFrame:
    with zipfile.ZipFile(path("Dati", year)) as z:
        occ = pd.read_csv(z.open(f"{year}Dati/OCCUPAZIONE_{year}.CSV"), sep=";", encoding="utf-8-sig", dtype={"CONTRATTO": str})
    with zipfile.ZipFile(path("Anagrafiche", year)) as z:
        cc = pd.read_csv(z.open(f"{year}Anagrafiche/CompartoContratto_{year}.CSV"), sep=";", encoding="utf-8-sig", dtype=str)
    cols = [c for c in occ.columns if c.startswith(("PERSONALE", "PART_TIME"))]
    if len(cols) != 6:
        raise ValueError(f"{year}: expected six headcount columns, found {cols}")
    heads = occ.assign(heads=occ[cols].sum(axis=1)).groupby("CONTRATTO").heads.sum().astype(int).rename("heads").reset_index()
    d = heads.merge(cc, left_on="CONTRATTO", right_on="CODI_CONTRATTO", how="left", validate="one_to_one")
    if d.DESC_CONTRATTO.isna().any():
        raise ValueError(f"{year}: contracts without a description: {d[d.DESC_CONTRATTO.isna()].CONTRATTO.tolist()}")
    d = d.rename(columns={"CONTRATTO": "contract", "CODI_COMPARTO": "area_code", "DESC_AGGREGATO": "area", "DESC_CONTRATTO": "contract_name"}).assign(year=year)
    return d[["year", "contract", "area_code", "area", "contract_name", "heads"]]


def build_csv(years: tuple[int, ...] = YEARS, out: Path = CSV) -> pd.DataFrame:
    d = pd.concat([_year_table(y) for y in years], ignore_index=True).sort_values(["year", "heads"], ascending=[True, False], ignore_index=True)
    d.to_csv(out, index=False)
    return d


def load(csv: Path = CSV, year: int | None = None) -> pd.DataFrame:
    d = pd.read_csv(csv, keep_default_na=False, dtype={"contract": str})
    return d if year is None else d[d.year == year].reset_index(drop=True)


def load_map() -> pd.DataFrame:
    return pd.read_csv(MAP, keep_default_na=False)


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "build") == "fetch":
        for y in YEARS:
            fetch(y, force="--force" in sys.argv)
    else:
        d = build_csv()
        print(d.groupby("year").agg(contracts=("contract", "size"), heads=("heads", "sum")))
