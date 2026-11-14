"""Conti Pubblici Territoriali (CPT, Agenzia per la Coesione Territoriale): public spending by region, sector and level of government, 2000 onwards, EUR million.

Dataset family `SP_SPA_SE_TS_*` = expenditure ('SP') of the extended public sector ('SPA', consolidated: central and local government plus public enterprises), by sector of intervention ('SE'), time series ('TS').
`CETOT` = total expenditure, `CETOTCAP` = capital account only (CAP is capital, not per capita), `CETOTCOR` = current account. The files are semicolon-separated with decimal commas and are hosted at
hashed `/media/<hash>/` paths that change between releases, so the fetcher finds them from the catalog page (single files until October 2026, one zip archive per family since). Spending is attributed to the region where it is incurred, including central-government spending (pensions, defence), so it is not what a regional government chooses to spend.
"""
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

HOST = "https://politichecoesione.governo.it"
CATALOG = f"{HOST}/it/politica-di-coesione/misurazione-valutazione-e-trasparenza/la-misurazione-delle-politiche-di-coesione/conti-pubblici-territoriali-cpt/i-dati/catalogo-open-cpt/"
RAW_DIR = Path("data/raw/cpt")
FILES = {"total": "sp_spa_se_ts_cetot.csv", "capital": "sp_spa_se_ts_cetotcap.csv", "current": "sp_spa_se_ts_cetotcor.csv"}
ARCHIVE = "spa_spese"  # archive of the SPA expenditure tables, from November 2026
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
PAGES = range(1, 9)  # the catalog table is paginated (7 pages in 2026; one spare)
NUTS2 = {  # ISTAT region code -> NUTS2 (the autonomous provinces 21 Trento and 22 Bolzano are NUTS2 regions of their own)
    "01": "ITC1", "02": "ITC2", "03": "ITC4", "05": "ITH3", "06": "ITH4", "07": "ITC3", "08": "ITH5", "09": "ITI1", "10": "ITI2", "11": "ITI3", "12": "ITI4",
    "13": "ITF1", "14": "ITF2", "15": "ITF3", "16": "ITF4", "17": "ITF5", "18": "ITF6", "19": "ITG1", "20": "ITG2", "21": "ITH2", "22": "ITH1",
}


def find_links(pages: list[str]) -> dict[str, str]:
    """file name -> URL, from the catalog pages' HTML."""
    links = {}
    for html in pages:
        for href in re.findall(r'href="(/media/[^"]+\.csv)"', html):
            links[href.rsplit("/", 1)[-1]] = HOST + href
    return links


def find_archive(pages: list[str]) -> str | None:
    """URL of the archive of the extended-public-sector expenditure tables (`/media/<hash>/spa_spese-<n>.zip`), None when the pages have none."""
    for html in pages:
        for href in re.findall(r'href="(/media/[^"]+\.zip)"', html):
            if href.rsplit("/", 1)[-1].lower().startswith(ARCHIVE):
                return HOST + href
    return None


def extract(archive: bytes) -> dict[str, bytes]:
    """file name (lower case, as in FILES) -> content, for the files of FILES found in the archive (its names are upper case, in subfolders)."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1].lower()
            if base in FILES.values():
                out[base] = z.read(name)
    return out


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not force and all((RAW_DIR / f).exists() for f in FILES.values()):
        return RAW_DIR
    pages = [requests.get(CATALOG, params={"page": p}, headers=UA, timeout=120) for p in PAGES]
    links = find_links([r.text for r in pages if r.ok])
    missing = [f for f in FILES.values() if f not in links]
    if missing:  # since November 2026 the catalog lists one archive per family instead of the single files
        archive = find_archive([r.text for r in pages if r.ok])
        if archive is None:
            raise RuntimeError(f"CPT catalog: files not found {missing} and no {ARCHIVE} archive")
        resp = requests.get(archive, headers=UA, timeout=600)
        resp.raise_for_status()
        members = extract(resp.content)
        if set(members) != set(FILES.values()):
            raise RuntimeError(f"CPT archive: files not found {sorted(set(FILES.values()) - set(members))}")
        for f, content in members.items():
            (RAW_DIR / f).write_bytes(content)
        record("cpt", source="politichecoesione.governo.it", dataset="Open CPT SP_SPA_SE_TS_*", url=archive)
        return RAW_DIR
    for f in FILES.values():
        resp = requests.get(links[f], headers=UA, timeout=300)
        resp.raise_for_status()
        (RAW_DIR / f).write_bytes(resp.content)
    record("cpt", source="politichecoesione.governo.it", dataset="Open CPT SP_SPA_SE_TS_*", url=CATALOG)
    return RAW_DIR


def load(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Long table `geo (NUTS2), year, sector, account (total/capital/current), level (government tier), value (EUR million)`."""
    frames = []
    for account, f in FILES.items():
        d = pd.read_csv(raw_dir / f, sep=";", decimal=",", encoding="utf-8-sig")
        d = d.rename(columns={"Regione per Dettaglio": "region", "Settore": "sector", "Tipologia di Soggetti": "level", "Anno": "year", "S - Consolidato SPA": "value"})
        frames.append(pd.DataFrame({"geo": d.region.str[:2].map(NUTS2), "year": d.year, "sector": d.sector.str.split(" - ", n=1).str[1], "account": account, "level": d.level, "value": d.value}))
    out = pd.concat(frames, ignore_index=True)
    if out.geo.isna().any():
        raise ValueError("CPT: unmapped region codes")
    return out
