"""Superbonus (SuperEcobonus) investments for completed works by region, from the ENEA annual reports on tax deductions.

Each report has a table of the investments for works with an end date up to 31 December of the reference year, by region and by type of building (condominiums, single-family houses, functionally independent units, A/9 open to the public), in euro:
the 2024 report (data 2023) Tabella 4.3 and the 2026 report (data 2025) Tabella 4.2. Trentino-Alto Adige is one row (the two provinces are separate NUTS2 regions in Eurostat). The investments are cumulative from the start of the Superbonus in July 2020 and by transmission of the final asseverations up to the extraction date (so late
transmissions raise the earlier snapshot); they cover the SuperEcobonus only (not the super sismabonus, the facade bonus or the other building bonuses). The PDFs go to `data/raw/enea/` (gitignored); parsing needs poppler's `pdftotext`; the committed table `data/external/enea_superbonus_regions.csv` is the source of truth.
`uv run python -m cifresignificative.core.sources.enea_regions fetch|build`.
"""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/enea")
CSV = Path("data/external/enea_superbonus_regions.csv")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
DL = "https://www.efficienzaenergetica.enea.it/component/jdownloads/?task=download.send&id={id}&catid=9&Itemid=101"
# snapshot -> (report, download id, table marker in the text, catid)
REPORTS = {
    "2023-12-31": ("Rapporto annuale detrazioni fiscali 2024 (dati 2023)", 720, 9, "Tabella 4.3 - Investimenti per lavori realizzati"),
    "2025-12-31": ("Rapporto annuale detrazioni fiscali 2026 (dati 2025)", 794, 40, "Tabella 4 .2 - Investimenti per lavori realizzati"),
}
# Italian region -> NUTS2 codes it contains
NUTS2 = {"Abruzzo": "ITF1", "Basilicata": "ITF5", "Calabria": "ITF6", "Campania": "ITF3", "Emilia-Romagna": "ITH5", "Friuli-Venezia Giulia": "ITH4", "Lazio": "ITI4", "Liguria": "ITC3", "Lombardia": "ITC4", "Marche": "ITI3", "Molise": "ITF2",
         "Piemonte": "ITC1", "Puglia": "ITF4", "Sardegna": "ITG2", "Sicilia": "ITG1", "Toscana": "ITI1", "Trentino-Alto Adige": "ITH1+ITH2", "Umbria": "ITI2", "Valle d'Aosta": "ITC2", "Veneto": "ITH3"}
ZONE = {"ITC": "North", "ITH": "North", "ITI": "Centre", "ITF": "South", "ITG": "South"}


def path(snapshot: str) -> Path:
    return RAW_DIR / f"enea_{snapshot}.pdf"


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for snap, (report, did, cat, _) in REPORTS.items():
        if path(snap).exists() and not force:
            continue
        url = f"https://www.efficienzaenergetica.enea.it/component/jdownloads/?task=download.send&id={did}&catid={cat}&Itemid=101"
        resp = requests.get(url, headers=UA, timeout=600)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            raise RuntimeError(f"{report}: not a PDF")
        path(snap).write_bytes(resp.content)
        record(f"enea_report_{snap}", source=f"ENEA, {report}", url=url)
    return RAW_DIR


def _number(tok: str) -> float:
    """Euro amount: `80.346.290.206` (no decimals) or `57.003.953.154,20` (comma decimals)."""
    return float(tok.replace(".", "").replace(",", ".")) if "," in tok else float(tok.replace(".", ""))


def parse_table(text: str, marker: str) -> pd.DataFrame:
    """The regional table that follows `marker`: one row per region with the investments for condominiums, single-family houses, functionally independent units, A/9 and the total (EUR); the total row is `Italia`."""
    lines = text.split("\n")
    i = next(k for k, ln in enumerate(lines) if marker in ln)
    rows = []
    for ln in lines[i: i + 40]:
        m = re.match(r"\s*([A-Za-zÀ-ú' \-]+?)\s{2,}((?:[\d.,]+\s+){3,4}[\d.,]+)\s*$", ln)
        if m:
            name = re.sub(r"\s*-\s*", "-", m.group(1).strip())
            v = [_number(t) for t in m.group(2).split()]
            if len(v) == 5:
                rows.append({"region": name, "condominiums": v[0], "single_family": v[1], "independent_units": v[2], "a9": v[3], "total": v[4]})
    d = pd.DataFrame(rows)
    parts = d[d.region != "Italia"].drop(columns="region").sum()
    tot = d[d.region == "Italia"].iloc[0]
    if len(d) != 21 or abs(parts.total - tot.total) > 100:
        raise ValueError(f"table not parsed: {len(d)} rows, regions sum {parts.total:.0f} against total {tot.total:.0f}")
    return d


def build_csv(out: Path = CSV) -> pd.DataFrame:
    parts = []
    for snap, (report, _, _, marker) in REPORTS.items():
        text = subprocess.run(["pdftotext", "-layout", str(path(snap)), "-"], capture_output=True, text=True, check=True).stdout
        d = parse_table(text, marker).assign(snapshot=snap, report=report, table=marker.split(" - ")[0].replace("  ", " ").replace("4 .", "4."))
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["nuts2"] = df.region.map(NUTS2)
    df.loc[df.region == "Italia", "nuts2"] = "IT"
    unknown = df[df.nuts2.isna()].region.unique()
    if len(unknown):
        raise ValueError(f"regions without a NUTS2 code: {unknown}")
    df = df[["snapshot", "region", "nuts2", "condominiums", "single_family", "independent_units", "a9", "total", "report", "table"]]
    df.to_csv(out, index=False)
    return df


def load(csv: Path = CSV) -> pd.DataFrame:
    return pd.read_csv(csv, parse_dates=["snapshot"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "fetch":
        print(fetch("--force" in sys.argv))
    else:
        d = build_csv()
        print(d.groupby("snapshot").total.apply(lambda s: s.iloc[-1] if False else s.max()).to_dict(), len(d), "rows")
