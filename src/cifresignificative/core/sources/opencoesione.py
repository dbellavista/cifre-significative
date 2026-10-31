"""OpenCoesione, cycle 2014-2020: public financing and payments of the cohesion projects by Italian region (the Italy-only track of notebook 34).

OpenCoesione publishes one row per project (`progetti_esteso_<fund>_2014-2020.zip`: ERDF (FESR), ESF (FSE), the national Development and Cohesion Fund (FSC), the Complementary Action Plan (PAC) and ordinary national resources (ORD)) and a separate localisation table (`localizzazioni.zip`: project x municipality). Money is not split by place in the files, so a project with several regions or provinces is divided **equally** among its distinct Italian regions (Trentino-Alto Adige is split into the two autonomous provinces, NUTS2 ITH1 Bolzano and ITH2 Trento); projects with no Italian localisation ("national scope", abroad, none) are kept apart and reported. Financing is the net of economies (`OC_FINANZ_*_NETTO`: EU part, total public, so the national part is the difference); payments are `TOT_PAGAMENTI`. Data as of the 30 June 2026 release.
Output `data/external/opencoesione_regions.csv` (committed; region x fund file) and `data/external/opencoesione_unallocated.csv`. The zips go to `data/raw/opencoesione/` (gitignored). `uv run python -m cifresignificative.core.sources.opencoesione fetch|build`.
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/opencoesione")
BASE = "https://opencoesione.gov.it/it/opendata/"
FUNDS = ["FESR", "FSE", "FSC", "PAC", "ORD"]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
REGIONS_CSV = Path("data/external/opencoesione_regions.csv")
UNALLOC_CSV = Path("data/external/opencoesione_unallocated.csv")
NUTS2 = {"001": "ITC1", "002": "ITC2", "003": "ITC4", "005": "ITH3", "006": "ITH4", "007": "ITC3", "008": "ITH5", "009": "ITI1", "010": "ITI2", "011": "ITI3", "012": "ITI4", "013": "ITF1", "014": "ITF2", "015": "ITF3", "016": "ITF4", "017": "ITF5", "018": "ITF6",
         "019": "ITG1", "020": "ITG2"}
TAA = {"021": "ITH1", "022": "ITH2"}  # province codes of Trentino-Alto Adige
COLS = ["COD_LOCALE_PROGETTO", "OC_FINANZ_UE_NETTO", "OC_FINANZ_TOT_PUB_NETTO", "TOT_PAGAMENTI"]


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in [(f"progetti_esteso_{f}_2014-2020.zip", f"{BASE}fondi/progetti_esteso_{f}_2014-2020.zip") for f in FUNDS] + [("localizzazioni.zip", f"{BASE}localizzazioni.zip")]:
        p = RAW_DIR / name
        if p.exists() and not force:
            continue
        r = requests.get(url, headers=UA, timeout=600)
        r.raise_for_status()
        if not r.content.startswith(b"PK"):
            raise RuntimeError(f"{name}: not a zip")
        p.write_bytes(r.content)
        record(f"opencoesione_{name}", source="OpenCoesione", dataset=name, url=url)
    return RAW_DIR


def _read(zpath: Path, usecols: list[str], **kw) -> pd.DataFrame:
    with zipfile.ZipFile(zpath) as z:
        member = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(member) as f:
            return pd.read_csv(f, sep=";", usecols=usecols, dtype=str, **kw)


def project_regions() -> pd.DataFrame:
    """Distinct Italian NUTS2 area of each project (Trentino-Alto Adige by province) and the number of such areas; projects with only national / abroad / no localisation get `area` = None."""
    L = _read(RAW_DIR / "localizzazioni.zip", ["COD_LOCALE_PROGETTO", "COD_REGIONE", "COD_PROVINCIA"])
    area = L.COD_REGIONE.map(NUTS2)
    taa = L.COD_REGIONE.eq("004")
    area = area.where(~taa, L.COD_PROVINCIA.map(TAA))  # TAA rows with 'all provinces' stay unassigned
    L = L.assign(area=area).dropna(subset=["area"]).drop_duplicates(["COD_LOCALE_PROGETTO", "area"])
    return L[["COD_LOCALE_PROGETTO", "area"]]


def build(out: Path = REGIONS_CSV, unalloc: Path = UNALLOC_CSV) -> pd.DataFrame:
    loc = project_regions()
    n_areas = pd.Series(loc.groupby("COD_LOCALE_PROGETTO").size(), name="n_areas")
    rows, un = [], []
    for f in FUNDS:
        P = _read(RAW_DIR / f"progetti_esteso_{f}_2014-2020.zip", COLS)
        for c in COLS[1:]:
            P[c] = pd.to_numeric(P[c].str.replace(",", "."), errors="coerce").fillna(0.0)  # decimal comma
        P["national_public"] = P.OC_FINANZ_TOT_PUB_NETTO - P.OC_FINANZ_UE_NETTO
        P = P.rename(columns={"OC_FINANZ_UE_NETTO": "eu", "OC_FINANZ_TOT_PUB_NETTO": "total_public", "TOT_PAGAMENTI": "payments"})
        m = P.merge(loc, on="COD_LOCALE_PROGETTO", how="left").merge(n_areas, left_on="COD_LOCALE_PROGETTO", right_index=True, how="left")
        un.append(pd.DataFrame({"fund": [f], "projects": [len(P)], "projects_unallocated": [int(m[m.area.isna()].COD_LOCALE_PROGETTO.nunique())], "total_public_meur": [P.total_public.sum() / 1e6], "unallocated_total_public_meur": [m[m.area.isna()].total_public.sum() / 1e6],
                                "multi_area_projects": [int((n_areas.reindex(P.COD_LOCALE_PROGETTO.unique()) > 1).sum())], "multi_area_total_public_meur": [m[m.n_areas > 1].drop_duplicates("COD_LOCALE_PROGETTO").total_public.sum() / 1e6]}))
        m = m.dropna(subset=["area"])
        for c in ("eu", "national_public", "total_public", "payments"):
            m[c] = m[c] / m.n_areas
        g = m.groupby("area").agg(projects=("COD_LOCALE_PROGETTO", "nunique"), eu_meur=("eu", lambda s: s.sum() / 1e6), national_public_meur=("national_public", lambda s: s.sum() / 1e6), total_public_meur=("total_public", lambda s: s.sum() / 1e6), payments_meur=("payments", lambda s: s.sum() / 1e6)).reset_index()
        rows.append(g.assign(fund=f))
    d = pd.concat(rows, ignore_index=True)[["fund", "area", "projects", "eu_meur", "national_public_meur", "total_public_meur", "payments_meur"]].sort_values(["fund", "area"], ignore_index=True)
    d.to_csv(out, index=False)
    pd.concat(un, ignore_index=True).to_csv(unalloc, index=False)
    return d


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "fetch":
        print(fetch("--force" in sys.argv))
    else:
        print(build().groupby("fund")[["eu_meur", "national_public_meur", "total_public_meur", "payments_meur"]].sum().round(0).to_string())
        print(pd.read_csv(UNALLOC_CSV).round(0).to_string())
