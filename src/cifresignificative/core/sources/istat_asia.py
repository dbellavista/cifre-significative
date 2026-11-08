"""ISTAT Esploradati SDMX: Active enterprises' local units and persons employed at municipal level by 2-digit ATECO (`183_285_DF_DICA_ASIAULP_7`).

Downloads the annual ASIA local-units series (2012-2024) for the 13 Emilia-Romagna peer municipalities (`core.geo.ER_COMUNI`), plus
the 9 Emilia-Romagna provinces (`ITD51`-`ITD59`), the region (`ITD5`) and Italy (`IT`) from dataflow `183_285_DF_DICA_ASIAULP_7` (`fetch_areas`; NUTS 2010 codes only).
Maps 2-digit ATECO 2007 divisions into Cesena's key specialization clusters (`agro_industrial`, `ict_software`, `wellness_biomed`,
`construction_re`, `tradable_manuf`, `commerce_hospitality`, `services_other`) for Shift-Share and average firm-size analysis.
"""
import sys
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.geo import (
    ALL_COMUNI_IDS,
    BELT_BY_ID,
    BELT_COMUNI_IDS,
    format_comune_id,
    harmonise_nuts3,
)
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/istat_comuni")
BASE = "https://esploradati.istat.it/SDMXWS/rest/data"
FLOW_MUNI_2D = "183_285_DF_DICA_ASIAULP_7"
FLOW_KEY = "183_285_DF_DICA_ASIAULP_1"
AREA_CODES: tuple[str, ...] = (*(f"ITD5{k}" for k in range(1, 10)), "ITD5", "IT")  # NUTS 2010 codes: the flow has no ITH*
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36", "Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}

# Cesena economic specialization clusters mapped from 2-digit ATECO 2007 divisions
ATECO_CLUSTERS: dict[str, str] = {
    "01": "agro_industrial", "02": "agro_industrial", "03": "agro_industrial",
    "10": "agro_industrial", "11": "agro_industrial", "12": "agro_industrial",
    "28": "agro_industrial",  # agricultural & food-packaging machinery (Cesena/Romagna mechatronics)
    "21": "wellness_biomed", "26": "wellness_biomed", "32": "wellness_biomed", "86": "wellness_biomed", "93": "wellness_biomed",
    "58": "ict_software", "61": "ict_software", "62": "ict_software", "63": "ict_software",
    "41": "construction_re", "42": "construction_re", "43": "construction_re", "68": "construction_re",
    "45": "commerce_hospitality", "46": "commerce_hospitality", "47": "commerce_hospitality", "55": "commerce_hospitality", "56": "commerce_hospitality",
}
SPECIALIZATION_LABELS: dict[str, str] = {
    "agro_industrial": "Agro-industrial & food machinery",
    "wellness_biomed": "Wellness, biomedical & sports",
    "ict_software": "ICT, software & digital services",
    "construction_re": "Construction & real estate",
    "tradable_manuf": "Other tradable manufacturing",
    "commerce_hospitality": "Commerce, tourism & hospitality",
    "services_other": "Professional, financial & other services",
    "TOTAL": "Total non-farm economy",
}

ATECO_2D_LABELS: dict[str, str] = {
    "01": "Crop & animal production", "10": "Food products", "11": "Beverages",
    "25": "Fabricated metal products", "26": "Computer, electronic & optical",
    "27": "Electrical equipment", "28": "Machinery & equipment n.e.c.",
    "32": "Other manufacturing (incl. medical/wellness)", "33": "Repair & installation of machinery",
    "41": "Construction of buildings", "42": "Civil engineering", "43": "Specialized construction",
    "45": "Motor vehicle trade/repair", "46": "Wholesale trade", "47": "Retail trade",
    "49": "Land transport", "52": "Warehousing & transport support",
    "55": "Accommodation", "56": "Food & beverage service",
    "58": "Publishing", "61": "Telecommunications", "62": "Software & IT consultancy", "63": "Information services",
    "64": "Financial services", "68": "Real estate activities", "69": "Legal & accounting",
    "70": "Head offices & management consultancy", "71": "Architecture & engineering",
    "72": "Scientific R&D", "73": "Advertising & market research", "74": "Other professional/technical",
    "85": "Education", "86": "Human health activities", "93": "Sports & recreation", "96": "Personal services",
    "TOTAL": "Total economy",
}


MANUF_DIVS = {f"{k:02d}" for k in range(13, 34)} - {"21", "26", "28", "32"}

SIZE_CLASS_ORDER: tuple[str, ...] = ("W0_9", "W10_49", "W50_249", "W_GE250", "TOTAL")
SIZE_CLASS_LABELS: dict[str, str] = {
    "W0_9": "Micro (0-9 employees)",
    "W10_49": "Small (10-49 employees)",
    "W50_249": "Medium (50-249 employees)",
    "W_GE250": "Large (250+ employees)",
    "TOTAL": "All size classes",
}


def classify_ateco2d(code: str) -> str:
    """Map a 2-digit ATECO 2007 code ('10', '62', ...) to one of the 6 structural clusters."""
    if code in ATECO_CLUSTERS:
        return ATECO_CLUSTERS[code]
    if code in MANUF_DIVS or code in ("05", "06", "07", "08", "09", "35", "36", "37", "38", "39"):
        return "tradable_manuf"
    return "services_other"


def raw_path(raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / "asia_muni_2d.csv"


def size_raw_path(raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / "asia_muni_size_2d.csv"


def belt_raw_path(raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / "asia_belt_total.csv"


def fetch(force: bool = False, raw_dir: Path = RAW_DIR) -> Path:
    """Download ASIA 2-digit ATECO local units (`LU`) and persons employed (`LUEMPDAA`) for the 13 ER municipalities (`PERS_EMPL_SIZE_CLASS=TOTAL`)."""
    out = raw_path(raw_dir)
    if not out.exists() or force:
        raw_dir.mkdir(parents=True, exist_ok=True)
        areas = "+".join(ALL_COMUNI_IDS)
        url = f"{BASE}/IT1,{FLOW_MUNI_2D},1.0/A.{areas}...TOTAL?startPeriod=2012"
        r = requests.get(url, headers=UA, timeout=600)
        r.raise_for_status()
        out.write_bytes(r.content)
        record("istat_asia_muni_2d", source="istat-sdmx", dataset=FLOW_MUNI_2D, url=url)
    fetch_size_classes(force=force, raw_dir=raw_dir)
    fetch_belt_totals(force=force, raw_dir=raw_dir)
    fetch_areas(force=force, raw_dir=raw_dir)
    return out


def fetch_belt_totals(force: bool = False, raw_dir: Path = RAW_DIR) -> Path:
    """Download ASIA total economy (`0010`) local units (`LU`) and persons employed (`LUEMPDAA`) for the 16 Cesena belt municipalities (`2012-2024`)."""
    out = belt_raw_path(raw_dir)
    if out.exists() and not force:
        return out
    raw_dir.mkdir(parents=True, exist_ok=True)
    areas = "+".join(BELT_COMUNI_IDS)
    url = f"{BASE}/IT1,{FLOW_MUNI_2D},1.0/A.{areas}.LU+LUEMPDAA.0010.TOTAL?startPeriod=2012"
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)
    record("istat_asia_belt_total", source="istat-sdmx", dataset=f"{FLOW_MUNI_2D}_belt", url=url)
    return out


def load_belt_totals(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load `comune_id, comune, belt_group, year, local_units, employees, avg_firm_size` (`2012-2024`) for the 16 Cesena belt municipalities."""
    p = belt_raw_path(raw_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found; run `uv run python -m cifresignificative.core.sources.istat_asia fetch`"
        )
    d = pd.read_csv(
        p,
        low_memory=False,
        usecols=["REF_AREA", "DATA_TYPE", "TIME_PERIOD", "OBS_VALUE"],
    )
    d["comune_id"] = d["REF_AREA"].map(format_comune_id)
    w = (
        d.pivot_table(
            index=["comune_id", "TIME_PERIOD"],
            columns="DATA_TYPE",
            values="OBS_VALUE",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"TIME_PERIOD": "year", "LU": "local_units", "LUEMPDAA": "employees"})
    )
    w.columns.name = None
    w["comune"] = w["comune_id"].map(lambda c: BELT_BY_ID[c].name if c in BELT_BY_ID else c)
    w["belt_group"] = w["comune_id"].map(lambda c: BELT_BY_ID[c].belt_group if c in BELT_BY_ID else "belt")
    w["avg_firm_size"] = w["employees"] / w["local_units"].replace(0, float("nan"))
    return w[["comune_id", "comune", "belt_group", "year", "local_units", "employees", "avg_firm_size"]].sort_values(
        ["comune_id", "year"], ignore_index=True
    )



def fetch_size_classes(force: bool = False, raw_dir: Path = RAW_DIR) -> Path:
    """Download ASIA 2-digit ATECO x employment size class (`W0_9`, `W10_49`, `W50_249`, `W_GE250`, `TOTAL`) for the 13 ER municipalities."""
    out = size_raw_path(raw_dir)
    if out.exists() and not force:
        return out
    raw_dir.mkdir(parents=True, exist_ok=True)
    areas = "+".join(ALL_COMUNI_IDS)
    classes = "+".join(SIZE_CLASS_ORDER)
    url = f"{BASE}/IT1,{FLOW_MUNI_2D},1.0/A.{areas}...{classes}?startPeriod=2012"
    r = requests.get(url, headers=UA, timeout=600)
    r.raise_for_status()
    out.write_bytes(r.content)
    record("istat_asia_muni_size_2d", source="istat-sdmx", dataset=f"{FLOW_MUNI_2D}_size", url=url)
    return out


def areas_raw_path(raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / "asia_areas_2d.csv"


def fetch_areas(force: bool = False, raw_dir: Path = RAW_DIR) -> Path:
    """Download ASIA 2-digit ATECO local units and persons employed for the 9 Emilia-Romagna provinces, the region and Italy (`2012-2024`).

    The flow answers only to NUTS 2010 codes (`ITD51`-`ITD59`, `ITD5`): `ITH58` gives HTTP 404. Written to its own file, so the municipal file is kept.
    """
    out = areas_raw_path(raw_dir)
    if out.exists() and not force:
        return out
    raw_dir.mkdir(parents=True, exist_ok=True)
    areas = "+".join(AREA_CODES)
    url = f"{BASE}/IT1,{FLOW_MUNI_2D},1.0/A.{areas}...TOTAL?startPeriod=2012"
    r = requests.get(url, headers=UA, timeout=600)
    r.raise_for_status()
    out.write_bytes(r.content)
    record("istat_asia_areas_2d", source="istat-sdmx", dataset=f"{FLOW_MUNI_2D}_areas", url=url)
    return out


def load_areas(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load `area, nuts, year, ateco_2d, local_units, employees` for the Emilia-Romagna provinces, the region (`ITH5`) and Italy (`IT`); `ateco_2d` has `'TOTAL'` for the whole economy."""
    p = areas_raw_path(raw_dir)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `uv run python -m cifresignificative.core.sources.istat_asia fetch`")
    # dtype=str: '0010' (total) must not be read as 10 (food), which a file with only numeric divisions would do
    d = pd.read_csv(p, low_memory=False, dtype={"ECON_ACTIVITY_NACE_2007": str}, usecols=["REF_AREA", "DATA_TYPE", "ECON_ACTIVITY_NACE_2007", "PERS_EMPL_SIZE_CLASS", "TIME_PERIOD", "OBS_VALUE"])
    d = d[d["PERS_EMPL_SIZE_CLASS"].astype(str) == "TOTAL"].copy()
    d["ateco_raw"] = d["ECON_ACTIVITY_NACE_2007"].str.strip()
    is_tot = d["ateco_raw"].isin(["0010", "TOTAL"])
    d = d[d["ateco_raw"].str.fullmatch(r"\d{2}") | is_tot].copy()
    d["ateco_2d"] = d["ateco_raw"].where(~is_tot, "TOTAL")
    w = (
        d.pivot_table(index=["REF_AREA", "TIME_PERIOD", "ateco_2d"], columns="DATA_TYPE", values="OBS_VALUE", aggfunc="first")
        .reset_index()
        .rename(columns={"REF_AREA": "area", "TIME_PERIOD": "year", "LU": "local_units", "LUEMPDAA": "employees"})
    )
    w.columns.name = None
    for col in ("local_units", "employees"):
        if col not in w:
            w[col] = float("nan")
    w["nuts"] = harmonise_nuts3(w["area"])
    return w[["area", "nuts", "year", "ateco_2d", "local_units", "employees"]].sort_values(["area", "year", "ateco_2d"], ignore_index=True)


def load(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load tidy `comune_id, year, ateco_2d, macro_cluster, local_units, employees, avg_firm_size` table (2-digit divisions + `'TOTAL'`)."""
    p = raw_path(raw_dir)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `uv run python -m cifresignificative.core.sources.istat_asia fetch`")
    d = pd.read_csv(p, low_memory=False, usecols=["REF_AREA", "DATA_TYPE", "ECON_ACTIVITY_NACE_2007", "PERS_EMPL_SIZE_CLASS", "TIME_PERIOD", "OBS_VALUE"])
    d = d[d["PERS_EMPL_SIZE_CLASS"].astype(str) == "TOTAL"].copy()
    d["comune_id"] = d["REF_AREA"].map(format_comune_id)
    d["ateco_raw"] = d["ECON_ACTIVITY_NACE_2007"].astype(str).str.strip()
    # Keep 2-digit divisions ('01'..'96') and the overall total ('0010' or 'TOTAL')
    is_2d = d["ateco_raw"].str.fullmatch(r"\d{2}")
    is_tot = d["ateco_raw"].isin(["0010", "TOTAL"])
    d = d[is_2d | is_tot].copy()
    d["ateco_2d"] = d["ateco_raw"].where(~is_tot, "TOTAL")
    w = (
        d.pivot_table(
            index=["comune_id", "TIME_PERIOD", "ateco_2d"],
            columns="DATA_TYPE",
            values="OBS_VALUE",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"TIME_PERIOD": "year", "LU": "local_units", "LUEMPDAA": "employees"})
    )
    w.columns.name = None
    for col in ("local_units", "employees"):
        if col not in w:
            w[col] = float("nan")
    # If 'TOTAL' is missing or partial in any municipality-year, also ensure we can sum the 2-digit divisions
    w["macro_cluster"] = w["ateco_2d"].map(lambda c: "TOTAL" if c == "TOTAL" else classify_ateco2d(str(c)))
    w["specialization"] = w["macro_cluster"]
    w["specialization_label"] = w["specialization"].map(lambda s: SPECIALIZATION_LABELS.get(str(s), str(s)))
    w["ateco_label"] = w["ateco_2d"].map(lambda c: ATECO_2D_LABELS.get(str(c), f"ATECO {c}"))
    w["is_total"] = w["ateco_2d"] == "TOTAL"
    w["avg_firm_size"] = w["employees"] / w["local_units"].replace(0, float("nan"))
    cols = [
        "comune_id", "year", "ateco_2d", "ateco_label", "macro_cluster",
        "specialization", "specialization_label", "is_total", "local_units", "employees", "avg_firm_size",
    ]
    return w[cols].sort_values(["comune_id", "year", "ateco_2d"], ignore_index=True)


load_all = load


def load_size_classes(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load tidy `comune_id, year, ateco_2d, specialization, size_class, size_label, is_total, local_units, employees, avg_firm_size` table."""
    p = size_raw_path(raw_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found; run `uv run python -m cifresignificative.core.sources.istat_asia fetch`"
        )
    d = pd.read_csv(
        p,
        low_memory=False,
        usecols=["REF_AREA", "DATA_TYPE", "ECON_ACTIVITY_NACE_2007", "PERS_EMPL_SIZE_CLASS", "TIME_PERIOD", "OBS_VALUE"],
    )
    d["size_class"] = d["PERS_EMPL_SIZE_CLASS"].astype(str).str.strip()
    d = d[d["size_class"].isin(SIZE_CLASS_ORDER)].copy()
    d["comune_id"] = d["REF_AREA"].map(format_comune_id)
    d["ateco_raw"] = d["ECON_ACTIVITY_NACE_2007"].astype(str).str.strip()
    is_2d = d["ateco_raw"].str.fullmatch(r"\d{2}")
    is_tot = d["ateco_raw"].isin(["0010", "TOTAL"])
    d = d[is_2d | is_tot].copy()
    d["ateco_2d"] = d["ateco_raw"].where(~is_tot, "TOTAL")
    w = (
        d.pivot_table(
            index=["comune_id", "TIME_PERIOD", "ateco_2d", "size_class"],
            columns="DATA_TYPE",
            values="OBS_VALUE",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"TIME_PERIOD": "year", "LU": "local_units", "LUEMPDAA": "employees"})
    )
    w.columns.name = None
    for col in ("local_units", "employees"):
        if col not in w:
            w[col] = float("nan")
        w[col] = w[col].fillna(0.0)
    w["specialization"] = w["ateco_2d"].map(lambda c: "TOTAL" if c == "TOTAL" else classify_ateco2d(str(c)))
    w["specialization_label"] = w["specialization"].map(lambda s: SPECIALIZATION_LABELS.get(str(s), str(s)))
    w["ateco_label"] = w["ateco_2d"].map(lambda c: ATECO_2D_LABELS.get(str(c), f"ATECO {c}"))
    w["size_label"] = w["size_class"].map(lambda sc: SIZE_CLASS_LABELS.get(str(sc), str(sc)))
    w["is_total"] = w["ateco_2d"] == "TOTAL"
    w["avg_firm_size"] = w["employees"] / w["local_units"].replace(0, float("nan"))
    cols = [
        "comune_id", "year", "ateco_2d", "ateco_label", "specialization", "specialization_label",
        "size_class", "size_label", "is_total", "local_units", "employees", "avg_firm_size",
    ]
    return w[cols].sort_values(["comune_id", "year", "ateco_2d", "size_class"], ignore_index=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        print(fetch("--force" in sys.argv))
    df = load()
    print(f"ASIA loaded: {len(df)} rows, years {df.year.min()}-{df.year.max()}, comuni {df.comune_id.nunique()}, divisions {df.ateco_2d.nunique()}")
