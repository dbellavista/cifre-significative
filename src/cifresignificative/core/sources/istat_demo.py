"""ISTAT Esploradati SDMX: Municipal demography (resident population by single-year age, old-age index, dependency ratio, and demographic balance).

Combines three official ISTAT municipal SDMX dataflows for the 13 Emilia-Romagna peer municipalities (`core.geo.ER_COMUNI`):
1. `164_164_DF_DCIS_RICPOPRES2011_11`: Intercensal reconstructed population by single-year age and demographic balance for Emilia-Romagna municipalities (2008-2019).
2. `22_289_DF_DCIS_POPRES1_11`: Resident population on 1 January by single-year age for Emilia-Romagna municipalities (2019-2026).
3. `22_315`: Municipal demographic balance (`LBIRTH`, `DEATH`, `REGM` inflows, `DEREGM` outflows, `INTNETMIGR_REGOF`, `INTERTNMIG_REGOF`, 2019-2025).

In the overlap year 2019, `22_289_DF_DCIS_POPRES1_11` and `164_164_DF_DCIS_RICPOPRES2011_11` match exactly on 1 January population (`JAN`);
we prefer the current census series from 2019 onwards.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.geo import (
    ALL_COMUNI_IDS,
    BELT_BY_ID,
    BELT_COMUNI_IDS,
    COMUNI_BY_ID,
    format_comune_id,
)
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/istat_comuni")
BASE = "https://esploradati.istat.it/SDMXWS/rest/data"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36", "Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}

FLOW_HIST = "164_164_DF_DCIS_RICPOPRES2011_11"  # 2008-2019 age structure & balance
FLOW_POP = "22_289_DF_DCIS_POPRES1_11"          # 2019-2026 age structure
FLOW_BAL = "22_315"                             # 2019-2025 demographic balance


BALANCE_RENAME = {"LBIRTH": "births", "DEATH": "deaths", "REGM": "inflow_total", "DEREGM": "outflow_total", "REGOTHC": "inflow_abroad", "DEREGOTHC": "outflow_abroad", "STAT_ADJUST": "stat_adjust"}


def _age_num(code: str) -> int | None:
    if code == "Y_GE100":
        return 100
    m = re.fullmatch(r"Y(\d+)", str(code).strip())
    return int(m.group(1)) if m else None


def fetch(force: bool = False, raw_dir: Path = RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    areas = "+".join(ALL_COMUNI_IDS)
    jobs = [
        ("demo_hist_2008_2019.csv", FLOW_HIST, f"A.{areas}...9.TOTAL?startPeriod=2008"),
        ("demo_pop_2019_2026.csv", FLOW_POP, f"A.{areas}.JAN.9..99?startPeriod=2019"),
        ("demo_bal_2019_2025.csv", FLOW_BAL, f"A.{areas}..9?startPeriod=2019"),
    ]
    for fname, flow, key in jobs:
        p = raw_dir / fname
        if p.exists() and not force:
            continue
        url = f"{BASE}/IT1,{flow},1.0/{key}"
        r = requests.get(url, headers=UA, timeout=600)
        r.raise_for_status()
        p.write_bytes(r.content)
        record(f"istat_{fname[:-4]}", source="istat-sdmx", dataset=flow, url=url)
    fetch_belt_balance(force=force, raw_dir=raw_dir)
    return raw_dir


def fetch_belt_balance(force: bool = False, raw_dir: Path = RAW_DIR) -> Path:
    """Download total population (`JAN`, `AGE=TOTAL`) and demographic balance (`LBIRTH`, `DEATH`, `REGM`, `DEREGM`, `INTNETMIGR_REGOF`, `INTERTNMIG_REGOF`) for the 16 Cesena belt municipalities (`2008-2025`)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    areas = "+".join(BELT_COMUNI_IDS)
    jobs = [
        ("demo_belt_hist_2008_2019.csv", FLOW_HIST, f"A.{areas}.JAN+LBIRTH+DEATH+REGM+DEREGM+REGOTHC+DEREGOTHC.TOTAL.9.TOTAL?startPeriod=2008"),
        ("demo_belt_pop_2019_2026.csv", FLOW_POP, f"A.{areas}.JAN.9.TOTAL.99?startPeriod=2019"),
        ("demo_belt_bal_2019_2025.csv", FLOW_BAL, f"A.{areas}.LBIRTH+DEATH+REGM+DEREGM+REGOTHC+DEREGOTHC+INTNETMIGR_REGOF+INTERTNMIG_REGOF+STAT_ADJUST.9?startPeriod=2019"),
    ]
    for fname, flow, key in jobs:
        p = raw_dir / fname
        if p.exists() and not force:
            continue
        url = f"{BASE}/IT1,{flow},1.0/{key}"
        r = requests.get(url, headers=UA, timeout=120)
        r.raise_for_status()
        p.write_bytes(r.content)
        record(f"istat_{fname[:-4]}", source="istat-sdmx", dataset=flow, url=url)
    return raw_dir


def load_belt_balance(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load `2008-2025` total resident population (`pop_total`), natural balance, and net migration (`net_migration`, `INTNETMIGR_REGOF`, `INTERTNMIG_REGOF`) for the 16 Cesena belt municipalities."""
    p_hist = raw_dir / "demo_belt_hist_2008_2019.csv"
    p_pop = raw_dir / "demo_belt_pop_2019_2026.csv"
    p_bal = raw_dir / "demo_belt_bal_2019_2025.csv"
    for p in (p_hist, p_pop, p_bal):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing; run `uv run python -m cifresignificative.core.sources.istat_demo fetch`"
            )
    hist = pd.read_csv(p_hist, low_memory=False, usecols=["REF_AREA", "DATA_TYPE", "TIME_PERIOD", "OBS_VALUE"])
    pop19 = pd.read_csv(p_pop, low_memory=False, usecols=["REF_AREA", "DATA_TYPE", "TIME_PERIOD", "OBS_VALUE"])
    bal19 = pd.read_csv(p_bal, low_memory=False, usecols=["REF_AREA", "DATA_TYPE", "TIME_PERIOD", "OBS_VALUE"])

    pop_h = hist[(hist["DATA_TYPE"] == "JAN") & (hist["TIME_PERIOD"].astype(int) < 2019)].copy()
    pop_c = pop19[pop19["DATA_TYPE"] == "JAN"].copy()
    pop = pd.concat([pop_h, pop_c], ignore_index=True)
    pop["comune_id"] = pop["REF_AREA"].map(format_comune_id)
    pop["year"] = pop["TIME_PERIOD"].astype(int)
    pop = pop.rename(columns={"OBS_VALUE": "pop_total"})[["comune_id", "year", "pop_total"]]

    bh = hist[(hist["DATA_TYPE"] != "JAN") & (hist["TIME_PERIOD"].astype(int) < 2019)].copy()
    bh["comune_id"] = bh["REF_AREA"].map(format_comune_id)
    bh["year"] = bh["TIME_PERIOD"].astype(int)
    bc = bal19.copy()
    bc["comune_id"] = bc["REF_AREA"].map(format_comune_id)
    bc["year"] = bc["TIME_PERIOD"].astype(int)
    b_all = pd.concat(
        [bh[["comune_id", "year", "DATA_TYPE", "OBS_VALUE"]], bc[["comune_id", "year", "DATA_TYPE", "OBS_VALUE"]]],
        ignore_index=True,
    )
    bw = b_all.pivot_table(index=["comune_id", "year"], columns="DATA_TYPE", values="OBS_VALUE", aggfunc="first").reset_index()
    bw.columns.name = None
    bw = bw.rename(columns=BALANCE_RENAME)

    out = pop.merge(bw, on=["comune_id", "year"], how="left")
    out = add_balance_components(out)
    out["comune"] = out["comune_id"].map(lambda c: BELT_BY_ID[c].name if c in BELT_BY_ID else c)
    out["belt_group"] = out["comune_id"].map(lambda c: BELT_BY_ID[c].belt_group if c in BELT_BY_ID else "belt")
    out["natural_balance_per_1000"] = out["natural_balance"] / out["pop_total"] * 1000.0
    out["net_migr_rate_per_1k"] = out["net_migration"] / out["pop_total"] * 1000.0
    return out



def add_balance_components(out: pd.DataFrame) -> pd.DataFrame:
    """Split the yearly population change into natural balance, internal and international migration, and a registry adjustment.

    ISTAT `REGM`/`DEREGM` are registrations from and to other Italian municipalities (their difference equals `INTNETMIGR_REGOF`) and `REGOTHC`/`DEREGOTHC` those from and to abroad (`INTERTNMIG_REGOF`).
    The adjustment is the remainder of the change to next January: census and registry revisions, published as `STAT_ADJUST` for 2019-2024 and about zero in the reconstructed 2008-2018 series. `net_migration` is internal + international.
    """
    out = out.sort_values(["comune_id", "year"], ignore_index=True)
    out["natural_balance"] = out["births"] - out["deaths"]
    out["net_migration_internal"] = out["inflow_total"] - out["outflow_total"]
    out["net_migration_international"] = out["inflow_abroad"] - out["outflow_abroad"]
    out["net_migration"] = out["net_migration_internal"] + out["net_migration_international"]
    next_pop = out.groupby("comune_id")["pop_total"].shift(-1)
    out["pop_change_next"] = next_pop - out["pop_total"]
    out["registry_adjustment"] = out["pop_change_next"] - out["natural_balance"] - out["net_migration"]
    return out


def _parse_age_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate single-year `JAN` population rows into cohorts (`pop_total`, `pop_0_14`, `pop_15_64`, `pop_25_44`, `pop_65_plus`)."""
    d = df[df["DATA_TYPE"].astype(str) == "JAN"].copy()
    d["comune_id"] = d["REF_AREA"].map(format_comune_id)
    d["year"] = d["TIME_PERIOD"].astype(int)
    tot = d[d["AGE"].astype(str) == "TOTAL"].set_index(["comune_id", "year"])["OBS_VALUE"].rename("pop_total")
    s = d.assign(age_n=d["AGE"].map(_age_num)).dropna(subset=["age_n"])
    s["age_n"] = s["age_n"].astype(int)

    def cohort_sum(lo: int, hi: int, name: str) -> pd.Series:
        return s[s["age_n"].between(lo, hi)].groupby(["comune_id", "year"])["OBS_VALUE"].sum().rename(name)

    out = pd.concat(
        [
            tot,
            cohort_sum(0, 14, "pop_0_14"),
            cohort_sum(15, 64, "pop_15_64"),
            cohort_sum(25, 44, "pop_25_44"),
            cohort_sum(65, 100, "pop_65_plus"),
        ],
        axis=1,
    ).reset_index()
    return out


def load(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load complete `2008-2025` municipal demographic table with age cohorts, aging/dependency indices, migration flows, and cohort 25-44 net retention proxy."""
    p_hist = raw_dir / "demo_hist_2008_2019.csv"
    p_pop = raw_dir / "demo_pop_2019_2026.csv"
    p_bal = raw_dir / "demo_bal_2019_2025.csv"
    for p in (p_hist, p_pop, p_bal):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run `uv run python -m cifresignificative.core.sources.istat_demo fetch`")

    hist = pd.read_csv(p_hist, low_memory=False, usecols=["REF_AREA", "DATA_TYPE", "AGE", "TIME_PERIOD", "OBS_VALUE"])
    pop19 = pd.read_csv(p_pop, low_memory=False, usecols=["REF_AREA", "DATA_TYPE", "AGE", "TIME_PERIOD", "OBS_VALUE"])
    bal19 = pd.read_csv(p_bal, low_memory=False, usecols=["REF_AREA", "DATA_TYPE", "TIME_PERIOD", "OBS_VALUE"])

    # 1. Age structure: 2008-2018 from hist, 2019+ from pop19
    age_h = _parse_age_structure(hist[hist["TIME_PERIOD"].astype(int) < 2019])
    age_c = _parse_age_structure(pop19)
    age = pd.concat([age_h, age_c], ignore_index=True).sort_values(["comune_id", "year"], ignore_index=True)

    # Also compute exact 1-year cohort net migration for 25-44: Pop(25..44, t+1) - Pop(24..43, t)
    # Note: neglecting mortality at ages 24-43 (<0.05% p.a.), the cohort change between Jan 1 year t and Jan 1 year t+1 equals net migration of the 25-44 working-age cohort during year t!
    raw_all_jan = pd.concat(
        [
            hist[(hist["DATA_TYPE"] == "JAN") & (hist["TIME_PERIOD"].astype(int) < 2019)],
            pop19[pop19["DATA_TYPE"] == "JAN"],
        ],
        ignore_index=True,
    )
    raw_all_jan["comune_id"] = raw_all_jan["REF_AREA"].map(format_comune_id)
    raw_all_jan["year"] = raw_all_jan["TIME_PERIOD"].astype(int)
    raw_all_jan["age_n"] = raw_all_jan["AGE"].map(_age_num)
    s_jan = raw_all_jan.dropna(subset=["age_n"]).copy()
    s_jan["age_n"] = s_jan["age_n"].astype(int)
    p_24_43_t = s_jan[s_jan["age_n"].between(24, 43)].groupby(["comune_id", "year"])["OBS_VALUE"].sum()
    p_25_44_t1 = s_jan[s_jan["age_n"].between(25, 44)].groupby(["comune_id", "year"])["OBS_VALUE"].sum()
    # Align t+1 back to year t
    p_25_44_next = p_25_44_t1.rename_axis(["comune_id", "year"]).reset_index()
    p_25_44_next["year"] = p_25_44_next["year"] - 1
    cohort_mig = (p_25_44_next.set_index(["comune_id", "year"])["OBS_VALUE"] - p_24_43_t).rename("net_cohort_25_44")

    # 2. Demographic balance (births, deaths, REGM inflows, DEREGM outflows)
    bh = hist[(hist["AGE"] == "TOTAL") & (hist["TIME_PERIOD"].astype(int) < 2019) & hist["DATA_TYPE"].isin(["LBIRTH", "DEATH", "REGM", "DEREGM", "REGOTHC", "DEREGOTHC"])].copy()
    bh["comune_id"] = bh["REF_AREA"].map(format_comune_id)
    bh["year"] = bh["TIME_PERIOD"].astype(int)
    bc = bal19[bal19["DATA_TYPE"].isin(["LBIRTH", "DEATH", "REGM", "DEREGM", "REGOTHC", "DEREGOTHC", "INTNETMIGR_REGOF", "INTERTNMIG_REGOF", "STAT_ADJUST"])].copy()
    bc["comune_id"] = bc["REF_AREA"].map(format_comune_id)
    bc["year"] = bc["TIME_PERIOD"].astype(int)
    b_all = pd.concat([bh[["comune_id", "year", "DATA_TYPE", "OBS_VALUE"]], bc[["comune_id", "year", "DATA_TYPE", "OBS_VALUE"]]], ignore_index=True)
    bw = b_all.pivot_table(index=["comune_id", "year"], columns="DATA_TYPE", values="OBS_VALUE", aggfunc="first").reset_index()
    bw.columns.name = None
    bw = bw.rename(columns=BALANCE_RENAME)

    out = age.merge(bw, on=["comune_id", "year"], how="left").merge(cohort_mig.reset_index(), on=["comune_id", "year"], how="left")
    out["comune"] = out["comune_id"].map(lambda c: COMUNI_BY_ID[c].name if c in COMUNI_BY_ID else c)
    out["old_age_index"] = out["pop_65_plus"] / out["pop_0_14"] * 100.0
    out["aging_index"] = out["old_age_index"]
    out["dependency_ratio"] = (out["pop_0_14"] + out["pop_65_plus"]) / out["pop_15_64"] * 100.0
    out["share_25_44_pct"] = out["pop_25_44"] / out["pop_total"] * 100.0
    out = add_balance_components(out)
    out["natural_balance_per_1000"] = out["natural_balance"] / out["pop_total"] * 1000.0
    out["net_migration_total"] = out["net_migration"]
    out["net_migration_per_1000"] = out["net_migration"] / out["pop_total"] * 1000.0
    out["net_migr_rate_per_1k"] = out["net_migration_per_1000"]
    out["net_migration_25_44_per_1000"] = out["net_cohort_25_44"] / out["pop_25_44"] * 1000.0
    out["net_cohort_25_44_per_1k"] = out["net_migration_25_44_per_1000"]
    return out.sort_values(["comune_id", "year"], ignore_index=True)


load_all = load


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        print(fetch("--force" in sys.argv))
    df = load()
    print(f"Demography loaded: {len(df)} rows, years {df.year.min()}-{df.year.max()}, comuni {df.comune_id.nunique()}")
