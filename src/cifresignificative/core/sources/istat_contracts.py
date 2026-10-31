"""Istat collective-bargaining data (esploradati SDMX): contractual hourly wages by agreement group, and the "bargaining tension" indicators on the waiting time for contract renewals.

- Wages (`155_318_DF_DCSC_RETRCONTR1C_4`, base December 2021 = 100): monthly index of contractual wages for 113 agreement groups (`MAIN_AGREEMENT_GROUP`, code list `CL_TIPO_CONTRATTO`, nested: total economy > private sector > industry > manufacturing > ...), 2005 to now.
  Contractual wages follow the collective agreements only (no wage drift, no bonuses outside the agreements), so the index moves in steps when an increment of an agreement takes effect.
- Bargaining tension (`149_319_DF_DCSC_RETRCONTR1T_1`, base 2015, 2015-2023, and `..._4`, base 2021, 2021-2026): share of employees with an expired contract (`DIP_ATT`), mean months of waiting of those employees (`MES_ATT`) and of all employees (`MES_TOT`),
  for six aggregates only (agriculture, industry, private services, private sector, public administration, total economy). The two bases differ by a few tenths in their overlap (2021-2023) and are **not spliced**.
Raw files go to `data/raw/istat/contracts/` (gitignored). Descriptive use only: the contracts' renewal dates and the number of employees per agreement are not in these flows.
"""
import re
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

BASE = "https://esploradati.istat.it/SDMXWS/rest"
RAW_DIR = Path("data/raw/istat/contracts")
WAGES = "155_318_DF_DCSC_RETRCONTR1C_4"
TENSION = {"2015": "149_319_DF_DCSC_RETRCONTR1T_1", "2021": "149_319_DF_DCSC_RETRCONTR1T_4"}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
SECTORS = {"Z0010": "agriculture", "Z0060": "industry", "Z1240": "private services", "Z2520": "private sector", "Z2540": "public administration", "Z3620": "total economy"}
EMPLOYEES_ALL = 10  # PROF_STATUS_EMP: all employees (23 and 6 are sub-groups)


def path(flow: str) -> Path:
    return RAW_DIR / f"{flow}.csv"


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for flow in (WAGES, *TENSION.values()):
        if path(flow).exists() and not force:
            continue
        url = f"{BASE}/data/IT1,{flow},1.0/all"
        resp = requests.get(url, headers={**UA, "Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}, timeout=900)
        resp.raise_for_status()
        path(flow).write_bytes(resp.content)
        record(f"istat_contracts_{flow}", source="istat-sdmx", dataset=flow, url=url)
    if force or not (RAW_DIR / "groups.csv").exists():
        url = f"{BASE}/codelist/IT1/CL_TIPO_CONTRATTO/latest"
        resp = requests.get(url, headers={**UA, "Accept": "application/xml"}, timeout=300)
        resp.raise_for_status()
        rows = []
        for m in re.finditer(r'<structure:Code id="([^"]+)"[^>]*>(.*?)</structure:Code>', resp.text, re.DOTALL):
            body = m.group(2)
            it, en = (re.search(rf'<common:Name xml:lang="{lang}">([^<]*)</common:Name>', body) for lang in ("it", "en"))
            par = re.search(r'<structure:Parent>\s*<Ref id="([^"]+)"', body)
            rows.append({"code": m.group(1), "name_it": it.group(1) if it else None, "name_en": en.group(1) if en else None, "parent": par.group(1) if par else None})
        pd.DataFrame(rows).to_csv(RAW_DIR / "groups.csv", index=False)
        record("istat_contracts_groups", source="istat-sdmx", dataset="CL_TIPO_CONTRATTO", url=url)
    return RAW_DIR


def load_groups(wage_groups: set[str] | None = None) -> pd.DataFrame:
    """The agreement groups present in the wage file with their parent, `leaf` (no group below it) and `branch` (`private`, `public` or `aggregate`: the total and the private sector itself)."""
    g = pd.read_csv(RAW_DIR / "groups.csv")
    if wage_groups is None:
        wage_groups = set(pd.read_csv(path(WAGES), usecols=["MAIN_AGREEMENT_GROUP"]).MAIN_AGREEMENT_GROUP)
    g = g[g.code.isin(wage_groups)].reset_index(drop=True)
    parent = dict(zip(g.code, g.parent, strict=True))
    g["leaf"] = ~g.code.isin(set(g.parent.dropna()))

    def branch(code: str) -> str:
        seen, c = [], code
        while isinstance(c, str) and c not in seen:
            seen.append(c)
            c = parent.get(c)
        return "private" if "Z2520" in seen and code not in ("Z2520", "Z3620") else ("aggregate" if code in ("Z2520", "Z3620") else "public")
    g["branch"] = g.code.map(branch)
    return g


def load_wages(p: Path | None = None) -> pd.DataFrame:
    """Contractual hourly wage index (December 2021 = 100) by month (period-end timestamps) and agreement group, all employees; `provisional` marks the latest, flagged values."""
    d = pd.read_csv(p or path(WAGES), low_memory=False)
    d = d[(d.DATA_TYPE == "WAGE_H_2021") & (d.FREQ == "M") & (d.PROF_STATUS_EMP == EMPLOYEES_ALL)]
    out = pd.DataFrame({"group": d.MAIN_AGREEMENT_GROUP, "month": pd.PeriodIndex(d.TIME_PERIOD, freq="M").to_timestamp("M"), "index": d.OBS_VALUE, "provisional": d.OBS_STATUS.eq("p")})
    return out.sort_values(["group", "month"], ignore_index=True)


def load_tension() -> pd.DataFrame:
    """Waiting-time indicators by month and sector for the two bases (`base` 2015 or 2021): `employees_expired_pct` (DIP_ATT), `months_expired` (MES_ATT, employees with an expired contract), `months_all` (MES_TOT, all employees)."""
    names = {"DIP_ATT": "employees_expired_pct", "MES_ATT": "months_expired", "MES_TOT": "months_all"}
    frames = []
    for base, flow in TENSION.items():
        d = pd.read_csv(path(flow), low_memory=False)
        d = d[d.FREQ == "M"]
        kind = d.DATA_TYPE.str.replace(r"_(2021|2)$", "", regex=True)
        frames.append(pd.DataFrame({"base": base, "sector": d.MAIN_AGREEMENT_GROUP.map(SECTORS), "month": pd.PeriodIndex(d.TIME_PERIOD, freq="M").to_timestamp("M"), "indicator": kind.map(names), "value": d.OBS_VALUE}))
    return pd.concat(frames, ignore_index=True).dropna(subset=["indicator"]).sort_values(["base", "indicator", "sector", "month"], ignore_index=True)
