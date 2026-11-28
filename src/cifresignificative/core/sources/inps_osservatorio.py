"""INPS Osservatorio sui lavoratori dipendenti (settore privato non agricolo): workers, pay and paid days by province of work.

Source: https://servizi2.inps.it/servizi/osservatoristatistici/ ("Lavoratori dipendenti per attività economica e provincia di lavoro"). The navigable tables have no API; the CSV exports are saved by hand into `data/raw/inps_osservatorio/inps_*.csv` (one file per block, because the site splits the years by ATECO classification: 2008-2013 ATECO 2002, 2014 onwards ATECO 2007; totals by province are not affected by that break). On 2026-11-24 INPS published the blocks "Anni 2019-2024" (ATECO 2007) and "Anno 2025" (ATECO 2025; totals by province are not affected), so the Osservatorio now covers 2008-2025 and 2022-2023 are no longer missing. The part-time cut (`No`/`Si`) exists for 2008-2018, 2024 and 2025 only: the 2019-2023 blocks have a single `-` value. The 2024 values of the old "Anno 2024" block equal those of the new block, which replaces it. The 2021 values were revised (Forlì-Cesena 122,340 workers against 122,085 in Open Data; Italy 16,274,677 against 16,226,252): the Osservatorio is used.
Second source, now only a cross-check (2014-2020 agree; 2021 is revised in the Osservatorio), INPS Open Data (dataflow `DFB_ST_DIP_ATECO07_PROV_ANAG_01`, "Lavoratori dipendenti per Province e Sezioni ateco 2007 - serie storica", 2014-2021, 103 provinces + Estero), saved as `data/raw/inps_osservatorio/opendata_*.csv` (not matched by `GLOB`). Long CSV, one row per territory x measure x ATECO section x year, no total row and no part-time cut: `load_totals()` sums the 18 sections. The site stops an export at 50,000 rows, so keep the three-measure export (44,892 rows; a bigger one with 70 provinces is truncated). Cells under the privacy threshold are `(*)` and empty (12, all in workers: Arezzo and Matera section T, Estero sections L and T); the sum then misses them (`masked_cells` > 0), which on 2014-2018 costs 1 worker in the Italy total of 2015 and 2018. 2014-2018 agree with the Osservatorio to the unit (checked by `check_overlap`).
Statistical unit: a worker with at least one paid day in the year, counted once, classified by the last job (province, working time) while pay and days sum all jobs. Public sector, agriculture and domestic work are out; paid days are not hours (horizontal part-time counts whole days).
Two layouts are read: totals (`Anno;Provincia;;lavoratori;retribuzione;giornate`) and the cut by `Presenza tempo parziale nell'anno` (`No`/`Si`/`Totale`; `Si` = at least one part-time relationship in the year, not only part-time). Cells carry HTML whitespace, the year and province appear only on the first row of a block, numbers use the dot as thousands separator.
`load()` returns a tidy frame: `year, province, part_time ("Totale" | "No" | "Si"), kind ("province" | "total" | "other"), workers, pay_eur, paid_days`. `kind="other"` is `Estero` / `Non ripartibili`.
"""
import csv
import io
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/inps_osservatorio")
GLOB = "inps_*.csv"
TOTAL = "Totale"
OTHER = {"Estero", "Non ripartibili"}
NAMES = {"Forli'-Cesena": "Forlì-Cesena"}
# Provinces whose territory changed between 2008 and 2024 in these tables: Monza (2009), Fermo (2009), Barletta-Andria-Trani (2009)
# were split from Milano, Ascoli Piceno, Bari and Foggia; Sud Sardegna (2016) came out of the Sardinian reorganisation. The four
# old Sardinian provinces are treated as changed too (conservative: the exports do not say how INPS mapped them).
REDRAWN = frozenset({"Milano", "Monza e della Brianza", "Ascoli Piceno", "Fermo", "Bari", "Foggia", "Barletta-Andria-Trani", "Cagliari", "Sassari", "Nuoro", "Oristano", "Sud Sardegna"})
REVISED_YEARS = frozenset({2021})  # Osservatorio revised on 2026-11-24; Open Data still has the old values
OPENDATA_GLOB = "opendata_*.csv"
OPENDATA_MEASURES = {"NUM_LAVORATORI": "workers", "RETR_ANNO": "pay_eur", "NUM_GG_RETR": "paid_days"}
# Open Data territory labels that differ from the Osservatorio ones (after `_unquote`).
OPENDATA_NAMES = {"Valle d'Aosta / Vallée d'Aoste": "Aosta", "Bolzano / Bozen": "Provincia Autonoma di Bolzano/Bozen", "Trento": "Provincia Autonoma di Trento", "Reggio di Calabria": "Reggio Calabria", "Reggio nell'Emilia": "Reggio Emilia"}
FIELDS = {"Numero lavoratori nell'anno": "workers", "Retribuzione nell'anno": "pay_eur", "Numero giornate retribuite nell'anno": "paid_days"}


def clean_name(raw: str) -> str:
    """Province label as printed by INPS (`Forli' -Cesena`, `Verbano -Cusio-Ossola`) -> one spelling."""
    s = re.sub(r"\s*-\s*", "-", re.sub(r"\s+", " ", raw).strip())
    return NAMES.get(s, s)


def _num(cell: str) -> float:
    return float(re.sub(r"\s+", "", cell).replace(".", ""))


def parse_export(text: str) -> pd.DataFrame:
    """One INPS CSV export -> tidy rows; raises if the header lacks a year, a province or one of the three measures."""
    rows = list(csv.reader(io.StringIO(re.sub(r"\s{3,}", " ", text)), delimiter=";"))
    head = next((i for i, r in enumerate(rows) if r and r[0].strip() == "Anno"), None)
    if head is None:
        raise ValueError("no header row starting with 'Anno'")
    cols = [c.strip() for c in rows[head]]
    pt_col = next((i for i, c in enumerate(cols) if c.startswith("Presenza tempo parziale")), None)
    idx = {name: cols.index(src) for src, name in FIELDS.items() if src in cols}
    if "Provincia" not in cols or len(idx) != 3:
        raise ValueError(f"unexpected header: {cols}")
    out, year, prov = [], None, None
    for r in rows[head + 1:]:
        if len(r) < len(cols) or not r[idx["workers"]].strip():
            continue
        year = int(r[0]) if r[0].strip() else year
        prov = clean_name(r[cols.index("Provincia")]) if r[cols.index("Provincia")].strip() else prov
        pt = r[pt_col].strip() if pt_col is not None else TOTAL
        out.append((year, prov, pt, *(_num(r[idx[k]]) for k in ("workers", "pay_eur", "paid_days"))))
    d = pd.DataFrame(out, columns=["year", "province", "part_time", "workers", "pay_eur", "paid_days"])
    d["kind"] = d.province.map(lambda p: "total" if p == TOTAL else "other" if p in OTHER else "province")
    return d


def load(raw: Path = RAW) -> pd.DataFrame:
    """All exports in `raw`; the totals appear in both layouts and must agree (checked), the first copy is kept."""
    files = sorted(raw.glob(GLOB))
    if not files:
        raise FileNotFoundError(f"no {GLOB} in {raw}: export the tables from the INPS Osservatorio (see the module docstring)")
    d = pd.concat([parse_export(f.read_text(encoding="utf-8")) for f in files], ignore_index=True)
    key = ["year", "province", "part_time"]
    dup = d[d.duplicated(key, keep=False)].sort_values(key)
    if len(dup) and (dup.groupby(key)[["workers", "pay_eur", "paid_days"]].nunique() > 1).any().any():
        raise ValueError("the same year/province/part_time differs between export files")
    return d.drop_duplicates(key).sort_values(key, ignore_index=True)


def _unquote(label: str) -> str:
    """Open Data text fields are wrapped in `'...'` and write an apostrophe as `"'`: `'L"'Aquila'` -> `L'Aquila`."""
    return label.replace("\"'", "'").strip("'")


def parse_opendata(text: str) -> pd.DataFrame:
    """One Open Data long CSV -> `year, province, section, workers, pay_eur, paid_days` (NaN for `(*)` masked cells).

    The file is not valid CSV (commas inside quoted ATECO descriptions, apostrophes written as `"'`), so each line is split by hand:
    territory code and name, indicator and the four trailing fields (`year, value, privacy flag, handling`) are plain; the section is the one-letter code after the indicator description."""
    lines = text.lstrip("\ufeff").splitlines()
    if not lines or not lines[0].startswith("TERRITORIO,"):
        raise ValueError("not an INPS Open Data export: header must start with TERRITORIO")
    out = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        p = ln.split(",")
        cut = ln.find("anno'")
        m = re.match(r",([A-U]),", ln[cut + 5:]) if cut >= 0 else None
        if m is None or p[2] not in OPENDATA_MEASURES or not p[-4].isdigit():
            raise ValueError(f"unexpected Open Data row: {ln[:120]}")
        val = p[-3].strip()
        name = _unquote(p[1])
        out.append((int(p[-4]), clean_name(OPENDATA_NAMES.get(name, name)), m.group(1), OPENDATA_MEASURES[p[2]], float(val) if re.fullmatch(r"\d+(\.\d+)?", val) else float("nan")))
    long = pd.DataFrame(out, columns=["year", "province", "section", "measure", "value"])
    if long.duplicated(["year", "province", "section", "measure"]).any():
        raise ValueError("duplicate territory/section/year/measure rows: more than one export in the file?")
    return long.pivot(index=["year", "province", "section"], columns="measure", values="value").reset_index().rename_axis(columns=None)


def opendata_totals(sections: pd.DataFrame) -> pd.DataFrame:
    """Sum the ATECO sections: `year, province, kind, workers, pay_eur, paid_days, masked_cells` (sections hidden by the privacy threshold)."""
    g = sections.groupby(["year", "province"])
    d = g[["workers", "pay_eur", "paid_days"]].sum().reset_index()
    d["masked_cells"] = g.workers.apply(lambda x: int(x.isna().sum())).to_numpy()
    d["kind"] = d.province.map(lambda p: "other" if p in OTHER else "province")
    return d[["year", "province", "kind", "workers", "pay_eur", "paid_days", "masked_cells"]]


def check_overlap(od: pd.DataFrame, osserv: pd.DataFrame) -> pd.DataFrame:
    """Rows where the Open Data totals differ from the Osservatorio ones on the shared years and provinces (empty = they agree).

    The only accepted gap is a province with masked cells (`masked_cells` > 0) whose Open Data workers are short and whose pay and days are equal: the hidden section is missing from the sum."""
    cols = ["workers", "pay_eur", "paid_days"]
    o = osserv[(osserv.part_time == TOTAL) & (osserv.kind != "total")].set_index(["year", "province"])[cols]
    a = od.set_index(["year", "province"])
    idx = a.index.intersection(o.index)
    gap = o.loc[idx] - a.loc[idx, cols]
    masked_short = (a.loc[idx, "masked_cells"] > 0) & (gap.workers > 0) & (gap.pay_eur == 0) & (gap.paid_days == 0)
    return gap[(gap != 0).any(axis=1) & ~masked_short]


def load_totals(raw: Path = RAW) -> pd.DataFrame:
    """Province totals: the Osservatorio years (`load()`, `part_time == "Totale"`) plus any Open Data year the Osservatorio lacks (none since 2026-11-24); column `source` says which.

    Raises if the two disagree on the shared years. Open Data has no part-time cut, so `province_panel()` and `hours.py` keep using `load()`. The Italy total is not published in Open Data: sum the provinces and `Estero` (`kind` in `province`, `other`)."""
    base = load(raw)
    base = base[(base.part_time == TOTAL) & (base.kind != "total")].assign(masked_cells=0, source="osservatorio")
    files = sorted(raw.glob(OPENDATA_GLOB))
    if not files:
        return base.drop(columns="part_time").reset_index(drop=True)
    od = pd.concat([opendata_totals(parse_opendata(f.read_text(encoding="utf-8-sig"))) for f in files], ignore_index=True).drop_duplicates(["year", "province"])
    shared = sorted(set(od.year) & set(base.year))
    names = {y: (set(od[od.year == y].province), set(base[base.year == y].province)) for y in shared}
    unmatched = {y: sorted(a ^ b) for y, (a, b) in names.items() if a != b}
    if unmatched:
        raise ValueError(f"province names differ between Open Data and Osservatorio: {unmatched}")
    bad = check_overlap(od[~od.year.isin(REVISED_YEARS)], base)
    if len(bad):
        raise ValueError(f"Open Data and Osservatorio totals differ:\n{bad.head()}")
    new = od[~od.year.isin(base.year)].assign(source="opendata")
    return pd.concat([base.drop(columns="part_time"), new], ignore_index=True).sort_values(["year", "province"], ignore_index=True)


WORKING_TIME_GLOB = "tipo_orario_*.csv"
PART_TIME_TYPES = ("Tempo parziale misto", "Tempo parziale orizzontale", "Tempo parziale verticale")


def parse_working_time(text: str) -> pd.DataFrame:
    """One monthly-observatory export (`Provincia x Tipologia orario di lavoro`, workers only) -> `year, province, working_time, workers`.

    `workers` is the mean over the twelve months: the last column of the 2019-2020 monthly tables (`Media`), the only column of the 2021-2025 tables downloaded with the filter `Mese: Media` (month by month the site blocks the table for statistical confidentiality from 2021). Not the same measure as the annual workers: a worker counts once per month present, so the mean is below the annual count (Forlì-Cesena 2019: 99,717 against 120,305)."""
    rows = list(csv.reader(io.StringIO(re.sub(r"\s{3,}", " ", text)), delimiter=";"))
    year = next(int(m.group(1)) for r in rows[:2] for c in r[:1] if (m := re.search(r"Anno (\d{4})", c)))
    head = next(i for i, r in enumerate(rows) if {"Provincia", "Tipologia orario di lavoro"} <= {c.strip() for c in r})
    cols = [c.strip() for c in rows[head]]
    ip, it = cols.index("Provincia"), cols.index("Tipologia orario di lavoro")  # the column order differs between exports; the first column carries its value down on blank cells
    out, prov, wtype = [], None, None
    for r in rows[head + 1:]:
        if len(r) < 4 or not r[-1].strip():
            continue
        prov = clean_name(r[ip]) if r[ip].strip() else prov
        wtype = r[it].strip() or wtype
        out.append((year, prov, wtype, _num(r[-1])))
    return pd.DataFrame(out, columns=["year", "province", "working_time", "workers"])


def load_working_time(raw: Path = RAW) -> pd.DataFrame:
    """Mean monthly workers by province and working time (`Totale`, `Tempo pieno`, three part-time types), 2019-2025, plus `part_time_share_pct`.

    Raises if the four types differ from `Totale` by more than 2 workers in any province-year (the monthly means are rounded). `kind` as in `load()` (`Non ripartibili` / `Estero` are `other`)."""
    files = sorted(raw.glob(WORKING_TIME_GLOB))
    if not files:
        raise FileNotFoundError(f"no {WORKING_TIME_GLOB} in {raw}")
    d = pd.concat([parse_working_time(f.read_text(encoding="utf-8")) for f in files], ignore_index=True)
    w = d.pivot(index=["year", "province"], columns="working_time", values="workers")
    parts = w[["Tempo pieno", *PART_TIME_TYPES]].sum(axis=1)
    if "Totale" not in w:
        w["Totale"] = float("nan")
    w["Totale"] = w["Totale"].fillna(parts)  # the older exports (2008-2011 at least) have no `Totale` row: the sum of the four types
    if not (parts - w["Totale"]).abs().le(2).all():
        raise ValueError("working-time types do not add up to Totale")
    w["part_time"] = w[list(PART_TIME_TYPES)].sum(axis=1)
    w["part_time_share_pct"] = w.part_time / w["Totale"] * 100
    w = w.reset_index().rename_axis(columns=None)
    w["kind"] = w.province.map(lambda p: "total" if p == TOTAL else "other" if p in OTHER else "province")
    return w
