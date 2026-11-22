"""INPS Osservatorio sui lavoratori dipendenti (settore privato non agricolo): workers, pay and paid days by province of work.

Source: https://servizi2.inps.it/servizi/osservatoristatistici/ ("Lavoratori dipendenti per attività economica e provincia di lavoro"). The navigable tables have no API; the CSV exports are saved by hand into `data/raw/inps_osservatorio/inps_*.csv` (one file per block, because the site splits the years by ATECO classification: 2008-2013 ATECO 2002, 2014 onwards ATECO 2007; totals by province are not affected by that break). Years 2019-2023 were not retrievable when the exports were made (the site returned an empty error).
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
