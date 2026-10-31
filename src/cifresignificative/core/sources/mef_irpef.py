"""MEF Dipartimento delle Finanze: personal income tax (IRPEF) return statistics by income class, from the open-data export of
https://www1.finanze.gov.it/finanze/analisi_stat/public/index.php?opendata=yes (manual export, `download.sh` is kept with the files).

One CSV per (filing year, table code); the prefix of the file name is the **filing year** (`Anno presentazione`), the income is that of the **tax year = filing year - 1**.
Tables: `AE*` = individuals by prevalent income, in the turnover classification only (entrepreneurs: business-income form sections RF / RG; professionals: RE; each also for total income of at least EUR 100,000), `AA*` = all individuals (`TOT`), employees (`DIP`), pensioners (`PEN`); `AB*` = individuals with a VAT number (`TIT`), of which entrepreneurs (`IMP`), professionals (`PRO`),
farmers (`AGR`). `01xx` = number of taxpayers by income class; `02xx`/`03xx` = number, amount and mean of each income type and tax item by income class.
Amounts and means are in **thousand euro**. Known quirks handled here: three encodings (UTF-8 with BOM; 14 files where the byte 0xA0 of "à" was lost, leaving a lone 0xC3),
Italian number format (`.` thousands, `,` decimals, a space before negatives), `***` = suppressed cell, footnote lines after the table, variable names that change over the years
(spacing, `*`, apostrophe vs accent, typos; real renames are NOT merged and are listed by `variable_spans`).
Empty files on the server (filing years 2005-2008, the `AB*0203/0303/0202` tables of 2009-2011, `AAPFDIP0202` and `AAPFPEN0202` of 2016) are simply absent.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/external/source/mef_irpef")
GROUPS = {"PFTOT": "TOT", "PFDIP": "DIP", "PFPEN": "PEN", "PFTIT": "TIT", "PFIMP": "IMP", "PFPRO": "PRO", "PFAGR": "AGR", "PFMIN": "MIN",
          "TIPFG": "PIMP", "TPFGA": "PIMP", "TPFGC": "PIMP", "TIPRE": "PPRO", "TPREA": "PPRO", "TPREB": "PPRO"}  # AE* tables (individuals by prevalent income): PIMP = entrepreneurs (form sections RF, RG), PPRO = regular self-employed with a VAT number (RE); the tokens changed in 2014; the variant for total income of at least EUR 100,000 is told apart by `topic`  # MIN = minimi / regime di vantaggio / regimi agevolati (flat-rate filers)
DIMENSIONS = {"Classi di reddito complessivo in euro": "income", "Ventili di reddito complessivo": "ventile", "Regione": "region", "Sezione di attivita'": "activity", "Classi di volume d'affari in euro": "turnover", "Classi di eta'": "age"}
MEASURES = {"Frequenza": "freq", "Ammontare": "amount_keur", "Media": "mean_keur", "Percentuale": "pct"}
VENTILE_BOUND = re.compile(r"estremo del ventile", re.IGNORECASE)  # extra column of the income-twentile tables: income at the upper end of each twentile, euro
LEAD = re.compile(r"^Numero [a-z][a-z ']+$", re.IGNORECASE)  # the count column: "Numero contribuenti", "Numero titolari di partita IVA", "Numero agricoltori", ...


def read_text(path: Path) -> str:
    b = path.read_bytes()
    try:
        return b.decode("utf-8-sig")
    except UnicodeDecodeError:  # "à" lost its 0xA0 byte in 14 files: a lone 0xC3 is always an "à" here (the only non-ASCII character in them)
        return re.sub(rb"\xc3(?![\x80-\xbf])", "à".encode(), b).decode("utf-8-sig")


def num(tok: str) -> float:
    tok = tok.strip()
    return np.nan if tok in ("", "***") else float(tok.replace(".", "").replace(",", "."))


def canon(name: str) -> str:
    """Variable name without formatting noise: footnote marks (`*`, `(*)`, `(1)`, `(2)`), accents turned into `letter'` followed by a space (`contabilita' ordinaria`), spacing, known typos."""
    s = unicodedata.normalize("NFC", name).replace("’", "'")
    s = re.sub(r"\s*\((?:\*|\d)\)|\*", "", s)
    s = "".join({"à": "a'", "è": "e'", "é": "e'", "ì": "i'", "ò": "o'", "ù": "u'"}.get(c, c) for c in s)
    s = re.sub(r"(?<=[aeiou])'\s*(?=[A-Za-z])", "' ", s)  # accent-derived apostrophe: always followed by one space (elisions end in a consonant: dell', l', d')
    s = re.sub(r"(?<=[dl])'\s+(?=[A-Za-z])", "'", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace("Imposa netta", "Imposta netta")


def class_bounds(label: str) -> tuple[float, float]:
    """Numeric bounds (lo inclusive, hi exclusive) of a class label in the unit of the classification (euro, or years for age): `da 6.000 a 7.500`, `minore di -1.000`, `minore di zero`,
    `zero`, `oltre 300.000`, `0 - 14`. NaN for labels without numbers (regions, sectors, `Assenza dichiarazione Iva`)."""
    low = label.lower().strip()
    nums = [float(x.replace(".", "")) for x in re.findall(r"-?\d[\d\.]*", label)]
    if low.startswith("zero"):
        return 0.0, 0.0
    if low.startswith(("minore", "fino")):
        return -np.inf, (nums[0] if nums else 0.0)
    if low.startswith(("oltre", "maggiore", "superiore")) and nums:
        return nums[0], np.inf
    return (nums[0], nums[1]) if len(nums) == 2 else (np.nan, np.nan)


def parse_file(path: Path) -> pd.DataFrame:
    """Long table for one file: class, variable, measure, value (+ `suppressed`), with the file's metadata columns."""
    lines = read_text(path).splitlines()
    meta = {}
    i = 0
    while i < len(lines) and lines[i].strip():
        if ":;" in lines[i]:
            k, v = lines[i].split(":;", 1)
            meta[k.strip()] = v.strip().strip(";").strip()
        i += 1
    m = re.search(r"Dichiarazioni (\d{4}) - Anno d'imposta (\d{4})", lines[0])
    assert m, f"{path}: no title line"
    while not lines[i].strip():
        i += 1
    header = [h.strip() for h in lines[i].split(";")]
    while header and not header[-1]:
        header.pop()
    body, notes = [], []
    for ln in lines[i + 1:]:
        if not ln.strip():
            continue
        cells = ln.split(";")
        while cells and not cells[-1].strip():  # the trailing ";" of every row
            cells.pop()
        if len(cells) > len(header):  # labels that contain ";" (NACE section names such as "Commercio ...; riparazione ..."): fold the extra pieces back into the label
            extra = len(cells) - len(header)
            cells = ["; ".join(cells[: extra + 1]), *cells[extra + 1:]]
        (body if len(cells) >= len(header) else notes).append(cells if len(cells) >= len(header) else ln.strip())
    cols = []  # (variable, measure) per column after the first two
    for h in header[2:]:
        mm = re.match(r"^(.*) - (Frequenza|Ammontare|Media|Percentuale)$", h)
        if mm:
            cols.append(("taxpayers" if LEAD.match(mm.group(1)) else mm.group(1), MEASURES[mm.group(2)], h))
        elif LEAD.match(h):
            cols.append(("taxpayers", "freq", h))
        elif VENTILE_BOUND.search(h):
            cols.append(("ventile_upper_eur", "level", h))
        else:
            raise ValueError(f"{path.name}: unrecognised column {h!r}")
    code = path.stem[4:]
    dimension = DIMENSIONS.get(canon(meta.get("Classificazione", "")), DIMENSIONS.get(meta.get("Classificazione", ""), "other"))
    rows = []
    for order, cells in enumerate(body):
        label, ccode = cells[0].strip(), cells[1].strip()
        total = label.upper() == "TOTALE"
        lo, hi = (np.nan, np.nan) if total else class_bounds(label)
        for (var, meas, raw), tok in zip(cols, cells[2:len(header)], strict=False):
            rows.append((order, label, ccode, total, lo, hi, var, canon(var), raw.endswith("*") or "(*)" in raw or "*" in var, meas, num(tok), tok.strip() == "***"))
    out = pd.DataFrame(rows, columns=["class_order", "class_label", "class_code", "is_total", "class_lo", "class_hi", "variable_raw", "variable", "includes_zeros", "measure", "value", "suppressed"])
    out["value"] = out["value"].astype(float)  # object dtype when a table has no numeric cell at all (empty AE table of 2023)
    is_tot = out["is_total"].astype(bool)  # an empty table has object dtype here
    tot = out[is_tot].assign(_na=lambda t: t["value"].isna()).sort_values("_na", kind="stable").drop_duplicates(["variable", "measure"]).drop(columns="_na")  # the AE tables end with a second TOTALE row for the items without a class split: keep the filled cell
    out = pd.concat([out[~is_tot], tot]).sort_values("class_order", kind="stable", ignore_index=True)
    return out.assign(filing_year=int(m.group(1)), tax_year=int(m.group(2)), code=code, model=code[:2], group=GROUPS[code[2:7]], table_kind="dist" if code[7:9] == "01" else "multi", dimension=dimension,
                      updated=meta.get("Data ultimo aggiornamento", ""), topic=meta.get("Tematica", ""), notes=" | ".join(notes))


def jump_flags(d: pd.DataFrame, factor: float = 3.0, floor_keur: float = 1000.0) -> pd.Series:
    """Review flag for amount cells that jump away from BOTH neighbouring tax years (same direction, by more than `factor` times the cell's typical size, at least `floor_keur`).
    A flag is not an error: it marks both genuine regime changes (flat-rate net income by class in 2019, the 2021 integration treatment) and data oddities (2017, lowest class)."""
    a = d[(d.measure == "amount_keur") & ~d.is_total & d.value.notna()].sort_values(["code", "variable", "class_label", "tax_year"])
    key = [a.code, a.variable, a.class_label]
    g = a.groupby(key, observed=True)
    prev, nxt, py, ny = g.value.shift(1), g.value.shift(-1), g.tax_year.shift(1), g.tax_year.shift(-1)
    scale = g.value.transform(lambda s: s.abs().median()).clip(lower=floor_keur)
    up, dn = a.value - prev, a.value - nxt
    flag = (a.tax_year - py == 1) & (ny - a.tax_year == 1) & (up.abs() > factor * scale) & (dn.abs() > factor * scale) & (np.sign(up) == np.sign(dn))
    return flag.reindex(d.index, fill_value=False)


def _concat(files: list[Path]) -> pd.DataFrame:
    d = pd.concat([parse_file(f) for f in files], ignore_index=True)
    d["variable"] = d["variable"].astype("category")
    d["jump"] = jump_flags(d)
    return d


def load(src: Path = SRC) -> pd.DataFrame:
    """The income-class tables (`mef_irpef/`), the table behind `data/processed/mef_irpef.parquet`."""
    return _concat(sorted(src.glob("20*/20*.csv")))


def load_dims(root: Path = SRC.parent) -> pd.DataFrame:
    """The other classifications downloaded with `scripts/fetch_mef_irpef.py` (`mef_irpef_<name>/`: ventili, regione, attivita, fatturato, eta), with a `dimension` column
    (ventile, region, activity, turnover, age). Table codes are only unique within a dimension (`0105` means different things in different classifications): always key by (dimension, code)."""
    return _concat(sorted(f for d in root.glob("mef_irpef_*") for f in d.glob("20*/20*.csv")))


def variable_spans(d: pd.DataFrame) -> pd.DataFrame:
    """First and last tax year and number of years of every (code, variable): shows renames and new items."""
    g = d[d.measure.isin(["freq", "pct"])].groupby(["code", "variable"], observed=True).tax_year.agg(first="min", last="max", years="nunique")
    return g.reset_index()


def qa(d: pd.DataFrame, rel_tol: float = 0.01) -> pd.DataFrame:
    """One row per (code, tax_year, variable) with: `class_sum_ratio` (sum of class values / TOTALE row, for freq and amount; 1 = consistent), `mean_check`
    (reported mean / (amount / freq) over classes with freq >= 1000 and an amount, as a median ratio), `suppressed_share`. Suppressed cells are excluded from the sums, so ratios below 1 may just mean suppression."""
    rows = []
    for (code, ty, var, meas), g in d[d.measure.isin(["freq", "amount_keur"])].groupby(["code", "tax_year", "variable", "measure"], observed=True):
        tot = g[g.is_total].value
        cls = g[~g.is_total].value
        rows.append({"code": code, "tax_year": ty, "variable": var, "measure": meas, "total": float(tot.iloc[0]) if len(tot) else np.nan, "class_sum": float(cls.sum(min_count=1)), "suppressed": int(g[~g.is_total].suppressed.sum())})
    out = pd.DataFrame(rows)
    out["class_sum_ratio"] = out.class_sum / out.total
    return out
