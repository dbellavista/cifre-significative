"""MEF Dipartimento delle Finanze: Municipal IRPEF declarations by income bracket ('Redditi e principali variabili IRPEF su base comunale').

Downloads the annual open-data CSV zips (tax years 2008-2024) from
`https://www1.finanze.gov.it/finanze/analisi_stat/public/v_4_0_0/contenuti/Redditi_e_principali_variabili_IRPEF_su_base_comunale_CSV_<year>.zip`
 into `data/raw/mef_comuni/` (gitignored) and extracts two tidy tables for the Emilia-Romagna peer municipalities (`core.geo.ER_COMUNI`):
1. `load_brackets()`: 8 income brackets of `Reddito complessivo` (`<=0`, `0-10k`, `10-15k`, `15-26k`, `26-55k`, `55-75k`, `75-120k`, `>120k`) with `taxpayers` (`Frequenza`) and `income_eur` (`Ammontare in euro`).
2. `load_all()` also returns municipality x tax_year totals and income components, with `_eur` amounts and `_freq` frequencies where the annual source provides them. Entrepreneur ordinary and simplified accounting and participation income include zero values in their frequencies.

Municipalities are matched primarily by `Codice catastale` (`C573` for Cesena, invariant across province re-numberings such as Rimini `099014` in 2009) and mapped to their canonical 6-digit ISTAT `comune_id`.

Blank brackets: since tax year 2023 the files blank the `>120k` cell of a municipality whenever its `<=0` cell is blank (416 of 808 municipalities with 10k+ taxpayers in 2023, 377 of 815 in 2024; apparently complementary suppression). Reading a blank as no income removes about 10% of the income of those municipalities. `Reddito complessivo` totals exist only from 2023, so a single blank positive bracket is recovered as total minus the published brackets (flag `imputed`; the residual includes the suppressed `<=0` cell, which is a few taxpayers). Before 2023 there are no totals, and blanks stay missing (small municipalities only, plus Niscemi at 10k+).
"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import truststore

from cifresignificative.core.geo import (
    BELT_BY_CATASTALE,
    BELT_BY_ID,
    COMUNI_BY_CATASTALE,
    COMUNI_BY_ID,
)
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/mef_comuni")
BASE_URL = "https://www1.finanze.gov.it/finanze/analisi_stat/public/v_4_0_0/contenuti"
YEARS = tuple(range(2008, 2025))
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}

BRACKET_RE = re.compile(
    r"^Reddito complessivo (minore (?:o uguale a|di) zero(?: euro)?|da (\d+) a (\d+) euro|oltre (\d+) euro)\s*-\s*(Frequenza|Ammontare(?: in euro)?)$",
    re.IGNORECASE,
)

SUMMARY_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("reddito_complessivo", re.compile(r"^Reddito complessivo\s*-\s*(Frequenza|Ammontare)", re.IGNORECASE), "both"),
    ("reddito_imponibile", re.compile(r"^Reddito imponibile\s*-\s*(Frequenza|Ammontare)", re.IGNORECASE), "both"),
    ("imposta_netta", re.compile(r"^Imposta netta\s*-\s*(Frequenza|Ammontare)", re.IGNORECASE), "both"),
    ("addizionale_comunale", re.compile(r"^Addizionale comunale dovuta\s*-\s*(Frequenza|Ammontare)", re.IGNORECASE), "both"),
    ("reddito_dipendente", re.compile(r"^Reddito da lavoro dipendente.*-\s*(Frequenza|Ammontare)", re.IGNORECASE), "both"),
    ("reddito_pensione", re.compile(r"^Reddito da pensione\s*-\s*(Frequenza|Ammontare)", re.IGNORECASE), "both"),
    ("reddito_autonomo", re.compile(r"^Reddito da lavoro autonomo.*-\s*(Frequenza|Ammontare)", re.IGNORECASE), "both"),
    ("reddito_fabbricati", re.compile(r"^Reddito da fabbricati\s*-\s*(Frequenza|Ammontare)", re.IGNORECASE), "both"),
    ("reddito_impresa_ordinaria", re.compile(r"^Reddito (?:di spettanza dell.imprenditore in contabilit.{0,3}|spettanza imprenditore) ordinaria.*-\s*(Frequenza|Ammontare)", re.IGNORECASE), "both"),
    ("reddito_impresa_semplificata", re.compile(r"^Reddito (?:di spettanza dell.imprenditore in contabilit.{0,3}|spettanza imprenditore) semplificata.*-\s*(Frequenza|Ammontare)", re.IGNORECASE), "both"),
    ("reddito_partecipazione", re.compile(r"^Reddito da partecipazione.*-\s*(Frequenza|Ammontare)", re.IGNORECASE), "both"),
]


def zip_path(year: int, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"irpef_comuni_{year}.zip"


def fetch(force: bool = False, years: tuple[int, ...] = YEARS, raw_dir: Path = RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for y in years:
        p = zip_path(y, raw_dir)
        if p.exists() and not force:
            continue
        url = f"{BASE_URL}/Redditi_e_principali_variabili_IRPEF_su_base_comunale_CSV_{y}.zip"
        r = requests.get(url, headers=UA, timeout=120)
        r.raise_for_status()
        if not r.content.startswith(b"PK"):
            raise RuntimeError(f"MEF IRPEF comuni {y}: response is not a ZIP archive")
        p.write_bytes(r.content)
        record(f"mef_comuni_{y}", source="MEF Finanze", dataset=f"IRPEF_comunale_{y}", url=url)
    return raw_dir


def _to_num(val: object) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip().replace(".", "").replace(",", ".")
    if not s or s.lower() in ("nan", "***", "n.d.", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def read_year_frame(zpath: Path, catastali: set[str] | None = None) -> pd.DataFrame:
    """Read one annual MEF CSV as strings, keeping `catastali` only when given (rows without a cadastral code are totals, and are dropped)."""
    with zipfile.ZipFile(zpath) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv_name) as f:
            first_line = f.readline().decode("latin1").rstrip("\r\n")
        n_cols = len(first_line.split(";"))
        with z.open(csv_name) as f:
            df = pd.read_csv(f, sep=";", encoding="latin1", dtype=str, usecols=range(n_cols), low_memory=False)
    df.columns = [re.sub(r"\s+", " ", c.strip()) for c in df.columns]
    cat_col = next(c for c in df.columns if "codice catastale" in c.lower())
    df = df[df[cat_col].notna()].copy()
    df["catastale"] = df[cat_col].astype(str).str.strip().str.upper()
    return df[df["catastale"].isin(catastali)].copy() if catastali is not None else df


def _bracket_spec(raw_lbl: str, lo_s: str | None, hi_s: str | None, top_s: str | None) -> tuple[float, float, int, str]:
    """Map a bracket column label to `(class_lo, class_hi, class_order, class_label)`."""
    if "minore" in raw_lbl.lower():
        return -np.inf, 0.0, 0, "<= 0"
    if top_s:
        return float(top_s), np.inf, 7, f"> {int(top_s)}"
    lo, hi = float(lo_s or 0), float(hi_s or 0)
    return lo, hi, {0.0: 1, 10000.0: 2, 15000.0: 3, 26000.0: 4, 55000.0: 5, 75000.0: 6}.get(lo, 9), f"{int(lo)}-{int(hi)}"


def parse_brackets(row: pd.Series, columns: pd.Index) -> list[dict]:
    """Return one dict per income bracket of a MEF row (`class_*`, `taxpayers`, `income_eur`, `imputed`), recovering a single blank positive bracket from the reported totals."""
    classes: dict[int, dict] = {}
    for col in columns:
        m = BRACKET_RE.match(col)
        if not m:
            continue
        raw_lbl, lo_s, hi_s, top_s, meas_raw = m.groups()
        lo, hi, order, lbl = _bracket_spec(raw_lbl, lo_s, hi_s, top_s)
        cl = classes.setdefault(order, {"class_order": order, "class_label": lbl, "class_lo": lo, "class_hi": hi, "taxpayers": np.nan, "income_eur": np.nan, "imputed": False})
        cl["taxpayers" if meas_raw.lower().startswith("freq") else "income_eur"] = _to_num(row[col])

    tot_f = next((_to_num(row[c]) for c in columns if re.match(r"^Reddito complessivo\s*-\s*Freq", c, re.IGNORECASE)), np.nan)
    tot_a = next((_to_num(row[c]) for c in columns if re.match(r"^Reddito complessivo\s*-\s*Ammontare", c, re.IGNORECASE)), np.nan)
    blank = [cl for order, cl in classes.items() if order >= 1 and np.isnan(cl["taxpayers"]) and np.isnan(cl["income_eur"])]
    if len(blank) == 1 and np.isfinite(tot_f) and np.isfinite(tot_a):
        rest_f = np.nansum([cl["taxpayers"] for cl in classes.values()])
        rest_a = np.nansum([cl["income_eur"] for cl in classes.values()])
        # A non-positive residual means the published cells already exhaust the total: leave the cell blank rather than invent income.
        if tot_f - rest_f > 0 and tot_a - rest_a > 0:
            blank[0].update(taxpayers=tot_f - rest_f, income_eur=tot_a - rest_a, imputed=True)
    return sorted(classes.values(), key=lambda cl: cl["class_order"])


def parse_year(zpath: Path, comuni_ids: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one annual MEF municipal ZIP into `(brackets_df, summary_df)` for the requested `comuni_ids`."""
    all_by_cat = {**BELT_BY_CATASTALE, **COMUNI_BY_CATASTALE}
    all_by_id = {**BELT_BY_ID, **COMUNI_BY_ID}
    target_ids = set(COMUNI_BY_ID) if comuni_ids is None else set(comuni_ids)
    cat_to_id = {c.catastale: c.id for c in all_by_cat.values() if c.id in target_ids}
    df = read_year_frame(zpath, set(cat_to_id))
    yr_col = next(c for c in df.columns if "anno di imposta" in c.lower())
    contrib_col = next(c for c in df.columns if "numero contribuenti" in c.lower())

    b_rows: list[dict] = []
    s_rows: list[dict] = []

    for _, r in df.iterrows():
        cid = cat_to_id[r["catastale"]]
        ty = int(str(r[yr_col]).strip())
        # 1. Bracket rows
        b_rows.extend({"comune_id": cid, "tax_year": ty, **cl} for cl in parse_brackets(r, df.columns))

        # 2. Summary row
        s_item: dict[str, object] = {
            "comune_id": cid,
            "comune": all_by_id[cid].name if cid in all_by_id else cid,
            "tax_year": ty,
            "taxpayers_total": _to_num(r[contrib_col]),
        }
        for key, pat, _ in SUMMARY_PATTERNS:
            for col in df.columns:
                pm = pat.match(col)
                if pm:
                    suffix = "freq" if pm.group(1).lower().startswith("freq") else "eur"
                    s_item[f"{key}_{suffix}"] = _to_num(r[col])
        s_rows.append(s_item)

    b_df = pd.DataFrame(b_rows)
    if len(b_df):
        b_df = b_df.sort_values(["comune_id", "tax_year", "class_order"], ignore_index=True)
    return b_df, pd.DataFrame(s_rows).sort_values(["comune_id", "tax_year"], ignore_index=True)


def load_all(raw_dir: Path = RAW_DIR, comuni_ids: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load `(brackets_df, summary_df)` across all cached annual ZIPs in `raw_dir`."""
    zfiles = sorted(raw_dir.glob("irpef_comuni_*.zip"))
    if not zfiles:
        raise FileNotFoundError(f"No MEF municipal IRPEF zips in {raw_dir}; run `uv run python -m cifresignificative.core.sources.mef_comuni fetch`")
    b_list, s_list = [], []
    for zp in zfiles:
        b, s = parse_year(zp, comuni_ids=comuni_ids)
        b_list.append(b)
        s_list.append(s)
    return pd.concat(b_list, ignore_index=True), pd.concat(s_list, ignore_index=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        print(fetch("--force" in sys.argv))
    b, s = load_all()
    print(f"brackets: {len(b)} rows ({b.tax_year.min()}-{b.tax_year.max()}), summary: {len(s)} rows, comuni: {s.comune_id.nunique()}")
