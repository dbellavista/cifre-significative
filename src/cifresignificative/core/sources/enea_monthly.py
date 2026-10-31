"""Superbonus (SuperEcobonus) monthly use by region, from the ENEA monthly reports (62 end-of-month PDFs, August 2021 to September 2026).

Each report has 22 pages: Italy, the 20 regions (Trentino-Alto Adige is one row) and a summary table. Per page: the number of buildings (asseverazioni early on, buildings from spring 2023), the investment admitted to the deduction, the investment of the works already completed and their share, the deductions accrued for the completed works,
and the same by type of building (condominiums, single-family houses, functionally independent units, and A/9 buildings open to the public from 2023-03, called castles until 2024). Everything is cumulative from the start of the Superbonus (July 2020) up to the month end, by transmission date of the asseverazioni, in euro.
The layout changed twice (label wording, a total including the sums not admitted to the deduction from 2023-03, ligatures such as `inves men` and `se embre` from 2025), so rows are matched by label prefix inside a section (total, condominiums, single-family houses, independent units, A/9), not by position.
Descriptive tool: cumulative series are revised upwards by late transmissions, so a monthly flow (difference of two reports) is a flow by transmission date, and can be slightly negative.
The PDFs go to `data/raw/enea/monthly/` (gitignored; links in `data/external/enea_monthly_urls.csv`, taken from the ENEA page `risultati-superbonus.html`; parsing needs poppler's `pdftotext`); the committed table `data/external/enea_superbonus_monthly.csv` is the source of truth.
`uv run python -m cifresignificative.core.sources.enea_monthly fetch|build|check`.
"""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record
from cifresignificative.core.sources.enea_regions import NUTS2

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/enea/monthly")
URLS = Path("data/external/enea_monthly_urls.csv")
CSV = Path("data/external/enea_superbonus_monthly.csv")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
AMOUNT = re.compile(r"\d{1,3}(?:\.\d{3})*,\d{2}(?![\d%])")
PCT = re.compile(r"(\d{1,3}(?:,\d+)?)%")
COUNT = re.compile(r"(?<![\d,.])\d+(?:\.\d{3})*(?![\d,.%])")
# section header (a line without digits) -> section key
SECTIONS = [(r"^condomini$", "condo"), (r"^ed\w*ci unifamiliari$", "single"), (r"^u\.i\. funzionalmente", "ui"), (r"^(castelli|a9)", "a9"), (r"^inves\S* ?mento medio", "avg")]
TYPES = ["condo", "single", "ui", "a9"]
FIELDS = ["n_buildings", "inv_incl_nonadmitted", "inv_admitted", "inv_completed", "pct_completed", "deductions_accrued"] + [f"{t}_{k}" for t in TYPES for k in ("n", "admitted", "completed")]


def path(date: str) -> Path:
    return RAW_DIR / f"{date}.pdf"


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for date, url in pd.read_csv(URLS).itertuples(index=False):
        if path(date).exists() and not force:
            continue
        resp = requests.get(url, headers=UA, timeout=600)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            raise RuntimeError(f"{date}: not a PDF")
        path(date).write_bytes(resp.content)
        record(f"enea_monthly_{date}", source="ENEA, Report dati mensili Superbonus", url=url)
    return RAW_DIR


def _number(tok: str) -> float:
    return float(tok.replace(".", "").replace(",", "."))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("ﬁ", "fi").replace("ﬂ", "fl")).strip()


def region_name(raw: str) -> str:
    """Page title -> region name (the title has ligature glitches `Tren no-Alto Adige`, `Tren#no`, and names the national page `Dato Nazionale` or `Totale nazionale`)."""
    s = _norm(raw)
    if re.match(r"(?i)(dato|totale) nazionale", s):
        return "Italia"
    if s.lower().startswith("tren"):
        return "Trentino-Alto Adige"
    return "Friuli-Venezia Giulia" if s.lower().startswith("friuli") else s


def parse_page(text: str) -> tuple[str, dict]:
    """One region (or national) page -> (region name, field dict). Amounts in euro, counts as numbers, `pct_completed` as a fraction."""
    lines = [_norm(ln) for ln in text.split("\n") if ln.strip()]
    name = region_name(lines[1])
    out: dict = {}
    sec = "total"
    for ln in lines[2:]:
        ln = re.sub(r"A/9", "A9", ln)
        low = ln.lower()
        if not re.search(r"\d", re.sub(r"A9", "", ln)):
            hit = next((k for pat, k in SECTIONS if re.match(pat, low)), None)
            if hit:
                sec = hit
            continue
        if sec == "avg":
            continue
        amounts = [_number(a) for a in AMOUNT.findall(ln)]
        pct = PCT.findall(ln)
        pre = "" if sec == "total" else f"{sec}_"
        if low.startswith("n. "):
            m = COUNT.search(re.sub(r"^n\. |\bA9\b", "", ln))
            if m:
                out[f"{pre}n" if sec != "total" else "n_buildings"] = _number(m.group(0) + ",0")
        elif low.startswith("totale inves") and "ammessi" not in low and sec == "total" and amounts:
            out["inv_incl_nonadmitted"] = amounts[0]
        elif low.startswith("totale inves") and "lavori" in low and amounts:
            out["inv_completed"], out["pct_completed"] = amounts[0], float(pct[0].replace(",", ".")) / 100
        elif low.startswith("totale inves") and "ammessi" in low and sec == "total" and amounts:
            out["inv_admitted"] = amounts[0]
        elif low.startswith("detrazioni maturate") and amounts:
            out["deductions_accrued"] = amounts[0]
        elif sec != "total" and amounts and re.search(r"tot\. inv\.", low):
            out[f"{sec}_admitted"] = amounts[0]
        elif sec != "total" and amounts and re.search(r"tot\. lavori", low):
            out[f"{sec}_completed"] = amounts[0]
    return name, out


def parse_report(file: Path) -> pd.DataFrame:
    """The 21 country and region pages of one monthly report; page 22 (summary) is not used."""
    rows = []
    for page in range(1, 22):
        text = subprocess.run(["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(file), "-"], capture_output=True, text=True, check=True).stdout
        name, vals = parse_page(text)
        rows.append({"region": name, **vals})
    d = pd.DataFrame(rows)
    for col in FIELDS:
        if col not in d:
            d[col] = float("nan")
    return d[["region"] + FIELDS]


def build_csv(out: Path = CSV) -> pd.DataFrame:
    parts = []
    for date in pd.read_csv(URLS).date:
        d = parse_report(path(date)).assign(date=date)
        if sorted(d.region) != sorted(["Italia", *NUTS2]):
            raise ValueError(f"{date}: unexpected regions {sorted(set(d.region) ^ {'Italia', *NUTS2})}")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["nuts2"] = df.region.map({**NUTS2, "Italia": "IT"})
    df = df[["date", "region", "nuts2"] + FIELDS].sort_values(["date", "region"])
    df.to_csv(out, index=False)
    return df


def load(csv: Path = CSV) -> pd.DataFrame:
    return pd.read_csv(csv, parse_dates=["date"])


def check(df: pd.DataFrame, tol: float = 1000.0) -> pd.DataFrame:
    """Per month: the regions' sum minus the national total for the main amounts (euro) and the number of regions with a decrease in the cumulative completed investment since the previous month."""
    reg = df[df.region != "Italia"].groupby("date")[["inv_admitted", "inv_completed", "deductions_accrued"]].sum()
    nat = df[df.region == "Italia"].set_index("date")[["inv_admitted", "inv_completed", "deductions_accrued"]]
    gap = (reg - nat).add_prefix("gap_")
    comp = df[df.region != "Italia"].pivot(index="date", columns="region", values="inv_completed").sort_index()
    gap["n_decreases"] = (comp.diff() < -tol).sum(axis=1)
    return gap


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "fetch":
        print(fetch("--force" in sys.argv))
    elif cmd == "check":
        print(check(load()).round(0).to_string())
    else:
        d = build_csv()
        print(len(d), "rows", d.date.nunique(), "months")
