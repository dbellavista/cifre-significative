"""European Commission autumn forecasts for general-government net lending, revenue, expenditure and interest (% of GDP), all member states: the cross-country reference "plan" for the deficit-surprise placebo ranking of notebook 41.

The statistical annex of each autumn forecast (autumn 2015 to autumn 2025, so budget years 2016-2026; the autumn 2015 annex has the old AMECO layout, one column per country and one row per year over two pages per table, read by `parse_old_table`; published late October or early November, after the draft budgetary plans are in and normally taking their measures into account) has one table per variable with a row per country. Each row gives
three period averages, the annual values (years y-4..y+2 or y-3..y+2, with y the forecast year) and the figures of the previous (spring) forecast. The year columns are read from the header, the budget year B = y+1 is the one-year-ahead forecast. PDFs go to `data/raw/ec_forecast/` (gitignored);
the committed table `data/external/ec_forecasts.csv` (vintage, variable, country, year, value) is the source of truth. The autumn 2023 annex is a stand-alone file, the 2024 and 2025 ones are pages of the full report.
`uv run python -m cifresignificative.core.sources.ec_forecasts fetch|build`.
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

RAW_DIR = Path("data/raw/ec_forecast")
CSV = Path("data/external/ec_forecasts.csv")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
DL = "https://economy-finance.ec.europa.eu/document/download/"
# autumn year -> document id of the statistical annex (2024 and 2025: the full report, which contains it)
ANNEX = {
    2015: "71966575-29ae-4d32-b2ab-b9406d238c2c", 2016: "a6076d5b-b061-47e8-8634-e9669caa3869", 2017: "3c93324b-63da-4a50-a40b-9e9637907a70", 2018: "986b8bb5-0657-41dc-8add-29628a0b419e",
    2019: "7c0a724f-e06c-4cb7-b889-24c595a66054", 2020: "90a85fd9-2cd8-416d-a026-fa28353158de", 2021: "ae687e86-3796-4a77-9de1-71155a7d4cc8", 2022: "8cb9dbe9-abff-41e6-aa3b-644656508d4f",
    2023: "1b2c61ac-8457-444b-a1b8-c987593f6d5c", 2024: "7173e7c9-3841-4660-8d6a-a80712932f81", 2025: "34538512-fff6-451a-8bbc-4c8d60e4d132",
}
# variable -> start of the table title (the table number changes between vintages)
TABLES = {
    "net_lending": r"Net lending \(\+\) or net borrowing \(-\), general government \(as a percentage of GDP",
    "revenue": r"Total revenue, general government \(as a percentage of GDP",
    "expenditure": r"Total expenditure, general government \(as a percentage of GDP",
    "interest": r"Interest expenditure, general government \(as a percentage of GDP",
    "real_growth": r"Gross domestic product, volume \(percentage change on preceding year",
    "deflator": r"Deflator of gross domestic product \(percentage change on preceding year",
}
ISO3 = {
    "Belgium": "BEL", "Germany": "DEU", "Estonia": "EST", "Ireland": "IRL", "Greece": "GRC", "Spain": "ESP", "France": "FRA", "Croatia": "HRV", "Italy": "ITA", "Cyprus": "CYP", "Latvia": "LVA", "Lithuania": "LTU", "Luxembourg": "LUX", "Malta": "MLT",
    "Netherlands": "NLD", "Austria": "AUT", "Portugal": "PRT", "Slovakia": "SVK", "Slovenia": "SVN", "Finland": "FIN", "Bulgaria": "BGR", "Czechia": "CZE", "Czech Republic": "CZE", "Denmark": "DNK", "Hungary": "HUN", "Poland": "POL",
    "Romania": "ROU", "Sweden": "SWE", "United Kingdom": "GBR", "Japan": "JPN", "United States": "USA",
}
NUM = r"-?\d+\.\d"


def path(vintage: int) -> Path:
    return RAW_DIR / f"autumn{vintage}.pdf"


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year, doc in ANNEX.items():
        if path(year).exists() and not force:
            continue
        resp = requests.get(f"{DL}{doc}_en", headers=UA, timeout=300)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            raise RuntimeError(f"autumn {year}: not a PDF: {resp.content[:60]!r}")
        path(year).write_bytes(resp.content)
        record(f"ec_forecast_{year}", source="European Commission autumn forecast, statistical annex", url=f"{DL}{doc}_en")
    return RAW_DIR


def _text(pdf: Path) -> str:
    return subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout


def parse_table(text: str, title: str) -> pd.DataFrame:
    """One table of an annex as (iso3 or EA/EU, year, value): the annual columns come from the header (period averages `2005-09` or `2005 - 09` first, then the years, then the previous forecast, whose years repeat and are not kept);
    a row must carry the years and the previous-forecast values plus at most three averages, otherwise the table is rejected."""
    lines = text.split("\n")
    start = next(i for i, ln in enumerate(lines) if re.search(r"Table \d+:\s+" + title, ln))
    hdr = next(i for i in range(start + 1, start + 12) if len(re.findall(r"\b(?:19|20)\d\d\b", lines[i])) >= 8)
    h = re.sub(r"(\d{4})\s*-\s*(\d{2})\b", r"\1-\2", lines[hdr])
    toks = [int(t) for t in re.findall(r"(?<![\d-])((?:19|20)\d\d)(?![\d-])", h)]
    years = [toks[0]]
    for t in toks[1:]:
        if t <= years[-1]:
            break
        years.append(t)
    n_prev = len(toks) - len(years)
    rows, seen = [], set()
    for ln in lines[hdr + 1: hdr + 45]:
        m = re.match(r"\s*([A-Za-z][A-Za-z .()0-9]*?)\s{2,}([-\d:].*)$", ln)
        if not m:
            continue
        name, vals = m.group(1).strip(), re.findall(rf"{NUM}|:", re.sub(r"(?<=\d) :", "", m.group(2)))  # a colon glued to a number is a flag
        key = ISO3.get(name) or ("EA" if name.startswith("Euro area") else "EU" if name == "EU" else None)
        if key is None:
            continue
        if key in seen:  # the next table has started (tables that end without the United States row)
            break
        seen.add(key)
        if not len(years) + n_prev <= len(vals) <= 3 + len(years) + n_prev:
            raise ValueError(f"{name}: {len(vals)} values for {len(years)} years")
        vals = vals[len(vals) - len(years) - n_prev:]  # right-aligned: a country without history (Croatia, early vintages) lacks some period averages
        rows += [{"iso3": key, "year": y, "value": float(v)} for y, v in zip(years, vals) if v != ":"]  # ":" = not available
        if name == "United States":
            break
    d = pd.DataFrame(rows)
    if not {"ITA", "DEU", "EA"} <= set(d.iso3):
        raise ValueError("table rows not found")
    return d


ISO2 = {"BE": "BEL", "DE": "DEU", "EE": "EST", "IE": "IRL", "EL": "GRC", "ES": "ESP", "FR": "FRA", "IT": "ITA", "CY": "CYP", "LV": "LVA", "LT": "LTU", "LU": "LUX", "MT": "MLT", "NL": "NLD", "AT": "AUT", "PT": "PRT", "SI": "SVN", "SK": "SVK", "FI": "FIN",
        "BG": "BGR", "CZ": "CZE", "DK": "DNK", "HR": "HRV", "HU": "HUN", "PL": "POL", "RO": "ROU", "SE": "SWE", "UK": "GBR", "US": "USA", "JP": "JPN", "EA-19": "EA"}
# autumn 2015 (AMECO layout): variable -> table number (two pages per table, one column per country, one row per year)
OLD_TABLES = {"net_lending": 76, "revenue": 72, "expenditure": 75, "interest": 66, "real_growth": 10, "deflator": 24}


def parse_old_table(text: str, number: int) -> pd.DataFrame:
    """One table of the autumn 2015 annex (country codes in the header, annual rows `2015  -2.7  0.9 ...`); the table appears on two pages (euro-area members first), after the contents list."""
    lines = text.split("\n")
    heads = [i for i, ln in enumerate(lines) if i > 300 and re.match(rf"{number}\. ", ln)]
    if len(heads) != 2:
        raise ValueError(f"table {number}: {len(heads)} pages")
    rows = []
    for h, end in zip(heads, heads[1:] + [len(lines)]):
        hdr = next(i for i in range(h + 1, h + 12) if len(re.findall(r"\b[A-Z]{2}\b", lines[i])) >= 6)
        codes = lines[hdr].split()
        codes = [c for c in codes if re.fullmatch(r"[A-Z]{2}(-\d+)?", c)]
        for ln in lines[hdr + 1: min(hdr + 80, end)]:
            m = re.match(r"\s*(20\d\d)\s+(\S.*)$", ln)
            if not m:
                continue
            vals = m.group(2).split()
            if len(vals) != len(codes):
                raise ValueError(f"table {number} {m.group(1)}: {len(vals)} values for {len(codes)} columns")
            rows += [{"iso3": ISO2[c], "year": int(m.group(1)), "value": float(v)} for c, v in zip(codes, vals) if c in ISO2 and v != ":"]
    d = pd.DataFrame(rows)
    if not {"ITA", "DEU", "EA", "USA"} <= set(d.iso3):
        raise ValueError("table rows not found")
    return d


def build_csv(out: Path = CSV) -> pd.DataFrame:
    """All vintages and variables; years from y-1 to y+2 (the estimate for the year just ended, the current year, the budget year and the year after)."""
    parts = []
    if path(2015).exists():
        text = _text(path(2015))
        for var, num in OLD_TABLES.items():
            parts.append(parse_old_table(text, num).query("2014 <= year <= 2017").assign(vintage=2015, variable=var))
    for y in ANNEX:
        if y == 2015:
            continue
        text = _text(path(y))
        for var, title in TABLES.items():
            d = parse_table(text, title)
            d = d[d.year.between(y - 1, y + 2)]
            parts.append(d.assign(vintage=y, variable=var))
    d = pd.concat(parts, ignore_index=True)[["vintage", "variable", "iso3", "year", "value"]].sort_values(["vintage", "variable", "iso3", "year"], ignore_index=True)
    d.to_csv(out, index=False)
    return d


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "fetch":
        print(fetch("--force" in sys.argv))
    else:
        d = build_csv()
        print(d.groupby(["vintage", "variable"]).iso3.nunique().unstack().to_string())
