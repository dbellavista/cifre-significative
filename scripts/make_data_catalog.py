"""Generate docs/data_catalog.md from the code definitions (catalog.py, source modules) and the built panel.

Usage: uv run python scripts/make_data_catalog.py          (after fetch + build) rewrites docs/data_catalog.md
       uv run python scripts/make_data_catalog.py --check  exits 1 when the file is stale: the whole file when the processed data exist,
                                                          otherwise only the data-independent sections (CI has no data)
"""
import sys
from pathlib import Path

import pandas as pd

from cifresignificative.core import catalog
from cifresignificative.core.sources import ameco, imf, oecd, procurement, worldbank

DOC = Path("docs/data_catalog.md")
PANEL_PATH = Path("data/processed/panel.parquet")
PANEL = pd.DataFrame()  # loaded by main() when the processed data exist


def coverage(name: str) -> str:
    d = PANEL[PANEL.indicator == name]
    if d.empty:
        return "not in panel"
    it = d[d.iso3 == "ITA"].year
    return f"{d.geo.nunique()} geos, {int(d.year.min())}-{int(d.year.max())}; Italy {int(it.min()) if len(it) else '-'}-{int(it.max()) if len(it) else '-'}"


def fmt_filters(f: dict) -> str:
    return ", ".join(f"{k}={v if isinstance(v, str) else '|'.join(v[:3]) + '...'}" for k, v in f.items())


def panel_rows() -> list[tuple]:
    """One row per panel indicator, from the code definitions and the built panel."""
    rows = []
    for i in catalog.EUROSTAT:
        rows.append((i.name, i.theme, f"Eurostat `{i.dataset}`", fmt_filters(i.filters), i.note, coverage(i.name)))
    for n, (code, theme) in worldbank.WDI.items():
        rows.append((n, theme, f"World Bank `{code}`", "", "", coverage(n)))
    for n, (code, theme) in imf.WEO.items():
        rows.append((n, theme, f"IMF WEO `{code}`", "", "general government unless stated", coverage(n)))
    for n, (suffix, theme) in ameco.SERIES.items():
        rows.append((n, theme, f"AMECO `{suffix}`", "", "", coverage(n)))
    for n, (flow, measure, sex) in oecd.SERIES.items():
        rows.append((n, "pensions", f"OECD Pensions at a Glance `{measure}`", f"sex={sex or 'all'}", "", coverage(n)))
    for code, (n, _) in procurement.INDICATORS.items():
        rows.append((n, "governance", f"Single Market Scoreboard indicator {code}", "", "Power BI report, 2021-2024", coverage(n)))
    rows.append(("vat_compliance_gap_pct", "governance", "EC VAT gap report 2025, Table 205", "", "% of VAT due", coverage("vat_compliance_gap_pct")))
    for n in ["women_managers_share_pct"]:
        rows.append((n, "gender", "Eurostat `lfsa_egais` (derived)", "", "women / all managers", coverage(n)))
    for ext in sorted(Path("data/external").glob("*.csv")):
        df = pd.read_csv(ext)
        if "indicator" not in df.columns:  # reference tables (governments, policies) are not panel series
            continue
        for n in df.indicator.unique():
            filled = int(df[(df.indicator == n)].value.notna().sum())
            rows.append((n, df[df.indicator == n].theme.iloc[0], f"Manual: `data/external/{ext.name}`", "", f"{filled} values entered", coverage(n)))

    known = {r[0] for r in rows}
    for n in sorted(set(PANEL.indicator) - known):  # derived series (computed in clean/build.py) and anything not listed above
        d = PANEL[PANEL.indicator == n]
        rows.append((n, d.theme.iloc[0], d.source.iloc[0], "", "derived in clean/build.py" if "derived" in d.source.iloc[0] else "", coverage(n)))
    return rows


def table(items: list[catalog.Indicator], with_freq: bool = False) -> list[str]:
    head = "| indicator | theme | dataset | filters | note |" + (" freq |" if with_freq else "")
    out = [head, "|---|---|---|---|---|" + ("---|" if with_freq else "")]
    for i in items:
        out.append(f"| {i.name} | {i.theme} | Eurostat `{i.dataset}` | {fmt_filters(i.filters).replace('|', '/')} | {i.note.replace('|', '/')} |" + (f" {i.freq} |" if with_freq else ""))
    return out


def static_sections() -> str:
    """Sections that depend only on the code (no processed data): safe to verify in CI."""
    lines = ["## Other processed tables", "",
             "| file | content |", "|---|---|",
             f"| `regional.parquet` | NUTS2 regions, long format: {len(catalog.EUROSTAT_REGIONAL)} indicators (table below) |",
             f"| `highfreq.parquet` | monthly and quarterly series, long format with period label and period-end date: {len(catalog.EUROSTAT_HIGHFREQ)} indicators (table below) |",
             "| `sectors.parquet` | industry value added (real and nominal) and hours, 10 NACE sections |",
             "| `spending.parquet` | government spending by function (COFOG) and by economic nature |",
             "| `pensions_oecd_2024.parquet` | OECD pension design cross-section (replacement rates, retirement ages) |", "",
             "## Regional indicators (NUTS2)", "", *table(catalog.EUROSTAT_REGIONAL), "",
             "## High-frequency indicators (monthly `m_`, quarterly `q_`)", "", *table(catalog.EUROSTAT_HIGHFREQ, with_freq=True), ""]
    return "\n".join(lines)


def render(rows: list[tuple]) -> str:
    out = ["# Data catalog", "", "_Generated by `scripts/make_data_catalog.py` from the code definitions and the built panel; do not edit by hand._", "",
           f"{len(rows)} indicators in `data/processed/panel.parquet` (long format). Column details and caveats: `data_dictionary.md`.", "",
           "## Sources", "",
           "| provider | access | code | refresh |", "|---|---|---|---|",
           "| Eurostat | Statistics API (JSON-stat) | `sources/eurostat.py`, `catalog.py` | `uv run python -m cifresignificative.core.fetch --force` |",
           "| World Bank | `wbgapi` (WDI, EdStats, WGI) | `sources/worldbank.py` | same |",
           "| IMF WEO | DataMapper API | `sources/imf.py` | same |",
           "| AMECO (EC) | bulk CSV zip | `sources/ameco.py` | same |",
           "| OECD | SDMX API (Pensions at a Glance) | `sources/oecd.py` | same |",
           "| EC VAT gap | Excel download (URL changes per edition) | `sources/vatgap.py` | edit `URL` when a new report appears |",
           "| Single Market Scoreboard | headless browser reading the embedded Power BI report | `sources/procurement.py`, `sources/powerbi.py`, `sources/browser.py` | same |",
           "| Hand-entered | CSV templates | `sources/manual.py`, `docs/manual_data.md` | manual |", "",
           "Provenance of every download (URL, source update date, timestamp) is in `data/raw/manifest.json`.", "",
           static_sections(),
           "## Indicators in the panel", "",
           "| indicator | theme | source | filters | note | coverage |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r[1], r[0])):
        out.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    return "\n".join(out) + "\n"


def main(check: bool = False) -> int:
    global PANEL
    if not PANEL_PATH.exists():
        if not check:
            print("processed panel missing: run fetch and build first", file=sys.stderr)
            return 1
        stale = static_sections() not in DOC.read_text()
        print("docs/data_catalog.md: static sections are " + ("STALE: run scripts/make_data_catalog.py" if stale else "up to date") + " (no processed data: full check skipped)")
        return int(stale)
    PANEL = pd.read_parquet(PANEL_PATH)
    text = render(panel_rows())
    if check:
        stale = DOC.read_text() != text
        print("docs/data_catalog.md is " + ("STALE: run scripts/make_data_catalog.py" if stale else "up to date"))
        return int(stale)
    DOC.write_text(text)
    print(f"wrote {DOC} ({text.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(check="--check" in sys.argv))
