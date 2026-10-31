"""Download MEF tax-return statistics (open-data portal, CC BY 3.0 IT) by classification.

Usage:
  uv run python scripts/fetch_mef_irpef.py list ventili                 # catalogue of tables (no download), printed
  uv run python scripts/fetch_mef_irpef.py get ventili [--tables REGEX] [--years 2012 2025]   # download into data/external/source/mef_irpef_<name>/
Classifications: classi_reddito, ventili, regione, attivita, eta, fatturato, comune. Default `--tables` keeps individuals (AAPF*, ABPF*, AE*). One request per second.
"""
import argparse
from pathlib import Path

from cifresignificative.core.sources import mef_portal as P


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "get"])
    ap.add_argument("name", choices=sorted(P.CLASSIFICATIONS))
    ap.add_argument("--tables", default=P.PERSONS, help="regex on the table code (year removed)")
    ap.add_argument("--years", nargs=2, type=int, default=[2005, 2030])
    a = ap.parse_args(argv)
    if a.cmd == "list":
        cat = P.list_tables(P.CLASSIFICATIONS[a.name])
        out = Path(f"data/external/source/mef_irpef_{a.name}")
        out.mkdir(parents=True, exist_ok=True)
        cat.to_csv(out / "catalog_all.csv", index=False)
        cat = cat.assign(title=cat.title.str.replace(r"^\d{4} a\.i\. \d{4} / ", "", regex=True))
        print(cat.groupby(["table", "title", "tematica", "classificazione"]).year.agg(["min", "max", "count"]).to_string(max_colwidth=90))
        return 0
    cat = P.fetch(P.CLASSIFICATIONS[a.name], Path(f"data/external/source/mef_irpef_{a.name}"), a.tables, tuple(a.years))
    print(f"{len(cat)} tables requested, {int((cat.file != '').sum())} downloaded, {int((cat.file == '').sum())} empty on the server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
