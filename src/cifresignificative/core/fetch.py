"""CLI: download raw data. `uv run python -m cifresignificative.core.fetch [--force] [--only NAME ...]`"""
import argparse
import sys
from collections.abc import Callable

from cifresignificative.core.catalog import (
    EUROSTAT,
    EUROSTAT_EARNINGS,
    EUROSTAT_HIGHERED,
    EUROSTAT_HIGHFREQ,
    EUROSTAT_INCOME,
    EUROSTAT_MIGRATION,
    EUROSTAT_PROJECTIONS,
    EUROSTAT_REGIONAL,
    EUROSTAT_REGIONAL_SBS,
    EUROSTAT_SECTORS,
    EUROSTAT_SECTORS64,
    EUROSTAT_SELFEMP,
    EUROSTAT_SIZE,
    EUROSTAT_SPENDING,
)
from cifresignificative.core.sources import (
    ageing,
    ameco,
    bready,
    cohesion,
    cpt,
    eba_sovereign,
    ecb_shss,
    eqi,
    eu_payments,
    euklems,
    eurostat,
    gisco_nuts,
    imf,
    istat,
    istat_contracts,
    istat_fpa,
    istat_migration,
    oecd,
    oecd_bargaining,
    oecd_regulation,
    procurement,
    vatgap,
    worldbank,
)
from cifresignificative.core.sources import browser as br


def _jobs() -> list[tuple[str, Callable[[bool], object]]]:
    jobs: list[tuple[str, Callable[[bool], object]]] = []
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force)) for i in EUROSTAT]
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force, regional=True)) for i in EUROSTAT_REGIONAL]
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force, regional=True, raw_dir=eurostat.REGIONAL_SBS_DIR)) for i in EUROSTAT_REGIONAL_SBS]
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force, raw_dir=eurostat.SECTOR_DIR)) for i in EUROSTAT_SECTORS]
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force, raw_dir=eurostat.SPENDING_DIR)) for i in EUROSTAT_SPENDING]
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force, raw_dir=eurostat.HIGHERED_DIR)) for i in EUROSTAT_HIGHERED]
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force, raw_dir=eurostat.PROJECTIONS_DIR)) for i in EUROSTAT_PROJECTIONS]
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force, raw_dir=eurostat.HIGHFREQ_DIR)) for i in EUROSTAT_HIGHFREQ]
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force, raw_dir=eurostat.SECTOR64_DIR)) for i in EUROSTAT_SECTORS64]
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force, raw_dir=eurostat.EARNINGS_DIR)) for i in EUROSTAT_EARNINGS]
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force, raw_dir=eurostat.MIGRATION_DIR)) for i in EUROSTAT_MIGRATION]
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force, raw_dir=eurostat.INCOME_DIR)) for i in EUROSTAT_INCOME]
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force, raw_dir=eurostat.SELFEMP_DIR)) for i in EUROSTAT_SELFEMP]
    jobs += [(i.name, lambda force, i=i: eurostat.fetch(i, force, raw_dir=eurostat.SIZE_DIR)) for i in EUROSTAT_SIZE]
    jobs += [(n, lambda force, n=n: worldbank.fetch(n, force)) for n in worldbank.WDI]
    jobs += [(n, lambda force, n=n: imf.fetch(n, force)) for n in imf.WEO]
    jobs += [("ameco", ameco.fetch), ("vat_gap", vatgap.fetch), ("ageing_report", ageing.fetch), ("istat_irregular_work", istat.fetch), ("istat_internal_migration", istat_migration.fetch), ("istat_fpa", istat_fpa.fetch), ("eqi", eqi.fetch), ("cpt", cpt.fetch), ("euklems", euklems.fetch), ("bready_italy", bready.fetch), ("cohesion_esif", cohesion.fetch), ("procurement", procurement.fetch), ("ecb_shss", ecb_shss.fetch), ("eba_sovereign", eba_sovereign.fetch), ("gisco_nuts", gisco_nuts.fetch), ("istat_contracts", istat_contracts.fetch), ("eu_payments", eu_payments.fetch)]
    jobs += [(f"oecd_{f}", lambda force, f=f: oecd.fetch(f, force)) for f in oecd.FLOWS]
    jobs += [("oecd_regulation", oecd_regulation.fetch), ("oecd_bargaining", oecd_bargaining.fetch)]
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download cached files")
    ap.add_argument("--only", nargs="*", help="job names")
    ap.add_argument("--interactive", action="store_true", help="allow a visible browser with human help when a site blocks the headless one")
    args = ap.parse_args(argv)
    br.INTERACTIVE = args.interactive
    failures = []
    for name, job in _jobs():
        if args.only and name not in args.only:
            continue
        try:
            print(f"ok   {name:36s} {job(args.force)}")
        except Exception as exc:  # noqa: BLE001 - report all failures, don't stop at the first
            failures.append(name)
            print(f"FAIL {name:36s} {str(exc)[:200]}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
