"""Fetch Eurostat datasets through the Statistics API (JSON-stat) and cache raw responses."""
import json
from datetime import UTC, datetime
from pathlib import Path

import requests
import truststore

from cifresignificative.catalog import Indicator
from cifresignificative.countries import GEO_ALL, START_YEAR

truststore.inject_into_ssl()  # use the OS certificate store (corporate/CA setups)

API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
RAW_DIR = Path("data/raw/eurostat")
REGIONAL_DIR = Path("data/raw/eurostat_nuts2")
REGIONAL_SBS_DIR = Path("data/raw/eurostat_nuts2_sbs")
SECTOR_DIR = Path("data/raw/eurostat_sectors")
SPENDING_DIR = Path("data/raw/eurostat_spending")
HIGHERED_DIR = Path("data/raw/eurostat_highered")
PROJECTIONS_DIR = Path("data/raw/eurostat_projections")
SECTOR64_DIR = Path("data/raw/eurostat_sectors64")
EARNINGS_DIR = Path("data/raw/eurostat_earnings")
MIGRATION_DIR = Path("data/raw/eurostat_migration")
INCOME_DIR = Path("data/raw/eurostat_income")
SELFEMP_DIR = Path("data/raw/eurostat_selfemp")
SIZE_DIR = Path("data/raw/eurostat_size")
HIGHFREQ_DIR = Path("data/raw/eurostat_highfreq")
FIRST_PERIOD = {"A": str(START_YEAR), "M": f"{START_YEAR}-01", "Q": f"{START_YEAR}-Q1"}
MANIFEST = Path("data/raw/manifest.json")


def build_params(ind: Indicator, geos: list[str] | None = GEO_ALL) -> list[tuple[str, str]]:
    params = [("format", "JSON"), ("lang", "EN"), ("sinceTimePeriod", FIRST_PERIOD[ind.freq])]
    params += [("geo", g) for g in geos or []]
    for key, value in ind.filters.items():
        params += [(key, v) for v in ([value] if isinstance(value, str) else value)]
    return params


def fetch(ind: Indicator, force: bool = False, regional: bool = False, raw_dir: Path | None = None) -> Path:
    """Download one indicator to data/raw/eurostat[_nuts2]/<name>.json and update the manifest.

    regional=True requests every geo (all NUTS levels) instead of the country list.
    """
    raw_dir = raw_dir or (REGIONAL_DIR if regional else RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{ind.name}.json"
    if out.exists() and not force:
        return out
    resp = requests.get(f"{API}/{ind.dataset}", params=build_params(ind, None if regional else GEO_ALL), timeout=300)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    _update_manifest(ind, resp.url, resp.json().get("updated"))
    return out


def _update_manifest(ind: Indicator, url: str, source_updated: str | None) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[ind.name] = {
        "source": "eurostat",
        "dataset": ind.dataset,
        "url": url,
        "source_updated": source_updated,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
