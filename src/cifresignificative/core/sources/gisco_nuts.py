"""GISCO NUTS level-2 boundaries (Eurostat, 20M generalisation, EPSG:3035) for the 2010, 2013, 2016, 2021 and 2024 classifications (2021 is the harmonisation target; 2010 and 2013 are older codes in the Eurostat history, 2024 holds the newest codes).

Used for the NUTS crosswalk (area overlap of an old region with the 2021 regions) and for the neighbour structure of the spatial checks. Boundaries are generalised: areas and contiguity are approximate, shared borders are matched with a small tolerance.
"""
from pathlib import Path

import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

YEARS = (2010, 2013, 2016, 2021, 2024)
URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_{year}_3035_LEVL_2.geojson"
RAW_DIR = Path("data/raw/gisco_nuts")


def path(year: int) -> Path:
    return RAW_DIR / f"NUTS_RG_20M_{year}_3035_LEVL_2.geojson"


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        if path(year).exists() and not force:
            continue
        resp = requests.get(URL.format(year=year), timeout=120)
        resp.raise_for_status()
        if b'"FeatureCollection"' not in resp.content[:200]:
            raise RuntimeError(f"GISCO NUTS {year}: not the expected GeoJSON: {resp.content[:80]!r}")
        path(year).write_bytes(resp.content)
        record(f"gisco_nuts_{year}", source="gisco-services.ec.europa.eu", dataset="NUTS_RG_20M level 2", url=URL.format(year=year))
    return RAW_DIR
