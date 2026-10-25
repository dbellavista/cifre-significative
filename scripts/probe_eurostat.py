"""Show the dimensions and codes of Eurostat datasets before adding them to the catalog.

Usage: uv run python scripts/probe_eurostat.py DATASET [DATASET ...] [--geo ITC4] [--since 2023] [--filter unit=PC ...] [--codes DIM]

Requests one geo (default IT) from `--since` onwards, so the response stays small, and prints every dimension other than geo/time with its
number of codes and the first few labels. Add `--filter key=value` when the API answers HTTP 413 (dataset too large unfiltered), and
`--codes DIM` to list every code of one dimension. Nothing is written to disk. The catalog needs one value per dimension (or a `keep`).
"""
import argparse

import requests
import truststore

truststore.inject_into_ssl()
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def probe(dataset: str, geo: str, since: str, filters: list[str], codes: str | None, head: int = 8) -> None:
    params = [("format", "JSON"), ("lang", "EN"), ("geo", geo), ("sinceTimePeriod", since), *[(k, v) for k, _, v in (f.partition("=") for f in filters)]]
    r = requests.get(f"{API}/{dataset}", params=params, timeout=120)
    if r.status_code != 200:
        hint = " (too large: add --filter key=value)" if r.status_code == 413 else ""
        print(f"{dataset}: HTTP {r.status_code}{hint} {r.text[:120]!r}")
        return
    js = r.json()
    times = list(js["dimension"]["time"]["category"]["index"])
    print(f"{dataset} | {js.get('label', '')[:80]} | {times[0]}..{times[-1]} ({len(times)} periods) | values: {len(js.get('value', {}))}")
    for d in js["id"]:
        if d in ("geo", "time", "freq"):
            continue
        cat = js["dimension"][d]["category"]
        labels, index = cat.get("label", {}), list(cat["index"])
        shown = index if d == codes else index[:head]
        print(f"   {d} ({len(index)}): " + ", ".join(f"{c}={labels.get(c, '')[:30]}" for c in shown) + ("" if d == codes or len(index) <= head else ", ..."))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--geo", default="IT")
    ap.add_argument("--since", default="2023")
    ap.add_argument("--filter", action="append", default=[], dest="filters")
    ap.add_argument("--codes", default=None)
    a = ap.parse_args()
    for ds in a.datasets:
        probe(ds, a.geo, a.since, a.filters, a.codes)
