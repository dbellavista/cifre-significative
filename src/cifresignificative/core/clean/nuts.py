"""NUTS 2013/2016 -> 2021 crosswalk from GISCO boundaries: area overlap of every old NUTS2 region with the 2021 regions.

Classes of an old code: `same` (also a 2021 code, the region kept its code), `nested` (at least `NEST` of the old area that lies inside any 2021 region is inside one of them: renamed or merged into it),
`split` (spread over several 2021 regions), `legacy` (NUTS 2006 codes mapped by hand, `LEGACY_2006`) and `later` (a code that exists only in the 2024 classification, so it cannot be mapped back to 2021). Splits are never allocated: no value is invented for a new region.
The shares are taken relative to the old area covered by 2021 regions at all, because the generalised boundaries leave slivers of coast uncovered (Corsica would otherwise look split at 93%).
`code_drift` lists codes whose area changed between classifications; on the 20M boundaries it is dominated by generalisation noise on tiny regions (Ceuta, Aland, Malta, Bremen), so read it only for large regions.
"""
import json
from itertools import pairwise

import pandas as pd
from shapely.geometry import shape
from shapely.validation import make_valid

from cifresignificative.core.sources import gisco_nuts

# NUTS 2006 codes (older than the GISCO files): Istat's ITD/ITE North-East and Centre, and Finland. Renames are 1:1 by name (ITD1 Bolzano-Bozen -> ITH1 ...); FI18 Etelä-Suomi was divided into FI1B and FI1C in 2010, so it is not mapped.
LEGACY_2006 = {"ITD1": "ITH1", "ITD2": "ITH2", "ITD3": "ITH3", "ITD4": "ITH4", "ITD5": "ITH5", "ITE1": "ITI1", "ITE2": "ITI2", "ITE3": "ITI3", "ITE4": "ITI4", "FI1A": "FI1D", "FI18": None}
NEST = 0.95  # share of the old region's (covered) area inside a single 2021 region for a rename or merge
TARGET = 2021


def regions(year: int) -> dict:
    """NUTS_ID -> shapely geometry (EPSG:3035, metres) of the level-2 regions of a classification year."""
    feats = json.loads(gisco_nuts.path(year).read_text())["features"]
    return {f["properties"]["NUTS_ID"]: make_valid(shape(f["geometry"])) for f in feats}


def overlaps(old_year: int, target: int = TARGET) -> pd.DataFrame:
    """One row per (old, new) pair that intersects with positive area: share of the old region and of the new region covered."""
    old, new = regions(old_year), regions(target)
    rows = []
    for o, go in old.items():
        for n, gn in new.items():
            if o[:2] != n[:2] and not (o[:2] in ("EL", "UK") and n[:2] in ("EL", "UK")):  # codes keep their country prefix (EL = Greece throughout)
                continue
            a = go.intersection(gn).area
            if a > 0:
                rows.append((o, n, a / go.area, a / gn.area))
    return pd.DataFrame(rows, columns=["old", "new", "share_of_old", "share_of_new"])


def crosswalk(old_years=(2016, 2013, 2010), target: int = TARGET, later: int = 2024) -> pd.DataFrame:
    """Old code -> 2021 code with the class of the link, taking the most recent classification that contains the code.
    Columns: old, new, kind (same, nested, split, later), share_of_old (of the covered area), share_of_new, from_year."""
    target_codes = set(regions(target))
    out = {}
    for y in old_years:
        for code, g in overlaps(y, target).groupby("old"):
            if code in out:
                continue
            best = g.loc[g.share_of_old.idxmax()]
            rel = best.share_of_old / g.share_of_old.sum()
            if code in target_codes:
                kind, best = "same", (g[g.new.eq(code)].iloc[0] if g.new.eq(code).any() else best)
            else:
                kind = "nested" if rel >= NEST else "split"
            out[code] = {"old": code, "new": best.new, "kind": kind, "share_of_old": best.share_of_old / g.share_of_old.sum(), "share_of_new": best.share_of_new, "from_year": y}
    for code, new in LEGACY_2006.items():
        out[code] = {"old": code, "new": new, "kind": "legacy" if new else "split", "share_of_old": float("nan"), "share_of_new": float("nan"), "from_year": 2006}
    for code in set(regions(later)) - target_codes - set(out):
        out[code] = {"old": code, "new": None, "kind": "later", "share_of_old": float("nan"), "share_of_new": float("nan"), "from_year": later}
    return pd.DataFrame(out.values()).sort_values("old", ignore_index=True)


def code_drift(years=gisco_nuts.YEARS, min_overlap: float = 0.9) -> pd.DataFrame:
    """Codes present in two classifications whose region changed shape: share of the older region lying inside the same-coded newer one, below `min_overlap`."""
    rows = []
    for a, b in pairwise(years):
        ra, rb = regions(a), regions(b)
        rows += [(c, a, b, ra[c].intersection(rb[c]).area / ra[c].area) for c in ra.keys() & rb.keys()]
    d = pd.DataFrame(rows, columns=["code", "from_year", "to_year", "share_inside"])
    return d[d.share_inside < min_overlap].sort_values("share_inside", ignore_index=True)


def harmonise(df: pd.DataFrame, cw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Regional long table with every region code translated to NUTS 2021.

    Codes kept in 2021 stay (`geo_method` native); renamed or merged ones (kind nested, legacy) take their 2021 code (`renamed`, original code in `geo_old`); splits, codes that exist only after 2021 and extra-regio codes
    are left out, so no value is invented for a region. Eurostat publishes both the old and the new code of a region in the transition years: the native row wins, then the code of the newest classification (`from_year`).
    The data hold no true merger (two old codes feeding one new region in the same year), so nothing is summed; `assert_no_merger` fails loudly if that changes."""
    cw = crosswalk() if cw is None else cw
    link = cw[cw.kind.isin(["nested", "legacy"])].set_index("old")
    d = df.assign(geo_old=df.geo, geo_method="renamed")
    native = d.geo.isin(set(regions(TARGET)))
    d["geo"] = d.geo.map(link.new).where(~native, d.geo)
    d.loc[native, "geo_method"] = "native"
    d = d.dropna(subset=["geo"])
    d["_rank"] = d.geo_method.eq("native").map({True: 0, False: 1}) * 10000 - d.geo_old.map(link.from_year).fillna(TARGET)  # native first, then the newest vintage
    d = d.sort_values("_rank").drop_duplicates(["geo", "indicator", "year"] + [c for c in ("period",) if c in d]).drop(columns="_rank")
    if "country" in d:
        d["country"] = d.geo.str[:2]
    return d.sort_values(["indicator", "geo", "year"], ignore_index=True)


def overlap_agreement(df: pd.DataFrame, cw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Where Eurostat publishes both the old and the 2021 code of a region in the same year: how far apart the two values are (relative difference), by indicator."""
    cw = crosswalk() if cw is None else cw
    link = cw[cw.kind.isin(["nested", "legacy"])].set_index("old").new
    d = df.assign(new=df.geo.map(link)).dropna(subset=["new"])
    both = d.merge(df[["geo", "indicator", "year", "value"]], left_on=["new", "indicator", "year"], right_on=["geo", "indicator", "year"], suffixes=("_old", "_new"))
    both["rel_diff"] = (both.value_old - both.value_new).abs() / both.value_new.abs().clip(lower=1e-9)
    return both.groupby("indicator").rel_diff.agg(pairs="size", median="median", p90=lambda s: s.quantile(0.9), max="max").reset_index()


def coverage(before: pd.DataFrame, after: pd.DataFrame, windows: dict[str, tuple[int, int]], indicators: list[str]) -> pd.DataFrame:
    """Regions with at least one value in each window, per indicator, before and after harmonisation (the loss the changes screens suffered from mixed codes)."""
    def n(t: pd.DataFrame, ind: str, a: int, b: int) -> int:
        return t[(t.indicator == ind) & t.year.between(a, b)].geo.nunique()

    rows = [(ind, name, n(before, ind, a, b), n(after, ind, a, b)) for ind in indicators for name, (a, b) in windows.items()]
    return pd.DataFrame(rows, columns=["indicator", "window", "regions_before", "regions_after"])
