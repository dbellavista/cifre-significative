"""Population-weighted benchmark: does Italy's gap to "the other EU27" depend on counting each country once or by population?"""
import numpy as np
import pandas as pd

from . import corr
from .data import (
    EU27_ISO3,
    ITALY,
    before_breaks,
    eu_median_ex_italy,
    load_panel,
    weighted_median_ex_italy,
    wide,
)

WINDOW = (2021, 2025)
MIN_YEARS = 3


def benchmark_gaps(panel: pd.DataFrame, indicators: list[str] | None = None, window: tuple[int, int] = WINDOW, min_years: int = MIN_YEARS) -> pd.DataFrame:
    """Italy's window mean minus the unweighted and the population-weighted median of the other members, per indicator, in cross-country sd.

    The weighted median is skipped (NaN) where `wide` has too few members in a year; both medians use the years where both exist.
    """
    panel = before_breaks(panel)  # a window mean must not mix two bases of a series (earnings from 2024)
    pop = wide(panel, "population", EU27_ISO3)
    if indicators is None:
        ind = sorted(set(panel.indicator) - {"population"})
        indicators = [i for i in ind if i not in corr.size_scaled([i])]
    rows = []
    for name in indicators:
        w = wide(panel, name, EU27_ISO3).loc[window[0]:window[1]]
        if ITALY not in w.columns or w.empty:
            continue
        um, wm = eu_median_ex_italy(w), weighted_median_ex_italy(w, pop)
        ok = w[ITALY].notna() & um.notna() & wm.notna()
        if ok.sum() < min_years:
            continue
        sd = w.drop(columns=ITALY).loc[ok].mean().std()
        if not sd > 0:
            continue
        it, u, p = w.loc[ok, ITALY].mean(), um[ok].mean(), wm[ok].mean()
        rows.append({"indicator": name, "italy": it, "median": u, "wmedian": p, "gap": (it - u) / sd, "wgap": (it - p) / sd, "n_years": int(ok.sum())})
    r = pd.DataFrame(rows)
    r["shift"] = r.wgap - r.gap
    r["flips"] = np.sign(r.gap) != np.sign(r.wgap)
    r["crosses_half_sd"] = (r.gap.abs() >= 0.5) != (r.wgap.abs() >= 0.5)
    return r.sort_values("shift", key=abs, ascending=False).reset_index(drop=True)


def summary(r: pd.DataFrame) -> dict[str, float]:
    """Counts for the notebook text: how many gaps flip sign or cross the 0.5 sd line, and the rank agreement of the two gap lists."""
    return {"n": len(r), "flips": int(r.flips.sum()), "crosses_half_sd": int(r.crosses_half_sd.sum()), "median_abs_shift": float(r["shift"].abs().median()),
            "spearman": float(r.gap.corr(r.wgap, method="spearman")), "gap_median": float(r.gap.median()), "wgap_median": float(r.wgap.median())}


def run_popweight_all(out=None) -> dict[str, pd.DataFrame]:
    from .experiments import OUT
    out = OUT if out is None else out
    out.mkdir(parents=True, exist_ok=True)
    r = benchmark_gaps(load_panel())
    res = {"p1_popweight_gaps": r, "p2_popweight_summary": pd.DataFrame([summary(r)])}
    for k, v in res.items():
        v.to_parquet(out / f"{k}.parquet")
    return res
