"""Italian governments and policy milestones: loading, and descriptive statistics by government.

Government statistics are *descriptive*: they weight each calendar year by the share of it spent in office and compare Italy with the median of
the other EU27 members over the same weighted period. They say nothing about causes (policy lags, external shocks, euro-area conditions).
"""
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from cifresignificative.studies.italy_productivity.analysis.data import (
    EU27_ISO3,
    ITALY,
    PROCESSED,
    long_wide,
)

TODAY = date(2026, 10, 12)  # end of the open-ended current government when the data were built


def load_governments(since: str = "1946-01-01") -> pd.DataFrame:
    g = pd.read_csv(PROCESSED.parent / "external" / "italy_governments.csv", parse_dates=["start", "end"])
    g["end_filled"] = g["end"].fillna(pd.Timestamp(TODAY))
    g["label"] = g["government"]
    return g[g.end_filled >= pd.Timestamp(since)].reset_index(drop=True)


def load_policies() -> pd.DataFrame:
    p = pd.read_csv(PROCESSED.parent / "external" / "italy_policies.csv", parse_dates=["date"])
    p["year"] = p.date.dt.year
    return p


def government_at(governments: pd.DataFrame, when: pd.Timestamp) -> str | None:
    m = governments[(governments.start <= when) & (governments.end_filled > when)]
    return None if m.empty else str(m.iloc[0].government)


def year_weights(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Share of each calendar year spent in office (index = year)."""
    out = {}
    for y in range(start.year, end.year + 1):
        y0, y1 = pd.Timestamp(y, 1, 1), pd.Timestamp(y + 1, 1, 1)
        days = (min(end, y1) - max(start, y0)).days
        if days > 0:
            out[y] = days / (y1 - y0).days
    return pd.Series(out)


def weighted_mean(w: pd.DataFrame | pd.Series, weights: pd.Series) -> pd.Series | float:
    idx = weights.index.intersection(w.index)
    ww = weights.loc[idx]
    if isinstance(w, pd.Series):
        v = w.loc[idx]
        ok = v.notna()
        return float((v[ok] * ww[ok]).sum() / ww[ok].sum()) if ok.sum() and ww[ok].sum() >= 0.5 else np.nan
    v = w.loc[idx]
    num = v.mul(ww, axis=0).sum(skipna=True)
    den = v.notna().mul(ww, axis=0).sum()
    return (num / den).where(den >= 0.5)


def interpolate_at(w: pd.DataFrame | pd.Series, when: pd.Timestamp) -> pd.Series | float:
    """Value at a date from year-end annual stocks (debt): linear between the previous and the current year-end."""
    frac = (when - pd.Timestamp(when.year, 1, 1)).days / 365.25
    prev, cur = w.loc[when.year - 1] if when.year - 1 in w.index else np.nan, w.loc[when.year] if when.year in w.index else np.nan
    return prev + (cur - prev) * frac


def merge_short_governments(governments: pd.DataFrame, min_months: float = 6.0) -> pd.DataFrame:
    """Fold a government that lasted under `min_months` into the one just before it when both have the same prime minister (D'Alema II, 4 months, into D'Alema I).

    A government with under half a year of data has no yearly-weighted statistic (`weighted_mean` returns NaN), so on its own it would drop out of the comparison.
    """
    rows: list[dict] = []
    for g in governments.to_dict("records"):
        months = (g["end_filled"] - g["start"]).days / 30.44
        if rows and months < min_months and rows[-1]["pm"] == g["pm"] and rows[-1]["end_filled"] == g["start"]:
            rows[-1].update(end=g["end"], end_filled=g["end_filled"], government=f"{rows[-1]['government']}-{g['government'].split()[-1]}")
        else:
            rows.append(g)
    return pd.DataFrame(rows)


def government_stats(lh: pd.DataFrame, governments: pd.DataFrame, since: str = "1960-01-01") -> pd.DataFrame:
    """One row per government (a short one is folded into the previous government of the same prime minister): months in office, Italy vs median of the other EU27 for growth, unemployment, investment and debt change."""
    governments = merge_short_governments(governments)
    gdp = long_wide(lh, "gdp_real_ncu_bn")
    pc = long_wide(lh, "gdp_real_per_head_ncu")
    metrics = {"gdp_growth_pa": gdp.pct_change(fill_method=None) * 100, "gdp_per_head_growth_pa": pc.pct_change(fill_method=None) * 100,
               "unemployment_rate": long_wide(lh, "unemployment_rate"), "gfcf_pct_gdp": long_wide(lh, "gfcf_pct_gdp")}
    debt = long_wide(lh, "gov_debt_pct_gdp")
    others = [c for c in EU27_ISO3 if c != ITALY and c in gdp.columns]
    rows = []
    for g in governments.to_dict("records"):
        start, end = g["start"], g["end_filled"]
        if end < pd.Timestamp(since) or start.year > gdp.index.max():
            continue
        w = year_weights(start, end)
        row = {"government": g["government"], "pm": g["pm"], "start": start.date(), "end": (g["end"].date() if pd.notna(g["end"]) else None), "months": round((end - start).days / 30.44, 1)}
        for name, tab in metrics.items():
            it = weighted_mean(tab[ITALY], w) if ITALY in tab else np.nan
            med = cast(pd.Series, weighted_mean(tab[others], w)).median()
            row[f"{name}_italy"], row[f"{name}_others"] = it, med
        if end.year <= debt.index.max() and start.year - 1 >= debt.index.min():
            di = interpolate_at(debt[ITALY], end) - interpolate_at(debt[ITALY], start)
            do = cast(pd.Series, interpolate_at(debt[others], end) - interpolate_at(debt[others], start)).median()
            row["debt_change_pp_italy"], row["debt_change_pp_others"] = di, do
        rows.append(row)
    return pd.DataFrame(rows)
