"""Income distribution, Italy vs the EU27 (EU-SILC via Eurostat): shares by quintile and decile, S80/S20, Ginis before and after transfers.

Descriptive and cross-country: Italy is ranked among the 27 members and judged by a placebo ranking (every country's gap to the median of the others, not a test
with Italy as a cluster). The survey year is Eurostat's reference year of the survey; incomes refer mostly to the previous calendar year. Breaks (flag `b`) and
provisional values (`p`) are dropped from window means. Redistribution is the proportional fall of the Gini from before transfers to disposable income.
"""
import numpy as np
import pandas as pd

from cifresignificative.studies.italy_productivity.analysis.data import (
    EU27_ISO3,
    ITALY,
    PROCESSED,
)

DROP_FLAGS = ("b", "p")
WINDOWS = {"2005-09": (2005, 2009), "2015-19": (2015, 2019), "2021-25": (2021, 2025)}
QUINTILES = ["QU1", "QU2", "QU3", "QU4", "QU5"]


def load_income(path=PROCESSED / "income_dist.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def wide(inc: pd.DataFrame, drop_flags=DROP_FLAGS, countries=EU27_ISO3) -> pd.DataFrame:
    """(iso3, year) x measure: income shares as `QU1`..`QU5`, `D1`, `D10`, plus `s80_s20`, `gini_disp`, `gini_pre_pens_incl`, `gini_pre_pens_excl` and the derived `redistribution_incl` / `redistribution_excl` (% fall of the Gini)."""
    d = inc[inc.iso3.isin(countries) & ~inc.flag.isin(drop_flags)]
    d = d.assign(measure=d.quant_inc.fillna(d.table))
    W = d.pivot_table(index=["iso3", "year"], columns="measure", values="value")
    for tag in ("incl", "excl"):
        W[f"redistribution_{tag}"] = (1 - W.gini_disp / W[f"gini_pre_pens_{tag}"]) * 100
    return W


def window_table(W: pd.DataFrame, window: tuple[int, int], min_years: int = 2) -> pd.DataFrame:
    """Country x measure means over the window (countries with fewer than `min_years` observed years of a measure get NaN)."""
    d = W[W.index.get_level_values("year").isin(range(window[0], window[1] + 1))]
    return d.groupby(level="iso3").mean().where(d.groupby(level="iso3").count() >= min_years)


def placebo_gap(X: pd.Series, country: str = ITALY) -> dict:
    """Gap of `country` to the median of the other countries, ranked against every country's own gap to the median of the rest (1 = largest gap).
    `p_abs` = share of countries whose absolute gap is at least as large as the country's (a placebo ranking, not a test with one cluster)."""
    x = X.dropna()
    if country not in x.index:
        return {}
    gaps = pd.Series({c: x[c] - x.drop(c).median() for c in x.index})
    g = gaps[country]
    return {"value": float(x[country]), "median_others": float(x.drop(country).median()), "gap": float(g), "rank_from_top": int((x > x[country]).sum() + 1), "n_countries": len(x),
            "p_abs": float((gaps.abs() >= abs(g) - 1e-12).mean())}


def italy_table(W: pd.DataFrame, windows: dict[str, tuple[int, int]] = WINDOWS, measures=None) -> pd.DataFrame:
    """One row per window and measure: Italy's value, the median of the others, the gap, Italy's rank (1 = highest value) and the placebo share `p_abs`."""
    rows = []
    for name, w in windows.items():
        T = window_table(W, w)
        for m in measures or T.columns:
            rows.append({"window": name, "measure": m, **placebo_gap(T[m])})
    return pd.DataFrame(rows)


def change_table(W: pd.DataFrame, start: tuple[int, int], end: tuple[int, int], measures=None) -> pd.DataFrame:
    """Same as `italy_table` for the change in the window mean from `start` to `end`: Italy's change against every country's change."""
    ch = window_table(W, end) - window_table(W, start)
    return pd.DataFrame([{"measure": m, **placebo_gap(ch[m])} for m in measures or ch.columns])


def shape_gap(W: pd.DataFrame, window: tuple[int, int], country: str = ITALY) -> pd.DataFrame:
    """Quintile shares of `country` minus the median of the others (the gaps sum to about zero): where in the distribution the country differs."""
    T = window_table(W, window)[QUINTILES].dropna()
    return pd.DataFrame({"italy": T.loc[country], "median_others": T.drop(country).median()}).assign(gap=lambda d: d.italy - d.median_others)


def shuffled_null(W: pd.DataFrame, window: tuple[int, int], n_rep: int = 500, seed: int = 0, measure: str = "s80_s20") -> dict:
    """Calibration: with country labels shuffled, how often does a random country's gap rank among the top `k` by absolute size? By construction ~k/n; reports the share of random countries with p_abs < 0.05 (expected ~0.05 or below)."""
    x = window_table(W, window)[measure].dropna()
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_rep):
        xs = pd.Series(rng.permutation(x.to_numpy()), index=x.index)
        hits += placebo_gap(xs, x.index[rng.integers(len(x))]).get("p_abs", 1) < 0.05
    return {"share_p_below_005": hits / n_rep, "n_rep": n_rep}
