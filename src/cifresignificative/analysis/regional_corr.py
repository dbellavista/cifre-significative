"""Regional (NUTS2) correlation experiments: the national funnel in `corr.py` applied to regions within countries.

Regions of one country share institutions, policy and the national cycle, and neighbours are not independent, so the default
unit is the region *demeaned by its country*, robustness drops whole countries, and the permutation null shuffles regions
inside each country. Associations only, never causal.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from cifresignificative.analysis import corr
from cifresignificative.analysis.data import ITALY_AREAS

GEO = "geo"
MIN_REGIONS = 3  # a country needs this many regions with data to contribute within-country variation
NO_FLAGS = ""  # flag characters are matched one by one (flags such as "bp" or "bdu" combine several)
CONTROL = "ln_gdp_pc"
DERIVED = ["gdp_per_hour_eur", "hours_per_inhabitant", "investment_rate_pct_gdp", "labour_share_pct_gdp", "compensation_per_hour_eur"]
MECHANICAL = {frozenset(p) for p in combinations(DERIVED, 2)}  # ratios built from the same GDP / hours / population are linked by construction

# Regional analogues of the literature hypotheses: (id, theme, y, x, expected sign, label)
HYPOTHESES = [
    ("RH1", "Education & productivity", "gdp_per_hour_eur", "tertiary_attainment_25_64", 1, "Graduate share goes with output per hour"),
    ("RH2", "Innovation", "gdp_per_hour_eur", "rd_pct_gdp", 1, "R&D intensity goes with output per hour"),
    ("RH3", "Digital", "gdp_per_hour_eur", "broadband_households_pct", 1, "Broadband coverage goes with output per hour"),
    ("RH4", "Labour dualism", "employment_rate_20_64_female", "tertiary_attainment_25_64", 1, "Graduate share goes with female employment"),
    ("RH5", "Education & skills", "neet_rate_15_29", "early_leavers_18_24", 1, "Early school leaving goes with NEET"),
    ("RH6", "Migration", "net_migration_rate", "ln_gdp_pc", 1, "Net migration goes with (log) regional income per head"),
    ("RH7", "Health", "life_expectancy_birth", "physicians_per_100k", 1, "Physician density goes with life expectancy"),
    ("RH8", "Wages", "compensation_per_hour_eur", "gdp_per_hour_eur", 1, "Output per hour goes with pay per hour (shares the hours denominator)"),
    ("RH9", "Investment", "investment_rate_pct_gdp", "unemployment_rate", -1, "Slack goes with lower investment"),
    ("RH10", "Ageing", "employment_rate_55_64", "old_age_dependency", -1, "Older regions have lower old-age employment"),
]


def _strip(c: str) -> str:
    return c.removeprefix("nuts2_")


def regional_matrix(regional: pd.DataFrame, years: tuple[int, int] | None = None, drop_flag_chars: str = "u", derive: bool = True) -> pd.DataFrame:
    """(geo, year) x indicator table with the `nuts2_` prefix removed. An observation is dropped when its flag contains any
    character of `drop_flag_chars` (u = low reliability, p = provisional, b = break, e = estimated). Adds derived ratios
    (output per hour, hours per inhabitant, investment rate, labour share, pay per hour)."""
    d = regional
    if drop_flag_chars:
        has = d.flag.fillna("").map(lambda f: any(ch in f for ch in drop_flag_chars))
        d = d[~has]
    if years:
        d = d[d.year.between(*years)]
    M = d.pivot_table(index=[GEO, "year"], columns="indicator", values="value").rename(columns=_strip)
    if derive and {"gdp_mio_eur", "hours_ths", "population"} <= set(M):
        M["gdp_per_hour_eur"] = M.gdp_mio_eur / M.hours_ths * 1e3
        M["hours_per_inhabitant"] = M.hours_ths * 1e3 / M.population
        if "gfcf_mio_eur" in M:
            M["investment_rate_pct_gdp"] = M.gfcf_mio_eur / M.gdp_mio_eur * 100
        if "coe_mio_eur" in M:
            M["labour_share_pct_gdp"] = M.coe_mio_eur / M.gdp_mio_eur * 100
            M["compensation_per_hour_eur"] = M.coe_mio_eur / M.hours_ths * 1e3
    if "gdp_pps_pc_eu27" in M:
        M[CONTROL] = np.log(M.gdp_pps_pc_eu27)
    return M


def country_of(index: pd.Index) -> pd.Series:
    """Country prefix (NUTS2 code[:2]) for an index of region codes."""
    return pd.Series(index.str[:2], index=index)


def country_demean(X: pd.DataFrame, min_regions: int = MIN_REGIONS) -> pd.DataFrame:
    """Subtract each country's mean of its regions per column; countries with fewer than `min_regions` regions with data
    in a column become NaN (a country with one region would otherwise be demeaned to zero)."""
    g = country_of(X.index)
    out = X - X.groupby(g.to_numpy()).transform("mean")
    return out.where(X.groupby(g.to_numpy()).transform("count") >= min_regions)


def within_country_table(R: pd.DataFrame, kind: str, window: tuple[int, int], window0: tuple[int, int] | None = None, income_control: bool = True) -> pd.DataFrame:
    """Region x indicator table, demeaned by country, optionally with the log-GDP-per-head fit removed (what is left beyond income)."""
    X = corr.cross_section(R, kind, window, window0, entity_level=GEO)
    X = country_demean(X)
    if not income_control:
        return X
    if CONTROL in X:
        X = corr.residualise(X, CONTROL)
    return X.drop(columns=["gdp_pps_pc_eu27"], errors="ignore")  # the control itself (and its level) is not screened


def screen_regions(X: pd.DataFrame, method: str = "spearman", q: float = 0.05, n_min: int = 80, check_robust: bool = True) -> pd.DataFrame:
    """The funnel of `corr.screen` on regions: leave-one-country-out robustness, 'without Ireland' = without IE."""
    return corr.screen(X, method, q, n_min, check_robust, groups=country_of(X.index), irl_label="IE", exclusions=MECHANICAL)


def region_permutation_null(X: pd.DataFrame, **kw) -> list[int]:
    return corr.permutation_null(X.drop(columns=corr.size_scaled(X.columns)), groups=country_of(X.index), **kw)


def between_country_table(R: pd.DataFrame, window: tuple[int, int]) -> pd.DataFrame:
    """Country means of region window means (one row per country; compare with the national screen)."""
    X = corr.cross_section(R, "level", window, entity_level=GEO)
    return X.groupby(country_of(X.index).to_numpy()).mean().drop(columns=["gdp_pps_pc_eu27"], errors="ignore")


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum()) if (xc ** 2).sum() > 0 else np.nan


def italy_slope_test(Xw: pd.DataFrame, x: str, y: str, min_regions: int = 5) -> dict:
    """Does Italy's within-country slope of y on x differ from the other countries'? Italy is a single cluster, so a clustered
    interaction test would be meaningless; instead the same statistic (country slope minus the slope of all the others, on
    country-demeaned regions) is computed for every country with at least `min_regions` regions, and Italy is ranked among
    them: p = share of countries with an equal or larger |difference| (smallest possible p = 1 / number of countries).
    Also Italy's own slope and its North vs South split (exploratory, 21 regions)."""
    d = Xw[[x, y]].dropna().rename(columns={x: "x", y: "y"})
    d["country"] = country_of(d.index).to_numpy()
    sizes = d.country.value_counts()
    eligible = sizes[sizes >= min_regions].index
    if "IT" not in eligible or len(eligible) < 5:
        return {}
    diff = {}
    for c in eligible:
        a, b = d[d.country == c], d[d.country != c]
        diff[c] = _slope(a.x.to_numpy(), a.y.to_numpy()) - _slope(b.x.to_numpy(), b.y.to_numpy())
    diffs = pd.Series(diff).dropna()
    it = d[d.country == "IT"]
    out = {"x": x, "y": y, "n": len(d), "n_italy": len(it), "n_countries": len(diffs), "slope_italy": _slope(it.x.to_numpy(), it.y.to_numpy()),
           "slope_others": _slope(d[d.country != "IT"].x.to_numpy(), d[d.country != "IT"].y.to_numpy()), "slope_diff_italy": float(diffs["IT"]),
           "p_diff": float((diffs.abs() >= abs(diffs["IT"])).mean())}
    area = pd.Series([ITALY_AREAS.get(g[:3], "other") for g in it.index], index=it.index)
    for name, mask in {"north": area.isin(["North-West", "North-East"]), "south": area.isin(["South", "Islands"])}.items():
        sub = it[mask]
        out[f"slope_{name}"] = _slope(sub.x.to_numpy(), sub.y.to_numpy()) if len(sub) >= 5 else np.nan
        out[f"rho_{name}"] = float(sub.x.corr(sub.y, method="spearman")) if len(sub) >= 5 else np.nan
    return out


def convergence(R: pd.DataFrame, start: tuple[int, int] = (2005, 2007), end: tuple[int, int] = (2017, 2019), indicator: str = "gdp_pps_pc_eu27",
                drop_top_region: bool = False) -> pd.DataFrame:
    """Sigma-free beta-convergence of regional income: annualised log growth between the two windows on the initial log level,
    country fixed effects (country-demeaned), SEs clustered by country. beta < 0 = poorer regions catch up. One row for all
    countries pooled, one for Italy, and one for each of the larger countries."""
    lv = corr.cross_section(R[[indicator]], "level", start, entity_level=GEO)[indicator]
    lv_end = corr.cross_section(R[[indicator]], "level", end, entity_level=GEO)[indicator]
    d = pd.DataFrame({"ln0": np.log(lv), "growth": (np.log(lv_end) - np.log(lv)) / (end[1] - start[1]) * 100}).dropna()
    d["country"] = country_of(d.index).to_numpy()
    if drop_top_region:
        d = d[d.groupby("country").ln0.transform("max") != d.ln0]  # the richest region of each country (often the capital)
    rows = []
    for name, sub in [("all countries (pooled, country FE)", d), *[(c, d[d.country == c]) for c in ("IT", "ES", "DE", "FR", "PL")]]:
        sub = sub[sub.groupby("country").ln0.transform("count") >= MIN_REGIONS]
        if len(sub) < 8:
            continue
        sub = sub.assign(ln0=sub.ln0 - sub.groupby("country").ln0.transform("mean"), growth=sub.growth - sub.groupby("country").growth.transform("mean"))
        m = smf.ols("growth ~ ln0 - 1", sub).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.country)[0]}) if sub.country.nunique() > 1 \
            else smf.ols("growth ~ ln0 - 1", sub).fit(cov_type="HC1")
        rows.append({"sample": name, "n_regions": len(sub), "beta": float(m.params["ln0"]), "se": float(m.bse["ln0"]), "p": float(m.pvalues["ln0"]), "drop_top_region": drop_top_region})
    return pd.DataFrame(rows)
