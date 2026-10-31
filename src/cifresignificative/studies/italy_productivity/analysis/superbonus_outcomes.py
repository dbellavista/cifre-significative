"""Did the regions that used the Superbonus more change differently? ENEA's investment per inhabitant by region (`superbonus_regions.region_table`, end-2023 snapshot) against the change of the notebook 34 regional outcomes (Eurostat NUTS2, summed or population-weighted for Trentino-Alto Adige).

Descriptive, 20 regions. One pre-specified test: Spearman of the investment per inhabitant against the change of the employment rate 20-64 between 2019 and 2023 (the outcome that tracks governance in notebook 34), permutation p over regions. Placebos: the same outcome over the four years before (2015-2019), which the credit cannot have affected, and the
test with the investment shuffled within the three zones. Robustness, not a family: the other five notebook 34 outcomes (BH over the six x two periods), the investment per euro of 2019 GDP, and the rank correlation beyond the 2019 income level and the 2015-19 change. Never causal: the credit followed the housing stock, ownership and the ability to advance the cost, and regions that were recovering from 2020 were also the ones that renovated.
"""
import numpy as np
import pandas as pd
from scipy import stats

from cifresignificative.studies.italy_productivity.analysis.regional_south import (
    OUTCOMES,
)
from cifresignificative.studies.italy_productivity.analysis.superbonus_monthly import (
    _bh,
)
from cifresignificative.studies.italy_productivity.analysis.superbonus_regions import (
    rank_test,
)

PRIMARY = "employment_rate_20_64"
WINDOWS = {"after": (2019, 2023), "before": (2015, 2019)}


LOGGED = {"gdp_pps_pc_eu27", "gdp_per_hour_eur", "hours_per_inhabitant"}  # changes in log points x 100; the rates change in percentage points


def outcome_levels(t: pd.DataFrame, regional: pd.DataFrame, years: tuple[int, ...] = (2015, 2019, 2023)) -> pd.DataFrame:
    """Long table (region, indicator, year, value) of the outcomes for the 20 regions. Trentino-Alto Adige's two NUTS2 regions: GDP, hours and population are summed (then output per hour and hours per inhabitant derived as in `regional_corr`), the rates and the income index are averaged with the population weights of the same year."""
    s = lambda ind: regional[regional.indicator.eq(f"nuts2_{ind}") & regional.year.isin(years)].set_index(["geo", "year"]).value
    pop = s("population")
    base = {i: s(i) for i in ("gdp_mio_eur", "hours_ths", "population", *[o for o in OUTCOMES if o not in ("gdp_per_hour_eur", "hours_per_inhabitant")])}
    rows = []
    for region, nuts2 in zip(t.region, t.nuts2, strict=True):
        codes = nuts2.split("+")
        for y in years:
            tot = {i: sum(base[i].get((c, y), np.nan) for c in codes) for i in ("gdp_mio_eur", "hours_ths", "population")}
            w = [pop.get((c, y), np.nan) for c in codes]
            v = {i: float(np.average([base[i].get((c, y), np.nan) for c in codes], weights=w)) for i in OUTCOMES if i in base}
            v["gdp_per_hour_eur"] = tot["gdp_mio_eur"] / tot["hours_ths"] * 1e3
            v["hours_per_inhabitant"] = tot["hours_ths"] * 1e3 / tot["population"]
            rows += [{"region": region, "indicator": i, "year": y, "value": v[i]} for i in OUTCOMES]
    return pd.DataFrame(rows)


def outcome_changes(levels: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per region, columns `<indicator>|after` and `<indicator>|before` (change over the windows, see `LOGGED`) and `<indicator>|2019` (the baseline level)."""
    p = levels.pivot_table(index="region", columns=["indicator", "year"], values="value")
    out = pd.DataFrame(index=p.index)
    for ind in OUTCOMES:
        for w, (a, b) in WINDOWS.items():
            out[f"{ind}|{w}"] = 100 * np.log(p[(ind, b)] / p[(ind, a)]) if ind in LOGGED else p[(ind, b)] - p[(ind, a)]
        out[f"{ind}|2019"] = p[(ind, 2019)]
    return out


def _perm_within(x: np.ndarray, y: np.ndarray, zone: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    """Pooled Spearman rho with a two-sided p from shuffling x inside each zone."""
    rho = float(stats.spearmanr(x, y)[0])
    null = np.empty(n_perm)
    for i in range(n_perm):
        xs = x.copy()
        for z in np.unique(zone):
            m = zone == z
            xs[m] = rng.permutation(x[m])
        null[i] = stats.spearmanr(xs, y)[0]
    return rho, float((1 + (np.abs(null) >= abs(rho) - 1e-12).sum()) / (1 + n_perm))


def partial_rank(x: pd.Series, y: pd.Series, controls: pd.DataFrame, n_perm: int = 20000, seed: int = 0) -> dict:
    """Spearman of the residuals of the ranks of x and y after a linear fit on the ranks of the controls; permutation p shuffles the residuals of x."""
    r = lambda s: s.rank().to_numpy(float)
    Z = np.column_stack([np.ones(len(x))] + [r(controls[c]) for c in controls])
    res = lambda v: r(v) - Z @ np.linalg.lstsq(Z, r(v), rcond=None)[0]
    ex, ey = res(x), res(y)
    rho = float(np.corrcoef(ex, ey)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rng.permutation(ex), ey)[0, 1] for _ in range(n_perm)])
    return {"rho": rho, "p": float((1 + (np.abs(null) >= abs(rho) - 1e-12).sum()) / (1 + n_perm))}


def link(t: pd.DataFrame, ch: pd.DataFrame, x: str = "per_inhabitant_eur", n_perm: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Every outcome x window: Spearman rho of x against the change over the 20 regions, permutation p over regions, permutation p within zones; BH q over the twelve tests; `primary` marks the pre-specified one."""
    d = t.set_index("region")
    rows, rng = [], np.random.default_rng(seed)
    for ind in OUTCOMES:
        for w in WINDOWS:
            y = ch[f"{ind}|{w}"]
            k = pd.concat([d[x], y], axis=1).dropna()
            tt = pd.DataFrame({"x": k.iloc[:, 0], "y": k.iloc[:, 1]})
            r = rank_test(tt, "x", "y", n_perm=n_perm, seed=seed)
            _, pz = _perm_within(tt.x.to_numpy(), tt.y.to_numpy(), d.zone.reindex(tt.index).to_numpy(), max(n_perm // 10, 1000), rng)
            rows.append({"outcome": ind, "window": w, "n": r["n"], "rho": r["rho"], "p": r["p"], "p_within_zone": pz, "primary": ind == PRIMARY and w == "after"})
    out = pd.DataFrame(rows)
    out["q"] = _bh(out.p.to_numpy())
    return out


def robustness(t: pd.DataFrame, ch: pd.DataFrame, n_perm: int = 20000, seed: int = 0) -> pd.DataFrame:
    """For the primary outcome: (a) investment per euro of 2019 GDP instead of per inhabitant; (b) beyond the 2019 income level; (c) beyond the 2019 income level and the 2015-19 change (the pre-trend); (d) 2019-2024 window is not available for the investment snapshot, so the end-2025 snapshot (whole programme) is `t25` in the notebook."""
    d = t.set_index("region").copy()
    d["per_gdp"] = d.investment_meur / d.gdp_meur
    y = ch[f"{PRIMARY}|after"]
    inc = pd.DataFrame({"gdp_per_inhabitant_eur": d.gdp_per_inhabitant_eur})
    both = inc.assign(pre=ch[f"{PRIMARY}|before"])
    rows = [{"variant": "per euro of GDP", **{k: v for k, v in rank_test(pd.DataFrame({"x": d.per_gdp, "y": y}), "x", "y", n_perm=n_perm, seed=seed).items() if k in ("rho", "p", "n")}}]
    rows.append({"variant": "beyond 2021 GDP per inhabitant", "n": len(d), **partial_rank(d.per_inhabitant_eur, y, inc, n_perm, seed)})
    rows.append({"variant": "beyond income and the 2015-19 change", "n": len(d), **partial_rank(d.per_inhabitant_eur, y, both, n_perm, seed)})
    return pd.DataFrame(rows)


def placebo(t: pd.DataFrame, ch: pd.DataFrame, x: str = "per_inhabitant_eur", n_shuffle: int = 300, n_perm: int = 1000, seed: int = 0) -> dict:
    """Calibration of the 12-test family: the investment is shuffled across regions, every test rerun (permutation p from `n_perm` permutations of the ranks, shared across tests), BH over the family. Returns the share of shuffles with any discovery at q < 0.05 and the mean number of discoveries; a calibrated family gives about 5% and about 0."""
    rng = np.random.default_rng(seed)
    d = t.set_index("region")
    Y = np.column_stack([ch[f"{i}|{w}"].rank().to_numpy(float) for i in OUTCOMES for w in WINDOWS])
    Y = (Y - Y.mean(0)) / np.sqrt(((Y - Y.mean(0)) ** 2).sum(0))
    xr = d[x].rank().to_numpy(float)
    perms = np.array([rng.permutation(len(xr)) for _ in range(n_perm)])
    hits = []
    for _ in range(n_shuffle):
        r = rng.permutation(xr); r = (r - r.mean()) / np.sqrt(((r - r.mean()) ** 2).sum())
        obs = np.abs(r @ Y)
        null = np.abs(r[perms] @ Y)  # n_perm x tests
        p = (1 + (null >= obs - 1e-12).sum(0)) / (1 + n_perm)
        hits.append(int((_bh(p) < 0.05).sum()))
    return {"n_shuffle": n_shuffle, "share_with_discovery": float(np.mean(np.array(hits) > 0)), "mean_discoveries": float(np.mean(hits))}
