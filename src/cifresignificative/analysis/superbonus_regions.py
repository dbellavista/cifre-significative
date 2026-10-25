"""Who received the Superbonus: ENEA's investments for completed works by region (`sources/enea_regions.py`) per inhabitant against regional income, and the shares of the North, Centre and South against their population shares.

Descriptive, 20 regions (Trentino-Alto Adige is one region: its two provinces are summed). Investments are the SuperEcobonus only, cumulative from July 2020 to the snapshot (end 2023: the years 2020-23 of the deficit story; end 2025: the whole programme), by end date of the works.
Income is the regional GDP per inhabitant of Eurostat (NUTS2, workplace based), a proxy of the region's economic level, not the income of the households that used the credits. One pre-specified test (Spearman of the investment per inhabitant of the end-2023 snapshot against GDP per inhabitant, permutation p over regions);
the end-2025 snapshot and the condominium share are shown as robustness, not as a family. Never causal: the credit followed the works, which followed the housing stock, ownership and the ability to advance the cost.
"""
import numpy as np
import pandas as pd
from scipy import stats

ZONE = {"ITC": "North", "ITH": "North", "ITI": "Centre", "ITF": "South and Islands", "ITG": "South and Islands"}


def region_table(enea: pd.DataFrame, regional: pd.DataFrame, snapshot: str, year: int = 2021) -> pd.DataFrame:
    """One row per region: investment (EUR million), per inhabitant (EUR), share of the national total, population share, GDP per inhabitant (EUR, `year`), condominium share of the region's investment, zone. Population and GDP are Eurostat NUTS2 (`regional`), summed for Trentino-Alto Adige."""
    e = enea[(enea.snapshot == pd.Timestamp(snapshot)) & enea.region.ne("Italia")].copy()
    def agg(ind: str) -> pd.Series:
        s = regional[(regional.indicator == ind) & regional.year.eq(year)].set_index("geo").value
        return e.nuts2.map(lambda codes: sum(s.get(c, np.nan) for c in codes.split("+")))
    pop, gdp = agg("nuts2_population"), agg("nuts2_gdp_mio_eur")
    out = pd.DataFrame({"region": e.region, "nuts2": e.nuts2, "zone": e.nuts2.map(lambda c: ZONE[c[:3]]), "investment_meur": e.total / 1e6, "population": pop, "gdp_meur": gdp, "condominium_share": e.condominiums / e.total})
    out["per_inhabitant_eur"] = out.investment_meur * 1e6 / out.population
    out["share_of_investment"] = out.investment_meur / out.investment_meur.sum()
    out["share_of_population"] = out.population / out.population.sum()
    out["gdp_per_inhabitant_eur"] = out.gdp_meur * 1e6 / out.population
    return out.reset_index(drop=True)


def zone_shares(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("zone").agg(investment_meur=("investment_meur", "sum"), population=("population", "sum"), gdp_meur=("gdp_meur", "sum"))
    g["share_of_investment"] = g.investment_meur / g.investment_meur.sum()
    g["share_of_population"] = g.population / g.population.sum()
    g["per_inhabitant_eur"] = g.investment_meur * 1e6 / g.population
    g["ratio_of_shares"] = g.share_of_investment / g.share_of_population
    return g.reindex(["North", "Centre", "South and Islands"])


def rank_test(t: pd.DataFrame, x: str = "gdp_per_inhabitant_eur", y: str = "per_inhabitant_eur", n_perm: int = 20000, seed: int = 0) -> dict:
    """Spearman rho and a two-sided permutation p over the regions."""
    d = t[[x, y]].dropna()
    rho = float(stats.spearmanr(d[x], d[y])[0])
    rng = np.random.default_rng(seed)
    rx = d[x].rank().to_numpy()
    ry = d[y].rank().to_numpy()
    null = np.array([np.corrcoef(rng.permutation(rx), ry)[0, 1] for _ in range(n_perm)])
    return {"x": x, "y": y, "n": len(d), "rho": rho, "p": float((1 + (np.abs(null) >= abs(rho) - 1e-12).sum()) / (1 + n_perm))}
