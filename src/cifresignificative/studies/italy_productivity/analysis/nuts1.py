"""NUTS1 aggregates of the NUTS2 regional table, for the countries whose quality-of-government index exists only at NUTS1 (Germany: 16 Laender, Belgium: 3 regions).

Extensive indicators (population, GDP, hours, investment, compensation) are summed over the NUTS2 children; every other indicator (rates, GDP per head in PPS, ...) is the population-weighted mean of the children. A NUTS1 value needs all its NUTS 2021
children in that year (and the population of each, for the weighted ones); the flag is the union of the children's flag characters. Population weights are an approximation for rates defined on a sub-population (employment rate 20-64, unemployment
rate on the active population), so the NUTS1 rates are close to, not equal to, Eurostat's own NUTS1 figures.
"""
import pandas as pd

from cifresignificative.core.clean import nuts

EXTENSIVE = {"nuts2_population", "nuts2_gdp_mio_eur", "nuts2_hours_ths", "nuts2_gfcf_mio_eur", "nuts2_coe_mio_eur"}


def to_nuts1(reg21: pd.DataFrame, countries: tuple[str, ...] = ("DE", "BE")) -> pd.DataFrame:
    """The regional long table at NUTS1 for the given countries (columns geo, year, value, flag, indicator, theme, source, country)."""
    kids = pd.Series([c for c in nuts.regions(nuts.TARGET) if c[:2] in countries])
    n_kids = kids.groupby(kids.str[:3]).size()
    r = reg21[reg21.geo.isin(set(kids))].copy()
    r["parent"] = r.geo.str[:3]
    pop = r[r.indicator == "nuts2_population"].set_index(["geo", "year"]).value
    r["pop"] = [pop.get((g, y)) for g, y in zip(r.geo, r.year, strict=True)]
    rows = []
    for (ind, parent, year), g in r.groupby(["indicator", "parent", "year"]):
        g = g.dropna(subset=["value"])
        if len(g) != n_kids[parent]:
            continue
        flag = "".join(sorted(set("".join(g.flag.dropna()))))
        if ind in EXTENSIVE:
            v = g.value.sum()
        elif g["pop"].notna().all():
            v = (g.value * g["pop"]).sum() / g["pop"].sum()
        else:
            continue
        rows.append({"geo": parent, "year": year, "value": v, "flag": flag or None, "indicator": ind, "theme": g.theme.iloc[0], "source": "aggregated from NUTS2", "country": parent[:2]})
    return pd.DataFrame(rows)
