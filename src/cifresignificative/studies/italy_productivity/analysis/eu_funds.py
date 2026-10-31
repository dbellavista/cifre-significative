"""EU cohesion payments and regional outcomes (the "why the South" family, Track A style): how much support a region received in the 2007-2013 programme against how its income and labour market changed afterwards.

Predictors (per region, time-invariant): cumulative payments of the ERDF, the Cohesion Fund and the ESF of the 2007-2013 programme per inhabitant and as a share of regional GDP over 2007-2013 (`sources/eu_payments.py`).
Outcomes: changes between the window means 2005-2007 (before the programme) and 2015-2019 of log GDP per head (PPS), the employment rate, the unemployment rate, GDP per hour and hours per inhabitant. Everything is demeaned by country (only differences between
regions of one country count) and purged of the initial log income, because funds are allocated to the poorer regions: what is asked is whether, among regions of one country with the same starting income, those with more support moved differently.
Spearman with the within-country permutation p, BH over the family, the leave-one-country-out range, spatial filtering of every column (`analysis.spatial`), Italy's slope against the other countries' (placebo ranking) and a shuffled null. Descriptive: funds go to
regions for reasons that also drive their outcomes, regions are not independent, and payments are the Commission's, not the spending on the ground. Never causal.
"""
import pandas as pd

from cifresignificative.studies.italy_productivity.analysis import corr
from cifresignificative.studies.italy_productivity.analysis import regional_corr as rc

EARLY, RECENT = (2005, 2007), (2015, 2019)
OUTCOMES = [rc.CONTROL, "employment_rate_20_64", "unemployment_rate", "gdp_per_hour_eur", "hours_per_inhabitant"]
XS = ["funds_cohesion_per_head", "funds_cohesion_pct_gdp"]
EU15 = {"AT", "BE", "DE", "DK", "EL", "ES", "FI", "FR", "IE", "IT", "LU", "NL", "PT", "SE"}  # the member states before 2004 (the United Kingdom is not in the sample)


def predictors(reg21: pd.DataFrame, funds: pd.DataFrame, years: tuple[int, int] = (2007, 2013)) -> pd.DataFrame:
    """Per region of the NUTS 2021 table: cumulative cohesion payments per inhabitant (EUR, mean population of the programme years) and as a share of the programme years' GDP (%), and all funds (with the EAFRD) per inhabitant."""
    r = reg21[reg21.year.between(*years)]
    pop = r[r.indicator == "nuts2_population"].groupby("geo").value.mean()
    gdp = r[r.indicator == "nuts2_gdp_mio_eur"].groupby("geo").value.sum()
    out = pd.DataFrame({"funds_cohesion_per_head": funds.cohesion / pop.reindex(funds.index), "funds_cohesion_pct_gdp": funds.cohesion / 1e6 / gdp.reindex(funds.index) * 100,
                        "funds_all_per_head": funds.all_funds / pop.reindex(funds.index)})
    return out.dropna(how="all")


def build_table(reg21: pd.DataFrame, funds: pd.DataFrame, income_control: bool = True, early=EARLY, recent=RECENT) -> pd.DataFrame:
    """Region x variable table: the funds predictors and the outcome changes, demeaned by country and (optionally) purged of the initial log income."""
    Rb = rc.regional_matrix(reg21, drop_flag_chars="ubp")
    ch = corr.cross_section(Rb, "change", recent, early, entity_level=rc.GEO)[OUTCOMES]
    T = ch.join(predictors(reg21, funds), how="inner").join(corr.cross_section(Rb, "level", early, entity_level=rc.GEO)[[rc.CONTROL]].rename(columns={rc.CONTROL: "initial_income"}))
    W = rc.country_demean(T)
    return corr.residualise(W, "initial_income") if income_control else W.drop(columns="initial_income")
