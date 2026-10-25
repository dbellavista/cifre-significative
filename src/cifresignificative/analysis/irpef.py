"""Declared income of the self-employed and the flat-rate regimes, from the MEF IRPEF return statistics (`sources/mef_irpef.py`, table `data/processed/mef_irpef.parquet`).

Keep in mind how the tables work: taxpayers are classed by **reddito complessivo** (ordinary IRPEF income). Income taxed under the flat-rate regimes (regime forfetario, regime di vantaggio, the older
regime dei minimi) is taxed separately and is *not* in reddito complessivo, so a flat-rate filer with no other income sits in the zero class whatever the size of the business. Distributions by class therefore
describe the ordinary-regime income; the flat-rate income is only available in aggregate (count and net income per class of reddito complessivo, or in total). Amounts are thousand euro in the source, euro here.
Classes below zero are left out of means (a 2017 anomaly and a definition change from 2018 make them non-comparable). Descriptive only.
"""
import numpy as np
import pandas as pd

GROUP_TABLES = {"TOT": ("AAPFTOT0106", "AAPFTOT0202"), "DIP": ("AAPFDIP0105", "AAPFDIP0202"), "PEN": ("AAPFPEN0105", "AAPFPEN0202"), "TIT": ("ABPFTIT0106", "ABPFTIT0203"),
                "IMP": ("ABPFIMP0106", "ABPFIMP0203"), "PRO": ("ABPFPRO0106", "ABPFPRO0203"), "AGR": ("ABPFAGR0105", "ABPFAGR0202")}
FLAT = {"forfetario": "Reddito netto (regime forfetario)", "vantaggio": "Reddito netto (regime di vantaggio)", "minimo": "Reddito minimo dichiarato"}


def _rows(d: pd.DataFrame, code: str, variable: str, measure: str, positive: bool = True) -> pd.DataFrame:
    x = d[(d.code == code) & (d.variable == variable) & (d.measure == measure) & ~d.is_total]
    return x[x.class_lo >= 0] if positive else x


def total(d: pd.DataFrame, code: str, variable: str, measure: str, positive: bool = True) -> pd.Series:
    """Sum over the income classes (by default the classes from zero up) per tax year; the TOTALE row is not used, so a lowest-class anomaly cannot enter."""
    x = _rows(d, code, variable, measure, positive)
    return x.groupby("tax_year").value.sum(min_count=1)


def taxpayers(d: pd.DataFrame, group: str, positive: bool = True) -> pd.Series:
    """Number of taxpayers of the group per tax year (from the distribution table, which covers every year since 2008)."""
    return total(d, GROUP_TABLES[group][0], "taxpayers", "freq", positive)


def mean_income(d: pd.DataFrame, group: str, variable: str = "Reddito complessivo") -> pd.Series:
    """Euro per taxpayer: amount of `variable` over the non-negative classes / taxpayers in the same classes (the multi-variable table must have the variable that year)."""
    code = GROUP_TABLES[group][1]
    amt = total(d, code, variable, "amount_keur") * 1000
    n = total(d, code, "taxpayers", "freq")
    return (amt / n).dropna()


def share_at_least(d: pd.DataFrame, group: str, threshold: float) -> pd.Series:
    """Share of the group's taxpayers (non-negative classes) whose reddito complessivo is in a class starting at or above `threshold` euro."""
    x = d[(d.code == GROUP_TABLES[group][0]) & (d.variable == "taxpayers") & (d.measure == "freq") & ~d.is_total & (d.class_lo >= 0)]
    n = x.groupby("tax_year").value.sum()
    return (x[x.class_lo >= threshold].groupby("tax_year").value.sum() / n).dropna()


def median_income(d: pd.DataFrame, group: str) -> pd.Series:
    """Median reddito complessivo by linear interpolation within the class that contains it (all classes counted, negative ones in the lower half)."""
    x = d[(d.code == GROUP_TABLES[group][0]) & (d.variable == "taxpayers") & (d.measure == "freq") & ~d.is_total]
    out = {}
    for ty, g in x.groupby("tax_year"):
        g = g.drop_duplicates("class_label").sort_values(["class_lo", "class_hi"]).reset_index(drop=True)
        c, n = g.value.cumsum(), g.value.sum()
        i = int((c >= n / 2).idxmax())
        lo, hi, k = g.class_lo[i], g.class_hi[i], g.value[i]
        out[ty] = lo + (n / 2 - (c[i] - k)) / k * (hi - lo) if np.isfinite(lo) and np.isfinite(hi) and k else np.nan
    return pd.Series(out).sort_index()


def flat_rate(d: pd.DataFrame) -> pd.DataFrame:
    """Per tax year, for individuals with a VAT number: taxpayers (all classes), and for each flat-rate regime the filers, net income (euro) and mean net income, the share of
    VAT-number holders in the regimes, and the filers with ordinary income of at least 1,000 euro (reddito complessivo class from 1,000 up)."""
    code = GROUP_TABLES["TIT"][1]
    out = pd.DataFrame({"tit_taxpayers": total(d, code, "taxpayers", "freq", positive=False)})
    for k, var in FLAT.items():
        n, a = total(d, code, var, "freq", positive=False), total(d, code, var, "amount_keur", positive=False) * 1000
        out[f"{k}_filers"], out[f"{k}_net_income"] = n, a
        out[f"{k}_mean"] = a / n
        x = _rows(d, code, var, "freq", positive=False)
        out[f"{k}_with_other_income"] = x[x.class_lo >= 1000].groupby("tax_year").value.sum()
    flat = out[[f"{k}_filers" for k in FLAT]].sum(axis=1, min_count=1)
    out["flat_share_of_tit"] = flat / out.tit_taxpayers
    return out.dropna(how="all", axis=1)


def mean_with_flat(d: pd.DataFrame) -> pd.Series:
    """Euro per individual with a VAT number: (reddito complessivo + net income taxed under the flat-rate regimes) / taxpayers, non-negative classes. The flat regimes' net income is added at
    the aggregate level only (it cannot be placed in a class), which assumes their filers are in the taxpayer count (they are: the zero class holds most of them)."""
    code = GROUP_TABLES["TIT"][1]
    amt = total(d, code, "Reddito complessivo", "amount_keur") * 1000
    flat = sum((total(d, code, v, "amount_keur") * 1000).reindex(amt.index).fillna(0) for v in FLAT.values())
    return ((amt + flat) / total(d, code, "taxpayers", "freq")).dropna()


BANDS = (0, 25_000, 50_000, 60_000, 75_000, 100_000, 150_000, 250_000, np.inf)  # turnover bands (euro); every edge is a class boundary of the MEF turnover classification in all years


def professional_costs(dims: pd.DataFrame, bands=BANDS) -> pd.DataFrame:
    """Declared fees (`Totale compensi`) and expenses (`Totale spese`) of regular self-employed professionals in the ordinary regime (form section RE), by **VAT turnover** band and tax year, from the AE tables
    (`mef_irpef_dims.parquet`, table `*0406`; the table code changed in 2014 and the file for tax year 2022 is empty on the server). Columns: `fees_keur`, `expenses_keur`, `filers` (taxpayers with fees), `cost_share` = expenses / fees.
    Flat-rate filers file another form section and are not in these tables, so what is shown is the *selected* group that stayed in the ordinary regime."""
    a = dims[(dims.model == "AE") & (dims.group == "PPRO") & dims.code.str.endswith("0406") & dims.variable.isin(["Totale compensi", "Totale spese"]) & ~dims.is_total & (dims.class_lo >= 0)]
    p = a[a.measure.isin(["freq", "amount_keur"])].pivot_table(index=["tax_year", "class_lo", "class_hi"], columns=["variable", "measure"], values="value", observed=True).reset_index()
    p.columns = ["tax_year", "class_lo", "class_hi", "fees_keur", "fees_n", "expenses_keur", "expenses_n"]  # pivot columns are sorted: (fees, amount), (fees, freq), (expenses, amount), (expenses, freq)
    edges = np.asarray(bands, dtype=float)
    i = np.searchsorted(edges, p.class_lo.to_numpy(), side="right") - 1
    assert (p.class_hi.to_numpy() <= edges[i + 1] + 1e-9).all(), "a class straddles a band edge"
    p["band"] = [f"{int(edges[k] / 1000)}k+" if np.isinf(edges[k + 1]) else f"{int(edges[k] / 1000)}-{int(edges[k + 1] / 1000)}k" for k in i]
    p["band_order"] = i
    g = p.groupby(["tax_year", "band_order", "band"]).agg(fees_keur=("fees_keur", "sum"), expenses_keur=("expenses_keur", "sum"), filers=("fees_n", "sum"))
    return g.assign(cost_share=g.expenses_keur / g.fees_keur).reset_index()
