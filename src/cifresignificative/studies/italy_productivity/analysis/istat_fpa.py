"""Capital transfers of the Italian general government by receiving sector and by Istat edition (`sources/istat_fpa.py`), the primary counterpart of the Eurostat D.9 numbers and of the judgment-based grouping of notebook 38.

Four uses: (1) `by_receiver`: investment grants (D.92) to households, corporations and non-profit institutions and other capital transfers (D.99) to corporations and households, EUR million, one edition; (2) `reclassification`: what the March 2023 edition (the Superbonus and the facade bonus booked in full as payable credits)
changed against the last edition before it (October 2022); (3) `hearing_check`: the Superbonus-plus-facade spending that Istat drew in its hearing of March 2023, read from a chart, against the households' investment grants of the same edition; (4) `facade_split`: the households' investment grants above their 2019 level minus Istat's Superbonus effect of
the note of 26 April 2024 gives an implied annual split of the facade bonus (an assumption: all the rise of the households' investment grants is Superbonus or facade bonus). Accrual, descriptive; the receiving sector is Istat's.
"""
import pandas as pd

from cifresignificative.core.sources.istat_fpa import latest_edition

RECEIVER_ITEMS = {"d92_households": "D92_D_W0_S14", "d92_corporations": "D92_D_W0_SIM", "d92_npish": "D92_D_W0_S15", "d92_rest_of_world": "D92_D_W1_S1", "d99_corporations": "D99_D_W0_SIM", "d99_households": "D99_D_W0_S14", "d99_rest_of_world": "D99_D_W1_S1", "subsidies": "D3_D_W0"}


def edition_table(d: pd.DataFrame, edition: str, items=None) -> pd.DataFrame:
    """Year x item (EUR million) for one edition; if an edition exists in both flows the later flow (`current`) wins."""
    x = d[d.edition.eq(edition)]
    if items is not None:
        x = x[x["item"].isin(items)]
    x = x.sort_values("flow").drop_duplicates(["item", "year"], keep="last")  # "b19" < "current"
    return x.pivot(index="year", columns="item", values="value_meur")


def by_receiver(d: pd.DataFrame, edition: str, years=range(2015, 2026)) -> pd.DataFrame:
    t = edition_table(d, edition, RECEIVER_ITEMS.values()).reindex(list(years))
    out = pd.DataFrame({name: t[item] for name, item in RECEIVER_ITEMS.items() if item in t})
    out["d92_total"] = out[[c for c in out if c.startswith("d92_")]].sum(axis=1, min_count=1)
    out["d99_total"] = out[[c for c in out if c.startswith("d99_")]].sum(axis=1, min_count=1)
    return out.rename_axis("year")


def increase_since(rec: pd.DataFrame, base_year: int = 2019) -> pd.DataFrame:
    """Rise of each column over its `base_year` level."""
    return (rec - rec.loc[base_year]).drop(index=base_year)


def reclassification(d: pd.DataFrame, before: str = "2022M10", after: str = "2023M3", years=(2020, 2021), items=("D92_D_W0_S14", "D92_D_W0_SIM", "D99_D_W0_SIM", "D3_D_W0", "B9_B_W0")) -> pd.DataFrame:
    """The same year in two editions (item, year, value before, value after, difference): the March 2023 booking of the Superbonus and the facade bonus as payable credits against the last edition before it."""
    b, a = edition_table(d, before, items), edition_table(d, after, items)
    rows = [{"item": it, "year": y, "before": b.loc[y, it], "after": a.loc[y, it], "change": a.loc[y, it] - b.loc[y, it]} for it in items for y in years]
    return pd.DataFrame(rows)


def hearing_check(d: pd.DataFrame, istat: pd.DataFrame, edition: str = "2023M3", base_year: int = 2019, years=(2020, 2021, 2022)) -> pd.DataFrame:
    """Households' investment grants of `edition` above their `base_year` level against the Superbonus-plus-facade spending of the hearing of March 2023 (`superbonus_facade_spending_2023_vintage`, read from a bar chart, +-0.5 billion), EUR million."""
    hh = edition_table(d, edition, ["D92_D_W0_S14"])["D92_D_W0_S14"]
    chart = istat[istat.series.eq("superbonus_facade_spending_2023_vintage")].set_index("year").value.astype(float) * 1000
    out = pd.DataFrame({"d92_households_meur": hh.reindex(list(years)), "above_base_meur": hh.reindex(list(years)) - hh.loc[base_year], "hearing_chart_meur": chart.reindex(list(years))})
    out["difference_meur"] = out.above_base_meur - out.hearing_chart_meur
    return out.rename_axis("year").reset_index()


def facade_split(rec: pd.DataFrame, istat: pd.DataFrame, gdp_meur: pd.Series, base_year: int = 2019, years=(2020, 2021, 2022, 2023)) -> pd.DataFrame:
    """Implied annual split of the households' investment grants above their `base_year` level into the Superbonus (Istat's effect on net borrowing of the note of 26 April 2024, % of GDP times GDP, rounded to 0.1 point so +-1 billion) and the rest, taken to be the facade bonus. The assumption: the rise
    of the households' investment grants is the Superbonus and the facade bonus (the 2019 level covers the ordinary bonuses booked as grants). Compare the sum of the implied facade bonus with the 'about 23 billion for 2020-22' of the Camera dossier."""
    inc = increase_since(rec[["d92_households"]], base_year).reindex(list(years))
    sb = istat[istat.series.eq("superbonus_incidence_on_net_borrowing")].set_index("year").value.astype(float).reindex(list(years)) / 100 * gdp_meur.reindex(list(years))
    out = pd.DataFrame({"households_d92_above_base_meur": inc.d92_households, "superbonus_istat_meur": sb})
    out["implied_facade_meur"] = out.households_d92_above_base_meur - out.superbonus_istat_meur
    return out.rename_axis("year").reset_index()


def deficit_vintages(d: pd.DataFrame, targets: pd.DataFrame, gdp_meur: pd.Series, editions=("2022M10", "2023M3", "2026M10"), years=(2021,)) -> pd.DataFrame:
    """Italy's deficit (net lending B9 as a positive deficit, EUR million and % of the *latest* nominal GDP so that only the vintage of the numerator changes) in a few editions, and its surprise against the programmatic target of the budget year (`budget.targets`, % of GDP):
    the surprise with the first release before the March 2023 reclassification and with the latest edition. The difference is the accounting treatment of the building and other payable credits, not a policy or forecast miss."""
    rows = []
    for ed in editions:
        t = edition_table(d, ed, ["B9_B_W0"])["B9_B_W0"]
        for y in years:
            if y in t.index and pd.notna(t.loc[y]):
                deficit = -float(t.loc[y])
                pct = deficit / float(gdp_meur.loc[y]) * 100
                rows.append({"edition": ed, "year": y, "deficit_meur": deficit, "deficit_pct_latest_gdp": pct, "target": float(targets.loc[y, "target"]), "surprise_pp": pct - float(targets.loc[y, "target"])})
    return pd.DataFrame(rows)


REAL_TIME_ITEMS = ["OTR_C_W0", "OTE_D_W0", "D41_D_W0", "B9_B_W0"]


def first_release(d: pd.DataFrame, years, items=REAL_TIME_ITEMS) -> pd.DataFrame:
    """For each year the values of the **first edition that contains it** (the first release: the March or April of the following year, from 2020M3 for 2019; the editions start in September 2019, so earlier years have no true first release here). Index = year; `edition` says which one; EUR million."""
    rows = []
    for y in years:
        x = d[d["item"].isin(items) & d.year.eq(y)].dropna(subset=["value_meur"])
        if x.empty:
            continue
        ed = x.loc[x.published.idxmin(), "edition"]
        t = edition_table(d, ed, items)
        rows.append({"year": y, "edition": ed, **{it: t.loc[y, it] for it in items}})
    return pd.DataFrame(rows).set_index("year")


def vintage_components(d: pd.DataFrame, latest_comp: pd.DataFrame, targets: pd.DataFrame, years=range(2019, 2026), which: str = "first") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The deficit table and the outturn components of `budget.decompose_surprise` with the Istat first release (`which` = `first`) or the latest edition: revenue, expenditure, interest and the deficit in % of the **latest** nominal GDP (the GDP of each edition is not in this account, so only
    the numerator changes); nominal GDP and growth are the latest, so the growth errors and the denominator effect are the same in both vintages. Returns (T, comp) with the shapes `decompose_surprise` expects."""
    if which == "first":
        v = first_release(d, years)
    else:
        e = edition_table(d, "2026M10", REAL_TIME_ITEMS)
        v = e.reindex(list(years))
    gdp = latest_comp.nominal_gdp.reindex(v.index)
    pct = lambda s: s / gdp * 100
    comp = pd.DataFrame({"revenue": pct(v.OTR_C_W0), "expenditure": pct(v.OTE_D_W0), "interest": pct(v.D41_D_W0)})
    comp["primary_expenditure"] = comp.expenditure - comp.interest
    comp = comp.join(latest_comp[["nominal_gdp", "real_growth", "nominal_growth"]])
    T = pd.DataFrame({"deficit": -pct(v.B9_B_W0)}).join(targets[["target"]])
    T["deficit_surprise"] = T.deficit - T.target
    return T, comp


def surprise_vintages(d: pd.DataFrame, latest_comp: pd.DataFrame, targets: pd.DataFrame, tc: pd.DataFrame, decompose, years=range(2019, 2026)) -> pd.DataFrame:
    """The deficit surprise and its parts (revenue shortfall, primary-expenditure overrun, interest overrun, denominator) with the first release and with the latest edition, and the revision (latest minus first release), by budget year. `decompose` is `budget.decompose_surprise`."""
    out = {}
    for which in ("first", "latest"):
        T, comp = vintage_components(d, latest_comp, targets, years, which)
        out[which] = decompose(T, comp, tc)[["deficit", "deficit_surprise", "revenue_shortfall", "primary_expenditure_overrun", "interest_overrun", "denominator_effect"]]
    f, l = out["first"], out["latest"]
    res = f.add_suffix("_first").join(l.add_suffix("_latest"))
    for c in ("deficit", "deficit_surprise", "revenue_shortfall", "primary_expenditure_overrun", "interest_overrun"):
        res[f"{c}_revision"] = res[f"{c}_latest"] - res[f"{c}_first"]
    res["first_edition"] = first_release(d, list(res.index)).edition
    return res.rename_axis("budget_year").reset_index()


def quarter_table(q: pd.DataFrame, edition: str, items, adjustment: str = "N") -> pd.DataFrame:
    """Period x item (EUR million) of the quarterly account for one edition and adjustment (`N` raw, `Y` seasonally adjusted); the later flow wins if an edition is in both."""
    x = q[q.edition.eq(edition) & q.adjustment.eq(adjustment) & q["item"].isin(items)].sort_values("flow").drop_duplicates(["item", "period"], keep="last")
    return x.pivot(index="period", columns="item", values="value_meur")


def capital_profile(q: pd.DataFrame, edition: str = "2026M10", years=range(2019, 2025)) -> pd.DataFrame:
    """Other capital expenditure payable (D.92 + D.99, item `OOKE_D_W0`) by quarter (raw series), its share of the calendar year and the net lending of the quarter, EUR million: when the credits and the grants were booked."""
    t = quarter_table(q, edition, ["OOKE_D_W0", "B9_B_W0", "P51G_D_W0"])
    t = t[t.index.str[:4].astype(int).isin(list(years))].copy()
    t["year"] = t.index.str[:4].astype(int)
    t["quarter"] = t.index.str[-1].astype(int)
    t["ooke_share_of_year"] = t.OOKE_D_W0 / t.groupby("year").OOKE_D_W0.transform("sum")
    return t.rename(columns={"OOKE_D_W0": "other_capital_expenditure", "B9_B_W0": "net_lending", "P51G_D_W0": "gross_fixed_capital_formation"}).reset_index()


def quarterly_reclassification(q: pd.DataFrame, before: str = "2023M1", after: str = "2023M4", items=("OOKE_D_W0", "B9_B_W0"), first="2020-Q1", last="2022-Q3") -> pd.DataFrame:
    """The same quarter in two quarterly editions: how the booking of the Superbonus and facade credits of March 2023 (the quarterly edition of April 2023) was spread over the quarters it covers (EUR million)."""
    b, a = quarter_table(q, before, items), quarter_table(q, after, items)
    idx = [p for p in a.index if first <= p <= last]
    rows = [{"period": p, "item": it, "before": b.loc[p, it] if p in b.index else float("nan"), "after": a.loc[p, it], "change": a.loc[p, it] - (b.loc[p, it] if p in b.index else float("nan"))} for it in items for p in idx]
    return pd.DataFrame(rows)


def eurostat_check(d: pd.DataFrame, spending: pd.DataFrame, balance_pct: pd.Series, gdp_meur: pd.Series, edition: str | None = None, first: int = 2005, last: int = 2024) -> pd.DataFrame:
    """Istat's latest edition against the Eurostat figures the project uses for Italy: net lending, total expenditure and revenue, compensation of employees, intermediate consumption, interest, subsidies and capital transfers (D.92 + D.99). Istat euro values are put in % of the project's nominal GDP (Eurostat `nama_10_gdp`)
    and compared with Eurostat's own % of GDP (`gov_10a_main`; net lending: the panel indicator): gap = Istat minus Eurostat in points of GDP. Both are the ESA transmission of the same accounts, so the gaps should be rounding (Eurostat rounds to 0.1) and GDP vintage; a gap above 0.25 would be a definition or vintage problem."""
    e = edition_table(d, edition or latest_edition(d))
    sp = spending[(spending.iso3 == "ITA") & (spending.kind == "nature")].pivot(index="year", columns="item", values="value")
    pairs = {"net lending": (balance_pct, e.B9_B_W0), "total expenditure": (sp.TE, e.OTE_D_W0), "total revenue": (sp.TR, e.OTR_C_W0), "compensation of employees": (sp.D1PAY, e.D1_D_W0), "intermediate consumption": (sp.P2, e.P2_D_W0),
             "interest": (sp.D41PAY, e.D41_D_W0), "subsidies": (sp.D3PAY, e.D3_D_W0), "capital transfers (D.92 + D.99)": (sp.D9PAY, e.D92_D_W0 + e.D99_D_W0)}
    rows = []
    for name, (eu, ist) in pairs.items():
        years = [y for y in eu.index if first <= y <= last and y in ist.index and y in gdp_meur.index]
        gap = (ist.loc[years] / gdp_meur.loc[years] * 100 - eu.loc[years]).dropna()
        rows.append({"item": name, "n_years": len(gap), "mean_gap_pp": gap.mean(), "max_abs_gap_pp": gap.abs().max(), "year_of_max": int(gap.abs().idxmax()), "share_of_years_above_0.1pp": float((gap.abs() > 0.1).mean())})
    return pd.DataFrame(rows)
