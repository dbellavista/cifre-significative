"""A lower bound for the building-credit part of Italy's capital-transfer overruns (notebook 38): the SuperEcobonus (Superbonus 110%) tax credits accrued each year, from ENEA's monthly bulletins, against the capital-transfer (ESA D.9) overruns of 2021-23.

ENEA publishes cumulative figures (`data/external/enea_superbonus.csv`, hand-transcribed from the annual reports, with report and table): investments admitted to deduction, investments for completed works and deductions accrued for completed works. Yearly flows are the differences between December
snapshots (2021 is the cumulative since July 2020, so it also holds the second half of 2020). The credits accrued are what Eurostat's 2023 decision brings into expenditure (capital transfers) when the works are done, so they are compared with the capital-transfer overrun of notebook 38 (outturn minus the target, % of the outturn GDP,
`budget.expenditure_misses`) in euro. This is a **lower bound**: ENEA covers only the Superbonus; the facade bonus, the ordinary ecobonus and sismabonus, and the other building bonuses are not in it, the overrun is measured against the plan (which may already contain part of the credits as lower revenue), and the capital-transfers category also holds other grants.
Snapshots are by transmission date of the asseverations (monthly bulletins), not by the date of the works, and later transmissions raise earlier years, so the flow of a year is understated and the flow of the next overstated.
"""
from pathlib import Path

import pandas as pd

CSV = Path("data/external/enea_superbonus.csv")
YEAR_ENDS = ["2021-12-31", "2022-12-31", "2023-12-31"]


def load(csv: Path = CSV) -> pd.DataFrame:
    return pd.read_csv(csv, parse_dates=["snapshot"])


def yearly_flows(snap: pd.DataFrame, years=(2021, 2022, 2023)) -> pd.DataFrame:
    """Flow of each year = December snapshot minus the previous December (2021: the whole cumulative); only the `monthly bulletin` snapshots are used. Columns: investments admitted, investments for completed works, deductions accrued (EUR) and the ratio of deductions to investments."""
    s = snap[snap.basis.eq("monthly bulletin") & snap.snapshot.dt.month.eq(12)]
    s = s.set_index(s.snapshot.dt.year).sort_index()
    cols = ["investments_admitted_eur", "investments_completed_eur", "deductions_accrued_eur"]
    cum = s[cols].reindex(list(years))
    flows = cum.diff()
    flows.iloc[0] = cum.iloc[0]
    flows["credit_rate"] = flows.deductions_accrued_eur / flows.investments_completed_eur
    return flows.rename_axis("budget_year")


def lower_bound(flows: pd.DataFrame, expenditure_misses: pd.DataFrame, gdp_meur: pd.Series) -> pd.DataFrame:
    """Superbonus credits accrued (EUR million) as a share of the capital-transfer overrun of the same year (EUR million = % of GDP x outturn GDP) and of the total expenditure miss; also the share if only the investment (without the 10% surcharge) is counted."""
    ct = expenditure_misses.set_index("budget_year").reindex(flows.index)
    out = pd.DataFrame(index=flows.index)
    out["capital_transfers_miss_pct_gdp"] = ct["capital transfers"]
    out["gdp_meur"] = gdp_meur.reindex(flows.index)
    out["capital_transfers_miss_meur"] = out.capital_transfers_miss_pct_gdp / 100 * out.gdp_meur
    out["total_expenditure_miss_meur"] = ct.total_expenditure_miss / 100 * out.gdp_meur
    out["superbonus_credits_meur"] = flows.deductions_accrued_eur / 1e6
    out["superbonus_investment_meur"] = flows.investments_completed_eur / 1e6
    out["credits_pct_gdp"] = out.superbonus_credits_meur / out.gdp_meur * 100
    out["share_of_capital_transfers_miss"] = out.superbonus_credits_meur / out.capital_transfers_miss_meur
    out["share_of_expenditure_miss"] = out.superbonus_credits_meur / out.total_expenditure_miss_meur
    out["share_investment_only"] = out.superbonus_investment_meur / out.capital_transfers_miss_meur
    out["remainder_meur"] = out.capital_transfers_miss_meur - out.superbonus_credits_meur
    return out.reset_index()


def cumulative(lb: pd.DataFrame) -> pd.Series:
    """The three years together: credits, capital-transfer overrun and their ratio."""
    c, m = lb.superbonus_credits_meur.sum(), lb.capital_transfers_miss_meur.sum()
    return pd.Series({"superbonus_credits_meur": c, "capital_transfers_miss_meur": m, "share_of_capital_transfers_miss": c / m, "remainder_meur": m - c, "credits_pct_gdp_sum": lb.credits_pct_gdp.sum(), "miss_pct_gdp_sum": lb.capital_transfers_miss_pct_gdp.sum()})


ISTAT_CSV = Path("data/external/istat_superbonus_deficit.csv")


def load_istat(csv: Path = ISTAT_CSV) -> pd.DataFrame:
    return pd.read_csv(csv)


def istat_bounds(lb: pd.DataFrame, istat: pd.DataFrame) -> pd.DataFrame:
    """The Superbonus effect on net borrowing in Istat's own accounts (April 2024 note, pp of GDP, 2021-23) in euro (times the outturn GDP) against the capital-transfer overrun, next to the ENEA lower bound: this one covers the Superbonus
    in all its forms (super ecobonus and super sismabonus) but not the facade bonus, whose total (about 23 billion for 2020-22, no annual split) is added in the cumulative row only. `lb` is `lower_bound`, indexed by year (2021-2023 rows, no cumulative row)."""
    s = istat[istat.series.eq("superbonus_incidence_on_net_borrowing")].set_index("year").value.astype(float)
    d = lb.set_index("budget_year").loc[[2021, 2022, 2023]]
    out = pd.DataFrame({"istat_superbonus_pct_gdp": s.reindex(d.index).to_numpy()}, index=d.index)
    out["istat_superbonus_meur"] = out.istat_superbonus_pct_gdp / 100 * d.gdp_meur
    out["capital_transfers_miss_meur"] = d.capital_transfers_miss_meur
    out["share_of_capital_transfers_miss"] = out.istat_superbonus_meur / out.capital_transfers_miss_meur
    out["enea_credits_meur"] = d.superbonus_credits_meur
    out["enea_over_istat"] = out.enea_credits_meur / out.istat_superbonus_meur
    return out.reset_index()


def istat_cumulative(b: pd.DataFrame, istat: pd.DataFrame) -> pd.Series:
    """2021-23 together: Istat's Superbonus effect, the same plus the facade bonus (about 23 billion, 2020-22) and their shares of the capital-transfer overrun."""
    facade = float(istat[istat.series.eq("facade_bonus_2020_22")].value.iloc[0]) * 1000
    sb, miss = b.istat_superbonus_meur.sum(), b.capital_transfers_miss_meur.sum()
    return pd.Series({"istat_superbonus_meur": sb, "facade_bonus_meur": facade, "capital_transfers_miss_meur": miss, "share_superbonus": sb / miss, "share_superbonus_and_facade": (sb + facade) / miss,
                      "enea_credits_meur": b.enea_credits_meur.sum(), "enea_over_istat": b.enea_credits_meur.sum() / sb})


def vintage_table(lb: pd.DataFrame, istat: pd.DataFrame) -> pd.DataFrame:
    """For 2021-23, side by side (EUR million): the capital-transfer overrun, the ENEA credits, Istat's Superbonus effect of April 2024, Istat's Superbonus plus facade-bonus spending of March 2023 (2021-22 only, read from a bar chart), the Bank of Italy's credits matured in % of GDP times the GDP,
    and the Istat firms' tax-credit spending of March 2023 (context for the remainder, all credits of firms, 2021-22 only). The vintages differ (March 2023 against April 2024 revisions of the same years), so the columns are alternatives, not parts that add up."""
    d = lb.set_index("budget_year").loc[[2021, 2022, 2023]]
    def get(series: str) -> pd.Series:
        t = istat[istat.series.eq(series) & istat.year.notna()].set_index("year").value.astype(float)
        return t.reindex(d.index)
    out = pd.DataFrame(index=d.index)
    out["capital_transfers_miss_meur"] = d.capital_transfers_miss_meur
    out["enea_credits_meur"] = d.superbonus_credits_meur
    out["istat_2024_superbonus_meur"] = get("superbonus_incidence_on_net_borrowing") / 100 * d.gdp_meur
    out["istat_2023_superbonus_facade_meur"] = get("superbonus_facade_spending_2023_vintage") * 1000
    out["bankitaly_credits_meur"] = get("credits_matured_pct_gdp_bankitaly") / 100 * d.gdp_meur
    out["istat_2023_firm_credits_meur"] = get("firm_tax_credits_spending_2023_vintage") * 1000
    for c in ("enea_credits_meur", "istat_2024_superbonus_meur", "istat_2023_superbonus_facade_meur", "bankitaly_credits_meur", "istat_2023_firm_credits_meur"):
        out["share_" + c.removesuffix("_meur")] = out[c] / out.capital_transfers_miss_meur
    return out.reset_index()


# grouping of the firms' tax credits of Istat's Tavola 1 (hearing of 15 March 2023) by the ESA category they most plausibly land in; Istat's table does not give the ESA category of each credit, so this is a judgment, shown in two widths:
# core = credits tied to an investment by the firm (investment grants, D.92), wide = core + R&D, deferred-tax-asset conversion and vehicle bonus (which may be subsidies or investment grants); the rest is treated as subsidy-type (D.3)
CREDIT_CORE = ["Transizione 4.0", "Investimenti Mezzogiorno"]
CREDIT_WIDE = [*CREDIT_CORE, "Ricerca e sviluppo", "DTA", "Ecobonus veicoli"]
CREDIT_PREFIX = "firm_tax_credit_composition_pct__"


def credit_attribution(vt: pd.DataFrame, istat: pd.DataFrame, expenditure_misses: pd.DataFrame, gdp_meur: pd.Series) -> pd.DataFrame:
    """For 2021 and 2022 (the years with Istat's March 2023 vintage): the capital-transfer overrun (EUR million) minus the Superbonus and facade spending of the same vintage, then minus the firms' investment-type tax credits (core and wide grouping of `CREDIT_*`, share of the firms' credit total of Figura 2),
    leaving what is not explained by building bonuses or firms' investment credits. A cross-check: the subsidy-type credits (the rest of the table) against the subsidies overrun of the same year (`expenditure_misses`, pp of GDP times GDP). Shares of the overrun are given for the remainders. The categories are a judgment (`CREDIT_CORE`, `CREDIT_WIDE`)."""
    comp = istat[istat.series.str.startswith(CREDIT_PREFIX)].assign(kind=lambda d: d.series.str.removeprefix(CREDIT_PREFIX)).pivot(index="year", columns="kind", values="value")
    v = vt.set_index("budget_year").loc[[2021, 2022]]
    total = v.istat_2023_firm_credits_meur
    out = pd.DataFrame(index=v.index)
    out["capital_transfers_miss_meur"] = v.capital_transfers_miss_meur
    out["superbonus_facade_meur"] = v.istat_2023_superbonus_facade_meur
    out["after_building_bonuses_meur"] = out.capital_transfers_miss_meur - out.superbonus_facade_meur
    for name, cols in (("core", CREDIT_CORE), ("wide", CREDIT_WIDE)):
        out[f"firm_investment_credits_{name}_meur"] = comp.loc[v.index, cols].sum(axis=1) / 100 * total
        out[f"remainder_{name}_meur"] = out.after_building_bonuses_meur - out[f"firm_investment_credits_{name}_meur"]
        out[f"remainder_{name}_share"] = out[f"remainder_{name}_meur"] / out.capital_transfers_miss_meur
    sub_cols = [c for c in comp.columns if c not in CREDIT_WIDE and c != "Enegetici"]
    out["subsidy_type_credits_meur"] = comp.loc[v.index, [*sub_cols, "Enegetici"]].sum(axis=1) / 100 * total
    out["energy_credits_meur"] = comp.loc[v.index, "Enegetici"] / 100 * total
    sm = expenditure_misses.set_index("budget_year").reindex(v.index)["subsidies"] / 100 * gdp_meur.reindex(v.index)
    out["subsidies_miss_meur"] = sm
    return out.reset_index()


def capital_transfer_split(spending: pd.DataFrame, gdp_meur: pd.Series, attribution: pd.DataFrame, years=range(2018, 2025), base_year: int = 2019, iso3: str = "ITA") -> pd.DataFrame:
    """Italy's capital transfers (D.9) split into investment grants (D.92) and other capital transfers (D.99), % of GDP (Eurostat gov_10a_main) and EUR million, with the change since `base_year` (the last year before the pandemic): the Superbonus, facade bonus and firms'
    investment tax credits are recorded as investment grants, cash grants such as the pandemic contributi a fondo perduto as other capital transfers. Columns for 2021 and 2022 compare the D.92 increase with the building bonuses plus the firms' core and wide investment credits of `credit_attribution`
    (EUR million, Istat March 2023 vintage), and the D.99 increase with the part of the overrun those credits leave unexplained."""
    n = spending[(spending.kind == "nature") & (spending.iso3 == iso3) & spending["item"].isin(["D9PAY", "D92PAY", "D99PAY"])].pivot(index="year", columns="item", values="value").reindex(list(years))
    out = n.rename(columns={"D9PAY": "d9_pct_gdp", "D92PAY": "d92_pct_gdp", "D99PAY": "d99_pct_gdp"})
    g = gdp_meur.reindex(out.index)
    for c in ("d9", "d92", "d99"):
        out[f"{c}_meur"] = out[f"{c}_pct_gdp"] / 100 * g
        out[f"{c}_change_since_{base_year}_meur"] = out[f"{c}_meur"] - out.loc[base_year, f"{c}_meur"]
    a = attribution.set_index("budget_year")
    out["building_plus_core_credits_meur"] = a.superbonus_facade_meur + a.firm_investment_credits_core_meur
    out["building_plus_wide_credits_meur"] = a.superbonus_facade_meur + a.firm_investment_credits_wide_meur
    out["unexplained_core_meur"], out["unexplained_wide_meur"] = a.remainder_core_meur, a.remainder_wide_meur
    return out.rename_axis("year").reset_index()


GRANTS_CSV = Path("data/external/covid_grants_2021.csv")


def load_grants(csv: Path = GRANTS_CSV) -> pd.DataFrame:
    return pd.read_csv(csv)


def grants_check(split: pd.DataFrame, attribution: pd.DataFrame, grants: pd.DataFrame, year: int = 2021, base_year: int = 2019) -> pd.Series:
    """The 2021 cash grants of the Sostegni decrees against the other capital transfers (D.99) and against what the building bonuses and firms' investment credits leave unexplained in the capital-transfer overrun. `new_money_meur` of the authorized rows adds up to the distinct money:
    DL 41 art. 1 (11,150 million) and DL 73 art. 1 as converted (5,873 automatic, 3,400 alternative, 850 new money of the 4,000 for the worst-hit operators since 3,150 are unused DL 41 resources, 452.1 added in conversion); the initial text of DL 73 (8,000 automatic) is superseded.
    The automatic tranche was paid at `paid_over_authorized` of its authorization (5,210 of 5,873 by 22 June 2021), used as a payment rate for the whole. Authorized is not booked: the point is the order of magnitude."""
    auth = grants[grants.basis.str.startswith("authorized")]
    new = float(auth.new_money_meur.sum())
    paid = float(grants[grants.basis.str.startswith("paid")].amount_meur.iloc[0])
    final_auto = float(auth[auth.measure.str.contains("automatic top-up") & auth.basis.str.contains("law as converted")].amount_meur.iloc[0])
    rate = paid / final_auto
    s, a = split.set_index("year"), attribution.set_index("budget_year")
    base = s.loc[base_year, "d99_meur"]
    return pd.Series({"d99_meur": s.loc[year, "d99_meur"], "d99_base_meur": base, "d99_change_meur": s.loc[year, "d99_change_since_2019_meur"], "grants_authorized_net_meur": new, "automatic_paid_over_authorized": rate,
                      "grants_at_payment_rate_meur": new * rate, "d99_other_meur": s.loc[year, "d99_meur"] - base - new, "d99_other_at_payment_rate_meur": s.loc[year, "d99_meur"] - base - new * rate,
                      "unexplained_core_meur": a.loc[year, "remainder_core_meur"], "unexplained_wide_meur": a.loc[year, "remainder_wide_meur"]})


def yearly_flows_monthly(em: pd.DataFrame, years=(2021, 2022, 2023)) -> pd.DataFrame:
    """The same yearly flows from ENEA's monthly reports (`sources/enea_monthly.py`, the Italy row at each December month end) instead of the transcribed annual-report snapshots: the monthly reports are later extractions (they include late transmissions), 3-5% above the annual-report figures in 2021-22. Same columns as `yearly_flows`."""
    it = em[em.region.eq("Italia") & em.date.dt.month.eq(12)].set_index(em[em.region.eq("Italia") & em.date.dt.month.eq(12)].date.dt.year).sort_index()
    cum = it[["inv_admitted", "inv_completed", "deductions_accrued"]].rename(columns={"inv_admitted": "investments_admitted_eur", "inv_completed": "investments_completed_eur", "deductions_accrued": "deductions_accrued_eur"}).reindex(list(years))
    flows = cum.diff()
    flows.iloc[0] = cum.iloc[0]
    flows["credit_rate"] = flows.deductions_accrued_eur / flows.investments_completed_eur
    return flows.rename_axis("budget_year")


def rebase(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Lower bounds on the two bases side by side (shares of the capital-transfer overrun, credits in EUR million) and the ratio of the credits (monthly over annual-report basis)."""
    o, n = old.set_index("budget_year"), new.set_index("budget_year")
    out = pd.DataFrame({"credits_annual_meur": o.superbonus_credits_meur, "credits_monthly_meur": n.superbonus_credits_meur, "share_annual": o.share_of_capital_transfers_miss, "share_monthly": n.share_of_capital_transfers_miss, "share_investment_annual": o.share_investment_only, "share_investment_monthly": n.share_investment_only})
    out["credits_ratio"] = out.credits_monthly_meur / out.credits_annual_meur
    tot = pd.Series({"credits_annual_meur": o.superbonus_credits_meur.sum(), "credits_monthly_meur": n.superbonus_credits_meur.sum(), "share_annual": o.superbonus_credits_meur.sum() / o.capital_transfers_miss_meur.sum(), "share_monthly": n.superbonus_credits_meur.sum() / n.capital_transfers_miss_meur.sum(), "share_investment_annual": o.superbonus_investment_meur.sum() / o.capital_transfers_miss_meur.sum(), "share_investment_monthly": n.superbonus_investment_meur.sum() / n.capital_transfers_miss_meur.sum()}, name="2021-23")
    tot["credits_ratio"] = tot.credits_monthly_meur / tot.credits_annual_meur
    return pd.concat([out, tot.to_frame().T])
