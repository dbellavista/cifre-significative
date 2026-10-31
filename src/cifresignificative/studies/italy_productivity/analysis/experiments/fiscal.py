"""Budget-law financing, EU-wide budget surprises and OpenCoesione experiments."""
from pathlib import Path

import pandas as pd

from cifresignificative.core.sources import enea_monthly as EM
from cifresignificative.core.sources import enea_regions as ER
from cifresignificative.core.sources import istat_fpa as FPA
from cifresignificative.studies.italy_productivity.analysis import budget as BU
from cifresignificative.studies.italy_productivity.analysis import budget_xc as BX
from cifresignificative.studies.italy_productivity.analysis import istat_fpa as FP
from cifresignificative.studies.italy_productivity.analysis import opencoesione as OCA
from cifresignificative.studies.italy_productivity.analysis import regional_corr as rc
from cifresignificative.studies.italy_productivity.analysis import regional_south as rs
from cifresignificative.studies.italy_productivity.analysis import superbonus as SB
from cifresignificative.studies.italy_productivity.analysis import (
    superbonus_monthly as SBM,
)
from cifresignificative.studies.italy_productivity.analysis import (
    superbonus_outcomes as SBO,
)
from cifresignificative.studies.italy_productivity.analysis import (
    superbonus_regions as SBR,
)
from cifresignificative.studies.italy_productivity.analysis import (
    superbonus_timing as SBT,
)
from cifresignificative.studies.italy_productivity.analysis.data import (
    PROCESSED,
    load_panel,
    load_regional,
    load_regional_extra,
    load_spending,
    wide,
)

from .common import OUT


def run_budget_all(out: Path = OUT, n_perm: int = 20000, n_shuffle: int = 300) -> dict[str, pd.DataFrame]:
    """Budget-law financing round (`analysis.budget`): B1 coverage measures of the first year of each law, B2 the same summed over the three-year horizon, B3 the manovra next to the realised outcomes, B4 the pre-specified link family
    (planned deficit effect and deficit-financed share against the change in net borrowing, primary deficit and debt/GDP), B5 without 2020-21, B6 with the pandemic decrees added, B7 the lead placebo (outcomes of the previous year),
    B8 the year-shuffle null, B9 the realised debt decomposition by year, B10-B13 the second family (the manovra against the deficit surprise = outturn minus programmatic target, with the final target, without 2020-21 and its null), B14 how well the target predicted the change in the deficit, B15 the surprise split into revenue shortfall, primary-expenditure overrun and interest overrun with the growth forecast errors, B16 its rank correlations with the growth errors, B17 the expenditure miss by economic category (compensation, intermediate consumption, social payments, interest, subsidies, investment, capital transfers, other). Exploratory: one country, about ten years, no placebo countries."""
    panel = load_panel()
    gdp = panel[(panel.iso3 == "ITA") & panel.indicator.eq("gdp_nominal_meur")].set_index("year").value
    d = BU.load()
    m = BU.measures(d, gdp)
    out_ = BU.outcomes(panel)
    T = BU.link_table(m, out_, "bill", BU.targets())
    rng_years = T.loc[2016:2025]
    Tall = BU.link_table(m, out_, "all").loc[2016:2025]
    first = BU.first_year(m)
    results = {"b1_first_year": first, "b2_three_year": BU.three_year(m), "b3_link_table": T.loc[2016:2025].reset_index(),
               "b4_family": BU.link_family(rng_years, n_perm=n_perm), "b5_family_ex_covid": BU.link_family(rng_years, n_perm=n_perm, drop_years=(2020, 2021)),
               "b6_family_with_pandemic": BU.link_family(Tall, n_perm=n_perm), "b7_lead_placebo": BU.lead_placebo(rng_years, n_perm=n_perm),
               "b8_null": BU.year_shuffle_null(rng_years, n_shuffle=n_shuffle), "b9_debt_decomposition": out_.reset_index(),
               "b10_surprise_family": BU.link_family(rng_years, BU.XS, BU.YS_SURPRISE, n_perm=n_perm), "b11_surprise_ex_covid": BU.link_family(rng_years, BU.XS, BU.YS_SURPRISE, n_perm=n_perm, drop_years=(2020, 2021)),
               "b12_surprise_final_target": BU.link_family(rng_years.assign(deficit_surprise=rng_years.deficit_surprise_final), BU.XS, BU.YS_SURPRISE, n_perm=n_perm),
               "b13_surprise_null": BU.year_shuffle_null(rng_years, BU.XS, BU.YS_SURPRISE, n_shuffle=n_shuffle), "b14_plan_vs_outturn": BU.plan_vs_outturn(rng_years, n_perm=n_perm)}
    dec = BU.decompose_surprise(rng_years, BU.outturn_components(panel), BU.target_components())
    results["b15_surprise_decomposition"] = dec.reset_index()
    results["b16_surprise_drivers"] = BU.surprise_drivers(dec, n_perm=n_perm)
    results["b17_expenditure_by_category"] = BU.expenditure_misses(rng_years, BU.target_components(), BU.outturn_components(panel)).reset_index()
    flows = SB.yearly_flows(SB.load())
    lb = SB.lower_bound(flows, results["b17_expenditure_by_category"], wide(panel, "gdp_nominal_meur")["ITA"])
    results["b18_superbonus_flows"] = flows.reset_index()
    results["b19_superbonus_lower_bound"] = pd.concat([lb.astype({"budget_year": str}), SB.cumulative(lb).to_frame().T.assign(budget_year="2021-23")], ignore_index=True)
    ib = SB.istat_bounds(lb, SB.load_istat())
    results["b20_superbonus_istat"] = pd.concat([ib.astype({"budget_year": str}), SB.istat_cumulative(ib, SB.load_istat()).to_frame().T.assign(budget_year="2021-23")], ignore_index=True)
    results["b21_superbonus_vintages"] = SB.vintage_table(lb, SB.load_istat())
    results["b22_credit_attribution"] = SB.credit_attribution(results["b21_superbonus_vintages"], SB.load_istat(), results["b17_expenditure_by_category"], wide(panel, "gdp_nominal_meur")["ITA"])
    results["b23_capital_transfer_split"] = SB.capital_transfer_split(load_spending(), wide(panel, "gdp_nominal_meur")["ITA"], results["b22_credit_attribution"])
    results["b24_grants_check"] = SB.grants_check(results["b23_capital_transfer_split"], results["b22_credit_attribution"], SB.load_grants()).to_frame("value").rename_axis("item").reset_index()
    fpa = FPA.load()
    rec = FP.by_receiver(fpa, FPA.latest_edition(fpa))
    results["b25_fpa_receivers"] = rec.reset_index()
    results["b26_fpa_increase"] = FP.increase_since(rec).reset_index()
    results["b27_fpa_reclassification"] = FP.reclassification(fpa)
    results["b28_fpa_hearing_check"] = FP.hearing_check(fpa, SB.load_istat())
    results["b29_fpa_facade_split"] = FP.facade_split(rec, SB.load_istat(), wide(panel, "gdp_nominal_meur")["ITA"])
    vint = fpa[fpa["item"].isin(["D92_D_W0_S14", "D92_D_W0_SIM", "D99_D_W0_SIM", "D3_D_W0", "B9_B_W0"]) & fpa.year.between(2019, 2023)].sort_values("published").drop_duplicates(["edition", "item", "year"], keep="last")
    results["b30_fpa_vintages"] = vint[["edition", "published", "item", "year", "value_meur"]].reset_index(drop=True)
    results["b31_fpa_deficit_vintages"] = FP.deficit_vintages(fpa, BU.targets(), wide(panel, "gdp_nominal_meur")["ITA"], years=(2021, 2022))
    results["b32_surprise_vintages"] = FP.surprise_vintages(fpa, BU.outturn_components(panel), BU.targets(), BU.target_components(), BU.decompose_surprise)
    fpq = FPA.load_quarterly()
    results["b33_quarterly_capital_profile"] = FP.capital_profile(fpq)
    results["b34_quarterly_reclassification"] = FP.quarterly_reclassification(fpq)
    enea_r, reg_ = ER.load(), pd.read_parquet(PROCESSED / "regional.parquet")
    tabs = {snap: SBR.region_table(enea_r, reg_, snap) for snap in ("2023-12-31", "2025-12-31")}
    results["b35_superbonus_regions"] = pd.concat([t.assign(snapshot=snap) for snap, t in tabs.items()], ignore_index=True)
    results["b36_superbonus_zones"] = pd.concat([SBR.zone_shares(t).reset_index().assign(snapshot=snap) for snap, t in tabs.items()], ignore_index=True)
    results["b37_superbonus_rank_tests"] = pd.DataFrame([{"snapshot": snap, **SBR.rank_test(t, y=y)} for snap, t in tabs.items() for y in ("per_inhabitant_eur", "condominium_share")])
    em = EM.load()
    wf = SBM.window_flows(em, reg_)
    results["b39_superbonus_zone_windows"] = SBM.zone_flow_shares(wf)
    results["b40_superbonus_window_gradient"] = SBM.window_gradient(wf, n_perm=n_perm).assign(basis="admitted").pipe(lambda d: pd.concat([d, SBM.window_gradient(SBM.window_flows(em, reg_, "inv_completed"), n_perm=n_perm).assign(basis="completed")], ignore_index=True))
    results["b41_superbonus_gradient_placebo"] = pd.DataFrame([SBM.gradient_placebo(wf, n_shuffle=n_shuffle * 2)])
    results["b42_superbonus_gradient_path"] = SBM.gradient_path(em, reg_)
    results["b43_superbonus_monthly_flows"] = SBM.monthly_flows(em).reset_index()
    results["b44_superbonus_completion"] = SBM.zone_completion(em, ["2021-08-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2026-09-30"]).reset_index()
    nat_check, reg_check = SBM.basis_check(em, enea_r, pd.read_csv("data/external/enea_superbonus.csv", parse_dates=["snapshot"]))
    results["b45_superbonus_basis_national"], results["b46_superbonus_basis_regions"] = nat_check, reg_check
    results["b47_superbonus_monthly_checks"] = EM.check(em).reset_index()
    ch = SBO.outcome_changes(SBO.outcome_levels(tabs["2023-12-31"], reg_))
    results["b48_superbonus_outcome_changes"] = ch.reset_index()
    results["b49_superbonus_outcome_link"] = SBO.link(tabs["2023-12-31"], ch, n_perm=n_perm)
    results["b50_superbonus_outcome_robustness"] = SBO.robustness(tabs["2023-12-31"], ch, n_perm=n_perm)
    results["b51_superbonus_outcome_placebo"] = pd.DataFrame([SBO.placebo(tabs["2023-12-31"], ch, n_shuffle=n_shuffle * 2)])
    tp = SBT.monthly_path(em, reg_)
    tp.columns = [f"{a}|{b}" for a, b in tp.columns]
    results["b52_superbonus_policy_dates"] = SBT.load_dates()
    results["b53_superbonus_monthly_path"] = tp.reset_index()
    results["b54_superbonus_event_tests"] = SBT.event_tests(SBT.monthly_path(em, reg_))
    flows_m = SB.yearly_flows_monthly(em)
    lb_m = SB.lower_bound(flows_m, results["b17_expenditure_by_category"], wide(panel, "gdp_nominal_meur")["ITA"])
    results["b58_superbonus_rebased_flows"] = flows_m.reset_index()
    results["b59_superbonus_rebased_bound"] = SB.rebase(lb, lb_m).rename_axis("budget_year").reset_index().astype({"budget_year": str})
    bc, bt = SBT.building_counts(em)
    results["b56_superbonus_new_buildings"], results["b57_superbonus_new_buildings_tests"] = bc.reset_index(), bt
    results["b55_superbonus_onset"] = pd.DataFrame([SBT.onset(SBT.monthly_path(em, reg_))])
    results["b38_istat_eurostat_check"] = FP.eurostat_check(fpa, load_spending(), wide(panel, "gov_balance_pct_gdp")["ITA"], wide(panel, "gdp_nominal_meur")["ITA"])
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_opencoesione_all(out: Path = OUT, n_perm: int = 2000, n_shuffle: int = 100) -> dict[str, pd.DataFrame]:
    """OpenCoesione 2014-2020 round (`analysis.opencoesione`, Italy-only track of notebook 34): O1 the allocation by region (euro per inhabitant), O2 by zone, O3 the pre-specified family (4 predictors x 6 outcomes) with within-zone permutation p, O4 the shuffled-predictor null,
    O5 what could not be allocated by fund (national scope, abroad, no localisation) and the multi-region projects."""
    regional, extra = load_regional(), load_regional_extra()
    flows, spending = pd.read_parquet(PROCESSED / "italy_internal_migration.parquet"), pd.read_parquet(PROCESSED / "italy_cpt_spending.parquet")
    R = rc.regional_matrix(pd.concat([regional, extra], ignore_index=True), drop_flag_chars="u")
    T = rs.italy_table(R, flows, spending)
    oc = pd.read_csv("data/external/opencoesione_regions.csv")
    pred = OCA.region_predictors(oc, regional)
    results = {"o1_allocation": pred.reset_index(), "o2_zones": OCA.zone_table(pred).reset_index(names="zone"), "o3_family": OCA.link(T, pred, n_perm=n_perm), "o4_null": pd.DataFrame([OCA.null(T, pred, n_shuffle=n_shuffle)]),
               "o5_unallocated": pd.read_csv("data/external/opencoesione_unallocated.csv")}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_budget_xc_all(out: Path = OUT, n_perm: int = 20000, n_shuffle: int = 200) -> dict[str, pd.DataFrame]:
    """Placebo ranking of the deficit surprise (`analysis.budget_xc`): X1 surprises by country, year and component against the Commission's autumn forecast of the year before, X2 Italy's rank by component, window and measure (BH over the primary family),
    X3 Italy's rank year by year, X4 the same ranks among the 20 euro-area members, X5 the horizon (lead 0, 1, 2), X6 Italy's miss against the Commission and against the government's own target, X7 the per-country sign-flip screen of the mean
    idiosyncratic deficit surprise (2017-25 without 2020), X8 the sign-randomised null of the screen, X9 the accounting identity of the parts. One-sided ranks among 27 countries: the smallest possible p is 1/27."""
    panel = load_panel()
    F, O = BX.load_forecasts(), BX.outturn_wide(panel)
    S = BX.surprises(F, O)
    long = pd.concat([s.rename_axis("year").reset_index().melt(id_vars="year", var_name="iso3", value_name="value").assign(component=c) for c, s in S.items()], ignore_index=True)
    ident = (S["revenue_shortfall"] + S["primary_expenditure_overrun"] + S["interest_overrun"] - S["deficit"]).abs().stack().describe().to_frame("abs_gap").T
    ea = BX.rank_table({"deficit": S["deficit"]}, countries=BX.EURO_AREA_2025).drop(columns="q_value")  # a q over four tests would be misleading: p only
    results = {"x1_surprises": long, "x2_italy_rank": BX.rank_table(S), "x3_yearly_ranks": BX.yearly_ranks(S), "x4_euro_area_rank": ea, "x5_lead": BX.lead_sensitivity(F, O),
               "x6_two_views": BX.italy_two_views(S["deficit"], O["net_lending"], BU.targets()).reset_index(), "x9_identity": ident}
    results["x10_first_estimate"], results["x10_revisions"] = BX.first_estimate_sensitivity(F, O)
    split = BX.denominator_split(S["deficit"], F, BX.gdp_ratio(F, wide(panel, "gdp_nominal_meur")))
    results["x11_denominator_ranks"] = pd.DataFrame([{"part": k, "measure": m, "window": wn, **BX.italy_rank(X, yrs)} for k, v in {"surprise": S["deficit"], **split}.items() for m, X in (("raw", v), ("idiosyncratic", BX.idiosyncratic(v)))
                                                      for wn, yrs in BX.WINDOWS.items() if wn in BX.PRIMARY])
    results["x11_italy_split"] = pd.DataFrame({"surprise": S["deficit"]["ITA"], **{k: v["ITA"] for k, v in split.items()}}).rename_axis("year").reset_index()
    w = BX.WINDOWS["2017-25 without 2020"]  # the screen needs more than a handful of years: 3 years give only 8 sign patterns
    results["x7_screen"], results["x8_null"] = BX.country_sign_flip(S["deficit"], w, n_perm=n_perm), BX.sign_null(S["deficit"], w, n_shuffle=n_shuffle)
    P = BX.dbp_plans(F, BU.targets())
    Sd = BX.dbp_surprises(P, O)
    results["x12_dbp_plans"] = P.rename_axis("year").reset_index().melt(id_vars="year", var_name="iso3", value_name="plan_net_lending").dropna().merge(Sd.rename_axis("year").reset_index().melt(id_vars="year", var_name="iso3", value_name="surprise_vs_plan").dropna(), on=["year", "iso3"])
    results["x13_dbp_ranks"] = BX.dbp_rank_table(Sd, S["deficit"])
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results
