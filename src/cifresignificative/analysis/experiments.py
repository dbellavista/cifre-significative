"""Experiment runner: each experiment is a config -> tidy results table; `run_all` writes them to data/processed/correlations/."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats

from cifresignificative.analysis import bargaining as BG
from cifresignificative.analysis import budget as BU
from cifresignificative.analysis import budget_xc as BX
from cifresignificative.analysis import contracts as CT
from cifresignificative.analysis import corr
from cifresignificative.analysis import daily as DY
from cifresignificative.analysis import eu_funds as EF
from cifresignificative.analysis import hf as H
from cifresignificative.analysis import industry_panel as IP
from cifresignificative.analysis import istat_fpa as FP
from cifresignificative.analysis import latent as LT
from cifresignificative.analysis import nuts1 as N1
from cifresignificative.analysis import opencoesione as OCA
from cifresignificative.analysis import regimes as RG
from cifresignificative.analysis import regional_corr as rc
from cifresignificative.analysis import regional_south as rs
from cifresignificative.analysis import regulation as RG_
from cifresignificative.analysis import spatial as SP
from cifresignificative.analysis import superbonus as SB
from cifresignificative.analysis import superbonus_monthly as SBM
from cifresignificative.analysis import superbonus_outcomes as SBO
from cifresignificative.analysis import superbonus_regions as SBR
from cifresignificative.analysis import superbonus_timing as SBT
from cifresignificative.analysis.data import (
    PROCESSED,
    load_panel,
    load_regional,
    load_regional_extra,
    load_sectors64,
    load_spending,
    wide,
)
from cifresignificative.clean import nuts
from cifresignificative.sources import cnel_contracts as CN
from cifresignificative.sources import cnel_workers as CW
from cifresignificative.sources import conto_annuale as CA
from cifresignificative.sources import enea_monthly as EM
from cifresignificative.sources import enea_regions as ER
from cifresignificative.sources import eqi as EQI
from cifresignificative.sources import eu_payments as EP
from cifresignificative.sources import istat_contracts as IC
from cifresignificative.sources import istat_fpa as FPA
from cifresignificative.sources import oecd_bargaining as OB
from cifresignificative.sources import oecd_regulation as OR

OUT = PROCESSED / "correlations"
INCOME = "gdp_real_pc"  # development control for cross-sections (logged)
WINDOWS = {"recent": (2015, 2019), "early": (2005, 2009)}

HYPOTHESES = [  # (id, theme, y, x, expected sign of the association, label) - from docs/literature/README.md
    ("H1", "Productivity & growth", "gdp_real_pc", "ameco_tfp_index", 1, "Higher TFP goes with higher GDP per head"),
    ("H2", "Euro divergence", "gfcf_pct_gdp", "long_term_yield", -1, "Higher long-term yields go with lower investment"),
    ("H3", "Debt", "gov_debt_pct_gdp", "gdp_growth", -1, "Faster growth goes with lower debt/GDP"),
    ("H4", "Public investment", "gov_pubinv_pct_gdp", "gov_balance_pct_gdp", 1, "Bigger deficits go with lower public investment"),
    ("H5", "Pensions", "edu_spend_pct_gdp", "pension_spend_pct_gdp", -1, "Pension spending crowds out education spending"),
    ("H6", "Pensions", "neet_rate_15_29", "pension_spend_pct_gdp", 1, "Pension burden goes with more NEET youth"),
    ("H7", "Ageing & pensions", "employment_rate_55_64", "pension_spend_pct_gdp", -1, "Higher pension spending goes with lower old-age employment"),
    ("H8", "Education & skills", "emigration_nationals_25_34_per_1000", "tertiary_attainment_25_34", 1, "Graduate-heavy cohorts emigrate more"),
    ("H9", "Labour dualism", "gdp_real_pc", "employment_rate_20_64_female", 1, "Female employment goes with GDP per head"),
    ("H10", "Labour dualism", "employment_rate_20_64_female", "childcare_under3_30h_plus_pct", 1, "Full-time childcare goes with female employment"),
    ("H11", "Wages", "net_earnings_pps", "ameco_tfp_index", 1, "Productivity goes with net earnings"),
    ("H12", "Governance", "gdp_real_pc", "wgi_government_effectiveness", 1, "Government effectiveness goes with GDP per head"),
    ("H13", "Education & skills", "neet_rate_15_29", "early_leavers_18_24", 1, "Early school leaving goes with NEET"),
]


@dataclass(frozen=True)
class Experiment:
    name: str  # e1_levels, e2_changes, e3_within
    window: tuple[int, int] = WINDOWS["recent"]
    window0: tuple[int, int] | None = None
    control_income: bool = True
    method: str = "spearman"
    q: float = 0.05
    n_min: int = 20


def run(exp: Experiment, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = load_panel() if panel is None else panel
    M = corr.panel_matrix(panel)
    if exp.name == "e1_levels":
        X = corr.cross_section(M, "level", exp.window)
        if exp.control_income:
            X = corr.residualise(X, INCOME, log=True)
        res = corr.screen(X, exp.method, exp.q, exp.n_min)
    elif exp.name == "e2_changes":
        res = corr.screen(corr.cross_section(M, "change", exp.window, exp.window0 or WINDOWS["early"]), exp.method, exp.q, exp.n_min)
    elif exp.name == "e3_within":
        X = corr.within_changes(M).drop(columns=corr.size_scaled(M.columns))
        res = corr.fdr(corr.pair_screen(X, "pearson", n_min=150), exp.q)
        res = res[res.significant & (res.rho.abs() < corr.DUPLICATE_RHO)].sort_values("rho", key=abs, ascending=False)
    else:
        raise ValueError(exp.name)
    return res.assign(experiment=exp.name, config=str(asdict(exp))).reset_index(drop=True)


def run_split_sample(panel: pd.DataFrame | None = None, income_control: bool = True) -> pd.DataFrame:
    """E1s: the e1_levels screen replicated in 2005-14 and 2015-24 separately; `replicated` pairs are significant in both halves with the same sign."""
    panel = load_panel() if panel is None else panel
    return corr.split_sample(corr.panel_matrix(panel), control=INCOME if income_control else None)


def run_regimes(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """E3r: regime stability of the E3 within-country co-movements (pooled Pearson per regime, Cochran Q, leave-one-regime-out)."""
    panel = load_panel() if panel is None else panel
    M = corr.panel_matrix(panel)
    D = corr.within_changes(M).drop(columns=corr.size_scaled(M.columns))
    return RG.regime_table(D, run(Experiment("e3_within"), panel))


def run_hypotheses(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """E7: panel FE (entity + year, clustered) for each literature hypothesis, with and without Ireland, and a cross-section Italy-fit."""
    panel = load_panel() if panel is None else panel
    M = corr.panel_matrix(panel)
    X = corr.cross_section(M, "level", WINDOWS["recent"])
    rows = []
    for hid, theme, y, x, sign, label in HYPOTHESES:
        r, r_ex = corr.panel_fe(M, y, [x]), corr.panel_fe(M, y, [x], exclude=(corr.IRELAND,))
        fit = corr.italy_fit(X, x, y)
        if not r:
            rows.append({"id": hid, "theme": theme, "y": y, "x": x, "label": label, "status": "insufficient data"})
            continue
        coef = r["coef"][x]
        rows.append({"id": hid, "theme": theme, "y": y, "x": x, "label": label, "expected_sign": sign, "coef": coef, "p": r["p"][x],
                     "n_obs": r["n_obs"], "n_countries": r["n_countries"], "coef_ex_irl": r_ex.get("coef", {}).get(x), "p_ex_irl": r_ex.get("p", {}).get(x),
                     "sign_agrees": bool(coef * sign > 0), "italy_resid_z": fit.get("resid_z"), "status": "ok"})
    out = pd.DataFrame(rows)
    ok = out.status == "ok"
    out.loc[ok, "q_value"] = corr.fdr(out[ok], 0.05).q_value
    return out


def run_lead_lag(panel: pd.DataFrame | None = None, lags=(1, 2, 3)) -> pd.DataFrame:
    """E4: for each hypothesis, FE coefficient of x(t-k) on y(t) vs y(t-k) on x(t), one row per hypothesis and lag, with a verdict:
    'x leads' / 'y leads' when only that direction is significant at 5%, 'both' / 'neither' otherwise. Predictive order, not causation."""
    panel = load_panel() if panel is None else panel
    M = corr.panel_matrix(panel)
    rows = []
    for hid, theme, y, x, sign, label in HYPOTHESES:
        ll = corr.lead_lag(M, x, y, lags)
        for k, g in ll.groupby("lag"):
            g = g.set_index("direction")
            if {"x_leads_y", "y_leads_x"} - set(g.index):
                continue
            sx, sy = g.p["x_leads_y"] < 0.05, g.p["y_leads_x"] < 0.05
            rows.append({"id": hid, "x": x, "y": y, "lag": k, "coef_x_leads_y": g.coef["x_leads_y"], "p_x_leads_y": g.p["x_leads_y"],
                         "coef_y_leads_x": g.coef["y_leads_x"], "p_y_leads_x": g.p["y_leads_x"], "n_obs": int(g.n_obs["x_leads_y"]),
                         "verdict": {(True, False): "x leads", (False, True): "y leads", (True, True): "both"}.get((bool(sx), bool(sy)), "neither")})
    return pd.DataFrame(rows)


def run_regional_screen(regional: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """R1 / R1b / R2: region screens. Within-country (country-demeaned, income removed), the same between countries, and changes."""
    regional = load_regional() if regional is None else regional
    R = rc.regional_matrix(regional, drop_flag_chars="u")
    Rb = rc.regional_matrix(regional, drop_flag_chars="ubp")  # changes across windows: drop breaks and provisional values too
    out = {
        "r1_levels_within": rc.screen_regions(rc.within_country_table(R, "level", WINDOWS["recent"])).assign(experiment="r1_levels_within"),
        "r1_levels_between": corr.screen(rc.between_country_table(R, WINDOWS["recent"]), exclusions=rc.MECHANICAL).assign(experiment="r1_levels_between"),
        "r2_changes_within": rc.screen_regions(rc.within_country_table(Rb, "change", WINDOWS["recent"], WINDOWS["early"])).assign(experiment="r2_changes_within"),
    }
    return out


def run_regional_perm_check(regional: pd.DataFrame | None = None, n_perm: int = 20000) -> dict[str, pd.DataFrame]:
    """RP1 / RP2: the R1 and R2 pair screens again with the within-country permutation p (`regional_south.screen_perm`); compare `significant` with `significant_asymptotic`."""
    regional = load_regional() if regional is None else regional
    R = rc.regional_matrix(regional, drop_flag_chars="u")
    Rb = rc.regional_matrix(regional, drop_flag_chars="ubp")
    return {
        "rp1_levels_perm": rs.screen_perm(rc.within_country_table(R, "level", WINDOWS["recent"]), n_perm).assign(experiment="rp1_levels_perm"),
        "rp2_changes_perm": rs.screen_perm(rc.within_country_table(Rb, "change", WINDOWS["recent"], WINDOWS["early"]), n_perm).assign(experiment="rp2_changes_perm"),
    }


def run_regional_hypotheses(regional: pd.DataFrame | None = None, screens: dict[str, pd.DataFrame] | None = None, n_extra: int = 8) -> dict[str, pd.DataFrame]:
    """R3 (region + year FE, SEs clustered by country, with and without Ireland, lead-lag verdicts) and R4 (does Italy's within-country
    slope differ?) for the regional analogues of the literature hypotheses plus the top `n_extra` robust R1 survivors."""
    regional = load_regional() if regional is None else regional
    R = rc.regional_matrix(regional, drop_flag_chars="ubp")
    Xw = rc.within_country_table(rc.regional_matrix(regional, drop_flag_chars="u"), "level", WINDOWS["recent"], income_control=False)
    pairs = [(h[0], h[1], h[2], h[3], h[4], h[5]) for h in rc.HYPOTHESES]
    if screens and "r1_levels_within" in screens:
        top = screens["r1_levels_within"]
        theme_of = regional.assign(indicator=regional.indicator.str.removeprefix("nuts2_")).groupby("indicator").theme.first()
        known = {frozenset((h[2], h[3])) for h in rc.HYPOTHESES}
        top = top[top.robust & (top.x.map(theme_of) != top.y.map(theme_of)) & ~top.apply(lambda r: frozenset((r.x, r.y)) in known, axis=1)].head(n_extra)  # same-theme pairs (labour vs labour) are near-definitional
        pairs += [(f"S{i + 1}", "R1 survivor", r.y, r.x, 0, f"{r.x} ~ {r.y} (R1)") for i, r in enumerate(top.itertuples())]
    irl = tuple(g for g in R.index.get_level_values("geo").unique() if g.startswith("IE"))
    rows, lag_rows, slope_rows = [], [], []
    for hid, theme, y, x, sign, label in pairs:
        if x not in R or y not in R:
            continue
        r, r_ex = corr.panel_fe(R, y, [x], entity_level="geo", cluster_country=True), corr.panel_fe(R, y, [x], exclude=irl, entity_level="geo", cluster_country=True)
        sl = rc.italy_slope_test(Xw, x, y)
        if sl:
            slope_rows.append({"id": hid, **sl})
        if not r:
            continue
        coef = r["coef"][x]
        rows.append({"id": hid, "theme": theme, "y": y, "x": x, "label": label, "expected_sign": sign, "selected_from_screen": hid.startswith("S"), "coef": coef, "p": r["p"][x], "n_obs": r["n_obs"],
                     "n_regions": r["n_countries"], "coef_ex_irl": r_ex.get("coef", {}).get(x), "p_ex_irl": r_ex.get("p", {}).get(x),
                     "sign_agrees": bool(coef * sign > 0) if sign else None})
        ll = corr.lead_lag(R, x, y, entity_level="geo", cluster_country=True)
        for k, g in ll.groupby("lag"):
            g = g.set_index("direction")
            if {"x_leads_y", "y_leads_x"} - set(g.index):
                continue
            sx, sy = g.p["x_leads_y"] < 0.05, g.p["y_leads_x"] < 0.05
            lag_rows.append({"id": hid, "x": x, "y": y, "lag": k, "p_x_leads_y": g.p["x_leads_y"], "p_y_leads_x": g.p["y_leads_x"], "coef_x_leads_y": g.coef["x_leads_y"],
                             "coef_y_leads_x": g.coef["y_leads_x"], "verdict": {(True, False): "x leads", (False, True): "y leads", (True, True): "both"}.get((bool(sx), bool(sy)), "neither")})
    panel_fe = pd.DataFrame(rows)
    if len(panel_fe):  # FDR over the pre-specified hypotheses only: pairs picked from the screen were selected on the same kind of data
        pre = ~panel_fe.selected_from_screen
        panel_fe.loc[pre, "q_value"] = corr.fdr(panel_fe[pre], 0.05).q_value
    slopes = pd.DataFrame(slope_rows)
    return {"r3_region_panel": panel_fe, "r3_lead_lag": pd.DataFrame(lag_rows), "r4_italy_slope": slopes}


def run_convergence(regional: pd.DataFrame | None = None) -> pd.DataFrame:
    """R5: beta-convergence of regional GDP per head (PPS), with and without the richest region of each country."""
    regional = load_regional() if regional is None else regional
    R = rc.regional_matrix(regional, drop_flag_chars="ubp")
    return pd.concat([rc.convergence(R, drop_top_region=False), rc.convergence(R, drop_top_region=True)], ignore_index=True)


def run_regional_all(regional: pd.DataFrame | None = None, out: Path = OUT) -> dict[str, pd.DataFrame]:
    regional = load_regional() if regional is None else regional
    out.mkdir(parents=True, exist_ok=True)
    results = run_regional_screen(regional)
    results.update(run_regional_perm_check(regional))
    results.update(run_regional_hypotheses(regional, results))
    results["r5_convergence"] = run_convergence(regional)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_south_all(regional: pd.DataFrame | None = None, extra: pd.DataFrame | None = None, flows: pd.DataFrame | None = None, spending: pd.DataFrame | None = None,
                  out: Path = OUT, n_perm: int = 2000) -> dict[str, pd.DataFrame]:
    """The "why the South" round (`regional_south`): S1 EU-wide family by EQI wave (plain and beyond income) with its permutation null, S2 Italy's rank among countries, S3 Italy-only exploratory family and null,
    S4 the two tracks side by side, S5 North-South gaps in standard deviations."""
    regional = load_regional() if regional is None else regional
    extra = load_regional_extra() if extra is None else extra
    flows = pd.read_parquet(PROCESSED / "italy_internal_migration.parquet") if flows is None else flows
    spending = pd.read_parquet(PROCESSED / "italy_cpt_spending.parquet") if spending is None else spending
    R = rc.regional_matrix(pd.concat([regional, extra], ignore_index=True), drop_flag_chars="u")
    s1 = pd.concat([rs.eu_wide(R, income_control=False), rs.eu_wide(R, income_control=True)], ignore_index=True)
    W = rs.within_country(rs.wave_table(R, rs.PRIMARY_WAVE), income_control=False)
    s1_null = rs.family_null(W, rs.EU_WIDE, rs.OUTCOMES, rc.country_of(W.index))
    s2 = rs.italy_placebo(R)
    T = rs.italy_table(R, flows, spending)
    s3 = rs.italy_exploratory(T, [*rs.EU_WIDE, *rs.ITALY_ONLY], n_perm=n_perm)
    s4 = rs.compare_tracks(s2, s3)
    s5 = rs.zone_gaps(T, [*rs.OUTCOMES, *rs.EU_WIDE, *rs.ITALY_ONLY])
    results = {"s1_eu_wide": s1, "s1_null": s1_null, "s2_italy_rank": s2, "s3_italy_only": s3, "s4_compare": s4, "s5_zone_gaps": s5, "s_italy_table": T.reset_index()}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


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


def _nuts2021_matrix() -> pd.DataFrame:
    """Region x year indicator table on the NUTS 2021 geography (harmonised `regional` plus `regional_extra`), the counterpart of the matrix `run_south_all` builds."""
    reg, ex = pd.read_parquet(PROCESSED / "regional_nuts2021.parquet"), pd.read_parquet(PROCESSED / "regional_extra_nuts2021.parquet")
    return rc.regional_matrix(pd.concat([reg, ex], ignore_index=True), drop_flag_chars="u")


def run_nuts_all(out: Path = OUT, n_perm: int = 2000) -> dict[str, pd.DataFrame]:
    """NUTS harmonisation round: N1 regions per window before and after, N2 agreement of old and new codes where Eurostat publishes both, N3 the crosswalk of the codes in use, N4 the "why the South" EU-wide family (`rs.eu_wide`)
    on the original and on the NUTS 2021 geography side by side (sensitivity of notebook 34 to the code mix)."""
    regional, extra = load_regional(), load_regional_extra()
    reg21 = pd.read_parquet(PROCESSED / "regional_nuts2021.parquet")
    cw = nuts.crosswalk()
    used = set(regional.geo) | set(extra.geo)
    inds = sorted(regional.indicator.unique())
    n1 = nuts.coverage(regional, reg21, {"early 2005-07": (2005, 2007), "recent 2015-19": (2015, 2019), "late 2021-23": (2021, 2023)}, inds)
    n2 = nuts.overlap_agreement(regional, cw)
    n3 = cw[cw.old.isin(used) & cw.kind.ne("same")].reset_index(drop=True)
    Ro, Rh = rc.regional_matrix(pd.concat([regional, extra], ignore_index=True), drop_flag_chars="u"), _nuts2021_matrix()
    frames = []
    for geography, R in (("original", Ro), ("nuts2021", Rh)):
        for ic in (False, True):
            frames.append(rs.eu_wide(R, income_control=ic, n_perm=n_perm).assign(geography=geography, income_control=ic))
    f = pd.concat(frames, ignore_index=True)
    key = ["x", "y", "wave", "income_control"]
    n4 = f[f.geography.eq("original")][[*key, "n", "rho", "q_value", "significant"]].merge(f[f.geography.eq("nuts2021")][[*key, "n", "rho", "q_value", "significant"]], on=key, suffixes=("_original", "_nuts2021"), how="outer")
    results = {"n1_coverage": n1, "n2_overlap": n2, "n3_crosswalk": n3, "n4_south_geography": n4}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_spatial_all(out: Path = OUT, n_perm: int = 2000, n_sim: int = 100) -> dict[str, pd.DataFrame]:
    """Spatial-dependence round on the NUTS 2021 geography (`analysis.spatial`): SP1 Moran's I of every column before and after filtering, SP2 the "why the South" EU-wide family plain and filtered for every EQI wave
    (plain and beyond income), SP3 Italy's rank among the countries on filtered data, SP4 calibration with simulated spatially smooth x (plain against filtered pipeline)."""
    R = _nuts2021_matrix()
    nb = SP.Neighbours(sorted(R.index.get_level_values(0).unique()))
    moran, fam, placebo = [], [], []
    for w in rs.WAVES:
        for ic in (False, True):
            X = rs.within_country(rs.wave_table(R, w), ic)
            t, d = SP.compare_family(X, nb, rs.EU_WIDE, rs.OUTCOMES, n_perm=n_perm)
            fam.append(t.assign(wave=w, income_control=ic, primary=(w == rs.PRIMARY_WAVE)))
            moran.append(d.assign(wave=w, income_control=ic))
            if w == rs.PRIMARY_WAVE and not ic:
                placebo.append(SP.italy_placebo_filtered(X, nb, rs.EU_WIDE, rs.OUTCOMES).assign(wave=w))
    X0 = rs.within_country(rs.wave_table(R, rs.PRIMARY_WAVE), False)
    sp4 = pd.concat([SP.spatial_null(X0, nb, rs.EU_WIDE, rs.OUTCOMES, n_sim=n_sim, rho=r) for r in (0.4, 0.8)], ignore_index=True)  # weak and strong autocorrelation (the EQI columns have Moran's I about 0.3-0.5)
    results = {"sp1_moran": pd.concat(moran, ignore_index=True), "sp2_family": pd.concat(fam, ignore_index=True), "sp3_italy_rank": pd.concat(placebo, ignore_index=True), "sp4_null": sp4}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def _screen_tables(R: pd.DataFrame, Rb: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The two within-country tables of the regional screens (R1 levels 2015-19, R2 changes between 2005-09 and 2015-19), as in `run_regional_screen`."""
    return {"levels": rc.within_country_table(R, "level", WINDOWS["recent"]), "changes": rc.within_country_table(Rb, "change", WINDOWS["recent"], WINDOWS["early"])}


def run_spatial_screens(out: Path = OUT, n_perm: int = 20000, n_sim: int = 50, n_perm_null: int = 500, seed: int = 0) -> dict[str, pd.DataFrame]:
    """The notebook 25 pair screens (levels and changes, within-country permutation p, BH over all pairs) on the original region codes, on the NUTS 2021 geography, and on the NUTS 2021 geography after spatial filtering of every column
    (SP5 pair table, SP6 counts, SP7 the null of the whole screen: columns shuffled within countries, and independent spatially smooth noise with autocorrelation 0.4 and 0.8, plain against filtered; BH discoveries per screen)."""
    regional = load_regional()
    reg21 = pd.read_parquet(PROCESSED / "regional_nuts2021.parquet")
    orig = _screen_tables(rc.regional_matrix(regional, drop_flag_chars="u"), rc.regional_matrix(regional, drop_flag_chars="ubp"))
    R21, Rb21 = rc.regional_matrix(reg21, drop_flag_chars="u"), rc.regional_matrix(reg21, drop_flag_chars="ubp")
    new = _screen_tables(R21, Rb21)
    nb = SP.Neighbours(sorted(set(R21.index.get_level_values(0)) | set(Rb21.index.get_level_values(0))))
    pairs, counts = [], []
    for name in ("levels", "changes"):
        a, b = rs.screen_perm(orig[name], n_perm), rs.screen_perm(new[name], n_perm)
        XF, diag = nb.filter_table(new[name])
        c = rs.screen_perm(XF, n_perm)
        keep = ["x", "y", "n", "rho", "p", "q_value", "significant"]
        t = b[keep].merge(c[keep], on=["x", "y"], suffixes=("", "_filtered"), how="outer")
        t["same_sign"] = np.sign(t.rho) == np.sign(t.rho_filtered)
        pairs.append(t.assign(screen=name))
        counts.append({"screen": name, "pairs_original": len(a), "significant_original": int(a.significant.sum()), "pairs_nuts2021": len(b), "significant_nuts2021": int(b.significant.sum()),
                       "pairs_filtered": len(c), "significant_filtered": int(c.significant.sum()), "both_nuts2021_and_filtered": int((t.significant & t.significant_filtered).sum()), "sign_changed": int((~t.same_sign.fillna(True)).sum()),
                       "columns_with_positive_autocorrelation": int((diag.moran_z > 1.645).sum()), "columns_after_filter": int((diag.moran_z_filtered > 1.645).sum()), "columns": len(diag)})
    rng = np.random.default_rng(seed)
    rows = []
    for name in ("levels", "changes"):
        X = new[name]
        g = rc.country_of(X.index)
        for k in range(n_sim):
            variants = {"shuffled within countries": corr.shuffle_within(X, g, rng)}
            for rho in (0.4, 0.8):
                S = {}
                for c in X:
                    sr = X[c].dropna()
                    sr = sr[sr.index.isin(nb.pos)]
                    if len(sr) >= 20:
                        S[c] = pd.Series(SP.smooth_noise(sr.to_numpy(), nb.sub(np.array([nb.pos[i] for i in sr.index]))[0], rho, rng), index=sr.index)
                variants[f"smooth noise {rho}"] = rc.country_demean(pd.DataFrame(S).reindex(X.index))
            for label, V in variants.items():
                VF, _ = nb.filter_table(V)
                for pipe, M in (("plain", V), ("filtered", VF)):
                    t = rs.screen_perm(M, n_perm_null, seed=k)
                    rows.append({"screen": name, "null": label, "pipeline": pipe, "sim": k, "discoveries": int(t.significant.sum()), "share_p_below_005": float((t.p < 0.05).mean()), "n_tests": len(t)})
    results = {"sp5_screen_pairs": pd.concat(pairs, ignore_index=True), "sp6_screen_counts": pd.DataFrame(counts), "sp7_screen_null": pd.DataFrame(rows)}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_contracts_all(out: Path = OUT, n_perm: int = 20000, n_shuffle: int = 200) -> dict[str, pd.DataFrame]:
    """Contract timing and the real wage loss of the 2021-23 inflation shock across Istat's agreement groups (`analysis.contracts`): C1 leaf groups (staleness, standstill, real change to 2024 and 2025), C2 the primary test on the parent
    groups (mean of the sub-sectors, x shuffled within the private and public branch, BH over staleness and standstill against the 2024 and 2025 outcomes), C3 the same on the leaves (branch shuffle) and C4 within parents (the conservative
    null), C5 private groups only, C6 step thresholds of 0.15% and 0.5%, C7 the calm-period placebo (December 2016 to 2019), C8 the shuffled-outcome null, C9 the waiting-time indicators by sector, C10 the aggregates and C11 persistence."""
    from cifresignificative.analysis.hf import load_highfreq
    wages, groups = IC.load_wages(), IC.load_groups()
    dec = CT.december_inflation(load_highfreq())
    d24, d25 = CT.window_table(wages, groups, dec, 2021, 2024), CT.window_table(wages, groups, dec, 2021, 2025)
    leaves = d24.join(d25[["real_change", "nominal_change"]].rename(columns=lambda c: f"{c}_25"))
    clusters = CT.cluster_table(d24).join(CT.cluster_table(d25)[["real_change"]].rename(columns={"real_change": "real_change_25"}))
    ys = ("real_change", "real_change_25")
    placebo = CT.cluster_table(CT.window_table(wages, groups, dec, *CT.PLACEBO))
    thresholds = pd.concat([CT.family(CT.cluster_table(CT.window_table(wages, groups, dec, 2021, 2024, step=st)), ("real_change",), n_perm=n_perm).assign(step=st) for st in (0.0015, 0.0025, 0.005)], ignore_index=True)
    results = {"c1_leaf_groups": leaves.reset_index(), "c2_cluster_family": CT.family(clusters, ys, n_perm=n_perm), "c3_leaf_family_branch": CT.family(leaves, ys, n_perm=n_perm),
               "c4_leaf_family_within_parent": CT.family(leaves, ys, by="parent", n_perm=n_perm), "c5_cluster_private": CT.family(clusters[clusters.branch == "private"], ("real_change",), by="branch", n_perm=n_perm, n_min=15),
               "c6_cluster_thresholds": thresholds, "c7_placebo_2016_19": CT.family(placebo, ("real_change",), n_perm=n_perm), "c8_null": CT.shuffle_null(clusters, ys, n_shuffle=n_shuffle),
               "c9_tension": CT.tension_table(IC.load_tension()), "c10_aggregates": pd.concat([CT.aggregates_table(wages, groups, dec, 2021, e).assign(end=e) for e in (2024, 2025)]).reset_index(),
               "c11_persistence": CT.persistence(clusters.assign(real_change=clusters.real_change)), "c12_clusters": clusters.reset_index()}
    cyc, cmap = CN.primary_cycles(CN.load()), CN.load_map()
    timing = CT.renewal_timing(cyc, cmap, 2021, CN.ASOF)
    xs = (*CT.XS_CNEL, *CT.XS)
    leaves_v = CT.with_renewals(leaves, timing)
    leaves_a = CT.with_renewals(leaves, timing, ("verified", "partly verified"))
    cl_v, cl_a = (CT.cluster_table(t, extra=(*CT.XS_CNEL, "months_to_renewal")) for t in (leaves_v, leaves_a))
    for cl, t in ((cl_v, leaves_v), (cl_a, leaves_a)):
        cl["real_change_25"] = CT.cluster_table(t.assign(real_change=t.real_change_25)).real_change
    p_t = CT.renewal_timing(cyc, cmap, CT.PLACEBO[0], CN.ASOF)
    p_cl = CT.cluster_table(CT.with_renewals(CT.window_table(wages, groups, dec, *CT.PLACEBO), p_t), extra=CT.XS_CNEL)
    results |= {"c13_renewal_leaves": timing.reset_index(), "c14_cnel_cluster_family": CT.family(cl_v, ys, CT.XS_CNEL, n_perm=n_perm, n_min=15), "c15_proxy_same_clusters": CT.family(cl_v, ys, CT.XS, n_perm=n_perm, n_min=15),
                "c16_cnel_leaf_family": CT.family(leaves_v, ys, CT.XS_CNEL, n_perm=n_perm, n_min=15), "c17_cnel_cluster_with_partly": CT.family(cl_a, ys, xs, n_perm=n_perm, n_min=15), "c18_proxy_agreement": CT.proxy_agreement(leaves_a),
                "c19_cnel_placebo_2016_19": CT.family(p_cl, ("real_change",), CT.XS_CNEL, n_perm=n_perm, n_min=15), "c20_cnel_null": CT.shuffle_null(cl_v, ys, CT.XS_CNEL, n_shuffle=n_shuffle)}
    hf = load_highfreq()
    decs = {k: CT.december_inflation(hf, indicator=v) for k, v in CT.DEFLATOR_INDICATORS.items()}
    defl = CT.deflator_of(groups)
    cl_sector = defl.reindex(leaves.index).groupby(leaves.parent).first().rename("sector")  # a parent's sector is its leaves' (the parent code itself can be a bare aggregate, as Z2520 for agriculture)
    sec = leaves.assign(real_change=CT.sector_real_change(leaves, defl, decs, 2021, 2024), real_change_25=CT.sector_real_change(leaves, defl, decs, 2021, 2025, "nominal_change_25"))
    sec_cl = CT.cluster_table(sec, extra=("real_change_25",))
    sec_v = CT.with_renewals(sec, timing)
    sec_v_cl = CT.cluster_table(sec_v, extra=(*CT.XS_CNEL, "real_change_25"))
    results |= {"c21_sector_deflator_proxy": CT.family(sec_cl, ys, CT.XS, n_perm=n_perm), "c22_sector_deflator_cnel": CT.family(sec_v_cl, ys, CT.XS_CNEL, n_perm=n_perm, n_min=15),
                "c24_proxy_within_sector": CT.family(clusters.join(cl_sector), ys, CT.XS, by="sector", n_perm=n_perm), "c25_cnel_within_sector": CT.family(cl_v.join(cl_sector), ys, CT.XS_CNEL, by="sector", n_perm=n_perm, n_min=15),
                "c26_timing_by_sector": clusters.join(cl_sector).join(cl_v[[*CT.XS_CNEL, "months_to_renewal"]]).groupby("sector").agg(n=("real_change", "size"), staleness=("staleness_months", "mean"), months_since_renewal=("months_since_renewal", "mean"), months_expired=("months_expired", "mean"), real_change_national=("real_change", "mean")).reset_index(),
                "c23_deflators": pd.DataFrame([{"component": k, "start": 2021, "end": e, "cumulative_pct": CT.cumulative_inflation(v, 2021, e)} for k, v in decs.items() for e in (2024, 2025)])}
    # employee weights (CNEL-INPS 2025 counts of the mapped agreement codes; public agreements have none, so the weighted tests are private-sector tests)
    lw = CT.leaf_weights(cmap, CW.load())
    pw = CT.parent_weights(leaves, lw).rename_axis("group")
    cl_n = leaves.groupby("parent").size()
    results |= {"c27_weighted_summary": pd.concat([CT.weighted_summary(leaves, lw, y).assign(outcome=y) for y in ys]), "c28_weighted_proxy": CT.weighted_family(clusters, pw, ys, CT.XS, n_perm=n_perm),
                "c29_unweighted_same_rows": CT.family(clusters[pw.reindex(clusters.index).notna()], ys, CT.XS, n_perm=n_perm, n_min=15), "c30_weighted_cnel": CT.weighted_family(cl_v, pw, ys, CT.XS_CNEL, n_perm=n_perm),
                "c31_weights": pd.DataFrame({"employees": pw, "n_leaves": cl_n}).join(clusters[["real_change", "branch"]]).rename_axis("group").reset_index(),
                "c32_weighted_leaf_proxy": CT.weighted_family(leaves, lw, ys, CT.XS, n_perm=n_perm), "c33_weighted_null": CT.weighted_null(clusters, pw, ys, CT.XS, n_shuffle=n_shuffle)}
    # all groups: the private CNEL-INPS counts plus the public Conto annuale heads (31 December 2021)
    law = CT.all_leaf_weights(lw, CT.public_leaf_weights(CA.load_map(), CA.load(year=2021)))
    paw = CT.parent_weights(leaves, law).rename_axis("group")
    results |= {"c34_weighted_summary_all": pd.concat([CT.weighted_summary(leaves, law, y).assign(outcome=y) for y in ys]), "c35_weights_all": pd.DataFrame({"employees": paw, "n_leaves": cl_n}).join(clusters[["real_change", "branch"]]).rename_axis("group").reset_index(),
                "c36_weighted_proxy_all": CT.weighted_family(clusters, paw, ys, CT.XS, n_perm=n_perm), "c37_unweighted_same_rows_all": CT.family(clusters[paw.reindex(clusters.index).notna()], ys, CT.XS, n_perm=n_perm, n_min=15),
                "c38_weighted_cnel_all": CT.weighted_family(cl_v, paw, ys, CT.XS_CNEL, n_perm=n_perm), "c39_weighted_leaf_proxy_all": CT.weighted_family(leaves, law, ys, CT.XS, n_perm=n_perm),
                "c40_weighted_null_all": CT.weighted_null(clusters, paw, ys, CT.XS, n_shuffle=n_shuffle)}
    # variants of the weighting: Conto annuale year (start or end of the window), special-statute regions and Bolzano/Trento schools added, parent outcomes as unweighted or employee-weighted means of the leaves
    rows = []
    for year in (2021, 2024):
        for ext in (False, True):
            wv = CT.all_leaf_weights(lw, CT.public_leaf_weights(CA.load_map(), CA.load(year=year), extended=ext))
            tags = {"year": year, "extended": ext}
            pwv = CT.parent_weights(leaves, wv)
            rows.append(CT.weighted_family(clusters, pwv.rename_axis("group"), ys, CT.XS, n_perm=n_perm).assign(**tags, parent_mean="unweighted", level="parent", family="proxy"))
            wc, wc_v = CT.weighted_cluster_table(leaves, wv), CT.weighted_cluster_table(leaves_v, wv, extra=CT.XS_CNEL)
            rows += [CT.weighted_family(wc, wc.employees, ys, CT.XS, n_perm=n_perm).assign(**tags, parent_mean="weighted", level="parent", family="proxy"),
                     CT.weighted_family(wc_v, wc_v.employees, ys, CT.XS_CNEL, n_perm=n_perm).assign(**tags, parent_mean="weighted", level="parent", family="cnel"),
                     CT.weighted_family(leaves, wv, ys, CT.XS, n_perm=n_perm).assign(**tags, parent_mean="", level="leaf", family="proxy")]
    results["c41_weighted_variants"] = pd.concat(rows, ignore_index=True)
    wc21 = CT.weighted_cluster_table(leaves, law)
    results["c42_parent_weighted_means"] = wc21.join(clusters[["real_change"]].rename(columns={"real_change": "real_change_unweighted_mean"})).reset_index()
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_funds_all(out: Path = OUT, n_perm: int = 20000, n_shuffle: int = 100) -> dict[str, pd.DataFrame]:
    """EU cohesion payments (2007-2013 programme) against the change of regional outcomes between 2005-07 and 2015-19, within countries and beyond the initial income (`analysis.eu_funds`): F1 the descriptive table of the
    support by region, F2 the pre-specified family (two funds measures x five outcomes, within-country permutation p, BH), F3 the same plain and after spatial filtering, F4 variants (all funds with the EAFRD, the pre-2004 member states
    only, no income control, a 2005-09 baseline), F5 Italy's slope against the other countries' (plain and filtered), F6 the shuffled null."""
    reg21 = pd.read_parquet(PROCESSED / "regional_nuts2021.parquet")
    funds = EP.load_period()
    pred = EF.predictors(reg21, funds)
    W = EF.build_table(reg21, funds)
    g = rc.country_of(W.index)
    nb = SP.Neighbours(sorted(set(reg21.geo)))
    fam, _ = SP.compare_family(W, nb, EF.XS, EF.OUTCOMES, n_perm=n_perm)
    plain = rs.pair_family(W, EF.XS, EF.OUTCOMES, g, n_perm=n_perm)
    variants = []
    for name, tab in (("all funds (with the EAFRD)", W), ("pre-2004 member states only", W[W.index.str[:2].isin(EF.EU15)]), ("no income control", EF.build_table(reg21, funds, income_control=False)), ("baseline 2005-09", EF.build_table(reg21, funds, early=(2005, 2009)))):
        xs = ["funds_all_per_head"] if name.startswith("all funds") else EF.XS
        variants.append(rs.pair_family(tab, xs, EF.OUTCOMES, rc.country_of(tab.index), n_perm=n_perm).assign(variant=name))
    desc = pred.assign(country=pred.index.str[:2]).reset_index()
    results = {"f1_funds_by_region": desc, "f2_family": plain, "f3_family_filtered": fam, "f4_variants": pd.concat(variants, ignore_index=True),
               "f5_italy_rank": SP.italy_placebo_filtered(W, nb, EF.XS, EF.OUTCOMES), "f6_null": rs.family_null(W, EF.XS, EF.OUTCOMES, g, n_shuffle=n_shuffle)}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_nuts1_south(out: Path = OUT, n_perm: int = 2000) -> dict[str, pd.DataFrame]:
    """The EU-wide "why the South" family (`regional_south.eu_wide`, EQI against the six regional gaps) with Germany (16 Laender) and Belgium (3 regions) added at NUTS1, where the EQI exists only at NUTS1 (`analysis.nuts1`: outcomes aggregated
    from NUTS2 by sums and population weights). The two countries enter only at NUTS1 (their NUTS2 rows and structural-business predictors are left out), and spatial filtering is not applied (it needs NUTS1 boundaries). N1 compares the family
    without and with them (all waves, plain and beyond income), N2 Italy's rank among the countries without and with them, N3 the within-country rank correlations of Germany and Belgium themselves."""
    reg21, extra21 = pd.read_parquet(PROCESSED / "regional_nuts2021.parquet"), pd.read_parquet(PROCESSED / "regional_extra_nuts2021.parquet")
    de_be = ("DE", "BE")
    R0 = rc.regional_matrix(pd.concat([reg21, extra21], ignore_index=True), drop_flag_chars="u")
    eqi1 = EQI.load(nuts1=de_be)
    eqi1 = eqi1[eqi1.geo.str.len().eq(3)]
    alt = pd.concat([reg21[~reg21.country.isin(de_be)], N1.to_nuts1(reg21, de_be), extra21[~extra21.country.isin(de_be)], eqi1], ignore_index=True)
    R1 = rc.regional_matrix(alt, drop_flag_chars="u")
    key = ["x", "y", "wave", "income_control"]
    rows = []
    for ic in (False, True):
        f0, f1 = rs.eu_wide(R0, income_control=ic, n_perm=n_perm).assign(geography="baseline"), rs.eu_wide(R1, income_control=ic, n_perm=n_perm).assign(geography="with DE and BE at NUTS1")
        rows += [f0, f1]
    f = pd.concat(rows, ignore_index=True)
    cols = ["n", "rho", "q_value", "significant"]
    n1 = f[f.geography == "baseline"][[*key, *cols]].merge(f[f.geography != "baseline"][[*key, *cols, "loo_min_abs_rho", "loo_sign_flips"]], on=key, suffixes=("_baseline", "_with_de_be"), how="outer")
    p0, p1 = rs.italy_placebo(R0).assign(geography="baseline"), rs.italy_placebo(R1).assign(geography="with DE and BE at NUTS1")
    W1 = rs.within_country(rs.wave_table(R1, rs.PRIMARY_WAVE), income_control=True)
    from scipy import stats
    rows = []
    for c in de_be:
        sub = W1[W1.index.str[:2] == c]
        for x in ("eqi", "eqi_quality", "eqi_impartiality", "eqi_corruption"):
            for y in rs.OUTCOMES:
                d = sub[[x, y]].dropna()
                if len(d) >= 3:
                    rows.append({"country": c, "x": x, "y": y, "n": len(d), "rho": float(stats.spearmanr(d[x], d[y])[0])})
    results = {"n1_family_comparison": n1, "n2_italy_rank": pd.concat([p0, p1], ignore_index=True), "n3_de_be_within": pd.DataFrame(rows)}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


HF_CYCLE_PAIRS = [("m_esi", "m_industrial_production"), ("m_industry_confidence", "m_industrial_production"), ("m_consumer_confidence", "m_retail_volume"),
                  ("m_industrial_production", "m_unemployment_rate"), ("m_esi", "m_unemployment_rate"), ("m_services_confidence", "m_unemployment_rate")]
HF_EVENT_SERIES = {"spread_ITA_DEU": "Italy-Germany 10y spread (pp)", "m_long_term_yield": "Italy 10y yield (%)", "m_esi": "Italy economic sentiment", "m_unemployment_rate": "Italy unemployment rate (%)"}


def italian_events(start: str = "2005-01-01") -> pd.DataFrame:
    """Dated events since `start`: Italian government changes, Italian policies (data/external) and ECB decisions."""
    ext = Path("data/external")
    gov = pd.read_csv(ext / "italy_governments.csv", parse_dates=["start"])
    pol = pd.read_csv(ext / "italy_policies.csv", parse_dates=["date"])
    ecb = pd.read_csv(ext / "ecb_events.csv", parse_dates=["date"])
    ev = pd.concat([gov.rename(columns={"government": "event", "start": "date"}).assign(type="government")[["event", "date", "type"]],
                    pol.rename(columns={"title": "event"}).assign(type="policy")[["event", "date", "type"]],
                    ecb.rename(columns={"title": "event"}).assign(type="ecb")[["event", "date", "type"]]], ignore_index=True)
    return ev[ev.date >= start].sort_values("date").reset_index(drop=True)


EVIDENCE_RANK = {"unverified": 0, "partly verified": 1, "verified": 2}


def evidence_levels() -> dict[str, pd.DataFrame]:
    """Rating changes at each distinct evidence level, broadest first: a level that keeps the same rows as the previous one is dropped (when every row is `verified` there is only "all"), so no family repeats the same events."""
    out: dict[str, pd.DataFrame] = {}
    for name, level in (("all", "unverified"), ("partly verified or better", "partly verified"), ("verified", "verified")):
        ra = rating_actions(level)
        if len(ra) and (not out or len(ra) < len(list(out.values())[-1])): out[name] = ra
    return out


def rating_actions(min_evidence: str = "unverified") -> pd.DataFrame:
    """Italy sovereign rating/outlook changes (data/external/rating_actions.csv; changes only, confirmations excluded) as `event, date, type, action, agency`. Kept apart from `italian_events` so the monthly event study is unchanged; `min_evidence` keeps rows at or above that level of the ladder `unverified` < `partly verified` (two sources agree on the day) < `verified` (the day is in the agency's own record)."""
    ra = pd.read_csv(Path("data/external/rating_actions.csv"), parse_dates=["date"], keep_default_na=False)
    ra = ra[ra.evidence.map(EVIDENCE_RANK) >= EVIDENCE_RANK[min_evidence]]
    score = {"negative": -1, "stable": 0, "positive": 1}
    better = np.where(ra.action == "upgrade", 1, np.where(ra.action == "downgrade", -1, ra.outlook_to.map(score) - ra.outlook_from.map(score)))
    ra = ra.assign(event=ra.agency + " " + ra.action + " " + ra.rating_to.where(ra.action != "outlook", ra.outlook_to), type="rating", direction=-np.sign(better).astype(int))  # +1: spread expected to widen
    return ra[["event", "date", "type", "action", "agency", "evidence", "direction"]].sort_values("date").reset_index(drop=True)


def run_hf_stationarity(hf: pd.DataFrame) -> pd.DataFrame:
    extra = [("m_hicp_inflation", "level"), ("q_labour_cost_yoy", "level"), ("m_esi", "diff"), ("m_consumer_confidence", "level")]  # alternative transforms for comparison
    return H.stationarity_summary(H.stationarity_report(hf, [*H.TRANSFORMS.items(), *extra]))


def run_hf_spread(hf: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([H.spread_model(hf, c) for c in ("ITA", "ESP", "FRA", "PRT", "GRC")], ignore_index=True)
    return out.assign(q_value=corr.fdr(out, 0.05).q_value)


def run_hf_cycle(hf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """H2: panel lead-lag (country + month effects), per-country prewhitened cross-correlation, and Italy's lead against the others."""
    M = H.hf_panel(hf, sorted({i for p in HF_CYCLE_PAIRS for i in p}), countries=[c for c in hf.iso3.unique() if c != "EU27"])
    panel = H.lead_lag_table(M, HF_CYCLE_PAIRS, lags=(1, 3, 6, 12))
    countries = pd.concat([H.country_lead(hf, x, y) for x, y in HF_CYCLE_PAIRS], ignore_index=True)
    italy = []
    for key, g in countries.groupby(["x", "y"]):
        x, y = cast(tuple[str, str], key)
        rk = H.italy_rank(g)
        it = g[g.iso3 == "ITA"]
        if rk and len(it):
            italy.append({"x": x, "y": y, "italy_lag": int(it.lag.iloc[0]), "italy_r": rk["italy_value"], "rank_by_abs_r": rk["rank"], "n_countries": rk["n_countries"],
                          "median_lag_all": float(g.lag.median()), "share_significant": float(g.significant.mean())})
    q = H.hf_panel(hf, ["q_gdp_growth", "q_employment_ths", "q_hours_ths"], countries=[c for c in hf.iso3.unique() if c != "EU27"])
    panel = pd.concat([panel, H.lead_lag_table(q, [("q_gdp_growth", "q_employment_ths"), ("q_gdp_growth", "q_hours_ths")], lags=(1, 2, 4), own_lags=2)], ignore_index=True)
    return {"hf2_panel": panel, "hf2_countries": countries, "hf2_italy": pd.DataFrame(italy)}


def run_hf_passthrough(hf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    windows = {"2005Q1-2021Q2": ("2005-01-01", "2021-06-30"), "2021Q3-2023Q4 (energy shock)": ("2021-07-01", "2023-12-31")}
    panel = pd.DataFrame([{"window": k, **H.passthrough_panel(hf, w)} for k, w in windows.items()])
    by = pd.concat([H.passthrough_by_country(hf, w).assign(window=k) for k, w in windows.items() if k.startswith("2005")], ignore_index=True)
    ranks = []
    for k, g in by.groupby("window"):
        rk = H.italy_rank(g, "pass_through")
        if rk:
            ranks.append({"window": k, **rk, "median_all": float(g.pass_through.median())})
    return {"hf3_passthrough_panel": panel, "hf3_passthrough_countries": by, "hf3_passthrough_italy": pd.DataFrame(ranks)}


def run_hf_events(hf: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    events = italian_events()
    dates = list(zip(events.event, events.date))
    kind = events.set_index("event").type.to_dict()
    series = {"spread_ITA_DEU": H.spread_series(hf, "ITA", "DEU"), "m_long_term_yield": H.wide_hf(hf, "m_long_term_yield")["ITA"],
              "m_esi": H.wide_hf(hf, "m_esi")["ITA"], "m_unemployment_rate": H.wide_hf(hf, "m_unemployment_rate")["ITA"]}
    rows = []
    for name, s in series.items():
        r = H.event_study(s, dates, window)
        rows.append(r.assign(series=name, label=HF_EVENT_SERIES[name], type=r.event.map(kind)))
    out = pd.concat(rows, ignore_index=True)
    return out.assign(q_value=corr.fdr(out, 0.05).q_value).sort_values("p").reset_index(drop=True)


def run_hf_all(hf: pd.DataFrame | None = None, out: Path = OUT) -> dict[str, pd.DataFrame]:
    hf = H.load_highfreq() if hf is None else hf
    out.mkdir(parents=True, exist_ok=True)
    results = {"hf0_stationarity": run_hf_stationarity(hf), "hf1_spread": run_hf_spread(hf), **run_hf_cycle(hf), **run_hf_passthrough(hf), "hf4_events": run_hf_events(hf)}
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def daily_events() -> pd.DataFrame:
    """All dated events of the daily study: government, policy, ECB (`italian_events`) plus Italian rating changes. Events on the same date give an
    identical statistic (the same ECB decision sits in both the ECB and policy files), so they are merged into one test: names joined, types joined."""
    ra = rating_actions()
    ev = pd.concat([italian_events(), ra[["event", "date", "type"]]], ignore_index=True)
    return ev.groupby("date", as_index=False).agg(event=("event", lambda e: " / ".join(dict.fromkeys(e))), type=("type", lambda t: "+".join(sorted(set(t))))).reset_index(drop=True)


def run_daily_events(dy: pd.DataFrame, horizons=DY.HORIZONS) -> pd.DataFrame:
    """Event study of the daily ITA-DEU spread for every dated event (one test per date), one BH family over all (event, horizon) tests."""
    ev = daily_events()
    r = DY.event_study_daily(DY.spread_daily(dy), list(zip(ev.event, ev.date, strict=True)), horizons)
    r = r.join(ev.set_index("date").type, on="date")
    return r.assign(q_value=corr.fdr(r, 0.05).q_value).sort_values("p").reset_index(drop=True)


def run_daily_pooled(dy: pd.DataFrame) -> pd.DataFrame:
    """Direction-aligned pooled test for rating changes (downgrade / worse outlook should widen the spread), with robustness over the volatility
    window and over evidence quality; one BH family over all rows. Placebo days avoid every dated event, not only the rating ones."""
    x = DY.spread_daily(dy)
    allev = daily_events()
    rows = []
    for evidence, ra in evidence_levels().items():
        for vol in (60, 20):
            rows.append(DY.pooled_signed(x, ra, vol_window=vol, exclude=allev.date).assign(evidence=evidence, vol_window=vol))
    out = pd.concat(rows, ignore_index=True)
    return out.assign(q_value=corr.fdr(out, 0.05, p_col="p_one_sided").q_value)


def run_daily_rating_support(dy: pd.DataFrame, vol_window: int = 60) -> pd.DataFrame:
    """Sensitivity of the pooled rating test to the evidence level of each date (descriptive, no new FDR family): the distinct levels of `evidence_levels()` x three horizons. With every date verified against the agencies' own records there is a single level."""
    x, allev = DY.spread_daily(dy), daily_events()
    return pd.concat([DY.pooled_signed(x, v, vol_window=vol_window, exclude=allev.date).assign(subset=k) for k, v in evidence_levels().items()], ignore_index=True)


def run_daily_placebo(dy: pd.DataFrame, n_rep: int = 200) -> pd.DataFrame:
    return pd.DataFrame([DY.placebo_discoveries(DY.spread_daily(dy), n_events=len(daily_events()), n_rep=n_rep)])


PEERS = ("ESP", "FRA", "PRT")  # Italy's spread over each peer; Germany stays the primary reference (d1-d3)
GREECE_PSI = (pd.Timestamp("2011-03-01"), pd.Timestamp("2013-03-31"))  # yields above 30%, a different benchmark bond and the March 2012 PSI exchange; the end leaves the 60-day volatility window clear of 2012


def run_daily_peer_events(dy: pd.DataFrame, horizons=DY.HORIZONS) -> pd.DataFrame:
    """Event study of the Italy-minus-peer spread (Spain, France, Portugal) for the same dated events as `run_daily_events`. `q_peer` = BH within each peer's family (279 tests), `q_value` = BH over all peers together (837 tests)."""
    ev = daily_events()
    rows = []
    for peer in PEERS:
        r = DY.event_study_daily(DY.spread_daily(dy, "ITA", peer), list(zip(ev.event, ev.date, strict=True)), horizons).assign(peer=peer)
        rows.append(r.join(ev.set_index("date").type, on="date").assign(q_peer=lambda d: corr.fdr(d, 0.05).q_value))
    out = pd.concat(rows, ignore_index=True)
    return out.assign(q_value=corr.fdr(out, 0.05).q_value).sort_values("p").reset_index(drop=True)


def run_daily_peer_pooled(dy: pd.DataFrame) -> pd.DataFrame:
    """Direction-aligned pooled test for the rating changes on each Italy-minus-peer spread (same events, horizons and volatility windows as `run_daily_pooled`). `q_peer`: BH within a peer (6 rows), `q_value`: over the three peers (18 rows);
    Germany is repeated as `reference` (not part of either family). `mde_z` = smallest mean signed z reachable with one-sided p < 0.05 (1.645 placebo sd), `mde_z_family` the same at Bonferroni over the 18 peer rows."""
    allev, ra = daily_events(), list(evidence_levels().values())[-1]
    rows = []
    for peer in (*PEERS, "DEU"):
        x = DY.spread_daily(dy, "ITA", peer)
        for vol in (60, 20):
            rows.append(DY.pooled_signed(x, ra, vol_window=vol, exclude=allev.date).assign(peer=peer, vol_window=vol, reference=peer == "DEU"))
    out = pd.concat(rows, ignore_index=True)
    fam = out[~out.reference]
    q_all = corr.fdr(fam, 0.05, p_col="p_one_sided").q_value
    q_peer = pd.concat([corr.fdr(g, 0.05, p_col="p_one_sided").q_value for _, g in fam.groupby("peer", sort=False)])
    out["q_value"], out["q_peer"] = q_all.reindex(out.index), q_peer.reindex(out.index)
    return out.assign(mde_z=stats.norm.isf(0.05) * out.null_sd, mde_z_family=stats.norm.isf(0.05 / len(fam)) * out.null_sd)


def run_daily_peer_placebo(dy: pd.DataFrame, n_rep: int = 200) -> pd.DataFrame:
    """Shuffled-date null for every peer spread (93 random dates, BH over the peer's 279 tests): the mean number of discoveries must be about zero."""
    n = len(daily_events())
    return pd.DataFrame([DY.placebo_discoveries(DY.spread_daily(dy, "ITA", peer), n_events=n, n_rep=n_rep) | {"peer": peer} for peer in PEERS])


def run_daily_greece(dy: pd.DataFrame, horizons=DY.HORIZONS) -> pd.DataFrame:
    """Descriptive block, no test family: the Greece-minus-Germany spread around the same Italian events and the pooled rating test, with and without the restructuring years
    (variant `after_psi`: events from 2013-04-01 only and no placebo day within `GREECE_PSI`, so the unstable 2011-12 volatility cannot enter the placebo). Greek quotes in 2011-12 are another benchmark bond (see `data/external/source/daily_yields/README.md`)."""
    x, ev, ra = DY.spread_daily(dy, "GRC", "DEU"), daily_events(), list(evidence_levels().values())[-1]
    psi_days = list(x.index[(x.index >= GREECE_PSI[0]) & (x.index <= GREECE_PSI[1])])
    rows = []
    for variant, keep, excl in (("full", ev, None), ("after_psi", ev[ev.date > GREECE_PSI[1]], psi_days)):
        r = DY.event_study_daily(x, list(zip(keep.event, keep.date, strict=True)), horizons, exclude=excl)
        rows.append(r.assign(variant=variant, block="events"))
        k = ra if variant == "full" else ra[ra.date > GREECE_PSI[1]]
        rows.append(DY.pooled_signed(x, k, exclude=[*ev.date, *(excl or [])]).assign(variant=variant, block="ratings_pooled"))
    return pd.concat(rows, ignore_index=True)


def run_daily_all(dy: pd.DataFrame | None = None, out: Path = OUT) -> dict[str, pd.DataFrame]:
    dy = DY.load_daily() if dy is None else dy
    out.mkdir(parents=True, exist_ok=True)
    results = {"d1_events": run_daily_events(dy), "d2_ratings_pooled": run_daily_pooled(dy), "d3_placebo": run_daily_placebo(dy), "d4_peer_events": run_daily_peer_events(dy),
               "d5_peer_pooled": run_daily_peer_pooled(dy), "d6_peer_placebo": run_daily_peer_placebo(dy), "d7_greece": run_daily_greece(dy)}
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_regulation_all(out: Path = OUT, n_perm: int = 20000, n_rep: int = 200) -> dict[str, pd.DataFrame]:
    """G1-G6: Italy's position in the OECD PMR and EPL, and the sector test (regulation of the sector in 2018 against productivity growth of the sector, 2005-07 to 2017-19 as the primary specification,
    robustness: early and late windows, PMR 2023). BH over the primary family (pooled + 6 sectors) only; the robustness rows are descriptive."""
    pmr, epl = OR.load_pmr(), OR.load_epl()
    s = load_sectors64()
    rows, primary = [], None
    for wname, w in IP.WINDOWS.items():
        for year in (2018, 2023):
            d = RG_.sector_data(pmr, s, w, year)
            if (wname, year) == ("full", 2018):
                primary = d
            rows.append(RG_.test(d, n_perm=n_perm).assign(window=wname, pmr_year=year, primary=(wname, year) == ("full", 2018)))
    assert primary is not None
    results = {"g1_pmr_position": RG_.position(pmr), "g2_epl_gap": RG_.epl_gap(epl).reset_index(), "g3_sector_test": pd.concat(rows, ignore_index=True), "g4_italy_sectors": RG_.italy_rows(primary),
               "g5_italy_placebo": RG_.italy_placebo(primary), "g6_null": pd.DataFrame([RG_.shuffled_null(primary, n_rep=n_rep)])}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_bargaining_all(out: Path = OUT, n_perm: int = 20000, n_rep: int = 500) -> dict[str, pd.DataFrame]:
    """W1-W4: bargaining coverage and union density against the country pass-through of notebook 26 (`hf3_passthrough_countries`, rebuilt from the monthly data). BH over the two primary tests; the robustness
    rows (without Greece and Portugal, 2010-19 means, at least 6 observations) are descriptive."""
    cbc, tud = OB.load("cbc"), OB.load("tud")
    pt = H.passthrough_by_country(H.load_highfreq(), ("2005-01-01", "2021-06-30"))
    specs = {"primary": (BG.institution_means(cbc, tud), ()), "without GRC and PRT": (BG.institution_means(cbc, tud), ("GRC", "PRT")),
             "2010-19 means": (BG.institution_means(cbc, tud, (2010, 2019)), ()), "at least 6 observations": (BG.institution_means(cbc, tud, min_years=6), ())}
    assoc = pd.concat([BG.association(pt, inst, n_perm, exclude=ex).assign(spec=k, primary=k == "primary") for k, (inst, ex) in specs.items()], ignore_index=True)
    inst = specs["primary"][0]
    results = {"w1_institutions": inst.rename_axis("iso3").reset_index(), "w2_association": assoc, "w3_italy": BG.italy_position(pt, inst), "w4_null": pd.DataFrame([BG.shuffled_null(pt, inst, n_rep=n_rep)])}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_industry_all(out: Path = OUT) -> dict[str, pd.DataFrame]:
    """I1: industry panel (country x ~55 industries): per-country growth-on-intensity slopes, Italy's placebo rank and leave-one-section-out check."""
    out.mkdir(parents=True, exist_ok=True)
    results = IP.run(pd.read_parquet(PROCESSED / "euklems_growth.parquet"))
    for name, df in results.items():
        df.to_parquet(out / f"i1_{name}.parquet")
    return results


def run_latent_all(panel: pd.DataFrame | None = None, out: Path = OUT, k_max: int = 4) -> dict[str, pd.DataFrame]:
    """L1: PCA of the 2015-19 country x indicator table: parallel analysis, bootstrap and split-half stability, top loadings, Italy's scores over four windows."""
    panel = load_panel() if panel is None else panel
    M = corr.panel_matrix(panel)
    X, mu, sd = LT.prepare(M)
    Z = LT.standardise(X, mu, sd)
    pa = LT.parallel_analysis(Z)
    k = min(max(pa["n_retain"], 1), k_max)
    fit = LT.pca(Z, k)
    n = min(len(Z.columns), 10)
    results = {
        "summary": pd.DataFrame({"component": [f"PC{i + 1}" for i in range(n)], "eigenvalue": pa["real"][:n], "null_p95": pa["threshold"][:n], "explained": (pa["real"] / pa["real"].sum())[:n]}).assign(n_countries=len(Z), n_indicators=Z.shape[1], n_retain=pa["n_retain"]),
        "loadings": fit["loadings"].reset_index(names="indicator"),
        "scores": fit["scores"].reset_index(names="iso3"),
        "bootstrap": LT.bootstrap_stability(Z, k),
        "split_half": LT.split_half_congruence(M, k),
        "italy": LT.italy_over_time(M, fit["loadings"], mu, sd),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"l1_{name}.parquet")
    return results


def run_all(panel: pd.DataFrame | None = None, out: Path = OUT) -> dict[str, pd.DataFrame]:
    panel = load_panel() if panel is None else panel
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "e1_levels": run(Experiment("e1_levels"), panel),
        "e1_levels_raw": run(Experiment("e1_levels", control_income=False), panel).assign(experiment="e1_levels_raw"),
        "e2_changes": run(Experiment("e2_changes"), panel),
        "e3_within": run(Experiment("e3_within"), panel),
        "e3_regimes": run_regimes(panel),
        "e1_split_sample": run_split_sample(panel),
        "e1_split_sample_raw": run_split_sample(panel, income_control=False),
        "e5_italy_gap_gdp": corr.italy_gap_corr(panel, "gdp_real_pc"),
        "e5_italy_gap_employment": corr.italy_gap_corr(panel, "employment_rate_20_64"),
        "e4_lead_lag": run_lead_lag(panel),
        "e7_hypotheses": run_hypotheses(panel),
    }
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results
