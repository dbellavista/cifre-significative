"""National (country-year) experiments: the correlation funnel, hypotheses, lead-lag, industry panel and latent structure."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cifresignificative.studies.italy_productivity.analysis import corr
from cifresignificative.studies.italy_productivity.analysis import industry_panel as IP
from cifresignificative.studies.italy_productivity.analysis import latent as LT
from cifresignificative.studies.italy_productivity.analysis import regimes as RG
from cifresignificative.studies.italy_productivity.analysis.data import (
    PROCESSED,
    load_panel,
)

from .common import INCOME, OUT, WINDOWS

GDP_CONTROL = "log_gdp_real_pc"  # robustness control for the ratio-to-GDP hypotheses (a country's own cycle moves both sides)

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
    MG = M.assign(**{GDP_CONTROL: np.log(M[INCOME])})
    X = corr.cross_section(M, "level", WINDOWS["recent"])
    rows = []
    for hid, theme, y, x, sign, label in HYPOTHESES:
        r, r_ex = corr.panel_fe(M, y, [x]), corr.panel_fe(M, y, [x], exclude=(corr.IRELAND,))
        fit = corr.italy_fit(X, x, y)
        if not r:
            rows.append({"id": hid, "theme": theme, "y": y, "x": x, "label": label, "status": "insufficient data"})
            continue
        coef = r["coef"][x]
        rg = corr.panel_fe(MG, y, [x, GDP_CONTROL]) if y != INCOME else {}  # robustness column, not part of the pre-specified test
        rows.append({"id": hid, "theme": theme, "y": y, "x": x, "label": label, "expected_sign": sign, "coef": coef, "p": r["p"][x],
                     "n_obs": r["n_obs"], "n_countries": r["n_countries"], "coef_ex_irl": r_ex.get("coef", {}).get(x), "p_ex_irl": r_ex.get("p", {}).get(x),
                     "sign_agrees": bool(coef * sign > 0), "italy_resid_z": fit.get("resid_z"), "status": "ok",
                     "coef_gdp_control": rg.get("coef", {}).get(x), "p_gdp_control": rg.get("p", {}).get(x)})
    out = pd.DataFrame(rows)
    ok = out.status == "ok"
    out.loc[ok, "q_value"] = corr.fdr(out[ok], 0.05).q_value
    out["gdp_control"] = [_gdp_control_verdict(r) for r in out.itertuples()]
    return out


def _gdp_control_verdict(r) -> str:
    """How a hypothesis that passes the pre-specified test reads once log real GDP per head is held fixed (5% level, same sign)."""
    if r.status != "ok" or not r.q_value <= 0.05:
        return "not significant before"
    if r.y == INCOME:
        return "not applicable (the outcome is GDP per head)"
    return "holds" if r.p_gdp_control < 0.05 and r.coef_gdp_control * r.coef > 0 else "not distinguishable from the cycle"


NON_EURO = ("BGR", "CZE", "DNK", "HUN", "POL", "ROU", "SWE")  # national currency for most of 2005-2025: their euro amounts move with the exchange rate
PENSION_REAL = "log_pension_real_per_65plus"
PENSION_TESTS = [("H5", "edu_spend_pct_gdp"), ("H6", "neet_rate_15_29"), ("H7", "employment_rate_55_64")]


def pension_real_per_elderly(M: pd.DataFrame) -> pd.Series:
    """Log of old-age public spending per person aged 65+, deflated with the national HICP: a pension measure without GDP in it.

    Spending in euro = % of GDP x nominal GDP; the HICP index is the cumulated annual rate from each country's first year (its level is absorbed by
    the country effect). For the members outside the euro the euro amount moves with the exchange rate: see NON_EURO."""
    euro = M["pension_spend_pct_gdp"] / 100 * M["gdp_nominal_meur"] * 1e6
    elderly = M["population"] * M["wb_pop_65plus_pct"] / 100
    hicp = np.log1p(M["hicp_inflation"] / 100).groupby(level="iso3").cumsum()
    return (np.log(euro / elderly) - hicp).rename(PENSION_REAL)


def run_pension_real(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """E7b (exploratory, added after the results of E7 were seen): the three pension hypotheses with pension spending per person aged 65+ in real
    euro instead of % of GDP, for all members and for the euro-area members, with and without log real GDP per head as a control."""
    panel = load_panel() if panel is None else panel
    M = corr.panel_matrix(panel)
    M = M.assign(**{PENSION_REAL: pension_real_per_elderly(M), GDP_CONTROL: np.log(M[INCOME])})
    rows = []
    for hid, y in PENSION_TESTS:
        for sample, exclude in (("all members", ()), ("euro area", NON_EURO)):
            for spec, xs in (("pension % of GDP", ["pension_spend_pct_gdp"]), ("real per person 65+", [PENSION_REAL]), ("real per person 65+, GDP control", [PENSION_REAL, GDP_CONTROL])):
                r = corr.panel_fe(M, y, xs, exclude=exclude)
                if r:
                    rows.append({"id": hid, "y": y, "sample": sample, "spec": spec, "coef": r["coef"][xs[0]], "p": r["p"][xs[0]], "n_obs": r["n_obs"], "n_countries": r["n_countries"]})
    return pd.DataFrame(rows)


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
        "e7_pension_real": run_pension_real(panel),
    }
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results
