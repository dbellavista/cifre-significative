"""Correlation framework: synthetic planted/null data (no processed data needed) plus a shuffled-panel check."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.studies.italy_productivity.analysis import corr

PANEL = Path("data/processed/panel.parquet")


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(1)
    idx = pd.Index([f"C{i:02d}" for i in range(40)], name="iso3")
    base = rng.normal(size=40)
    X = pd.DataFrame({"a": base, "b": base + rng.normal(scale=0.6, size=40)}, index=idx)
    for i in range(30):
        X[f"noise{i}"] = rng.normal(size=40)
    return X


def test_planted_correlation_survives_and_noise_does_not(synthetic):
    res = corr.screen(synthetic)
    assert set(zip(res["x"], res["y"])) == {("a", "b")}
    assert res.robust.iloc[0] and res.rho.iloc[0] > 0.7


def test_fdr_monotone_and_bounded():
    df = pd.DataFrame({"p": [0.001, 0.04, 0.03, 0.5, 0.9]})
    out = corr.fdr(df)
    order = out.sort_values("p")
    assert (order.q_value.diff().dropna() >= 0).all() and (out.q_value >= out.p).all() and out.q_value.max() <= 1
    assert list(out.significant) == [True, False, False, False, False]


def test_size_scaled_and_near_duplicates():
    assert corr.size_scaled(["a_ths", "population", "gini", "x"], {"x": "thousand persons", "gini": "0-100"}) == ["a_ths", "population", "x"]
    rng = np.random.default_rng(3)
    base = rng.normal(size=40)
    X = pd.DataFrame({"p": base, "q": base + rng.normal(scale=0.02, size=40), "r": base + rng.normal(scale=0.5, size=40)},
                     index=pd.Index([f"C{i}" for i in range(40)], name="iso3"))
    res = corr.screen(X, drop_size_scaled=False)
    assert "q" not in set(res.x) | set(res.y) or not ((res.x == "p") & (res.y == "q")).any()  # the near-copy is not reported
    assert ((res.x == "p") & (res.y == "r")).any()


def test_residualise_removes_common_factor():
    rng = np.random.default_rng(4)
    idx = pd.Index([f"C{i}" for i in range(60)], name="iso3")
    dev = rng.normal(size=60)
    X = pd.DataFrame({"dev": dev, "u": dev + rng.normal(size=60), "v": dev + rng.normal(size=60)}, index=idx)
    assert corr.pair_screen(X[["u", "v"]], "pearson").rho.iloc[0] > 0.3  # shared factor makes them correlate
    R = corr.residualise(X, "dev")
    assert "dev" not in R and abs(corr.pair_screen(R, "pearson").rho.iloc[0]) < 0.3


def test_fdr_empty():
    assert corr.fdr(pd.DataFrame({"p": []})).empty


@pytest.mark.parametrize("x,y,trivial", [("gini", "wb_gini", True), ("arope", "arope_old", True), ("gdp_real_pc", "gini", False),
                                         ("wgi_x", "x", True), ("employment_rate_55_64", "employment_rate_60_64", True), ("gov_debt_pct_gdp", "gov_balance_pct_gdp", False), ("a", "b", False)])
def test_trivial_pair(x, y, trivial):
    assert corr.is_trivial_pair(x, y) is trivial


def test_min_coverage(synthetic):
    X = synthetic[["a", "b"]].copy()
    X.iloc[:25, 1] = np.nan  # only 15 overlapping rows
    assert corr.pair_screen(X, n_min=20).empty
    assert len(corr.pair_screen(X, n_min=10)) == 1


def test_loo_flags_single_country_driven_pair():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(30, 2)), columns=["u", "v"], index=pd.Index([f"C{i}" for i in range(30)], name="iso3"))
    X.iloc[0] = [30, 30]  # one extreme point manufactures a correlation
    pairs = corr.robustness(corr.pair_screen(X, "pearson"), X, "pearson")
    assert not pairs.robust.iloc[0]


def test_within_changes_remove_common_year_shock():
    idx = pd.MultiIndex.from_product([["AAA", "BBB"], range(2000, 2010)], names=["iso3", "year"])
    M = pd.DataFrame({"i": np.arange(20, dtype=float) ** 1.5}, index=idx)
    w = corr.within_changes(M)
    assert np.allclose(w.groupby(level="year").mean().dropna().to_numpy(), 0)  # a common shock is removed


@pytest.mark.skipif(not PANEL.exists(), reason="panel not built")
def test_shuffled_panel_yields_no_discoveries():
    panel = pd.read_parquet(PANEL)
    X = corr.cross_section(corr.panel_matrix(panel), "level", (2015, 2019))
    assert max(corr.permutation_null(X, n_perm=2)) <= 1  # FDR 5%: essentially nothing from pure noise


def test_italy_fit_flags_outlier():
    rng = np.random.default_rng(5)
    idx = pd.Index(["ITA"] + [f"C{i}" for i in range(29)], name="iso3")
    x = rng.normal(size=30)
    X = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(scale=0.3, size=30)}, index=idx)
    X.loc["ITA", "y"] += 5
    f = corr.italy_fit(X, "x", "y")
    assert f["resid_z"] > 5 and 1.5 < f["slope"] < 2.5


def _fake_panel_matrix(lead: bool):
    rng = np.random.default_rng(6)
    rows = []
    for c in range(20):
        x = rng.normal(size=25).cumsum() * 0.1 + rng.normal(size=25)
        y = np.r_[0, x[:-1]] * 1.0 + rng.normal(scale=0.3, size=25) if lead else rng.normal(size=25)
        rows += [(f"C{c:02d}", 2000 + t, x[t], y[t]) for t in range(25)]
    return pd.DataFrame(rows, columns=["iso3", "year", "x", "y"]).set_index(["iso3", "year"])


def test_lead_lag_detects_direction():
    ll = corr.lead_lag(_fake_panel_matrix(True), "x", "y", lags=(1,))
    p = ll.set_index("direction").p
    assert p["x_leads_y"] < 0.001 < p["y_leads_x"]


def test_panel_fe_returns_empty_for_thin_data():
    assert corr.panel_fe(_fake_panel_matrix(True).head(30), "y", ["x"]) == {}


@pytest.mark.skipif(not PANEL.exists(), reason="panel not built")
def test_run_lead_lag_verdicts():
    from cifresignificative.studies.italy_productivity.analysis import experiments as ex
    ll = ex.run_lead_lag(pd.read_parquet(PANEL), lags=(1,))
    assert set(ll.verdict) <= {"x leads", "y leads", "both", "neither"} and len(ll) >= 10


def _split_panel(flip: bool = False):
    """40 countries x 2005-2024: pair (a, b) correlated in every year; pair (c, d) only in 2005-14 (a 'period' effect); noise columns."""
    rng = np.random.default_rng(7)
    idx = pd.MultiIndex.from_product([[f"C{i:02d}" for i in range(40)], range(2005, 2025)], names=["iso3", "year"])
    cty = np.repeat(rng.normal(size=(40, 4)), 20, axis=0)  # persistent country components
    M = pd.DataFrame(cty + rng.normal(scale=0.3, size=cty.shape), index=idx, columns=["a", "b", "c", "d"])
    M["b"] = M.a + rng.normal(scale=1.5, size=len(M))  # noisy enough that window means stay below the duplicate cut-off
    late = M.index.get_level_values("year") >= 2015
    M.loc[late, "d"] = rng.normal(size=int(late.sum()))  # c-d link breaks in the second half
    M["c"] = M.d.where(~late, rng.normal(size=len(M)))
    for i in range(20):
        M[f"n{i}"] = np.repeat(rng.normal(size=40), 20) + rng.normal(size=len(M))
    return M


def test_split_sample_keeps_stable_pair_and_drops_period_pair():
    res = corr.split_sample(_split_panel(), n_min=20)
    rep = set(zip(res[res.replicated].x, res[res.replicated].y))
    assert ("a", "b") in rep and ("c", "d") not in rep
    assert {"rho_a", "rho_b", "q_value_a", "q_value_b", "replicated"} <= set(res.columns)


def test_split_sample_null_gives_no_replication():
    assert max(corr.split_sample_null(_split_panel(), n_perm=3, n_min=20)) == 0


def test_gdp_control_and_real_pension_measure_separate_the_denominator():
    """A ratio to GDP correlates with a cyclical outcome through the denominator alone; the GDP control and a measure without GDP in it remove that."""
    import numpy as np
    import pandas as pd

    from cifresignificative.studies.italy_productivity.analysis import corr
    from cifresignificative.studies.italy_productivity.analysis.experiments import (
        national as N,
    )
    rng = np.random.default_rng(0); rows = []
    for c in corr.COUNTRIES[:20]:
        cycle = rng.normal(0, 0.05, 20).cumsum()
        for t in range(20):
            gdp = 20000 * np.exp(0.02 * t + cycle[t]); pop = 1e7
            pension_euro = 0.12 * 20000 * np.exp(0.02 * t) * pop * np.exp(rng.normal(0, 0.01))  # follows the trend, not the cycle
            v = {"gdp_real_pc": gdp, "gdp_nominal_meur": gdp * pop / 1e6, "population": pop, "wb_pop_65plus_pct": 20.0, "hicp_inflation": 0.0,
                 "pension_spend_pct_gdp": pension_euro / (gdp * pop) * 100, "neet_rate_15_29": 15 - 60 * cycle[t] + rng.normal(0, 0.5),
                 "employment_rate_55_64": 50 + 60 * cycle[t] + rng.normal(0, 0.5), "edu_spend_pct_gdp": 5 + rng.normal(0, 0.1)}
            rows += [{"iso3": c, "year": 2005 + t, "indicator": k, "value": x, "flag": None} for k, x in v.items()]
    r = N.run_pension_real(pd.DataFrame(rows)).set_index(["id", "sample", "spec"])
    assert r.loc[("H6", "all members", "pension % of GDP"), "p"] < 0.01  # the ratio picks up the cycle
    assert r.loc[("H6", "all members", "real per person 65+"), "p"] > 0.05 and r.loc[("H7", "all members", "real per person 65+, GDP control"), "p"] > 0.05
