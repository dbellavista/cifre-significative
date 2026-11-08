"""Tests for Phase 3 (`studies.cesena.analysis.experiments` Two-Way FE lagged regressions, BH-FDR, and shuffled null)."""
import numpy as np
import pandas as pd

from cifresignificative.core.geo import ER_COMUNI
from cifresignificative.studies.cesena.analysis.experiments import (
    lagged_fe_screen,
    shuffled_null_check,
    size_class_growth,
    synthetic_cesena,
    synthetic_control_placebos,
    twoway_fe_regression,
)


def _make_synthetic_panel(seed: int = 99, planted_beta: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    comuni = [c.id for c in ER_COMUNI]
    years = list(range(2009, 2024))
    for i, cid in enumerate(comuni):
        ent_fe = (i - 6) * 1.5
        x_series = rng.normal(loc=200.0 + i * 10, scale=25.0, size=len(years))
        y_noise = rng.normal(loc=0.0, scale=0.35, size=len(years))
        for t_idx, yr in enumerate(years):
            yr_fe = np.sin(t_idx / 2.0) * 0.8
            # Planted effect at lag 2: y_{c, t} depends on x_{c, t-2}
            lag2_signal = planted_beta * (x_series[t_idx - 2] - np.mean(x_series)) if t_idx >= 2 else 0.0
            rows.append({
                "comune_id": cid,
                "year": yr,
                "regime": "dpr194" if yr <= 2015 else "dlgs118",
                "spese_capitale_comp_real_pc": float(x_series[t_idx]),
                "dlog_local_units": float(ent_fe + yr_fe + lag2_signal + y_noise[t_idx]),
            })
    return pd.DataFrame(rows)


def test_twoway_fe_recovers_planted_lag2_effect():
    df = _make_synthetic_panel(seed=42, planted_beta=0.08)
    df["y_lead2"] = df.groupby("comune_id")["dlog_local_units"].shift(-2)
    res = twoway_fe_regression(df, x_col="spese_capitale_comp_real_pc", y_col="y_lead2", n_perm=300, seed=42)
    assert np.isclose(res["beta"], 0.08, atol=0.015)
    assert res["p_cluster"] < 0.001
    assert res["p_perm"] < 0.01


def test_shuffled_null_produces_zero_discoveries():
    df = _make_synthetic_panel(seed=123, planted_beta=0.0)
    res = lagged_fe_screen(
        df,
        lags=(1, 2, 3, 4),
        predictors=(("spese_capitale_comp_real_pc", "Capital spending", "all"),),
        outcomes=(("dlog_local_units", "Local units growth"),),
        n_perm=200,
        seed=123,
    )
    assert len(res) == 4
    assert (res["q_bh_perm"] >= 0.05).all()
    null_chk = shuffled_null_check(df, lags=(1, 2, 3, 4), seed=777)
    assert null_chk["n_discoveries_q_perm_05"] == 0


def test_synthetic_cesena_recovers_exact_pre_match_and_post_gap():
    years = list(range(2008, 2025))
    rows = []
    for y in years:
        t = y - 2008
        # Donor 1 ('040012'): grows +1% per year; Donor 2 ('039014'): grows +3% per year
        v1 = 100.0 * (1.0 + 0.01 * t)
        v2 = 100.0 * (1.0 + 0.03 * t)
        v3 = 100.0 * (1.0 - 0.01 * t)
        v4 = 100.0 * (1.0 + 0.05 * t)
        # Cesena ('040007') equals 0.5*Donor1 + 0.5*Donor2 (+2% per year) through 2018, then +5% extra after 2018
        v_ces = 0.5 * v1 + 0.5 * v2 + (5.0 if y > 2018 else 0.0)
        for cid, val in [("040007", v_ces), ("040012", v1), ("039014", v2), ("099014", v3), ("037032", v4)]:
            rows.append({"comune_id": cid, "year": y, "mean_income_real_eur": val})
    df = pd.DataFrame(rows)
    series_df, sum_df = synthetic_cesena(df, focal_id="040007", t_pre_end=2018)
    assert len(series_df) == len(years)
    assert len(sum_df) == 1
    row = sum_df.iloc[0]
    assert row["pre_rmse_pp"] < 1e-3
    assert np.isclose(row["gap_vs_synthetic_pp"], 5.0, atol=1e-2)


def test_synthetic_control_placebos_in_space_and_in_time():
    years = list(range(2008, 2025))
    rows = []
    for y in years:
        t = y - 2008
        v1 = 100.0 * (1.0 + 0.01 * t)
        v2 = 100.0 * (1.0 + 0.03 * t)
        v3 = 100.0 * (1.0 - 0.01 * t)
        v4 = 100.0 * (1.0 + 0.02 * t)
        v_ces = 0.5 * v1 + 0.5 * v2 + (20.0 if y > 2018 else 0.0)
        for cid, val in [("040007", v_ces), ("040012", v1), ("039014", v2), ("099014", v3), ("037032", v4)]:
            rows.append({"comune_id": cid, "year": y, "mean_income_real_eur": val})
    df = pd.DataFrame(rows)
    in_space_df, in_time_df = synthetic_control_placebos(df, focal_id="040007", t_pre_end=2018, pseudo_t_end=2014)
    assert len(in_space_df) == 5
    ces_sp = in_space_df[in_space_df["is_cesena"]].iloc[0]
    assert ces_sp["rank_gap_desc"] == 1
    assert np.isclose(ces_sp["perm_p_gap"], 1.0 / 5.0)
    assert len(in_time_df) == 1
    it_row = in_time_df.iloc[0]
    assert np.isclose(it_row["pseudo_post_mean_gap_pp"], 0.0, atol=0.1)
    assert it_row["true_post_mean_gap_pp"] > 5.0



def test_size_class_growth_excludes_focal_and_counts_those_above():
    m = pd.DataFrame({"comune": ["FOCAL", "TINY A", "TINY B", "MID", "BIG"], "taxpayers_total": [77000, 500, 1500, 12000, 90000],
                      "mean_income_growth_08_24_pct": [4.0, 9.0, -6.0, 5.0, 1.0]}, index=["C573", "A", "B", "C", "D"])
    out = size_class_growth(m).set_index("lo")
    assert out["n"].sum() == 4  # the focal municipality is not its own peer
    assert out.loc[0, "n_above"] == 1 and out.loc[0, "top_above"] == ["Tiny A"] and out.loc[0, "median"] == 1.5
    assert out.loc[10000, "n_above"] == 1 and out.loc[40000, "n_above"] == 0
    assert 2000 not in out.index  # empty classes are not reported
