"""Tests for Pillar 4 (`studies.cesena.analysis.outcomes` bracket Gini and median interpolation)."""
import numpy as np
import pandas as pd

from cifresignificative.studies.cesena.analysis.outcomes import (
    bracket_gini_and_median,
    summarize_comprensorio_spillovers,
)


def test_bracket_gini_zero_under_equal_incomes_and_increases_with_top_concentration():
    # Case 1: every taxpayer in a single bracket -> trapezoidal Gini = 0
    eq_df = pd.DataFrame([
        {"class_order": 1, "class_lo": 0.0, "class_hi": 10000.0, "taxpayers": 0.0, "income_eur": 0.0},
        {"class_order": 3, "class_lo": 15000.0, "class_hi": 26000.0, "taxpayers": 1000.0, "income_eur": 20_000_000.0},
    ])
    m_eq = bracket_gini_and_median(eq_df)
    assert np.isclose(m_eq["gini_trap"], 0.0)
    assert np.isclose(m_eq["mean_income_eur"], 20_000.0)
    assert np.isclose(m_eq["median_income_eur"], 20_500.0)

    # Case 2: realistic unequal distribution across brackets including >120k top tail
    uneq_df = pd.DataFrame([
        {"class_order": 1, "class_lo": 0.0, "class_hi": 10000.0, "taxpayers": 200.0, "income_eur": 1_000_000.0},
        {"class_order": 2, "class_lo": 10000.0, "class_hi": 15000.0, "taxpayers": 200.0, "income_eur": 2_500_000.0},
        {"class_order": 3, "class_lo": 15000.0, "class_hi": 26000.0, "taxpayers": 300.0, "income_eur": 6_000_000.0},
        {"class_order": 4, "class_lo": 26000.0, "class_hi": 55000.0, "taxpayers": 200.0, "income_eur": 7_500_000.0},
        {"class_order": 7, "class_lo": 120000.0, "class_hi": np.inf, "taxpayers": 100.0, "income_eur": 20_000_000.0},
    ])
    m_uneq = bracket_gini_and_median(uneq_df)
    assert 0.30 < m_uneq["gini_trap"] < 0.65
    # Pareto top-tail adjustment must exceed the trapezoidal lower bound when mean_top (200k) > lo_top (120k)
    assert m_uneq["gini_pareto"] > m_uneq["gini_trap"]
    # Median taxpayer (#500 of 1000) is one-third into bracket 3 (15,000 to 26,000)
    expected_med = 15000.0 + (100.0 / 300.0) * (26000.0 - 15000.0)
    assert np.isclose(m_uneq["median_income_eur"], expected_med)


def test_summarize_comprensorio_spillovers_identities():
    comp = pd.DataFrame([
        {"comune_id": "040007", "comune": "Cesena", "belt_group": "cesena_core", "belt_label": "Cesena", "year": 2008, "pop_total": 90000.0, "net_migration": 200.0, "natural_balance": -300.0, "employees": np.nan, "local_units": np.nan, "income_pos_real_eur": 1.0e9, "taxpayers_pos": 50000.0},
        {"comune_id": "040007", "comune": "Cesena", "belt_group": "cesena_core", "belt_label": "Cesena", "year": 2012, "pop_total": 91000.0, "net_migration": 200.0, "natural_balance": -300.0, "employees": 35000.0, "local_units": 9000.0, "income_pos_real_eur": 1.05e9, "taxpayers_pos": 50000.0},
        {"comune_id": "040007", "comune": "Cesena", "belt_group": "cesena_core", "belt_label": "Cesena", "year": 2024, "pop_total": 91500.0, "net_migration": 300.0, "natural_balance": -400.0, "employees": 40000.0, "local_units": 9100.0, "income_pos_real_eur": 1.15e9, "taxpayers_pos": 50000.0, "addizionale_comunale_real_eur": 4.83e6},
        {"comune_id": "040007", "comune": "Cesena", "belt_group": "cesena_core", "belt_label": "Cesena", "year": 2025, "pop_total": 91500.0, "net_migration": np.nan, "natural_balance": np.nan, "employees": np.nan, "local_units": np.nan, "income_pos_real_eur": np.nan, "taxpayers_pos": np.nan},
        {"comune_id": "040008", "comune": "Cesenatico", "belt_group": "rubicone_mare", "belt_label": "Rubicone", "year": 2008, "pop_total": 24000.0, "net_migration": 300.0, "natural_balance": -50.0, "employees": np.nan, "local_units": np.nan, "income_pos_real_eur": 3.0e8, "taxpayers_pos": 15000.0},
        {"comune_id": "040008", "comune": "Cesenatico", "belt_group": "rubicone_mare", "belt_label": "Rubicone", "year": 2012, "pop_total": 25000.0, "net_migration": 300.0, "natural_balance": -50.0, "employees": 10000.0, "local_units": 3000.0, "income_pos_real_eur": 3.0e8, "taxpayers_pos": 15000.0},
        {"comune_id": "040008", "comune": "Cesenatico", "belt_group": "rubicone_mare", "belt_label": "Rubicone", "year": 2024, "pop_total": 26000.0, "net_migration": 400.0, "natural_balance": -50.0, "employees": 10200.0, "local_units": 3000.0, "income_pos_real_eur": 3.15e8, "taxpayers_pos": 15000.0, "addizionale_comunale_real_eur": 1.575e6},
        {"comune_id": "040008", "comune": "Cesenatico", "belt_group": "rubicone_mare", "belt_label": "Rubicone", "year": 2025, "pop_total": 26400.0, "net_migration": np.nan, "natural_balance": np.nan, "employees": np.nan, "local_units": np.nan, "income_pos_real_eur": np.nan, "taxpayers_pos": np.nan},
    ])
    by_grp, by_muni = summarize_comprensorio_spillovers(comp)
    assert set(by_grp["belt_group"]) == {"cesena_core", "rubicone_mare", "COMMUTER_BELT_16", "COMPRENSORIO_17"}
    tot = by_grp[by_grp["belt_group"] == "COMPRENSORIO_17"].iloc[0]
    ces = by_grp[by_grp["belt_group"] == "cesena_core"].iloc[0]
    belt = by_grp[by_grp["belt_group"] == "COMMUTER_BELT_16"].iloc[0]
    assert np.isclose(tot["pop_delta_08_25"], ces["pop_delta_08_25"] + belt["pop_delta_08_25"])
    assert np.isclose(tot["emp_delta_12_24"], ces["emp_delta_12_24"] + belt["emp_delta_12_24"])
    assert len(by_muni) == 2


def test_summarize_university_benchmarks():
    from cifresignificative.studies.cesena.analysis.outcomes import (
        summarize_university_benchmarks,
    )

    uni_df = pd.DataFrame([
        {"calendar_year": 2024, "comune_id": "040007", "comune_name": "Cesena", "students_total": 4700.0, "students_stem": 3100.0, "students_ict": 640.0, "students_engineering": 2000.0, "students_psychology": 1400.0, "stem_share": 0.66, "ict_share": 0.136, "engineering_share": 0.425, "psychology_share": 0.298, "graduates_total": 1100.0},
        {"calendar_year": 2024, "comune_id": "040012", "comune_name": "Forlì", "students_total": 6500.0, "students_stem": 1100.0, "students_ict": 0.0, "students_engineering": 1100.0, "students_psychology": 0.0, "stem_share": 0.17, "ict_share": 0.0, "engineering_share": 0.17, "psychology_share": 0.0, "graduates_total": 1700.0},
        {"calendar_year": 2024, "comune_id": "099014", "comune_name": "Rimini", "students_total": 4500.0, "students_stem": 650.0, "students_ict": 0.0, "students_engineering": 0.0, "students_psychology": 0.0, "stem_share": 0.14, "ict_share": 0.0, "engineering_share": 0.0, "psychology_share": 0.0, "graduates_total": 1200.0},
    ])
    demo_df = pd.DataFrame([
        {"comune_id": "040007", "year": 2024, "pop_total": 96000.0},
        {"comune_id": "040012", "year": 2024, "pop_total": 117000.0},
        {"comune_id": "099014", "year": 2024, "pop_total": 150000.0},
    ])

    bench = summarize_university_benchmarks(uni_df, demo_df, focal_id="040007", core_only=False, year=2024)
    assert not bench.empty
    assert "stem_share" in set(bench["metric"])
    stem_row = bench[bench["metric"] == "stem_share"].iloc[0]
    assert stem_row["cesena"] == 0.66
    assert stem_row["gap"] > 0


