"""Programmatic deficit targets: the row parser on both template formats (offline) and the committed table."""
from pathlib import Path

import pandas as pd
import pytest

from cifresignificative.sources import deficit_targets as dt


def test_parse_dpb_reads_both_template_formats(monkeypatch):
    en = """ 3. Net lending/net borrowing of general government 1 B.9 -3.0 -2.6 -2.2
Net lending (+) / net borrowing (-) ( B.9) by sub-sector
1. General government S.13 -2.6 -2.2 -1.1 -0.2 0.3
1a. Central government S.1311 -2.9 -2.4 -1.3 -0.4 0.1
"""
    it = """ 3. Indebitamento/surplus delle Amministrazioni pubbliche EDP B.9 -9,6 -9,4 -5,6
Ac c redi t am en t o n et t o( +) /i n debi t am en t o n et t o ( B.9) per s ot t os et t ore
1. Am m i n i s t razi on i pu bbl i c he S.13 -9,4 -5,6 -3,9 -3,3
1a. Am m i n i s t razi on i c en t rali S.1311 -9,3 -5,6 -3,9 -3,4
"""
    for text, want in ((en, -2.2), (it, -5.6)):
        monkeypatch.setattr(dt, "_text", lambda pdf, t=text: t)
        assert dt.parse_dpb(Path("x.pdf")) == (want, want)  # the macro table and the S.13 row agree; S.1311 (central government) is not mistaken for it


def test_committed_targets():
    d = pd.read_csv(dt.CSV).set_index("budget_year")
    assert list(d.index) == list(range(2016, 2027)) and (d.target_pct_gdp < 0).all()
    assert d.loc[2019, "target_pct_gdp"] == -2.4 and d.loc[2019, "target_final_pct_gdp"] == -2.04  # the 2019 target was lowered after the talks with the Commission
    assert (d.drop(index=2019).target_pct_gdp == d.drop(index=2019).target_final_pct_gdp).all()
    assert d.loc[2021, "target_pct_gdp"] == -7.0 and d.loc[2022, "target_pct_gdp"] == -5.6 and d.loc[2025, "target_pct_gdp"] == -3.3


def test_deficit_surprise_columns():
    from cifresignificative.analysis import budget as B
    out = pd.DataFrame({"deficit": [2.0, 9.4], "primary_deficit": [0.0, 6.0], "debt": [130.0, 150.0], "real_growth": [1.0, -8.0], "delta_deficit": [None, 7.4], "delta_primary_deficit": [None, 6.0], "delta_debt": [None, 20.0]}, index=[2019, 2020])
    m = pd.DataFrame({"scope": "bill", "budget_year": [2020], "horizon_year": 2020, "h": 0, "ref_date": pd.to_datetime(["2019-10-15"]), "deficit_pct_gdp": 1.0, "deficit_financed_share": 0.5, "coverage_ratio": 0.5, "uses_pct_gdp": 2.0})
    T = B.link_table(m, out, "bill", pd.DataFrame({"target": [2.2], "target_final": [2.2]}, index=pd.Index([2020], name="budget_year")))
    assert T.loc[2020, "deficit_surprise"] == pytest.approx(7.2)  # outturn deficit minus the target: positive = larger than planned
    assert T.loc[2020, "planned_delta_deficit"] == pytest.approx(0.2)  # the change the plan implied: target minus last year's outturn


def test_level_decomposition_identity_on_a_toy_year():
    from cifresignificative.analysis import budget as B
    T = pd.DataFrame({"deficit": [5.0], "target": [3.0], "deficit_surprise": [2.0]}, index=[2030])
    # target ratios on a forecast GDP of 100 (revenue 40, primary expenditure 40, interest 3: deficit 3); outturn GDP 110 with revenue 41.8 (38.0%), primary expenditure 49.5 (45.0%), interest 3.3 (3.0%)
    tc = pd.DataFrame({"revenue_target": [40.0], "primary_expenditure_target": [40.0], "interest_target": [3.0], "real_growth_forecast": [1.0], "nominal_growth_forecast": [2.0], "nominal_gdp_forecast": [100.0]}, index=[2030])
    comp = pd.DataFrame({"revenue": [41.8 / 110 * 100], "primary_expenditure": [49.5 / 110 * 100], "interest": [3.3 / 110 * 100], "nominal_gdp": [110.0], "real_growth": [2.0], "nominal_growth": [4.0]}, index=[2030])
    d = B.decompose_surprise(T.assign(deficit=lambda x: x.deficit), comp, tc).iloc[0]
    r = 100 / 110
    assert d.revenue_shortfall == pytest.approx(-(41.8 / 110 * 100 - 40 * r)) and d.primary_expenditure_overrun == pytest.approx(49.5 / 110 * 100 - 40 * r) and d.interest_overrun == pytest.approx(3.3 / 110 * 100 - 3 * r)
    assert d.denominator_effect == pytest.approx((40 + 3 - 40) * (r - 1))  # target deficit ratio times the GDP forecast error
    assert d.real_growth_error == pytest.approx(1.0) and d.nominal_growth_error == pytest.approx(2.0)


def test_committed_decomposition_adds_up():
    import pytest as _pytest

    from cifresignificative.analysis import budget as B
    from cifresignificative.analysis.data import load_panel
    try:
        panel = load_panel()
    except FileNotFoundError:
        _pytest.skip("processed data not built")
    m = B.measures(B.load())
    T = B.link_table(m, B.outcomes(panel), "bill", B.targets()).loc[2016:2025]
    D = B.decompose_surprise(T, B.outturn_components(panel), B.target_components())
    assert D.rounding.abs().max() < 0.25  # the three level misses and the denominator effect reproduce the surprise up to the rounding of the published figures
    assert D.loc[2020, "primary_expenditure_overrun"] > 3 and D.loc[2023, "primary_expenditure_overrun"] > 3 and (D.gdp_ratio.between(0.9, 1.1)).all()


def test_category_targets_add_up_to_total_expenditure():
    d = pd.read_csv(dt.CSV).set_index("budget_year").loc[2016:2025]
    cats = ["interest_target", "target_D1", "target_P2", "target_D62_D632", "target_D3", "target_P51G", "target_D9", "target_other"]
    assert not d[cats].isna().any().any()
    assert (d[cats].sum(axis=1) - d.expenditure_target).abs().max() <= 0.15  # the eight rows of the plan reproduce the total up to rounding


def test_expenditure_misses_by_category():
    import pytest as _pytest

    from cifresignificative.analysis import budget as B
    from cifresignificative.analysis.data import load_panel
    try:
        panel = load_panel()
    except FileNotFoundError:
        _pytest.skip("processed data not built")
    T = B.link_table(B.measures(B.load()), B.outcomes(panel), "bill", B.targets()).loc[2016:2025]
    M = B.expenditure_misses(T, B.target_components(), B.outturn_components(panel))
    assert (M.total - M.total_expenditure_miss).abs().max() < 0.15  # the categories add up to the miss of total expenditure
    assert (M.loc[2021:2023, "capital transfers"] > 3).all() and M.loc[2021:2023, "capital transfers"].gt(M.loc[2021:2023].drop(columns=["capital transfers", "total", "total_expenditure_miss"]).max(axis=1)).all()
