"""Sanity checks on the built panel (skipped until `fetch` + `build` have been run)."""
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest


def _ts(s: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(s))


PANEL = Path("data/processed/panel.parquet")
pytestmark = pytest.mark.skipif(not PANEL.exists(), reason="panel not built")


@pytest.fixture(scope="module")
def panel():
    return pd.read_parquet(PANEL)


def test_unique_keys(panel):
    assert not panel.duplicated(["indicator", "geo", "year"]).any()


def test_italy_present_for_every_indicator(panel):
    missing = set(panel.indicator) - set(panel[panel.geo == "IT"].indicator)
    assert not missing - {"wb_gov_debt_pct_gdp"}


@pytest.mark.parametrize("indicator,year,expected,tol", [
    ("gov_debt_pct_gdp", 2020, 154.4, 1.5),  # Eurostat EDP debt, Italy (may be revised)
    ("unemployment_rate", 2014, 12.7, 0.3),
    ("employment_rate_20_64", 2005, 61.5, 0.5),
])
def test_italy_spot_checks(panel, indicator, year, expected, tol):
    v = panel[(panel.geo == "IT") & (panel.indicator == indicator) & (panel.year == year)].value.item()
    assert abs(v - expected) <= tol


def test_imf_and_eurostat_debt_agree_for_italy(panel):
    def v(ind):
        return panel[(panel.geo == "IT") & (panel.indicator == ind) & (panel.year == 2020)].value.item()
    assert abs(v("imf_gov_debt_pct_gdp") - v("gov_debt_pct_gdp")) < 3  # different vintages/definitions, same ballpark


def test_benchmarks_have_fiscal_data(panel):
    for geo in ("US", "JP"):
        assert not panel[(panel.geo == geo) & (panel.indicator == "imf_gov_debt_pct_gdp")].empty


def test_pisa_italy_2018(panel):
    v = panel[(panel.geo == "IT") & (panel.indicator == "wb_pisa_math") & (panel.year == 2018)].value.item()
    assert abs(v - 487) < 2


def test_regional_italy_has_21_nuts2():
    path = Path("data/processed/regional.parquet")
    if not path.exists():
        pytest.skip("regional not built")
    r = pd.read_parquet(path)
    it = r[(r.country == "IT") & (r.indicator == "nuts2_gdp_pps_pc_eu27")]
    assert it.geo.nunique() >= 19  # 21 NUTS2 regions; some code revisions over time


def test_vat_gap_italy_2023(panel):
    v = panel[(panel.geo == "IT") & (panel.indicator == "vat_compliance_gap_pct") & (panel.year == 2023)].value.item()
    assert abs(v - 15.0) < 0.5  # EU VAT gap report 2025


def test_wgi_scores_in_range(panel):
    w = panel[panel.indicator.str.startswith("wgi_")]
    assert w.value.between(0, 100).all()


def test_sector_table_covers_italy():
    path = Path("data/processed/sectors.parquet")
    if not path.exists():
        pytest.skip("sectors not built")
    s = pd.read_parquet(path)
    it = s[(s.iso3 == "ITA") & (s.year == 2019)]
    assert it.nace_r2.nunique() == 10 and it[["gva_clv20_meur", "hours_ths"]].notna().all().all()


def test_earnings_break_year_is_flagged_and_on_the_annual_scale(panel):
    """2024 starts a new basis of the average wage (catalog.SERIES_BREAKS): the point is kept, flagged, and annual (it was published monthly until November 2026)."""
    from cifresignificative.core.catalog import EXCLUDED_POINTS, SERIES_BREAKS
    for (ind, year) in EXCLUDED_POINTS:
        assert not ((panel.indicator == ind) & (panel.year == year)).any()
    for ind, year in SERIES_BREAKS.items():
        d = panel[(panel.indicator == ind) & (panel.iso3 == "ITA")].set_index("year")
        assert "b" in str(d.flag[year]) and 0.8 < d.value[year] / d.value[year - 1] < 1.25


def test_women_managers_share_is_a_percentage(panel):
    v = panel[panel.indicator == "women_managers_share_pct"].value
    assert len(v) > 100 and v.between(0, 100).all()


def test_cofog_groups_add_up_to_total():
    path = Path("data/processed/spending.parquet")
    if not path.exists():
        pytest.skip("spending not built")
    from cifresignificative.studies.italy_productivity.analysis.data import (
        cofog_groups,
        load_spending,
    )
    sp = load_spending()
    groups = cofog_groups(sp, 2019)
    total = sp[(sp.kind == "cofog") & (sp.item == "TOTAL") & (sp.unit == "PC_GDP") & (sp.year == 2019)].set_index("iso3").value
    diff = (groups.sum(axis=1) - total).dropna().abs()
    assert diff.median() < 0.3  # rounding to one decimal in 13 groups; unallocated items can differ slightly


def test_eu_median_needs_broad_coverage():
    from cifresignificative.studies.italy_productivity.analysis.data import (
        eu_median_ex_italy,
    )
    w = pd.DataFrame({"ITA": [1.0, 1.0], **{f"C{i}": [5.0, np.nan if i else 5.0] for i in range(10)}}, index=[2023, 2024])
    w = w.rename(columns=dict(zip([f"C{i}" for i in range(10)], ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA"], strict=True)))
    m = eu_median_ex_italy(w)
    assert m.loc[2023] == 5.0 and np.isnan(m.loc[2024])


def test_pisa_2022_italy_matches_oecd_country_note(panel):
    for ind, expected in [("wb_pisa_math", 471), ("wb_pisa_reading", 482), ("wb_pisa_science", 477)]:
        v = panel[(panel.geo == "IT") & (panel.indicator == ind) & (panel.year == 2022)].value.item()
        assert abs(v - expected) < 1


def test_procurement_italy_matches_scoreboard(panel):
    def v(ind, year=2024):
        return panel[(panel.geo == "IT") & (panel.indicator == ind) & (panel.year == year)].value.item()
    assert abs(v("proc_single_bidder_pct") - 34) < 1 and abs(v("proc_direct_awards_pct") - 3) < 1 and abs(v("proc_decision_speed_days") - 123) < 1


def test_higher_education_indicators_in_range(panel):
    for ind in ("stem_share_grads_pct", "women_share_stem_grads_pct", "women_share_grads_pct", "phd_women_share_pct", "bachelor_grads_30plus_pct"):
        v = panel[panel.indicator == ind].value
        assert len(v) > 100 and v.between(0, 100).all(), ind
    it = panel[(panel.geo == "IT") & (panel.year == 2024)].set_index("indicator").value
    assert 20 < it["stem_share_grads_pct"] < 27 and 30 < it["tertiary_attainment_25_34"] < 33 and 26 < it["mean_age_grads_bachelor"] < 29


def test_vintage_diff_detects_revisions_and_additions():
    from cifresignificative.core.clean.vintages import diff
    old = pd.DataFrame({"indicator": ["a", "a", "b"], "geo": ["IT", "IT", "IT"], "year": [2020, 2021, 2020], "value": [10.0, 10.0, 5.0]})
    new = pd.DataFrame({"indicator": ["a", "a", "a"], "geo": ["IT", "IT", "IT"], "year": [2020, 2021, 2022], "value": [10.0, 11.0, 12.0]})
    res = diff(old, new)
    assert len(res["revised"]) == 1 and len(res["added"]) == 1 and len(res["dropped"]) == 1


def test_vintage_snapshots_per_table(tmp_path, monkeypatch):
    from cifresignificative.core.clean import vintages
    monkeypatch.setattr(vintages, "VINTAGE_DIR", tmp_path)
    reg = pd.DataFrame({"geo": ["ITC1"], "country": ["IT"], "year": [2020], "indicator": ["a"], "value": [1.0], "flag": [None], "source": ["s"]})
    hf = pd.DataFrame({"iso3": ["ITA"], "geo": ["IT"], "period": ["2020-01"], "date": [pd.Timestamp("2020-01-31")], "freq": ["M"], "indicator": ["a"], "value": [1.0], "flag": [None], "source": ["s"]})
    assert vintages.snapshot(reg, "regional") is not None and vintages.snapshot(reg, "regional") is None  # same content: no new file
    assert vintages.snapshot(reg, "regional_extra") is not None  # tables do not see each other's snapshots
    assert vintages.snapshot(hf, "highfreq") is not None
    assert vintages.snapshot(reg.assign(value=2.0), "regional") is not None
    assert len(list(tmp_path.glob("regional_2*.parquet"))) == 2 and len(list(tmp_path.glob("regional_extra_2*.parquet"))) == 1
    res = vintages.diff(hf, hf.assign(value=5.0), key=vintages.TABLES["highfreq"][0])
    assert len(res["revised"]) == 1


def test_new_series_present_and_sane(panel):
    it = panel[(panel.geo == "IT") & (panel.year == 2023)].set_index("indicator").value
    assert 82 < it["life_expectancy_birth"] < 85 and 40 < it["treatable_mortality_rate"] < 90
    assert 0 < it["childcare_under3_30h_plus_pct"] < 60 and it["employment_rate_60_64"] > 40


def test_long_history_and_projection_tables():
    lh = pd.read_parquet("data/processed/long_history.parquet")
    ita = lh[(lh.iso3 == "ITA") & (lh.variable == "gov_debt_pct_gdp")].set_index("year").value
    assert ita.index.min() <= 1962 and abs(ita.loc[2005] - 106.2) < 1.0
    ar = pd.read_parquet("data/processed/ageing_report.parquet")
    pen = ar[(ar.iso3 == "ITA") & (ar.variable == "Public pensions, gross") & (ar.year == 2022)].value.item()
    assert abs(pen - 15.6) < 0.2


def test_weighted_median_follows_the_weights():
    from cifresignificative.studies.italy_productivity.analysis.data import (
        weighted_median_ex_italy,
    )
    cols = ["AUT", "BEL", "BGR", "HRV", "CYP"]
    w = pd.DataFrame({c: [v] for c, v in zip(cols, [1.0, 2.0, 3.0, 4.0, 5.0], strict=True)}, index=[2023])
    heavy_top = pd.DataFrame({c: [x] for c, x in zip(cols, [1, 1, 1, 1, 100], strict=True)}, index=[2023])
    assert weighted_median_ex_italy(w, heavy_top, min_share=0.5).loc[2023] == 5.0
    equal = pd.DataFrame({c: [1] for c in cols}, index=[2023])
    assert weighted_median_ex_italy(w, equal, min_share=0.5).loc[2023] == 3.0


def test_crime_and_migration_series(panel):
    it = panel[(panel.geo == "IT") & (panel.year == 2024)].set_index("indicator").value
    assert 0.3 < it["homicide_victims_per_100k"] < 1.0 and 100 < it["prison_occupancy_pct"] < 150
    assert 5 < it["immigration_per_1000"] < 12 and it["net_migration_nationals_25_34_per_1000"] < 0
    assert not panel[panel.indicator == "serious_assault_rate_police"].shape[0]


def test_governments_and_policies_files():
    from cifresignificative.studies.italy_productivity.analysis.timeline import (
        government_at,
        load_governments,
        load_policies,
        year_weights,
    )
    g = load_governments("2001-01-01")
    assert list(g.government[-3:]) == ["Conte II", "Draghi", "Meloni"]
    assert (g.start.diff().dropna() >= pd.Timedelta(0)).all()  # chronological, no overlap
    assert government_at(g, _ts("2012-07-26")) == "Monti" and government_at(g, _ts("2020-03-09")) == "Conte II"
    from cifresignificative.studies.italy_productivity.analysis.timeline import (
        merge_short_governments,
    )
    m = merge_short_governments(load_governments("1989-01-01"))
    d = m[m.government == "D'Alema I-II"].iloc[0]  # D'Alema II (4 months) folded into D'Alema I
    assert len(m) == len(load_governments("1989-01-01")) - 1 and d.start == _ts("1998-10-21") and d.end == _ts("2000-04-26")
    assert "D'Alema II" not in set(m.government) and (m.start.diff().dropna() > pd.Timedelta(0)).all()
    p = load_policies()
    assert p.evidence.isin(["verified", "partly verified", "unverified"]).all() and (p.evidence == "verified").sum() >= 30
    assert p.date.is_monotonic_increasing and p.sources.notna().sum() >= 35
    w = year_weights(_ts("2011-11-16"), _ts("2013-04-28"))
    assert abs(w.sum() - (_ts("2013-04-28") - _ts("2011-11-16")).days / 365.25) < 0.02


def test_manual_series_survive_the_build_and_reference_tables_are_ignored():
    from cifresignificative.core.sources import manual
    d = manual.load()  # must not raise on italy_governments.csv / italy_policies.csv
    assert {"wb_pisa_math", "piaac_literacy"} <= set(d.indicator)
    assert d[(d.iso3 == "ITA") & (d.indicator == "piaac_literacy") & (d.year == 2023)].value.item() == 245.0


def test_piaac_italy_matches_the_oecd_table(panel):
    def v(ind, year):
        return panel[(panel.geo == "IT") & (panel.indicator == ind) & (panel.year == year)].value.item()
    assert v("piaac_literacy", 2023) == 245 and v("piaac_numeracy", 2023) == 244 and abs(v("piaac_low_literacy_pct", 2023) - 34.7) < 0.1
    assert v("piaac_literacy", 2012) == 250


@pytest.mark.skipif(not Path("data/processed/italy_tax_gap_mef.parquet").exists(), reason="the MEF report PDF is not redistributed (docs/reproducibility.md)")
def test_mef_tax_gap_matches_report_tables():
    d = pd.read_parquet("data/processed/italy_tax_gap_mef.parquet")
    def v(tax, year, measure):
        return d[(d.tax == tax) & (d.year == year) & (d.measure == measure)].value.item()
    assert v("IVA", 2022, "gap_meur") == 28966 and v("Totale entrate tributarie", 2018, "gap_meur") == 95573
    assert v("IRPEF lavoro autonomo e impresa", 2018, "propensity_pct") == 60.2 and v("Totale entrate tributarie", 2022, "propensity_pct") == 17.0


def test_euklems_growth_accounts_if_available():
    path = Path("data/processed/euklems_growth.parquet")
    if not path.exists():
        pytest.skip("EU KLEMS raw file not downloaded")
    e = pd.read_parquet(path)
    t = e[(e.iso3 == "ITA") & (e.nace_r2_code == "MARKT") & e.year.between(1996, 2019)].groupby("variable").value.mean()
    assert abs(t["LP1_G"] - 0.60) < 0.05 and t["LP1ConTFP"] < 0


def test_bready_italian_cities():
    path = Path("data/processed/italy_bready_cities.parquet")
    if not path.exists():
        pytest.skip("B-READY file not downloaded")
    d = pd.read_parquet(path)
    assert d.city.nunique() == 13 and d.score.between(0, 100).all()
    t = d[(d.level == "topic") & (d.city == "Milan") & (d.topic == "Business Entry")].score.item()
    assert abs(t - 84.0) < 0.5


def test_esif_absorption_table():
    d = pd.read_parquet("data/processed/esif_absorption.parquet")
    it = d[(d.geo == "IT") & d.fund.isin(["ERDF", "ESF", "CF", "YEI"])][["planned", "spent"]].sum()
    assert 90 < it.spent / it.planned * 100 < 100 and it.planned > 6e10


@pytest.mark.skipif(not Path("data/processed/italy_unobserved_economy.parquet").exists(), reason="the Istat workbook is not redistributed (docs/reproducibility.md)")
def test_istat_unobserved_economy():
    d = pd.read_parquet("data/processed/italy_unobserved_economy.parquet")
    t = d[d.component == "non_observed_total"].set_index("year")
    assert t.index.min() == 2011 and t.index.max() == 2023 and 10.0 < t.pct_gdp[2022] < 10.3
    parts = d[d.component.isin(["underground", "illegal"])].groupby("year").pct_gdp.sum()
    assert (parts - t.pct_gdp).abs().max() < 0.01


def test_cofog_euro_groups_add_up_to_total():
    if not Path("data/processed/spending.parquet").exists():
        pytest.skip("spending not built")
    from cifresignificative.studies.italy_productivity.analysis.data import (
        ITALY,
        italy_cofog_euro,
        load_spending,
        spend_wide,
    )
    sp = load_spending()
    if "MIO_EUR" not in set(sp.unit):
        pytest.skip("spending.parquet built before MIO_EUR was added")
    t = italy_cofog_euro(sp, [2019, 2024])
    for y in (2019, 2024):
        assert abs(t[f"{y} MEUR"].sum() - spend_wide(sp, "TOTAL", unit="MIO_EUR").loc[y, ITALY]) < 1.0   # groups partition the total (rounding)
        assert abs(t[f"{y} % of total"].sum() - 100) < 1e-6


def test_ameco_series_cover_every_member(panel):
    """AMECO uses its own country codes (`UK`, `ROM`): a code missing from `AMECO_ISO` is dropped silently, so check the members."""
    from cifresignificative.core.countries import EU27

    members = set(EU27.values())
    for ind, sub in panel[panel.source.str.startswith("ameco:")].groupby("indicator"):
        missing = members - set(sub.iso3)
        assert not missing, f"{ind} has no AMECO values for {sorted(missing)}"


def test_before_breaks_drops_the_years_on_a_new_basis(panel):
    from cifresignificative.core.catalog import SERIES_BREAKS
    from cifresignificative.studies.italy_productivity.analysis import corr
    from cifresignificative.studies.italy_productivity.analysis.data import (
        before_breaks,
    )
    d = before_breaks(panel)
    for ind, year in SERIES_BREAKS.items():
        assert d[d.indicator == ind].year.max() == year - 1 and panel[panel.indicator == ind].year.max() >= year
    assert len(panel) - len(d) == sum(((panel.indicator == i) & (panel.year >= y)).sum() for i, y in SERIES_BREAKS.items())
    assert corr.panel_matrix(panel)["net_earnings_pps"].dropna().index.get_level_values("year").max() < SERIES_BREAKS["net_earnings_pps"]
