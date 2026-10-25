"""Istat general-government account by receiving sector (committed S13 table): structure, the March 2023 reclassification and the checks against the hearing and the note."""

import pandas as pd
import pytest

from cifresignificative.analysis import istat_fpa as A
from cifresignificative.analysis import superbonus as SB
from cifresignificative.sources import istat_fpa as F


def test_committed_table_structure():
    d = F.load()
    assert set(d.flow) == {"current", "b19"} and d.edition.nunique() == 30 and d.published.min() == pd.Timestamp("2019-09-01") and F.latest_edition(d) == "2026M10"
    assert not d.duplicated(["flow", "edition", "item", "year"]).any() and {"D92_D_W0_S14", "D92_D_W0_SIM", "D99_D_W0_SIM", "B9_B_W0"} <= set(d["item"])
    lab = pd.read_csv(F.LABELS_CSV).set_index("item").label
    assert lab["D92_D_W0_S14"] == "investment grants to households" and lab["D99_D_W0_SIM"] == "other capital transfers to corporations" and set(d["item"]) <= set(lab.index)


def test_receivers_add_up_and_the_2021_jump_is_corporate_cash_grants():
    d = F.load()
    rec = A.by_receiver(d, "2026M10")
    assert rec.loc[2021, "d92_households"] == pytest.approx(38024) and rec.loc[2021, "d99_corporations"] == pytest.approx(31613)
    assert rec.loc[2021, "d92_total"] == pytest.approx(rec.loc[2021, [c for c in rec if c.startswith("d92_") and c != "d92_total"]].sum())
    inc = A.increase_since(rec)
    assert inc.loc[2021, "d99_corporations"] > 0.9 * inc.loc[2021, "d99_total"] and inc.loc[2021, "d92_households"] > 0.8 * inc.loc[2021, "d92_total"]  # corporate cash grants; households' building bonuses


def test_march_2023_reclassification_reproduces_istat_revisions():
    d = F.load()
    r = A.reclassification(d).set_index(["item", "year"])
    assert r.loc[("B9_B_W0", 2021), "change"] == pytest.approx(-32308) and r.loc[("B9_B_W0", 2020), "change"] == pytest.approx(-2738)  # the revisions Istat announced on 1 March 2023
    assert r.loc[("D92_D_W0_S14", 2021), "change"] > 35000 and abs(r.loc[("D99_D_W0_SIM", 2021), "change"]) < 1000  # the grants were already D.99 before


def test_hearing_check_and_facade_split():
    d, ist = F.load(), SB.load_istat()
    h = A.hearing_check(d, ist).set_index("year")
    assert (h.difference_meur.abs() < 1000).all()  # the chart of the hearing within the chart reading error (+-0.5 bn) and the base-year level
    gdp = pd.Series({2020: 1_660_000.0, 2021: 1_838_674.8, 2022: 1_998_269.1, 2023: 2_141_760.8})
    f = A.facade_split(A.by_receiver(d, "2026M10"), ist, gdp).set_index("year")
    assert 18000 < f.loc[2021, "implied_facade_meur"] < 23000 and abs(f.loc[2022, "implied_facade_meur"]) < 2000 and 20000 < f.loc[[2020, 2021, 2022], "implied_facade_meur"].sum() < 25000  # about 23 bn for 2020-22 in the Camera dossier


def test_deficit_vintages_before_and_after_the_reclassification():
    from cifresignificative.analysis import budget as BU
    from cifresignificative.analysis.data import PROCESSED, load_panel, wide
    if not (PROCESSED / "panel.parquet").exists():
        pytest.skip("processed data missing")
    v = A.deficit_vintages(F.load(), BU.targets(), wide(load_panel(), "gdp_nominal_meur")["ITA"], years=(2021,)).set_index("edition")
    assert abs(v.loc["2022M10", "surprise_pp"]) < 0.1 and 1.5 < v.loc["2023M3", "surprise_pp"] < 2.0 and v.loc["2026M10", "surprise_pp"] > v.loc["2023M3", "surprise_pp"]  # the 2021 surprise appears with the March 2023 edition


def test_first_release_and_surprise_vintages():
    from cifresignificative.analysis import budget as BU
    from cifresignificative.analysis.data import PROCESSED, load_panel
    if not (PROCESSED / "panel.parquet").exists() or not (PROCESSED / "spending.parquet").exists():
        pytest.skip("processed data missing")
    d = F.load()
    fr = A.first_release(d, range(2019, 2026))
    assert fr.loc[2019, "edition"] == "2020M3" and fr.loc[2021, "edition"] == "2022M3" and fr.loc[2022, "edition"] == "2023M3" and pd.Timestamp("2019-09-01") == d.published.min()  # first editions 2020-03 onwards; 2022 only appears after the reclassification
    r = A.surprise_vintages(d, BU.outturn_components(load_panel()), BU.targets(), BU.target_components(), BU.decompose_surprise).set_index("budget_year")
    assert r.loc[2021, "deficit_surprise_revision"] == pytest.approx(1.99, abs=0.05) and abs(r.loc[2021, "deficit_surprise_first"]) < 0.2 and r.loc[2021, "primary_expenditure_overrun_revision"] > 2.0
    assert r.loc[2022, "deficit_surprise_first"] > 1.9 and abs(r.loc[2020, "deficit_surprise_revision"]) < 0.1
    assert (r.deficit_latest + r.deficit_surprise_revision * 0 - r.deficit_first - r.deficit_revision).abs().max() < 1e-9  # revision = latest - first


def test_quarterly_profile_and_reclassification():
    q = F.load_quarterly()
    p = A.capital_profile(q).set_index("period")
    assert p.loc["2023-Q4", "other_capital_expenditure"] == pytest.approx(51410, abs=100) and p.loc["2023-Q4", "ooke_share_of_year"] > 0.35 and p.loc["2024-Q1", "other_capital_expenditure"] < 10000  # the end-2023 rush and the stop
    annual = F.load()
    a = A.edition_table(annual, "2026M10", ["D92_D_W0", "D99_D_W0"]).loc[2021].sum()
    assert abs(p.loc["2021-Q1":"2021-Q4", "other_capital_expenditure"].sum() / a - 1) < 0.02  # four quarters add to the annual D.92 + D.99
    r = A.quarterly_reclassification(q).pivot(index="period", columns="item", values="change")
    assert (r.loc["2021-Q1":"2021-Q4", "B9_B_W0"] < -6000).all() and abs(r.loc["2021-Q1":"2021-Q4", "B9_B_W0"].sum() + 32308) < 3000  # the four quarters add up about to the annual revision of -32.3 billion


def test_istat_agrees_with_the_eurostat_figures_used():
    from cifresignificative.analysis.data import (
        PROCESSED,
        load_panel,
        load_spending,
        wide,
    )
    if not (PROCESSED / "panel.parquet").exists() or not (PROCESSED / "spending.parquet").exists():
        pytest.skip("processed data missing")
    p = load_panel()
    t = A.eurostat_check(F.load(), load_spending(), wide(p, "gov_balance_pct_gdp")["ITA"], wide(p, "gdp_nominal_meur")["ITA"]).set_index("item")
    assert (t.max_abs_gap_pp < 0.25).all() and t.loc["net lending", "max_abs_gap_pp"] < 0.1 and t.loc["capital transfers (D.92 + D.99)", "max_abs_gap_pp"] < 0.1 and (t.n_years >= 19).all()  # same accounts: rounding and GDP vintage only
