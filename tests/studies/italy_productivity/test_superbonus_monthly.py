"""ENEA monthly Superbonus reports by region: parser on both layouts, the committed table against the national totals, and the window gradient family."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.core.sources import enea_monthly as EM
from cifresignificative.core.sources import enea_regions as ER
from cifresignificative.studies.italy_productivity.analysis import (
    superbonus_monthly as SM,
)

OLD = """Super Ecobonus 110%                            31 dicembre 2022
                                  Abruzzo
                                    % lavori
                                   realizzati   % edifici   % Invest.
N. di assevrerzioni                                  10.080
Totale investimenti ammessi a detrazione      2.131.615.679,54 €
Totale investimenti per lavori conclusi ammessi a detrazione      1.499.868.830,23 €    70,4%
Detrazioni maturate per i lavori conclusi      1.649.855.713,25 €
    Condomini
    N. di asseverazioni condominiali                  1.893     18,8%
    Tot. Inv. Condominiali      1.236.216.165,66 €      58,0%
    Tot. Lavori Condominiali realizzati     826.553.055,02 €    66,9%
    Edifici unifamiliari
    N. di asseverazioni in edifici unifamiliari     6.078     60,3%
    Tot. Inv. in edifici unifamiliari      689.396.447,23 €     32,3%
    Tot. Lavori in edifici unifamiliari realizzati      508.375.391,69 €    73,7%
    Investimento medio
    Condomini     653.046,05 €
"""
NEW = """SuperEcobonus                                31 dicembre 2025
                                  Tren no-Alto Adige
N. di ediﬁci                                  11451
Totale inves men (*)                  3.141.716.498,54 €
Totale inves men ammessi a detrazione      3.141.716.498,54 €
Totale inves men per lavori conclusi ammessi a detrazione      3.105.026.791,12 €      98,8%
Detrazioni maturate per i lavori conclusi      3.300.000.000,00 €
    Condomini
    N. di ediﬁci condominiali       6.434     56,2%
    Totale inves men (*)       2.582.673.139,76 €
    Tot. Inv. Condominiali ammessi a detrazione      2.582.673.139,76 €     82,2%
    Tot. Lavori Condominiali realizza ammessi a detrazione      2.550.191.599,17 €    98,7%
    A/9 aper al pubblico
    N. ediﬁci A/9 aper al pubblico        5      0,0%
    Totale inves men (*)       1.056.066,97 €
    Tot. Inv. ammessi a detrazione      1.056.066,97 €     0,0%
    Tot. Lavori realizza ammessi a detrazione      1.056.066,97 €     0,0%
"""


def test_parse_page_old_and_new_layout():
    name, v = EM.parse_page(OLD)
    assert name == "Abruzzo" and v["n_buildings"] == 10080 and v["inv_admitted"] == pytest.approx(2131615679.54) and v["inv_completed"] == pytest.approx(1499868830.23) and v["pct_completed"] == pytest.approx(0.704)
    assert v["condo_n"] == 1893 and v["condo_admitted"] == pytest.approx(1236216165.66) and v["single_completed"] == pytest.approx(508375391.69) and "inv_incl_nonadmitted" not in v and "avg" not in "".join(v)  # the average-investment block is skipped
    name, v = EM.parse_page(NEW)  # ligatures, glued thousands, the sum including non-admitted amounts, the A/9 section (its header holds a digit)
    assert name == "Trentino-Alto Adige" and v["n_buildings"] == 11451 and v["inv_incl_nonadmitted"] == pytest.approx(3141716498.54) and v["condo_n"] == 6434
    assert v["a9_n"] == 5 and v["a9_completed"] == pytest.approx(1056066.97) and v["condo_completed"] == pytest.approx(2550191599.17)


@pytest.mark.parametrize("raw,expected", [("Dato Nazionale", "Italia"), ("Totale nazionale", "Italia"), ("Tren#no-Alto Adige", "Trentino-Alto Adige"), ("Friuli Venezia Giulia", "Friuli-Venezia Giulia"), ("Veneto", "Veneto")])
def test_region_name(raw, expected):
    assert EM.region_name(raw) == expected


def test_committed_table_adds_up_and_matches_national_snapshots():
    d = EM.load()
    assert d.date.nunique() == 62 and (d.groupby("date").size() == 21).all() and d.date.min() == pd.Timestamp("2021-08-31") and d.date.max() == pd.Timestamp("2026-09-30")
    gap = EM.check(d)
    assert gap.filter(like="gap_").abs().max().max() < 1e5  # the regions add up to Italy (the largest gap is 2023-03, ENEA's own rounding)
    nat = pd.read_csv("data/external/enea_superbonus.csv", parse_dates=["snapshot"]).set_index("snapshot")
    it = d[d.region == "Italia"].set_index("date")
    for snap in ("2023-12-31", "2025-03-31"):  # same bulletin basis
        assert it.loc[snap, "inv_completed"] == pytest.approx(nat.loc[snap, "investments_completed_eur"], abs=1e5)  # within 0.0001%: the transcription is from the annual report
    assert it.n_buildings.is_monotonic_increasing is False and it.inv_completed.iloc[:40].is_monotonic_increasing  # revisions only from 2024
    assert d[d.region != "Italia"].a9_n.notna().sum() > 0 and d.n_buildings.notna().all()


def test_basis_check_annual_regional_table_differs_but_ranks_agree():
    d = EM.load()
    nat, reg = SM.basis_check(d, ER.load(), pd.read_csv("data/external/enea_superbonus.csv", parse_dates=["snapshot"]))
    assert nat.set_index("snapshot").ratio.loc["2023-12-31"] == pytest.approx(1.0, abs=1e-5) and nat.ratio.max() < 1.06
    assert (reg.rho_between_bases > 0.97).all() and reg.ratio_min.min() > 0.7


def test_bh_and_gradient_family_with_planted_effect_and_null():
    assert np.allclose(SM._bh(np.array([0.01, 0.04, 0.03, 0.5])), [0.04, 0.0533333, 0.0533333, 0.5])
    rng = np.random.default_rng(3)
    regions = [f"R{i}" for i in range(20)]
    inc = rng.uniform(20000, 40000, 20)
    rows = []
    for k, (label, _, _) in enumerate(SM.WINDOWS):
        y = inc * 0.05 + rng.normal(0, 150, 20) if k == 4 else rng.uniform(500, 2500, 20)  # window 4 carries the planted gradient
        rows += [{"window": label, "region": r, "gdp_per_inhabitant_eur": i, "per_inhabitant_eur": v} for r, i, v in zip(regions, inc, y, strict=True)]
    g = SM.window_gradient(pd.DataFrame(rows), n_perm=4000)
    assert g.loc[4, "q"] < 0.01 and g.loc[4, "rho"] > 0.7 and (g.drop(4).q > 0.05).sum() >= 6
    pl = SM.gradient_placebo(pd.DataFrame(rows).assign(per_inhabitant_eur=lambda x: rng.uniform(500, 2500, len(x))), n_shuffle=60, n_perm=1000)
    assert pl["share_with_discovery"] <= 0.2  # the shuffled null gives about zero discoveries (nominal 5%)


@pytest.mark.skipif(not Path("data/processed/regional.parquet").exists(), reason="processed regional data missing")
def test_window_flows_and_gradient_on_committed_data():
    d, reg = EM.load(), pd.read_parquet("data/processed/regional.parquet")
    f = SM.window_flows(d, reg)
    tot = f.groupby("window").flow_eur.sum()
    cum = d[d.region == "Italia"].set_index("date").inv_admitted
    assert tot.sum() == pytest.approx(cum.loc["2026-09-30"], rel=1e-6) and tot["H1 2023"] == pytest.approx(cum.loc["2023-06-30"] - cum.loc["2022-12-31"], rel=1e-6)
    z = SM.zone_flow_shares(f)
    assert abs(z.groupby("window").share_of_flow.sum() - 1).max() < 1e-9
    g = SM.window_gradient(f, n_perm=4000).set_index("window")
    assert g.loc["H1 2023", "rho"] > 0.6 and g.loc["H1 2023", "q"] < 0.05 and g.loc["H1 2022", "q"] > 0.5  # the income gradient of the new works arises in 2023, not before
    path = SM.gradient_path(d, reg).set_index("date").rho
    assert path.loc["2022-11-30"] < 0.3 and path.loc["2023-03-31"] > 0.4
