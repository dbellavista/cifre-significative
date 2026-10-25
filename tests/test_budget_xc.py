"""Commission forecast parser and the cross-country placebo ranking of the deficit surprise."""
import numpy as np
import pandas as pd
import pytest

from cifresignificative.analysis import budget_xc as BX
from cifresignificative.sources import ec_forecasts as ec

TABLE = """Table 36:   Net lending (+) or net borrowing (-), general government (as a percentage of GDP, 1997-2018)                                             31.10.2016
                                         averages                                                                        forecast            forecast
                             1997-01      2002-06    2007-11         2012        2013        2014          2015    2016       2017   2018   2016        2017
Germany                           -1.9        -3.4        -1.7         0.0         -0.2        0.3          0.7     0.6       0.4     0.3      0.2         0.1
Latvia                            -1.4        -1.2        -5.1 :      -0.8         -0.9        -1.6         -1.3    -0.8     -1.1    -1.2     -1.0        -1.0
Italy                             -2.5        -3.6        -3.5        -2.9         -2.7        -3.0         -2.6    -2.4     -2.4    -2.5     -2.4        -1.9
Euro area                         -1.9        -2.6        -3.9        -3.6         -3.0        -2.6         -2.1    -1.8     -1.5    -1.5     -1.9        -1.6
Croatia                              :        -4.1        -5.0        -5.3         -5.3        -5.4         -3.3    -2.1     -1.8    -1.4     -2.7        -2.3
United States                     -5.0         -4.0       -9.0        -8.0         -4.0        -4.0         -3.5    -3.0     -3.0    -3.0     -3.0        -3.0
Belgium                           -1.0         -1.0       -3.0        -4.0         -3.0        -3.0         -2.5    -2.0     -2.0    -2.0     -2.0        -2.0
"""


def test_parse_table_years_flags_and_stops():
    d = ec.parse_table(TABLE, ec.TABLES["net_lending"])
    it = d[d.iso3 == "ITA"].set_index("year").value
    assert list(it.index) == [2012, 2013, 2014, 2015, 2016, 2017, 2018] and it[2016] == -2.4 and it[2018] == -2.5  # the previous-forecast columns are not read as years
    assert d[d.iso3 == "LVA"].set_index("year").value[2012] == -0.8  # a flag glued to a number ("-5.1 :") is not a value
    assert len(d[d.iso3 == "HRV"]) == 7 and d[d.iso3 == "HRV"].set_index("year").value[2012] == -5.3  # a country without history lacks an average
    assert "BEL" not in set(d.iso3)  # rows after the United States row belong to the next table


def test_committed_forecasts():
    F = BX.load_forecasts()
    assert sorted(F.vintage.unique()) == list(range(2015, 2026)) and set(F.variable) == set(ec.TABLES)
    g = F[(F.variable == "real_growth") & (F.iso3 == "ITA") & (F.vintage == 2022)].set_index("year").value
    assert g[2022] == 3.8 and g[2023] == 0.3  # autumn 2022 growth forecasts for Italy
    assert not F.duplicated(["vintage", "variable", "iso3", "year"]).any()
    nl = BX.forecast_wide(F, "net_lending")
    assert list(nl.index) == list(range(2016, 2027)) and nl.loc[2020, "ITA"] == -2.3 and nl.loc[2021, "ITA"] == -7.8  # autumn 2019 and autumn 2020 forecasts for Italy
    assert nl.loc[2017:2025, BX.EU27_ISO3].notna().all().all()


def synthetic(n=27, shift=3.0, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"C{i:02d}" for i in range(n - 1)] + ["ITA"]
    S = pd.DataFrame(rng.normal(0, 1, (len(BX.YEARS), n)), index=BX.YEARS, columns=cols)
    S.loc[2021:2023, "ITA"] += shift
    return S


def test_italy_rank_and_idiosyncratic():
    S = synthetic()
    r = BX.italy_rank(S, [2021, 2022, 2023])
    assert r["rank"] == 1 and r["n"] == 27 and r["p"] == pytest.approx(1 / 27)
    common = S + np.linspace(-2, 2, len(S))[:, None]  # a common shock moves the raw surprise of everyone, not the idiosyncratic one
    a, b = BX.idiosyncratic(S), BX.idiosyncratic(common)
    assert np.allclose(a, b)


def test_window_mean_drops_incomplete_countries():
    S = synthetic()
    S.loc[2022, "C00"] = np.nan
    m = BX.window_mean(S, [2021, 2022, 2023])
    assert np.isnan(m["C00"]) and m.notna().sum() == 26


def test_sign_null_gives_no_discoveries_and_screen_finds_a_planted_country():
    S = synthetic(shift=0.0)
    S["ITA"] += 2.5  # a persistent overrun in every year
    scr = BX.country_sign_flip(S, [y for y in BX.YEARS if y != 2020], n_perm=4000)
    assert scr.iso3.iloc[0] == "ITA" and scr.p.iloc[0] < 0.02
    assert not scr.significant.any()  # with 8 years the smallest sign-flip p is 2/256, so the smallest q over 27 countries is about 0.2: the screen orders countries, it cannot discover
    nl = BX.sign_null(synthetic(shift=0.0), BX.YEARS, n_shuffle=40, n_perm=500)
    assert nl.discoveries.mean() < 0.25 and 0.0 <= nl.share_p_below_005.mean() < 0.12


def test_surprise_identity_on_committed_data():
    pytest.importorskip("pyarrow")
    from cifresignificative.analysis.data import PROCESSED, load_panel
    if not (PROCESSED / "panel.parquet").exists() or not (PROCESSED / "spending.parquet").exists():
        pytest.skip("processed data missing")
    S = BX.surprises(BX.load_forecasts(), BX.outturn_wide(load_panel()))
    gap = (S["revenue_shortfall"] + S["primary_expenditure_overrun"] + S["interest_overrun"] - S["deficit"]).abs()
    assert gap.stack().max() < 0.5 and gap.stack().mean() < 0.1  # the parts add up to the deficit surprise up to rounding of the published figures
    assert S["deficit"].loc[2022:2023, "ITA"].min() > 2.0


def test_first_estimate_sensitivity_on_committed_data():
    pytest.importorskip("pyarrow")
    from cifresignificative.analysis.data import PROCESSED, load_panel
    if not (PROCESSED / "panel.parquet").exists() or not (PROCESSED / "spending.parquet").exists():
        pytest.skip("processed data missing")
    ranks, rev = BX.first_estimate_sensitivity(BX.load_forecasts(), BX.outturn_wide(load_panel()))
    r = ranks[(ranks.window == "2021-23") & (ranks.measure == "raw")].set_index("outturn")
    assert r.loc["early estimate", "rank"] == 1 and r.loc["latest vintage", "rank"] == 1  # the result does not depend on the outturn vintage
    r21 = rev.set_index("year").loc[2021]
    assert r21.italy > 1.0 and r21.max_others < 0.5  # the 2021 revision (building credits) is Italian, not a common one


def test_denominator_split_on_committed_data():
    pytest.importorskip("pyarrow")
    from cifresignificative.analysis.data import PROCESSED, load_panel, wide
    if not (PROCESSED / "panel.parquet").exists() or not (PROCESSED / "spending.parquet").exists():
        pytest.skip("processed data missing")
    P, F = load_panel(), BX.load_forecasts()
    S = BX.surprises(F, BX.outturn_wide(P))["deficit"]
    r = BX.gdp_ratio(F, wide(P, "gdp_nominal_meur"))
    assert r.loc[2017:2025, BX.EU27_ISO3].stack().between(0.8, 1.2).all() and r.loc[2022, "ITA"] < 1  # inflation made the 2022 economy larger than forecast
    sp = BX.denominator_split(S, F, r)
    assert np.allclose((sp["denominator"] + sp["level"]).stack(), S.stack())  # the two parts add up
    assert sp["denominator"].loc[2021:2023, "ITA"].max() < 0 and sp["level"].loc[2021:2023, "ITA"].mean() > S.loc[2021:2023, "ITA"].mean()  # at constant GDP the Italian overrun is larger, not smaller


OLD = """
""" + "\n".join(f"filler {i}" for i in range(310)) + """
76. Net lending (+) or net borrowing (-); general government 1)

                       BE         DE         EE         ES         FR         IT         EA-19
        2016           -2.6        0.5        0.2       -3.6       -3.4       -2.3       -1.9
        2017           -2.3        0.4        0.1       -2.6       -3.3       -1.6       -1.5
76. Net lending (+) or net borrowing (-); general government 1)

                       BG         CZ         DK         HR         HU         PL         US         JP
        2016           -2.7       -1.3       -2.5       -4.7       -2.1       -2.8       -3.5       -5.7
        2017           -2.7       -1.1       -1.7       -4.1       -2.0       -2.8       -3.2       -5.1
"""


def test_parse_old_table_two_pages():
    d = ec.parse_old_table(OLD, 76)
    assert d[(d.iso3 == "ITA") & (d.year == 2016)].value.item() == -2.3 and d[(d.iso3 == "EA") & (d.year == 2017)].value.item() == -1.5
    assert d[(d.iso3 == "JPN") & (d.year == 2017)].value.item() == -5.1  # second page (non-euro members)
