"""Regional correlation framework: synthetic regions with planted within- and between-country structure, plus real-data checks."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.studies.italy_productivity.analysis import corr
from cifresignificative.studies.italy_productivity.analysis import regional_corr as rc

REGIONAL = Path("data/processed/regional.parquet")


def synthetic_regions(n_countries=14, n_regions=12, seed=0):
    """Regions in countries. x -> y holds within countries; a and b only share a country-level factor (a between-country confound)."""
    rng = np.random.default_rng(seed)
    rows, idx = [], []
    for c in range(n_countries):
        cc = f"{chr(65 + c)}{chr(65 + c)}"
        level = rng.normal(scale=5)
        for r in range(n_regions):
            x = rng.normal()
            rows.append({"x": x, "y": 0.8 * x + rng.normal(scale=0.5) + level, "a": level + rng.normal(), "b": level + rng.normal()})
            idx.append(f"{cc}{r:02d}"[:4])
    X = pd.DataFrame(rows, index=pd.Index(idx, name="geo"))
    return X + 0.0


def test_country_demean_zero_means_and_small_countries_nan():
    X = synthetic_regions().iloc[:30]
    X.index = [i[:2] + str(k) for k, i in enumerate(X.index)]
    d = rc.country_demean(X)
    assert np.allclose(d.groupby(rc.country_of(d.index).to_numpy()).mean().dropna().to_numpy(), 0)
    one = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=["XX01", "YY01", "YY02"])
    assert rc.country_demean(one).v.isna().all()  # no country has MIN_REGIONS regions


def test_within_screen_keeps_within_relation_and_rejects_between_confound():
    X = synthetic_regions()
    within = rc.country_demean(X)
    res = corr.screen(within, n_min=60, groups=rc.country_of(within.index), irl_label="IE")
    pairs = {frozenset(p) for p in zip(res["x"], res["y"])}
    assert frozenset(("x", "y")) in pairs and frozenset(("a", "b")) not in pairs
    between = X.groupby(rc.country_of(X.index).to_numpy()).mean()
    assert frozenset(("a", "b")) in {frozenset(p) for p in zip(*[corr.screen(between, n_min=10, dup_rho=None)[c] for c in ("x", "y")])}  # ...but it is there between countries


def test_shuffle_within_keeps_group_values():
    X = synthetic_regions()
    g = rc.country_of(X.index)
    S = corr.shuffle_within(X, g, np.random.default_rng(1))
    for c in g.unique():
        rows = (g == c).to_numpy()
        assert np.allclose(np.sort(S[rows].x.to_numpy()), np.sort(X[rows].x.to_numpy()))
    assert not np.allclose(S.x, X.x)


def test_leave_one_country_out_flags_single_country_pair():
    rng = np.random.default_rng(3)
    idx = pd.Index([f"{c}{c}{r:02d}"[:4] for c in "ABCDEFGH" for r in range(12)], name="geo")
    X = pd.DataFrame(rng.normal(size=(len(idx), 2)), columns=["u", "v"], index=idx)
    first = X.index.str[:2] == "AA"
    X.loc[first, "v"] = X.loc[first, "u"] * 3  # one country manufactures the correlation
    X.loc[first, "u"] += 20
    X.loc[first, "v"] += 20
    pairs = corr.robustness(corr.pair_screen(X, "pearson", n_min=50), X, "pearson", n_min=50, groups=rc.country_of(X.index))
    assert not pairs.robust.iloc[0]


def test_size_scaled_catches_regional_totals():
    assert set(corr.size_scaled(["nuts2_gdp_mio_eur", "nuts2_population", "gfcf_mio_eur", "fertility_rate"], {})) == {"nuts2_gdp_mio_eur", "nuts2_population", "gfcf_mio_eur"}


def test_regional_matrix_drops_flagged_characters_and_derives_ratios():
    rows = []
    for ind, v in [("nuts2_gdp_mio_eur", 100.0), ("nuts2_hours_ths", 10.0), ("nuts2_population", 5.0), ("nuts2_gdp_pps_pc_eu27", 20000.0)]:
        rows.append({"geo": "ITC4", "year": 2019, "indicator": ind, "value": v, "flag": None})
        rows.append({"geo": "ITC4", "year": 2020, "indicator": ind, "value": v, "flag": "bp"})
    R = rc.regional_matrix(pd.DataFrame(rows), drop_flag_chars="p")
    assert list(R.index.get_level_values("year")) == [2019]
    assert R.gdp_per_hour_eur.iloc[0] == pytest.approx(10000.0) and "ln_gdp_pc" in R
    assert len(rc.regional_matrix(pd.DataFrame(rows), drop_flag_chars="u")) == 2  # 'bp' is kept when only 'u' is dropped


def test_mechanical_ratio_pairs_excluded_from_screen():
    X = synthetic_regions()
    X["gdp_per_hour_eur"], X["compensation_per_hour_eur"] = X.x, X.x + 0.5 * np.random.default_rng(0).normal(size=len(X))
    res = rc.screen_regions(rc.country_demean(X), n_min=60)
    assert frozenset(("gdp_per_hour_eur", "compensation_per_hour_eur")) not in {frozenset(p) for p in zip(res["x"], res["y"])}


def _slope_data(italy_slope, other_slope, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for c in ["IT", "AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "II"]:
        for r in range(8):
            x = rng.normal()
            rows.append({"geo": f"{c}{r:02d}", "x": x, "y": (italy_slope if c == "IT" else other_slope) * x + rng.normal(scale=0.3)})
    return rc.country_demean(pd.DataFrame(rows).set_index("geo"))


def test_italy_slope_placebo_test():
    off = rc.italy_slope_test(_slope_data(3.0, 1.0), "x", "y")
    same = rc.italy_slope_test(_slope_data(1.0, 1.0), "x", "y")
    assert off["p_diff"] == pytest.approx(1 / 10) and off["slope_diff_italy"] > 1.5  # smallest possible p with 10 countries
    assert same["p_diff"] > 0.3


def test_convergence_negative_beta_when_poor_regions_catch_up():
    rng = np.random.default_rng(7)
    rows = []
    for c in ["AA", "BB", "CC"]:
        for r in range(10):
            lv0 = 10 + rng.normal(scale=0.3)
            for yr, lv in [(2005, lv0), (2006, lv0), (2007, lv0), (2017, lv0 + 0.4 - 0.3 * (lv0 - 10) + rng.normal(scale=0.02)), (2018, 0), (2019, 0)]:
                rows.append({"geo": f"{c}{r:02d}", "year": yr, "indicator": "gdp_pps_pc_eu27", "value": np.exp(lv0 + 0.4 - 0.3 * (lv0 - 10)) if yr >= 2017 else np.exp(lv0), "flag": None})
    R = rc.regional_matrix(pd.DataFrame(rows), derive=False)
    res = rc.convergence(R)
    assert res.beta.iloc[0] < -0.5


@pytest.mark.skipif(not REGIONAL.exists(), reason="regional not built")
def test_real_regional_table_is_clean():
    r = pd.read_parquet(REGIONAL)
    assert not r.geo.str.endswith("ZZ").any()
    assert r[(r.country == "IT") & (r.indicator == "nuts2_gdp_pps_pc_eu27")].geo.nunique() == 21
    assert r.indicator.nunique() >= 26


@pytest.mark.skipif(not REGIONAL.exists(), reason="regional not built")
def test_real_within_country_shuffle_gives_no_discoveries():
    X = rc.within_country_table(rc.regional_matrix(pd.read_parquet(REGIONAL)), "level", (2015, 2019))
    assert max(rc.region_permutation_null(X, n_min=80, n_perm=2)) <= 1
