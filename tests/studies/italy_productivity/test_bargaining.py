"""Bargaining coverage and union density: loader, institution means, the cross-country association test and Italy's position."""

import numpy as np
import pandas as pd
import pytest

from cifresignificative.core.sources import oecd_bargaining as OB
from cifresignificative.studies.italy_productivity.analysis import bargaining as B


def test_load_reads_codes_and_drops_empty(tmp_path):
    p = tmp_path / "cbc.csv"
    pd.DataFrame({"DATAFLOW": "x", "REF_AREA": ["ITA", "ITA", "DEU"], "MEASURE": "ERB", "TIME_PERIOD": [2019, 2020, 2020], "OBS_VALUE": [100.0, np.nan, 49.0], "OBS_STATUS": ["A", "A", "B"]}).to_csv(p, index=False)
    d = OB.load("cbc", p)
    assert d.shape[0] == 2 and d.flag.tolist() == ["A", "B"] and set(d.source) == {"oecd:CBC"}


def _series(values: dict[str, list[float]], start=2005):
    return pd.DataFrame([{"iso3": c, "year": start + k, "value": v} for c, vs in values.items() for k, v in enumerate(vs)])


def test_institution_means_need_min_observations_per_variable():
    cbc = _series({"ITA": [100, 100, 100], "DEU": [50, 40, 60], "FRA": [98, 98]})
    tud = _series({"ITA": [30, 31, 32], "DEU": [20, 21, 22], "FRA": [10, 10, 10]})
    m = B.institution_means(cbc, tud, (2005, 2007), min_years=3)
    assert m.loc["ITA", "cbc"] == 100 and m.loc["DEU", "cbc"] == 50 and pd.isna(m.loc["FRA", "cbc"]) and m.loc["FRA", "tud"] == 10  # FRA has two coverage values only


def _data(effect: float, seed: int, n=24):
    rng = np.random.default_rng(seed)
    iso = [f"C{k:02d}" for k in range(n)]
    cbc, tud = rng.uniform(10, 100, n), rng.uniform(5, 60, n)
    pt = effect * cbc / 50 + rng.normal(0, 1, n)
    return pd.DataFrame({"iso3": iso, "pass_through": pt}), pd.DataFrame({"cbc": cbc, "tud": tud}, index=iso)


def test_association_finds_planted_effect_and_not_noise():
    pt, inst = _data(2.0, 1)
    hit = B.association(pt, inst, n_perm=2000).set_index("variable")
    assert hit.loc["cbc", "rho"] > 0.5 and hit.loc["cbc", "p"] < 0.01 and hit.loc["cbc", "significant"] and not hit.loc["tud", "significant"]
    pt0, inst0 = _data(0.0, 2)
    assert not B.association(pt0, inst0, n_perm=2000).significant.any()


def test_association_exclude_drops_countries():
    pt, inst = _data(0.0, 3)
    r = B.association(pt, inst, n_perm=200, exclude=("C00", "C01")).set_index("variable")
    assert r.loc["cbc", "n"] == 22 and r.loc["cbc", "loo_sign_flips"] >= 0


def test_shuffled_null_about_zero_discoveries():
    pt, inst = _data(0.0, 4)
    out = B.shuffled_null(pt, inst, n_rep=80, n_perm=200)
    assert out["mean_discoveries"] < 0.3 and out["mean_share_p_below_005"] < 0.15


def test_italy_position_ranks():
    pt = pd.DataFrame({"iso3": ["ITA", "A", "B"], "pass_through": [0.3, 0.9, -0.1]})
    inst = pd.DataFrame({"cbc": [100.0, 50.0, 20.0], "tud": [30.0, 40.0, 10.0]}, index=["ITA", "A", "B"])
    r = B.italy_position(pt, inst).set_index("variable")
    assert r.loc["pass_through", "rank_1_highest"] == 2 and r.loc["cbc", "rank_1_highest"] == 1 and r.loc["tud", "median_others"] == 25.0


@pytest.mark.skipif(not (OB.RAW_DIR / "cbc.csv").exists(), reason="OECD files not downloaded")
def test_real_italy_coverage_is_100_every_year_and_density_30():
    c, t = OB.load("cbc"), OB.load("tud")
    assert (c[c.iso3 == "ITA"].value == 100).all() and 25 < t[(t.iso3 == "ITA") & (t.year == 2024)].value.iloc[0] < 35
