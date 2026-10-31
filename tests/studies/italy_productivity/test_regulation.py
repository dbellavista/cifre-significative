"""OECD regulation: loaders on small fixtures, the sector test (planted effect, calibrated null), Italy's rankings."""
import numpy as np
import pandas as pd
import pytest

from cifresignificative.core.sources import oecd_regulation as OR
from cifresignificative.studies.italy_productivity.analysis import regulation as R


def test_load_pmr_drops_empty_cells(tmp_path):
    p = tmp_path / "pmr.csv"
    pd.DataFrame({"DATAFLOW": "x", "REF_AREA": ["ITA", "ITA", "DEU"], "MEASURE": ["PMR", "SECTPMR", "PMR"], "TIME_PERIOD": [2018, 2018, 2023], "OBS_VALUE": [1.29, np.nan, 1.1]}).to_csv(p, index=False)
    d = OR.load_pmr(p)
    assert d.shape[0] == 2 and set(d.measure) == {"PMR"} and set(d.columns) >= {"iso3", "measure", "year", "value", "source"}


def test_load_epl_keeps_one_version(tmp_path):
    p = tmp_path / "epl.csv"
    pd.DataFrame({"REF_AREA": ["ITA"] * 3, "MEASURE": ["EPL_OV"] * 3, "TIME_PERIOD": [2013, 2013, 2014], "OBS_VALUE": [3.0, 3.1, 3.2], "VERSION": ["VERSION1", "VERSION4", "VERSION1"]}).to_csv(p, index=False)
    d = OR.load_epl(p)
    assert d.value.tolist() == [3.0, 3.2] and set(d.source) == {"oecd:EPL:VERSION1"}
    assert OR.load_epl(p, "VERSION4").value.tolist() == [3.1]


def _pmr(values: dict[str, dict[str, float]], year=2018):
    return pd.DataFrame([{"iso3": c, "measure": m, "year": year, "value": v} for c, d in values.items() for m, v in d.items()])


def test_position_rank_and_median():
    pmr = _pmr({"ITA": {"PMR": 1.0}, "DEU": {"PMR": 2.0}, "FRA": {"PMR": 0.5}, "ESP": {"PMR": 0.7}})
    r = R.position(pmr, ["PMR"], (2018,), eu=["ITA", "DEU", "FRA", "ESP"]).iloc[0]
    assert r["rank"] == 2 and r.median_others == 0.7 and r.n_eu == 4 and r.italy == 1.0


def test_epl_gap_rank_is_per_year_and_stops_at_last():
    rows = [{"iso3": c, "measure": "EPL_OV", "year": y, "value": v} for y in (2018, 2019, 2020) for c, v in (("ITA", 3.0), ("DEU", 2.0 + (y == 2019) * 2), ("FRA", 2.5))]
    g = R.epl_gap(pd.DataFrame(rows), last=2019, eu=["ITA", "DEU", "FRA"])
    assert g.index.tolist() == [2018, 2019] and g["rank"].tolist() == [1, 2] and g.median_others.tolist() == [2.25, 3.25]


def test_sector_regulation_averages_matched_indicators():
    pmr = _pmr({"ITA": {"LAWYERS": 3.0, "ACCOUNTANTS": 1.0, "RETAIL_TRADE": 2.0}, "DEU": {"LAWYERS": 1.0, "ACCOUNTANTS": 1.0}})
    r = R.sector_regulation(pmr).set_index(["iso3", "nace_r2"]).reg
    assert r[("ITA", "M69_M70")] == 2.0 and r[("ITA", "G47")] == 2.0 and ("DEU", "G47") not in r.index  # a sector needs all its indicators


def _panel(effect: float, seed: int, n=20):
    rng = np.random.default_rng(seed)
    rows = []
    for sec in ["G47", "M71", "D"]:
        reg = rng.normal(size=n)
        g = effect * reg + rng.normal(size=n) + {"G47": 0, "M71": 3, "D": -2}[sec]  # sector levels differ and must drop out
        rows += [{"iso3": f"C{k:02d}", "nace_r2": sec, "reg": reg[k], "g": g[k]} for k in range(n)]
    d = pd.DataFrame(rows)
    return d.assign(rg=lambda t: R._within_rank(t, "g"), rr=lambda t: R._within_rank(t, "reg"))


def test_within_rank_has_mean_zero_unit_sd_per_sector():
    d = _panel(0.0, 0)
    assert np.allclose(d.groupby("nace_r2").rr.mean(), 0) and np.allclose(d.groupby("nace_r2").rr.std(ddof=0), 1)


def test_pooled_test_finds_planted_effect_and_not_noise():
    hit = R.test(_panel(-1.0, 1), n_perm=2000).set_index("sector")
    null = R.test(_panel(0.0, 2), n_perm=2000).set_index("sector")
    assert hit.loc["pooled", "rho"] < -0.4 and hit.loc["pooled", "p"] < 0.01 and hit.loc["pooled", "significant"]
    assert null.loc["pooled", "p"] > 0.05 and not null.significant.any()


def test_shuffled_null_gives_about_zero_discoveries():
    out = R.shuffled_null(_panel(-1.0, 3), n_rep=30, n_perm=200)
    assert out["mean_discoveries"] < 0.3 and out["mean_share_p_below_005"] < 0.2


def test_italy_rows_and_placebo_rank():
    rows = []
    for c, reg, g in [("ITA", 3.0, -1.0), ("A", 1.0, 1.0), ("B", 2.0, 0.5), ("C", 4.0, 0.0)]:
        for sec in ["G47", "D", "M71", "J61"]:
            rows.append({"iso3": c, "nace_r2": sec, "reg": reg, "g": g})
    d = pd.DataFrame(rows)
    d = d.assign(rg=lambda t: R._within_rank(t, "g"), rr=lambda t: R._within_rank(t, "reg"))
    it = R.italy_rows(d)
    assert (it.reg_rank == 2).all() and (it.growth_rank == 4).all() and len(it) == 4
    pl = R.italy_placebo(d)
    assert pl.iso3.iloc[0] == "ITA" and pl["rank"].iloc[0] == 1 and pl.n_countries.iloc[0] == 4


@pytest.mark.skipif(not (OR.RAW_DIR / "pmr.csv").exists(), reason="OECD files not downloaded")
def test_real_pmr_has_italy_and_two_waves():
    pmr = OR.load_pmr()
    assert set(pmr.year) == {2018, 2023} and "ITA" in set(pmr.iso3)
    pos = R.position(pmr, ["ACCOUNTANTS"], (2018,)).iloc[0]
    assert pos["rank"] == 1 and pos.n_eu == 27  # Italy is the most restrictive EU member for accountants
