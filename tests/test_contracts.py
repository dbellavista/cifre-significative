"""Contract-timing measures on toy staircases (offline) and the committed analysis results when the Istat files are present."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.analysis import contracts as C
from cifresignificative.sources import istat_contracts as ic


def _wages(series: dict[str, list[float]], start="2019-01-31") -> pd.DataFrame:
    months = pd.date_range(start, periods=len(next(iter(series.values()))), freq="ME")
    return pd.DataFrame([{"group": g, "month": m, "index": v, "provisional": False} for g, vals in series.items() for m, v in zip(months, vals, strict=True)])


def _staircase(n: int, steps: dict[int, float], base: float = 100.0) -> list[float]:
    out, level = [], base
    for i in range(n):
        level *= 1 + steps.get(i, 0.0)
        out.append(level)
    return out


def test_timing_counts_months_since_the_last_step():
    n = 60  # January 2019 to December 2023; month 22 = November 2020, month 26 = March 2021
    w = _wages({"A": _staircase(n, {22: 0.01, 26: 0.01}), "B": _staircase(n, {5: 0.01})})
    t = C.timing(C.wide_index(w, ["A", "B"]), 2021)
    assert t.loc["A", "staleness_months"] == 9 and t.loc["A", "standstill_24m"] == 22  # last step in March 2021; two steps in the 24 months to December 2021
    assert t.loc["B", "staleness_months"] == 30 and t.loc["B", "standstill_24m"] == 24  # one step in June 2019, outside the last 24 months


def test_a_rise_below_the_threshold_is_not_a_step():
    w = _wages({"A": _staircase(60, {26: 0.001})})
    t = C.timing(C.wide_index(w, ["A"]), 2021)
    assert np.isnan(t.loc["A", "staleness_months"]) and t.loc["A", "standstill_24m"] == 24


def test_cumulative_inflation_and_real_change():
    dec = pd.Series({2022: 10.0, 2023: 5.0})
    assert C.cumulative_inflation(dec, 2021, 2023) == pytest.approx(15.5)
    n = 60  # the index ends 2021 at its level and rises 20% in 2022
    series = _staircase(n, {5: 0.01, 36 + 2: 0.2})  # a step in 2019, so staleness is defined; month 38 = March 2022
    w = _wages({"A": series})
    groups = pd.DataFrame({"code": ["A"], "name_it": ["a"], "name_en": ["a"], "parent": ["P"], "leaf": [True], "branch": ["private"]})
    d = C.window_table(w, groups, dec, 2021, 2023)
    assert d.loc["A", "nominal_change"] == pytest.approx(20.0) and d.loc["A", "real_change"] == pytest.approx((1.2 / 1.155 - 1) * 100)
    assert d.loc["A", "steps_after"] == 1


def test_cluster_table_averages_the_sub_sectors():
    d = pd.DataFrame({"parent": ["P", "P", "Q"], "branch": ["private"] * 3, "staleness_months": [10.0, 20.0, 5.0], "standstill_24m": [22, 24, 23], "steps_after": [2, 4, 1], "nominal_change": [5.0, 7.0, 6.0],
                      "real_change": [-8.0, -6.0, -7.0], "inflation": [14.0] * 3}, index=["a", "b", "c"])
    c = C.cluster_table(d)
    assert c.loc["P", "n_leaves"] == 2 and c.loc["P", "staleness_months"] == 15 and c.loc["P", "real_change"] == -7 and c.loc["Q", "n_leaves"] == 1


def test_family_finds_a_planted_association_and_not_a_shuffled_one():
    rng = np.random.default_rng(0)
    n = 40
    x = rng.permutation(n).astype(float)
    d = pd.DataFrame({"staleness_months": x, "standstill_24m": rng.permutation(n), "real_change": -x + rng.normal(0, 3, n), "branch": ["private"] * 30 + ["public"] * 10})
    f = C.family(d, ("real_change",), n_perm=2000).set_index("x")
    assert f.loc["staleness_months", "rho"] < -0.8 and f.loc["staleness_months", "significant"] and not f.loc["standstill_24m", "significant"]


def test_load_wages_keeps_the_hourly_index_of_all_employees(tmp_path):
    rows = [{"MAIN_AGREEMENT_GROUP": g, "DATA_TYPE": dt, "FREQ": "M", "PROF_STATUS_EMP": pe, "TIME_PERIOD": "2021-12", "OBS_VALUE": 100.0, "OBS_STATUS": None}
            for g in ("Z1", "Z2") for dt in ("WAGE_H_2021", "WAGE_E_2021") for pe in (10, 23)]
    p = tmp_path / "w.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    w = ic.load_wages(p)
    assert len(w) == 2 and set(w.group) == {"Z1", "Z2"} and w["month"].iloc[0] == pd.Timestamp("2021-12-31")


def _cycles(rows):
    d = pd.DataFrame(rows, columns=["code", "signed", "expiry"])
    d["signed"], d["expiry"] = pd.to_datetime(d.signed), pd.to_datetime(d.expiry)
    return d


def test_renewal_timing_months_since_signing_expiry_and_next_renewal():
    cyc = _cycles([("X", "2019-06-15", "2021-06-30"), ("X", "2023-03-10", "2025-12-31"), ("Y", "2020-01-10", "2022-12-31"), ("Z", "2022-05-01", "2024-12-31")])
    m = pd.DataFrame({"istat_group": ["g1", "g2", "g3", "g4"], "cnel_code": ["X", "Y", "Z", ""], "evidence": ["verified", "partly verified", "verified", "unverified"]})
    t = C.renewal_timing(cyc, m, 2021, pd.Timestamp("2026-10-17"))
    assert list(t.index) == ["g1", "g2", "g3"]  # the unverified group has no code
    assert t.loc["g1", "months_since_renewal"] == 30 and t.loc["g1", "months_expired"] == 6 and t.loc["g1", "months_to_renewal"] == 15 and t.loc["g1", "renewed_after"]
    assert t.loc["g2", "months_expired"] == 0 and not t.loc["g2", "renewed_after"] and t.loc["g2", "months_to_renewal"] == 58  # still valid in December 2021; censored at the archive date
    assert np.isnan(t.loc["g3", "months_since_renewal"]) and t.loc["g3", "months_to_renewal"] == 5  # nothing signed before the start of the window
    d = pd.DataFrame({"real_change": [1.0, 2.0, 3.0]}, index=pd.Index(["g1", "g2", "g3"], name="group"))
    assert list(C.with_renewals(d, t).index) == ["g1", "g3"] and list(C.with_renewals(d, t, ("verified", "partly verified")).index) == ["g1", "g2", "g3"]


def test_sector_deflator_assignment_and_real_change():
    g = pd.DataFrame({"code": ["Z0010", "Z2520", "Z1240", "Z1260", "Z0060", "Z0160", "Z2560", "Z2700"], "parent": ["Z2520", "Z3620", "Z2520", "Z1240", "Z2520", "Z0060", "Z2540", "Z2560"]})
    d = C.deflator_of(g)
    assert d["Z0010"] == "goods" and d["Z0160"] == "goods" and d["Z1260"] == "services" and d["Z2700"] == "all-items" and d["Z2520"] == "all-items"
    decs = {"goods": pd.Series({2022: 10.0, 2023: 10.0}), "services": pd.Series({2022: 0.0, 2023: 0.0}), "all-items": pd.Series({2022: 5.0, 2023: 5.0})}
    leaves = pd.DataFrame({"nominal_change": [0.0, 0.0, 0.0]}, index=["Z0010", "Z1260", "Z2700"])
    r = C.sector_real_change(leaves, d, decs, 2021, 2023)
    assert r["Z0010"] == pytest.approx((1 / 1.21 - 1) * 100) and r["Z1260"] == pytest.approx(0.0) and r["Z2700"] == pytest.approx((1 / 1.1025 - 1) * 100)


def test_proxy_agreement_reports_the_rank_correlations():
    d = pd.DataFrame({"staleness_months": [1, 2, 3, 4], "months_since_renewal": [10, 20, 30, 40], "standstill_24m": [1, 2, 3, 4], "months_expired": [4, 3, 2, 1], "months_to_renewal": [1, 2, 3, 4], "steps_after": [4, 3, 2, 1]})
    a = C.proxy_agreement(d).set_index("x")
    assert a.loc["staleness_months", "rho"] == pytest.approx(1.0) and a.loc["standstill_24m", "rho"] == pytest.approx(-1.0) and (a.n == 4).all()


@pytest.mark.skipif(not Path("data/raw/istat/contracts/groups.csv").exists(), reason="Istat contract files not downloaded")
def test_agreement_hierarchy_and_results():
    g = ic.load_groups()
    assert len(g) == 113 and g.leaf.sum() == 80 and (g.branch == "public").sum() == 16 and set(g[g.branch == "aggregate"].code) == {"Z2520", "Z3620"}
    from cifresignificative.analysis import experiments as ex
    r = ex.run_contracts_all(out=Path("/tmp/contracts_test"), n_perm=1000, n_shuffle=3)
    f = r["c2_cluster_family"].set_index(["x", "y"])
    assert (f.n == 30).all() and (f.rho < -0.4).all() and r["c1_leaf_groups"].shape[0] == 78
    assert (r["c7_placebo_2016_19"].rho.abs() < 0.3).all()  # the calm-period placebo is flat
    assert r["c13_renewal_leaves"].shape[0] == 74 and set(r["c13_renewal_leaves"].evidence) == {"verified", "partly verified"}
    c14 = r["c14_cnel_cluster_family"]
    assert len(c14) == 4 and (c14.n == 24).all() and (r["c20_cnel_null"].discoveries.mean() < 0.5)
    assert (r["c24_proxy_within_sector"].n == 30).all() and set(r["c26_timing_by_sector"].sector) == {"goods", "services", "all-items"}  # every parent has a sector, Z2720 and the bare aggregate Z2520 included
    assert r["c28_weighted_proxy"].n.eq(26).all() and r["c31_weights"].employees.notna().sum() == 26 and r["c27_weighted_summary"].query("group == 'public'").n_with_count.eq(0).all()  # public agreements have no CNEL-INPS count
    assert r["c33_weighted_null"].discoveries.mean() < 0.5 and r["c36_weighted_proxy_all"].n.eq(30).all() and r["c34_weighted_summary_all"].query("group == 'public'").n_with_count.eq(13).all()
    v = r["c41_weighted_variants"]
    assert len(v) == 64 and v.groupby(["year", "extended"]).size().eq(16).all() and v[(v.parent_mean == "weighted") & (v.family == "proxy") & (v.x == "staleness_months")].q_value.max() < 0.1
    cum = r["c23_deflators"].query("end == 2024").set_index("component").cumulative_pct
    assert cum["goods"] > cum["all-items"] > cum["services"]


def test_weighted_family_and_null():
    rng = np.random.default_rng(1)
    n = 30
    d = pd.DataFrame({"x": rng.normal(size=n), "branch": "private"})
    d["real_change"] = -d.x + rng.normal(0, 0.3, n)
    d = d.rename(columns={"x": "staleness_months"})
    w_even = pd.Series(1.0, index=d.index)
    t = C.weighted_family(d, w_even, xs=("staleness_months",), n_perm=2000)
    assert t.rho_weighted.iloc[0] == pytest.approx(t.rho.iloc[0], abs=1e-9) and t.n_eff.iloc[0] == pytest.approx(n) and t.p.iloc[0] < 0.01  # equal weights give the unweighted Spearman
    w_one = pd.Series(1e-3, index=d.index); w_one.iloc[0] = 1e3
    assert C.weighted_family(d, w_one, xs=("staleness_months",), n_perm=200).n_eff.iloc[0] < 1.01  # one huge agreement leaves one effective group
    nl = C.weighted_null(d.assign(real_change=rng.normal(size=n)), pd.Series(rng.lognormal(size=n), index=d.index), xs=("staleness_months",), n_shuffle=40, n_perm=200)
    assert nl.discoveries.mean() < 0.2


def test_leaf_weights_split_shared_codes():
    m = pd.DataFrame({"istat_group": ["a", "b", "c", "d"], "cnel_code": ["X1", "X1", "Y2", ""]})
    wk = pd.DataFrame({"code": ["X1", "Y2"], "employees": [100, 50]})
    w = C.leaf_weights(m, wk)
    assert w["a"] == 50 and w["b"] == 50 and w["c"] == 50 and "d" not in w.index


@pytest.mark.skipif(not Path("data/external/cnel_workers.csv").exists(), reason="cnel_workers.csv is not redistributed: rebuild it from the CNEL Excel files")
def test_committed_workers_table():
    from cifresignificative.sources import cnel_contracts as cn
    from cifresignificative.sources import cnel_workers as cw
    w = cw.load()
    assert w.code.is_unique and 15e6 < w.employees.sum() < 16e6 and w.set_index("code").employees["A011"] == 5196  # the operai agricoli contract
    lw = C.leaf_weights(cn.load_map(), w)
    assert lw.notna().sum() == 64 and lw.reindex(["Z2700", "Z3000"]).isna().all()  # ministries and local authorities (ARAN) have no count


@pytest.mark.skipif(not Path("data/external/cnel_workers.csv").exists(), reason="cnel_workers.csv is not redistributed: rebuild it from the CNEL Excel files")
def test_public_weights_from_committed_headcount():
    from cifresignificative.sources import cnel_contracts as cn
    from cifresignificative.sources import cnel_workers as cw
    from cifresignificative.sources import conto_annuale as ca
    hc = ca.load(year=2021)
    assert hc.contract.is_unique and 3.2e6 < hc.heads.sum() < 3.3e6 and hc.set_index("contract").heads["SCLA"] > 1.1e6  # schools are the largest contract
    pub = C.public_leaf_weights(ca.load_map(), hc)
    assert len(pub) == 13 and pub["Z3340"] == hc.set_index("contract").heads[["CCSS", "PPSS", "GGFF", "PPEN", "CM09"]].sum()
    allw = C.all_leaf_weights(C.leaf_weights(cn.load_map(), cw.load()), pub)
    assert allw.index.is_unique and len(allw) == 77  # every leaf but one private group has a weight
    with pytest.raises(ValueError):
        C.public_leaf_weights(pd.DataFrame({"istat_group": ["x"], "ca_contract": ["NOPE"]}), hc)


def test_weighted_cluster_table_and_extended_weights():
    d = pd.DataFrame({"parent": ["P", "P", "Q"], "branch": ["private", "private", "public"], "staleness_months": [0.0, 10.0, 5.0], "standstill_24m": [1.0, 1.0, 2.0], "real_change": [-10.0, 0.0, -4.0], "real_change_25": [-8.0, 0.0, -2.0]},
                     index=pd.Index(["a", "b", "c"], name="group"))
    t = C.weighted_cluster_table(d, pd.Series({"a": 1.0, "b": 3.0, "c": 5.0}))
    assert t.loc["P", "real_change"] == pytest.approx(-2.5) and t.loc["P", "staleness_months"] == pytest.approx(7.5) and t.loc["P", "employees"] == 4 and t.loc["Q", "n_leaves"] == 1
    t2 = C.weighted_cluster_table(d, pd.Series({"a": 1.0, "c": 5.0}))  # a leaf without a weight is left out of the mean
    assert t2.loc["P", "real_change"] == -10.0 and "P" in t2.index
    assert "P" not in C.weighted_cluster_table(d, pd.Series({"c": 5.0})).index  # a parent without any weight is dropped
    hc = pd.DataFrame({"contract": ["AA", "BB", "CC"], "heads": [10, 20, 5]})
    m = pd.DataFrame({"istat_group": ["x"], "ca_contract": ["AA+BB"], "extra_contracts": ["CC"]})
    assert C.public_leaf_weights(m, hc)["x"] == 30 and C.public_leaf_weights(m, hc, extended=True)["x"] == 35


def test_conto_annuale_years():
    from cifresignificative.sources import conto_annuale as ca
    h21, h24 = ca.load(year=2021), ca.load(year=2024)
    assert 3.2e6 < h21.heads.sum() < 3.3e6 and 3.3e6 < h24.heads.sum() < 3.5e6 and {"SCLA", "SSNA", "RALN"} <= set(h24.contract)  # contract codes are stable between the two years
    m = ca.load_map()
    assert len(C.public_leaf_weights(m, h24, extended=True)) == 13
