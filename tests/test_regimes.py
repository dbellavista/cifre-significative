"""Regime tests on synthetic within-change data: a stable pair, a pair living only in the crisis, and a shuffled null."""
import numpy as np
import pandas as pd

from cifresignificative.analysis import regimes as RG


def _D():
    rng = np.random.default_rng(11)
    idx = pd.MultiIndex.from_product([[f"C{i:02d}" for i in range(30)], range(2005, 2025)], names=["iso3", "year"])
    n = len(idx)
    D = pd.DataFrame({"a": rng.normal(size=n), "c": rng.normal(size=n)}, index=idx)
    D["b"] = D.a + rng.normal(scale=1.0, size=n)  # stable link in every regime
    crisis = (idx.get_level_values("year") >= 2008) & (idx.get_level_values("year") <= 2013)
    D["d"] = np.where(crisis, D.c + rng.normal(scale=0.3, size=n), rng.normal(size=n))  # link only in the crisis
    for i in range(10):
        D[f"n{i}"] = rng.normal(size=n)
    return D


PAIRS = pd.DataFrame({"x": ["a", "c"], "y": ["b", "d"]})


def test_stable_pair_is_not_break_driven_and_crisis_pair_is():
    r = RG.regime_table(_D(), PAIRS).set_index(["x", "y"])
    assert not r.loc[("a", "b"), "break_driven"] and r.loc[("c", "d"), "break_driven"]
    assert r.loc[("c", "d"), "rho_crisis"] > 0.8 and abs(r.loc[("c", "d"), "rho_recovery"]) < 0.3


def test_regime_null_gives_no_breaks():
    D = _D()
    pairs = pd.DataFrame([(f"n{i}", f"n{j}") for i in range(10) for j in range(i + 1, 10)], columns=["x", "y"])
    assert max(RG.regime_null(D, pairs, n_perm=2)) == 0
