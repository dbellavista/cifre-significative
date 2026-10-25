"""Regime / break tests for the national year-on-year co-movements (E3): is a pair's correlation stable across the macro regimes, or carried by one?

The panel starts in 2005, so the euro start (1999) cannot be tested; the regimes are pre-crisis, the 2008-13 crisis (incl. the sovereign episode),
recovery and the 2020 shock with its aftermath. Descriptive: a break flags where an association lives, it explains nothing.
"""
import numpy as np
import pandas as pd
from scipy import stats

from cifresignificative.analysis import corr

REGIMES = {"pre_crisis": (2005, 2007), "crisis": (2008, 2013), "recovery": (2014, 2019), "covid": (2020, 2024)}


def _pearson(d: pd.DataFrame, x: str, y: str) -> tuple[float, int]:
    d = d[[x, y]].dropna()
    return (corr._corr(d[x].to_numpy(), d[y].to_numpy(), "pearson")[0], len(d)) if len(d) >= 3 else (np.nan, len(d))


def regime_table(D: pd.DataFrame, pairs: pd.DataFrame, regimes: dict[str, tuple[int, int]] = REGIMES, n_min: int = 40, strength: float = 0.5, q: float = 0.05) -> pd.DataFrame:
    """D = `corr.within_changes` table; pairs has columns x, y. Per pair: pooled Pearson rho in each regime (`rho_<regime>`, needs `n_min` rows),
    `rho_full`, Cochran Q test that the regime correlations are equal (Fisher z, weights n-3; BH over pairs; rows are demeaned panel
    observations, so the n are approximate), `loo_min_abs` / `loo_sign_stable` leaving each regime out in turn, and `break_driven` =
    significant Q (q_value <= q) and a leave-one-regime-out rho falls below `strength` x |rho_full| or flips sign."""
    yr = D.index.get_level_values("year")
    parts = {k: D[(yr >= a) & (yr <= b)] for k, (a, b) in regimes.items()}
    rows = []
    for x, y in zip(pairs.x, pairs.y):
        full, n_full = _pearson(D, x, y)
        rho, n = {}, {}
        for k, P in parts.items():
            rho[k], n[k] = _pearson(P, x, y)
        ok = [k for k in parts if n[k] >= n_min and np.isfinite(rho[k]) and abs(rho[k]) < 1]
        if len(ok) < 2 or not np.isfinite(full):
            continue
        z, w = np.array([np.arctanh(rho[k]) for k in ok]), np.array([n[k] - 3 for k in ok], dtype=float)
        Q = float((w * (z - (w * z).sum() / w.sum()) ** 2).sum())
        loo = np.array([_pearson(pd.concat([P for k2, P in parts.items() if k2 != k]), x, y)[0] for k in ok])
        rows.append({"x": x, "y": y, "n": n_full, "rho_full": full, **{f"rho_{k}": rho[k] for k in parts}, "n_regimes": len(ok),
                     "Q": Q, "p": float(stats.chi2.sf(Q, len(ok) - 1)), "loo_min_abs": float(np.nanmin(np.abs(loo))),
                     "loo_sign_stable": bool(np.all(np.sign(loo) == np.sign(full)))})
    out = corr.fdr(pd.DataFrame(rows), q)
    if out.empty:
        return out.assign(break_driven=pd.Series(dtype=bool))
    weak = (out.loo_min_abs < strength * out.rho_full.abs()) | ~out.loo_sign_stable
    return out.assign(break_driven=out.significant & weak).drop(columns="significant").sort_values("p").reset_index(drop=True)


def regime_null(D: pd.DataFrame, pairs: pd.DataFrame, n_perm: int = 3, seed: int = 0, **kw) -> list[int]:
    """Number of `break_driven` pairs after shuffling each column of D across all rows (destroys both the association and any regime structure); should be ~0."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        S = D.apply(lambda c: pd.Series(rng.permutation(c.to_numpy()), index=c.index))
        out.append(int(regime_table(S, pairs, **kw).break_driven.sum()))
    return out
