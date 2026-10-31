"""Synthetic control for a country-level outcome: non-negative weights summing to one that best reproduce the treated unit before the event."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_weights(y_treated: pd.Series, donors: pd.DataFrame) -> pd.Series:
    """Weights over donor columns minimising the pre-period squared error (series must share the pre-period index)."""
    x, y = donors.values, y_treated.values
    n = x.shape[1]
    res = minimize(lambda w: np.sum((y - x @ w) ** 2), np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, options={"maxiter": 500, "ftol": 1e-12})
    return pd.Series(res.x, index=donors.columns)


def synthetic_control(panel: pd.DataFrame, treated: str, pre: tuple[int, int], donors: list[str]) -> tuple[pd.Series, pd.Series]:
    """panel: year x unit outcome table. Returns (weights, synthetic series for all years)."""
    pre_idx = panel.index[(panel.index >= pre[0]) & (panel.index <= pre[1])]
    w = fit_weights(panel.loc[pre_idx, treated], panel.loc[pre_idx, donors])
    return w, panel[donors] @ w


def rmspe(actual: pd.Series, synth: pd.Series, years: tuple[int, int]) -> float:
    idx = actual.index[(actual.index >= years[0]) & (actual.index <= years[1])]
    return float(np.sqrt(np.mean((actual.loc[idx] - synth.loc[idx]) ** 2)))


def placebo_ratios(panel: pd.DataFrame, units: list[str], pre: tuple[int, int], post: tuple[int, int]) -> pd.Series:
    """Post/pre RMSPE ratio when each unit in turn is treated and the others form the donor pool (in-space placebo)."""
    out = {}
    for u in units:
        others = [x for x in units if x != u]
        _, s = synthetic_control(panel, u, pre, others)
        out[u] = rmspe(panel[u], s, post) / max(rmspe(panel[u], s, pre), 1e-9)
    return pd.Series(out)
