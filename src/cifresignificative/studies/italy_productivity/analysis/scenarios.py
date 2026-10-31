"""Debt scenarios: simple transparent debt-dynamics simulation (not a forecast).

d_t = d_{t-1} * (1 + r_t) / (1 + g_t) - pb_t + drift_t, all in % of GDP, with r the implicit interest rate, g nominal GDP growth, pb the primary
balance and drift an extra deficit-increasing term (e.g. ageing costs).
"""
import numpy as np
import pandas as pd


def simulate_debt(d0: float, years: range, pb: np.ndarray | float, r: np.ndarray | float, g: np.ndarray | float, drift: np.ndarray | float = 0.0) -> pd.Series:
    n = len(years)
    pb, r, g, drift = (np.broadcast_to(np.asarray(x, dtype=float), n) for x in (pb, r, g, drift))
    d = np.empty(n + 1)
    d[0] = d0
    for t in range(n):
        d[t + 1] = d[t] * (1 + r[t]) / (1 + g[t]) - pb[t] + drift[t]
    return pd.Series(d[1:], index=list(years))


def stabilising_primary_balance(d: float, r: float, g: float) -> float:
    """Primary balance (% GDP) that keeps the debt ratio constant: pb* = d * (r - g) / (1 + g)."""
    return d * (r - g) / (1 + g)


def scenario_inputs(panel: pd.DataFrame, ageing: pd.DataFrame, iso3: str = "ITA", first: int = 2025, last: int = 2050, rate_convergence_years: int = 8) -> dict:
    """Starting point and paths for `simulate_debt`, from Eurostat 2024 data and the 2024 Ageing Report assumptions.

    Returns d0 (debt %GDP 2024), pb0 (primary balance 2024), r path (implicit rate converging to the Commission's nominal interest
    assumption), g path (nominal GDP growth = potential real growth x HICP) and drift path (ageing cost change vs 2024, % GDP).
    """
    from cifresignificative.studies.italy_productivity.analysis.data import (  # local import: data imports nothing from here
        ageing_series,
        wide,
    )

    bal, intr, debt = (wide(panel, k)[iso3] for k in ("gov_balance_pct_gdp", "gov_interest_pct_gdp", "gov_debt_pct_gdp"))
    d0, pb0, r0 = float(debt.loc[2024]), float((bal + intr).loc[2024]), float(intr.loc[2024] / debt.loc[2023])
    real = ageing_series(ageing, "Macroeconomic assumptions", "Potential GDP (growth rate)")[iso3] / 100
    infl = ageing_series(ageing, "Macroeconomic assumptions", "HICP (growth rate)")[iso3] / 100
    r_ar = ageing_series(ageing, "Macroeconomic assumptions", "Nominal interest rate")[iso3] / 100
    cost = ageing_series(ageing, "Total spending as % of GDP", "Baseline")[iso3]
    years = range(first, last + 1)
    return {"years": years, "d0": d0, "pb0": pb0, "r0": r0,
            "g": np.array([(1 + real.loc[y]) * (1 + infl.loc[y]) - 1 for y in years]),
            "r": np.array([r0 + (r_ar.loc[y] - r0) * min((y - 2024) / rate_convergence_years, 1) for y in years]),
            "drift": np.array([cost.loc[y] - cost.loc[2024] for y in years])}
