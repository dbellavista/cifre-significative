"""Dating the Superbonus policy changes of the winter 2022-23 against the ENEA monthly reports (`data/external/superbonus_policy_dates.csv`, each date from two independent sources or labelled `partly verified`).

ENEA counts the investment when the works are transmitted (by month end, cumulative), so a rule that governs the start of works (the CILA of 25 November 2022, the end of transfer and invoice discount on 17 February 2023) can only show up in these flows with a lag of months: the timing can rule an explanation out (a change that starts before the rule cannot be caused by it) but cannot date a rule's effect. Two events are pre-specified (the first full month after each: December 2022 and March 2023),
two measures each (the income gradient of the month's admitted investment per inhabitant against GDP per inhabitant, Spearman over the 20 regions, and the national monthly flow (log ratio of the two means)): contrast D(c) = mean of the six months from c minus the mean of the five months before c. The placebo ranks |D| at the event month among every month from January 2022 to March 2026 (the data allow five before and six after); months are serially dependent and other rule changes sit among them, so the rank is a rough yardstick, not a p-value in the strict sense.
Descriptive, never causal. Building-type flows (condominiums, single-family houses, independent units) show what the gradient is made of.
"""
import numpy as np
import pandas as pd
from scipy import stats

from cifresignificative.analysis import superbonus_monthly as SBM

CSV = "data/external/superbonus_policy_dates.csv"
EVENTS = {"CILA cut-off, 25 Nov 2022": "2022-12-31", "end of transfer / invoice discount, 17 Feb 2023": "2023-03-31"}  # month end of the first full month after the event
KINDS = {"all": "inv_admitted", "condominiums": "condo_admitted", "single-family": "single_admitted", "independent units": "ui_admitted"}
BEFORE, AFTER = 5, 6


def load_dates() -> pd.DataFrame:
    return pd.read_csv(CSV, parse_dates=["date"])


def monthly_path(em: pd.DataFrame, regional: pd.DataFrame) -> pd.DataFrame:
    """Month x kind: national flow (EUR million) and the income gradient (Spearman of the per-inhabitant flow of the 20 regions against GDP per inhabitant); the first month end has no flow."""
    att = SBM.attributes(em, regional)
    rows = {}
    for kind, field in KINDS.items():
        fl = SBM.cumulative(em, field).diff().dropna(how="all")
        pi = fl.div(att.population, axis=1)
        rows[(kind, "flow_meur")] = fl.sum(axis=1) / 1e6
        rows[(kind, "rho")] = pi.apply(lambda r: stats.spearmanr(r, att.gdp_per_inhabitant_eur, nan_policy="omit")[0], axis=1)
    return pd.DataFrame(rows)


def contrast(s: pd.Series, cut, log: bool = False) -> float:
    """Mean of the `AFTER` months from `cut` minus the mean of the `BEFORE` months before it (NaN when the series does not reach); with `log`, the log ratio of the two means (a flow: single negative months of ENEA revisions do not matter)."""
    i = s.index.get_loc(cut)
    if i < BEFORE or i + AFTER > len(s):
        return np.nan
    a, b = s.iloc[i:i + AFTER].mean(skipna=False), s.iloc[i - BEFORE:i].mean(skipna=False)
    return float(np.log(a / b) if log and a > 0 and b > 0 else (np.nan if log else a - b))


def event_tests(path: pd.DataFrame, first: str = "2022-01-31", last: str = "2026-03-31") -> pd.DataFrame:
    """Pre-specified family: 2 events x 2 measures (gradient of all admitted investment, log national flow). Rank of |D| at the event month among the months `first`..`last` that have the full window; BH over the four."""
    series = {"gradient (rho)": path[("all", "rho")], "flow (log ratio of means)": path[("all", "flow_meur")]}
    rows = []
    for ev, me in EVENTS.items():
        for m, s in series.items():
            lg = m.startswith("flow")
            allc = pd.Series({c: contrast(s, c, lg) for c in s.loc[first:last].index}).dropna()
            d = contrast(s, pd.Timestamp(me), lg)
            rows.append({"event": ev, "measure": m, "cut_month_end": me, "D": d, "n_cuts": len(allc), "rank": int((allc.abs() >= abs(d) - 1e-12).sum()), "p_rank": float((allc.abs() >= abs(d) - 1e-12).sum() / len(allc)), "max_abs_D": float(allc.abs().max()), "cut_of_max": str(allc.abs().idxmax())[:10]})
    out = pd.DataFrame(rows)
    out["q"] = SBM._bh(out.p_rank.to_numpy())
    return out


def onset(path: pd.DataFrame, threshold: float = 0.5, run: int = 2) -> dict:
    """First month end from which the gradient stays at or above `threshold` for `run` months in a row (all admitted investment), with the national flow of the months around it and the first month of the single-family flow below half of its June-August 2022 mean."""
    r = path[("all", "rho")].loc["2022-01-31":]
    ok = (r >= threshold).rolling(run).sum().eq(run)
    first = ok[ok].index[0] - pd.offsets.MonthEnd(run - 1) if ok.any() else None
    sf = path[("single-family", "flow_meur")]
    base = sf.loc["2022-06-30":"2022-08-31"].mean()
    after = sf.loc["2022-10-31":]
    drop = after[after < base / 2]
    return {"gradient_onset": str(first.date()) if first is not None else None, "single_family_flow_base_meur": float(base), "single_family_halved": str(drop.index[0].date()) if len(drop) else None}


def type_gradients(path: pd.DataFrame, start: str = "2022-07-31", end: str = "2023-06-30") -> pd.DataFrame:
    """Month x kind table of the gradient over the window, to see which building type carries it."""
    return pd.DataFrame({k: path[(k, "rho")] for k in KINDS}).loc[start:end]


def building_counts(em: pd.DataFrame, first: str = "2022-01-31", last: str = "2026-03-31") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exploratory proxy for the starts of works (ENEA publishes no project start dates): the number of buildings that enter the monthly report, by type (the first asseveration of a building, so closer to the start than the euro flows, which follow the progress of the works). Returns the monthly flows of new buildings (all, condominiums, single-family, independent units) and, for each type and event, the log-ratio contrast (six months from the cut against five before) with its rank among every cut month. Not part of the pre-specified family: no BH, read as description."""
    r = SBM.regions_only(em).groupby("date")[["n_buildings", "condo_n", "single_n", "ui_n"]].sum().diff().dropna()
    r.columns = ["all", "condominiums", "single-family", "independent units"]
    rows = []
    for ev, me in EVENTS.items():
        for kind in r.columns:
            s = r[kind]
            allc = pd.Series({c: contrast(s, c, True) for c in s.loc[first:last].index}).dropna()
            d = contrast(s, pd.Timestamp(me), True)
            rows.append({"event": ev, "type": kind, "D_log_ratio": d, "n_cuts": len(allc), "rank": int((allc.abs() >= abs(d) - 1e-12).sum()), "p_rank": float((allc.abs() >= abs(d) - 1e-12).mean())})
    return r, pd.DataFrame(rows)
