"""Sensitivity tables fixed by the 2026-10-30 review (`studies.cesena.analysis.robustness`), so that a headline is never read without its composition, period or counter-evidence.

- `income_composition`: mean and median income next to taxpayers, aggregate income and income per resident, by period. A mean can rise because low earners stopped arriving, so growth in the mean alone is not growth in the local income base.
- `population_balance`: population change split into natural balance, internal and international migration and registry adjustment, by area and period.
- `cohort_by_period`: 25-44 cohort change by sub-period. It contains registry revisions as well as moves (`STAT_ADJUST` reached -889 in Cesena in 2020), so it is not pure migration.
- `housing_gradient_test`: does cheaper housing go with more residents? Cross-sections only: a hypothesis check, not a test of a mechanism.
"""
import numpy as np
import pandas as pd
from scipy import stats

from cifresignificative.core.geo import CESENA, COMUNI_BY_ID

INCOME_PERIODS: tuple[tuple[int, int], ...] = ((2008, 2024), (2008, 2018), (2018, 2024))
BALANCE_PERIODS: tuple[tuple[int, int], ...] = ((2008, 2024), (2008, 2011), (2012, 2018), (2019, 2024))
COHORT_PERIODS: tuple[tuple[int, int], ...] = ((2012, 2024), (2012, 2016), (2017, 2024), (2017, 2019), (2022, 2024))


def _rank_row(ser: pd.Series, focal_id: str, **keys: object) -> dict[str, object]:
    ser = ser.dropna()
    return {
        **keys,
        "cesena": float(ser.loc[focal_id]),
        "rank": int((ser > ser.loc[focal_id]).sum() + 1),
        "n": len(ser),
        "peer_median": float(ser.drop(index=focal_id).median()),
    }


def income_composition(panel_df: pd.DataFrame, focal_id: str = CESENA) -> pd.DataFrame:
    """Growth (%) and rank (1 = highest) of `focal_id` by period on mean, median, taxpayers, aggregate income and income per resident."""
    measures = {
        "mean_income": "mean_income_real_eur",
        "median_income": "median_income_real_eur",
        "taxpayers": "taxpayers_pos",
        "aggregate_income": "income_pos_real_eur",
        "income_per_resident": "irpef_taxbase_real_pc",
    }
    rows: list[dict[str, object]] = []
    for name, col in measures.items():
        wide = panel_df.pivot(index="year", columns="comune_id", values=col)
        for t0, t1 in INCOME_PERIODS:
            rows.append(_rank_row((wide.loc[t1] / wide.loc[t0] - 1.0) * 100.0, focal_id, measure=name, t0=t0, t1=t1))
    return pd.DataFrame(rows)


def population_balance(belt_df: pd.DataFrame, focal_id: str = CESENA) -> pd.DataFrame:
    """Sum of the balance components per area and period, also per 1,000 residents at the start of the period; `pop_change` is January of `t1 + 1` minus January of `t0`."""
    comps = ["natural_balance", "net_migration_internal", "net_migration_international", "registry_adjustment"]
    areas = {
        "cesena": belt_df[belt_df["comune_id"] == focal_id],
        "belt_16": belt_df[belt_df["comune_id"] != focal_id],
        **{str(g): sub for g, sub in belt_df[belt_df["comune_id"] != focal_id].groupby("belt_group")},
    }
    rows: list[dict[str, object]] = []
    for area, sub in areas.items():
        pops = sub.groupby("year")["pop_total"].sum()
        for t0, t1 in BALANCE_PERIODS:
            flows = sub[sub["year"].between(t0, t1)][comps].sum()
            start = float(pops.loc[t0])
            row: dict[str, object] = {"area": area, "t0": t0, "t1": t1, "pop_start": start, "pop_change": float(pops.loc[t1 + 1]) - start}
            for c in comps:
                row[c] = float(flows[c])
                row[f"{c}_per_1k"] = float(flows[c]) / start * 1000.0
            row["pop_change_per_1k"] = float(row["pop_change"]) / start * 1000.0
            rows.append(row)
    return pd.DataFrame(rows)


def cohort_by_period(demo_df: pd.DataFrame, focal_id: str = CESENA) -> pd.DataFrame:
    """Mean yearly 25-44 cohort change per 1,000 cohort members, with rank (1 = highest), by sub-period."""
    wide = demo_df.pivot(index="year", columns="comune_id", values="net_cohort_25_44_per_1k")
    return pd.DataFrame([_rank_row(wide.loc[t0:t1].mean(), focal_id, t0=t0, t1=t1) for t0, t1 in COHORT_PERIODS])


def housing_gradient_test(omi_df: pd.DataFrame, belt_df: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of the OMI residential price with population growth (16 belt towns) and with 25-44 cohort change (13 benchmark cities).

    The suburbanisation story predicts a negative belt correlation (cheaper towns gain more) and a negative one across cities (dearer cities retain fewer 25-44).
    """
    prices = omi_df.set_index("comune_id")["compr_mid_eur_m2"]
    belt_only = belt_df[belt_df["comune_id"] != CESENA]
    pops = belt_only.pivot_table(index="comune_id", columns="year", values="pop_total")
    rows: list[dict[str, object]] = []
    for t0, t1 in ((2008, 2025), (2012, 2024), (2019, 2025)):
        growth = (pops[t1] / pops[t0] - 1.0) * 100.0
        rho, p = stats.spearmanr(prices.reindex(growth.index), growth)
        rows.append({"sample": "belt_16", "x": "price", "y": f"pop_growth_{t0}_{t1}", "n": len(growth), "spearman": float(rho), "p": float(p)})
    city_ids = [c for c in prices.index if c in COMUNI_BY_ID]
    wide = demo_df.pivot(index="year", columns="comune_id", values="net_cohort_25_44_per_1k")
    for t0, t1 in ((2012, 2024), (2017, 2024)):
        cohort = wide.loc[t0:t1].mean().reindex(city_ids)
        rho, p = stats.spearmanr(prices.reindex(city_ids), cohort)
        rows.append({"sample": "cities_13", "x": "price", "y": f"cohort_25_44_{t0}_{t1}", "n": len(city_ids), "spearman": float(rho), "p": float(p)})
    return pd.DataFrame(rows).replace({np.nan: None})
