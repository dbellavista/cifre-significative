"""Hours versus pay in INPS private-sector employee data, by province of work (research track R3).

The MEF income per recipient (post 01, -1.68% real, 2008-2024) puts a full-time year and a part-time or short contract on the same footing. INPS counts, per province, workers with at least one paid day in the year, their pay and their paid days, split by `part_time` (`Si` = at least one part-time relationship in the year, not only part-time). Descriptive only: paid days are not hours, the province is the workplace not the residence, and the two universes (INPS private non-agricultural employees; MEF "lavoro dipendente e assimilati" of residents) differ.

Decomposition of the mean nominal pay per worker, w = (1-s)*m_No + s*m_Si (s = share of workers with part-time in the year, m = pay per worker of the group), between years 0 and 1:
- `within_pct`: w at year-1 group pay and year-0 share versus w0: pay growth at constant share;
- `mix_pct`: w1 versus w at year-1 group pay and year-0 share: what the change in the share alone does to the mean (negative = drag);
- exact: (1 + within) * (1 + mix) = 1 + total.
"""
import numpy as np
import pandas as pd

from cifresignificative.core.sources.inps_osservatorio import REDRAWN

UNIT = "Forlì-Cesena"
ER_PROVINCES = ("Piacenza", "Parma", "Reggio Emilia", "Modena", "Bologna", "Ferrara", "Ravenna", UNIT, "Rimini")


def province_panel(d: pd.DataFrame) -> pd.DataFrame:
    """Wide table indexed by (year, province) from `inps_osservatorio.load()`: counts by part-time group and derived ratios.

    Requires No + Si == Totale for workers, pay and days (the exports agree to the unit); raises otherwise."""
    p = d[d.kind == "province"].pivot_table(index=["year", "province"], columns="part_time", values=["workers", "pay_eur", "paid_days"], aggfunc="first")
    p.columns = [f"{m}_{g.lower()}" for m, g in p.columns]
    for m in ("workers", "pay_eur", "paid_days"):
        if not np.allclose(p[f"{m}_no"] + p[f"{m}_si"], p[f"{m}_totale"]):
            raise ValueError(f"No + Si differs from Totale for {m}")
    p = p.rename(columns={"workers_totale": "workers", "pay_eur_totale": "pay_eur", "paid_days_totale": "paid_days"})
    p["share_pt_pct"] = p.workers_si / p.workers * 100
    p["days_per_worker"] = p.paid_days / p.workers
    p["pay_per_worker"] = p.pay_eur / p.workers
    p["pay_per_day"] = p.pay_eur / p.paid_days
    for g in ("no", "si"):
        p[f"days_per_worker_{g}"] = p[f"paid_days_{g}"] / p[f"workers_{g}"]
        p[f"pay_per_worker_{g}"] = p[f"pay_eur_{g}"] / p[f"workers_{g}"]
        p[f"pay_per_day_{g}"] = p[f"pay_eur_{g}"] / p[f"paid_days_{g}"]
    p["pt_pay_per_day_ratio"] = p.pay_per_day_si / p.pay_per_day_no
    return p


def comparable_provinces(panel: pd.DataFrame, years: tuple[int, int], strict: bool = True) -> list[str]:
    """Provinces present in both years; `strict` also drops those whose territory changed (`REDRAWN`)."""
    sets = [set(panel.xs(y, level="year").index) for y in years]
    keep = set.intersection(*sets)
    return sorted(keep - REDRAWN if strict else keep)


def mix_decomposition(panel: pd.DataFrame, y0: int, y1: int, provinces: list[str] | None = None) -> pd.DataFrame:
    """One row per province: changes between `y0` and `y1` (percent, share in points) and the within / mix split of the mean pay per worker."""
    a, b = panel.xs(y0, level="year"), panel.xs(y1, level="year")
    idx = provinces if provinces is not None else sorted(set(a.index) & set(b.index))
    a, b = a.loc[idx], b.loc[idx]
    s0 = a.share_pt_pct / 100
    w_cf = (1 - s0) * b.pay_per_worker_no + s0 * b.pay_per_worker_si  # year-1 group pay at the year-0 share
    out = pd.DataFrame({
        "share_pt_0": a.share_pt_pct, "share_pt_1": b.share_pt_pct, "share_pt_change_pp": b.share_pt_pct - a.share_pt_pct,
        "workers_change_pct": (b.workers / a.workers - 1) * 100,
        "days_per_worker_change_pct": (b.days_per_worker / a.days_per_worker - 1) * 100,
        "pay_per_day_change_pct": (b.pay_per_day / a.pay_per_day - 1) * 100,
        "pay_per_worker_change_pct": (b.pay_per_worker / a.pay_per_worker - 1) * 100,
        "within_pct": (w_cf / a.pay_per_worker - 1) * 100,
        "mix_pct": (b.pay_per_worker / w_cf - 1) * 100,
        "pt_pay_per_day_ratio_0": a.pt_pay_per_day_ratio, "pt_pay_per_day_ratio_1": b.pt_pay_per_day_ratio,
    })
    return out


def unit_vs_others(dec: pd.DataFrame, unit: str = UNIT, others: list[str] | None = None) -> pd.DataFrame:
    """For every column of a decomposition: the unit value, the median of the other provinces, the rank of the unit (1 = highest) and the share of others below it.

    This is the placebo ranking among peers for a single-unit claim: where the unit sits in the distribution of the provinces, not a test."""
    pool = dec.drop(index=unit, errors="ignore") if others is None else dec.loc[[p for p in others if p != unit]]
    rows = {}
    for c in dec.columns:
        v = dec.loc[unit, c]
        rows[c] = {"unit": v, "median_others": pool[c].median(), "rank": int((pool[c] > v).sum() + 1), "n_others": int(pool[c].notna().sum()), "share_others_below_pct": float((pool[c] < v).mean() * 100)}
    return pd.DataFrame(rows).T
