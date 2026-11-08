"""Housing Affordability, Real Estate Prices/Rents & Suburbanization Gradient (`studies.cesena.analysis.housing`).

Investigates Track 4.1: The Housing Affordability & Suburbanization mechanism resolving
Cesena's demographic paradox (economic outperformance vs. young-adult retention lag in the core municipality).

Integrates:
- Agenzia delle Entrate OMI (Osservatorio del Mercato Immobiliare) residential purchase quotations (€/m²)
  and rental rates (€/m²/month) across the 13 Emilia-Romagna benchmark peers and 16 Comprensorio Cesenate municipalities.
- MEF Department of Finance municipal IRPEF mean income and municipal tax surcharges.
- ISTAT Demo population growth and net migration rates (2008–2025).

Functions:
- `compute_housing_affordability`: Computes Price-to-Income ratios, Rent burdens, and financial suburbanization incentives.
- `comprensorio_housing_gradient`: Aggregates the 17 Comprensorio Cesenate municipalities across 4 functional urban rings.
- `peer_housing_comparison`: Ranks Cesena against the 12 Emilia-Romagna peer municipalities.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cifresignificative.core.geo import BELT_GROUP_LABELS
from cifresignificative.studies.cesena.analysis.data import (
    load_belt_comuni,
    load_demography,
    load_irpef_summary,
    load_omi_municipal,
)


def compute_housing_affordability(
    omi_df: pd.DataFrame | None = None,
    irpef_df: pd.DataFrame | None = None,
    demo_df: pd.DataFrame | None = None,
    income_year: int = 2024,
) -> pd.DataFrame:
    """Merge OMI residential prices and rents with MEF declared incomes and demographic balances."""
    if omi_df is None:
        omi_df = load_omi_municipal()
    if irpef_df is None:
        irpef_df = load_irpef_summary()
    if demo_df is None:
        demo_df = load_demography()

    # Combine irpef_summary (13 ER peers) with belt_comuni (17 Comprensorio municipalities)
    irp_sub = irpef_df[irpef_df["year"] == income_year].copy()
    if irp_sub.empty:
        irp_sub = irpef_df[irpef_df["year"] == irpef_df["year"].max()].copy()
    b_df = load_belt_comuni()
    b_sub = b_df[b_df["year"] == income_year].copy()
    if b_sub.empty:
        b_sub = b_df[b_df["year"] == b_df["year"].max()].copy()
    if "mean_income_eur" not in b_sub.columns and "mean_income_real_eur" in b_sub.columns:
        b_sub["mean_income_eur"] = b_sub["mean_income_real_eur"]
    if "addizionale_comunale_eur" not in b_sub.columns and "addizionale_comunale_real_eur" in b_sub.columns:
        b_sub["addizionale_comunale_eur"] = b_sub["addizionale_comunale_real_eur"]
    cols = ["comune_id", "mean_income_eur", "mean_income_real_eur", "taxpayers_total", "addizionale_comunale_eur"]
    inc_combined = pd.concat([
        irp_sub[[c for c in cols if c in irp_sub.columns]],
        b_sub[[c for c in cols if c in b_sub.columns]],
    ]).drop_duplicates("comune_id")

    # Extract demographic metrics (2008-2025 growth and recent net migration)
    d_comb = pd.concat([
        demo_df[["comune_id", "year", "pop_total", "net_migr_rate_per_1k"]],
        b_df[["comune_id", "year", "pop_total", "net_migr_rate_per_1k"]],
    ]).drop_duplicates(["comune_id", "year"])
    d_wide = d_comb.pivot(index="comune_id", columns="year", values=["pop_total", "net_migr_rate_per_1k"])
    pop_08 = d_wide[("pop_total", 2008)] if ("pop_total", 2008) in d_wide.columns else pd.Series(dtype=float)
    pop_25 = d_wide[("pop_total", 2025)] if ("pop_total", 2025) in d_wide.columns else pd.Series(dtype=float)
    migr_recent = d_wide[("net_migr_rate_per_1k", 2024)] if ("net_migr_rate_per_1k", 2024) in d_wide.columns else pd.Series(dtype=float)

    merged = omi_df.merge(inc_combined, on="comune_id", how="left")

    # Attach demographic indicators
    merged["pop_2008"] = merged["comune_id"].map(pop_08)
    merged["pop_2025"] = merged["comune_id"].map(pop_25)
    merged["pop_growth_08_25_pct"] = (merged["pop_2025"] / merged["pop_2008"] - 1.0) * 100.0
    merged["net_migr_rate_2024"] = merged["comune_id"].map(migr_recent)

    # Calculate municipal IRPEF surcharge per taxpayer
    merged["surcharge_per_taxpayer_eur"] = (
        merged["addizionale_comunale_eur"] / merged["taxpayers_total"].replace(0, np.nan)
    )

    # Housing Affordability Metrics:
    # 1. Price-to-Income Ratio: Years of gross mean income to buy a 100 m² apartment
    merged["price_to_income_ratio_100m2"] = (
        (merged["compr_mid_eur_m2"] * 100.0) / merged["mean_income_eur"]
    )

    # 2. Rent Burden: % of annual gross mean income to rent an 80 m² apartment
    merged["annual_rent_80m2_eur"] = merged["loc_mid_eur_m2_mo"] * 80.0 * 12.0
    merged["rent_burden_pct_80m2"] = (
        merged["annual_rent_80m2_eur"] / merged["mean_income_eur"] * 100.0
    )

    # 3. Savings from living in commuter belt / peer vs Cesena core
    ces_row = merged[merged["is_cesena"]].iloc[0]
    ces_rent_80m2 = ces_row["annual_rent_80m2_eur"]
    ces_price_100m2 = ces_row["compr_mid_eur_m2"] * 100.0
    ces_surcharge = ces_row["surcharge_per_taxpayer_eur"]

    merged["annual_rent_savings_vs_cesena_eur"] = ces_rent_80m2 - merged["annual_rent_80m2_eur"]
    merged["purchase_savings_100m2_vs_cesena_eur"] = ces_price_100m2 - (merged["compr_mid_eur_m2"] * 100.0)
    merged["surcharge_diff_vs_cesena_eur"] = merged["surcharge_per_taxpayer_eur"] - ces_surcharge

    # Net annual rental incentive = annual rent savings - extra municipal surcharge
    merged["net_annual_suburban_incentive_eur"] = (
        merged["annual_rent_savings_vs_cesena_eur"] - merged["surcharge_diff_vs_cesena_eur"].fillna(0)
    )

    return merged.sort_values(["belt_group", "comune_name"]).reset_index(drop=True)


def comprensorio_housing_gradient(affordability_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Group the 17 Comprensorio Cesenate municipalities across the 4 functional rings."""
    if affordability_df is None:
        affordability_df = compute_housing_affordability()

    # Filter to Comprensorio Cesenate municipalities
    comp = affordability_df[affordability_df["belt_group"] != "er_peer"].copy()

    ring_order = ["cesena_core", "rubicone_mare", "via_emilia_west", "unione_valle_savio"]
    records: list[dict[str, Any]] = []

    for ring in ring_order:
        sub = comp[comp["belt_group"] == ring]
        if sub.empty:
            continue
        records.append({
            "belt_group": ring,
            "belt_label": BELT_GROUP_LABELS.get(ring, ring),
            "n_municipalities": len(sub),
            "comuni_list": ", ".join(sub["comune_name"]),
            "compr_mid_median_eur_m2": float(sub["compr_mid_eur_m2"].median()),
            "loc_mid_median_eur_mo": float(sub["loc_mid_eur_m2_mo"].median()),
            "price_diff_vs_cesena_pct_median": float(sub["diff_vs_cesena_pct"].median()),
            "rent_diff_vs_cesena_pct_median": float(sub["diff_rent_vs_cesena_pct"].median()),
            "price_to_income_ratio_median": float(sub["price_to_income_ratio_100m2"].median()),
            "rent_burden_pct_median": float(sub["rent_burden_pct_80m2"].median()),
            "annual_rent_savings_80m2_median_eur": float(sub["annual_rent_savings_vs_cesena_eur"].median()),
            "purchase_savings_100m2_median_eur": float(sub["purchase_savings_100m2_vs_cesena_eur"].median()),
            "mean_income_median_eur": float(sub["mean_income_eur"].median()),
            "pop_growth_08_25_pct_median": float(sub["pop_growth_08_25_pct"].median()),
            "net_migr_rate_2024_median": float(sub["net_migr_rate_2024"].median()),
        })

    return pd.DataFrame(records)


def peer_housing_comparison(affordability_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Benchmark Cesena against the 12 Emilia-Romagna peer municipalities."""
    if affordability_df is None:
        affordability_df = compute_housing_affordability()

    peers = affordability_df[affordability_df["is_er_peer"] | affordability_df["is_cesena"]].copy()

    # Compute rankings across the 13 ER benchmark cities (1 = highest / most expensive)
    peers["rank_price_desc"] = peers["compr_mid_eur_m2"].rank(ascending=False, method="min").astype(int)
    peers["rank_rent_desc"] = peers["loc_mid_eur_m2_mo"].rank(ascending=False, method="min").astype(int)
    peers["rank_affordability_burden_desc"] = peers["price_to_income_ratio_100m2"].rank(ascending=False, method="min").astype(int)
    peers["rank_gross_yield_desc"] = peers["gross_rental_yield_pct"].rank(ascending=False, method="min").astype(int)

    # Core peer median (excluding Cesena)
    core_ex = peers[peers["is_core_peer"] & ~peers["is_cesena"]]
    all_ex = peers[~peers["is_cesena"]]

    ces = peers[peers["is_cesena"]].iloc[0]

    out_metrics = [
        {
            "metric": "compr_mid_eur_m2",
            "metric_label": "Residential Purchase Price (€/m²)",
            "cesena_value": float(ces["compr_mid_eur_m2"]),
            "core_peer_median": float(core_ex["compr_mid_eur_m2"].median()),
            "all_er_median": float(all_ex["compr_mid_eur_m2"].median()),
            "cesena_rank_1_to_13": int(ces["rank_price_desc"]),
        },
        {
            "metric": "loc_mid_eur_m2_mo",
            "metric_label": "Residential Monthly Rent (€/m²/month)",
            "cesena_value": float(ces["loc_mid_eur_m2_mo"]),
            "core_peer_median": float(core_ex["loc_mid_eur_m2_mo"].median()),
            "all_er_median": float(all_ex["loc_mid_eur_m2_mo"].median()),
            "cesena_rank_1_to_13": int(ces["rank_rent_desc"]),
        },
        {
            "metric": "price_to_income_ratio_100m2",
            "metric_label": "Price-to-Income Ratio (100 m² apartment / Mean Income, years)",
            "cesena_value": float(ces["price_to_income_ratio_100m2"]),
            "core_peer_median": float(core_ex["price_to_income_ratio_100m2"].median()),
            "all_er_median": float(all_ex["price_to_income_ratio_100m2"].median()),
            "cesena_rank_1_to_13": int(ces["rank_affordability_burden_desc"]),
        },
        {
            "metric": "gross_rental_yield_pct",
            "metric_label": "Gross Rental Yield (%)",
            "cesena_value": float(ces["gross_rental_yield_pct"]),
            "core_peer_median": float(core_ex["gross_rental_yield_pct"].median()),
            "all_er_median": float(all_ex["gross_rental_yield_pct"].median()),
            "cesena_rank_1_to_13": int(ces["rank_gross_yield_desc"]),
        },
    ]

    res_df = pd.DataFrame(out_metrics)
    res_df["gap_vs_core"] = res_df["cesena_value"] - res_df["core_peer_median"]
    res_df["gap_vs_all"] = res_df["cesena_value"] - res_df["all_er_median"]
    return res_df
