"""Agricultural land specialization (ATECO A) and provincial export analysis (Coeweb / ISTAT).

Empirically investigates the territorial foundation and trade performance of the Cesena economy:
1. Provincial foreign trade comparison (2025): Export volume, trade surplus, coverage ratio,
   and per-capita metrics across the 9 Emilia-Romagna provinces.
2. Agricultural land specialization (7th Agricultural Census 2020): Fruit orchards ('Frutta'),
   vineyards ('Vite'), arable crops, and farm size fragmentation (<10 ha vs >=100 ha).
3. Agro-industrial cluster vertical integration: Synthesizing the continuum between primary
   production (orchards, smallholder fragmentation), industrial food processing (ATECO 10 LQ = 4.44),
   packaging machinery (ATECO 28 LQ = 1.07), and global trade surplus.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cifresignificative.core.geo import ER_PROVINCES
from cifresignificative.studies.cesena.analysis.data import (
    load_agri_land_use,
    load_agri_size_classes,
    load_coeweb_commodities,
    load_coeweb_provinces,
)


@dataclass(frozen=True)
class ProvincialExportSummary:
    provinces_df: pd.DataFrame
    forli_cesena_export_eur: float
    forli_cesena_balance_eur: float
    forli_cesena_coverage_pct: float
    forli_cesena_rank_export: int
    forli_cesena_rank_balance: int
    forli_cesena_export_per_capita: float


@dataclass(frozen=True)
class AgriculturalSpecializationSummary:
    land_specialization_df: pd.DataFrame
    forli_cesena_frutta_ha: float
    forli_cesena_frutta_share_sau_pct: float
    forli_cesena_frutta_share_er_pct: float
    forli_cesena_frutta_rank_er: int
    romagna_frutta_share_er_pct: float
    forli_cesena_under10_ha: float
    forli_cesena_under10_share_sau_pct: float


@dataclass(frozen=True)
class ClusterIntegrationSummary:
    export_summary: ProvincialExportSummary
    agricultural_summary: AgriculturalSpecializationSummary
    agrifood_commodities: pd.DataFrame
    frutta_ha: float
    frutta_er_share_pct: float
    smallholder_under10_ha: float
    smallholder_under10_share_pct: float
    provincial_export_eur: float
    provincial_surplus_eur: float
    coverage_ratio_pct: float


def provincial_export_comparison(processed_dir: Path | None = None) -> ProvincialExportSummary:
    """Compare foreign trade performance across all 9 Emilia-Romagna provinces (2025 Coeweb)."""
    df = load_coeweb_provinces(processed_dir)
    fc_row = df[df["provincia"] == "Forlì-Cesena"].iloc[0]

    return ProvincialExportSummary(
        provinces_df=df,
        forli_cesena_export_eur=float(fc_row["export_eur"]),
        forli_cesena_balance_eur=float(fc_row["trade_balance_eur"]),
        forli_cesena_coverage_pct=float(fc_row["coverage_ratio_pct"]),
        forli_cesena_rank_export=int(fc_row["rank_export"]),
        forli_cesena_rank_balance=int(fc_row["rank_balance"]),
        forli_cesena_export_per_capita=float(fc_row["export_per_capita_eur"]),
    )


def agricultural_land_specialization(processed_dir: Path | None = None) -> AgriculturalSpecializationSummary:
    """Analyze agricultural land use, fruit specialization, and farm fragmentation across ER provinces."""
    df_sat = load_agri_land_use(processed_dir).set_index("category")
    df_sau = load_agri_size_classes(processed_dir).set_index("size_class")

    # Extract acreage series
    sau = df_sat.loc["Superficie agricola utilizzata"]
    frutta = df_sat.loc["Frutta"]
    vite = df_sat.loc["Vite"]
    legnose = df_sat.loc["Legnose agrarie"]
    seminativi = df_sat.loc["Seminativi"]

    under10 = df_sau.loc["< 10 ha"]
    over100 = df_sau.loc[">= 100 ha"]
    sau_tot = df_sau.loc["Totale"]

    # Filter to the 9 ER provinces and total
    provinces = list(ER_PROVINCES.keys()) + ["Emilia-Romagna"]

    rows = []
    for p in provinces:
        p_sau = float(sau[p])
        p_frutta = float(frutta[p])
        p_vite = float(vite[p])
        p_legnose = float(legnose[p])
        p_seminativi = float(seminativi[p])
        p_under10 = float(under10[p])
        p_over100 = float(over100[p])
        p_sau_tot = float(sau_tot[p])

        er_frutta = float(frutta["Emilia-Romagna"])
        er_vite = float(vite["Emilia-Romagna"])

        rows.append({
            "provincia": p,
            "sau_ha": p_sau,
            "frutta_ha": p_frutta,
            "frutta_share_sau_pct": round(p_frutta / p_sau * 100, 2),
            "frutta_share_er_pct": round(p_frutta / er_frutta * 100, 2),
            "vite_ha": p_vite,
            "vite_share_sau_pct": round(p_vite / p_sau * 100, 2),
            "vite_share_er_pct": round(p_vite / er_vite * 100, 2),
            "legnose_ha": p_legnose,
            "legnose_share_sau_pct": round(p_legnose / p_sau * 100, 2),
            "seminativi_ha": p_seminativi,
            "seminativi_share_sau_pct": round(p_seminativi / p_sau * 100, 2),
            "under10_ha": p_under10,
            "under10_share_sau_pct": round(p_under10 / p_sau_tot * 100, 2),
            "over100_ha": p_over100,
            "over100_share_sau_pct": round(p_over100 / p_sau_tot * 100, 2),
        })

    spec_df = pd.DataFrame(rows)

    # Prov-only rank for fruit acreage
    prov_mask = spec_df["provincia"] != "Emilia-Romagna"
    spec_df.loc[prov_mask, "rank_frutta_ha"] = spec_df.loc[prov_mask, "frutta_ha"].rank(ascending=False).astype(int)

    fc_row = spec_df[spec_df["provincia"] == "Forlì-Cesena"].iloc[0]
    ra_row = spec_df[spec_df["provincia"] == "Ravenna"].iloc[0]
    romagna_frutta = float(fc_row["frutta_ha"]) + float(ra_row["frutta_ha"])
    er_frutta_tot = float(spec_df[spec_df["provincia"] == "Emilia-Romagna"]["frutta_ha"].iloc[0])
    romagna_pct = round(romagna_frutta / er_frutta_tot * 100, 2)

    return AgriculturalSpecializationSummary(
        land_specialization_df=spec_df,
        forli_cesena_frutta_ha=float(fc_row["frutta_ha"]),
        forli_cesena_frutta_share_sau_pct=float(fc_row["frutta_share_sau_pct"]),
        forli_cesena_frutta_share_er_pct=float(fc_row["frutta_share_er_pct"]),
        forli_cesena_frutta_rank_er=int(fc_row["rank_frutta_ha"]),
        romagna_frutta_share_er_pct=romagna_pct,
        forli_cesena_under10_ha=float(fc_row["under10_ha"]),
        forli_cesena_under10_share_sau_pct=float(fc_row["under10_share_sau_pct"]),
    )


def agroindustrial_cluster_integration(processed_dir: Path | None = None) -> ClusterIntegrationSummary:
    """Synthesizes the vertical integration architecture of the Cesena/Romagna agrifood cluster."""
    export_sum = provincial_export_comparison(processed_dir)
    agri_sum = agricultural_land_specialization(processed_dir)
    commodities = load_coeweb_commodities(processed_dir)

    agrifood_comms = commodities[commodities["is_agrifood"]].copy()

    return ClusterIntegrationSummary(
        export_summary=export_sum,
        agricultural_summary=agri_sum,
        agrifood_commodities=agrifood_comms,
        frutta_ha=agri_sum.forli_cesena_frutta_ha,
        frutta_er_share_pct=agri_sum.forli_cesena_frutta_share_er_pct,
        smallholder_under10_ha=agri_sum.forli_cesena_under10_ha,
        smallholder_under10_share_pct=agri_sum.forli_cesena_under10_share_sau_pct,
        provincial_export_eur=export_sum.forli_cesena_export_eur,
        provincial_surplus_eur=export_sum.forli_cesena_balance_eur,
        coverage_ratio_pct=export_sum.forli_cesena_coverage_pct,
    )
