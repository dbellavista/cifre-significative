"""SOSE / MEF OpenCivitas Standard Expenditure Needs & Service Level Analysis.

Empirically tests whether municipal expenditure differences reflect true operational efficiency
vs. service under-provision by comparing historical expenditure against standard needs (Fabbisogni Standard)
and quantitative service output levels (Livelli Quantitativi delle Prestazioni, LQP).

Provides three analytical views:
1. `peer_standard_needs_comparison()`: Benchmarks Cesena against the 12 Emilia-Romagna peers on total
   fundamental functions, spending gaps, and 0-10 service delivery levels.
2. `comprensorio_standard_needs_comparison()`: Examines the 16 belt municipalities and the effect
   of inter-municipal unions (*Unione Valle del Savio*, *Unione Rubicone e Mare*).
3. `social_and_childcare_benchmarks()`: Deep-dives into social services, childcare coverage (*asili nido*),
   and material deprivation indices.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cifresignificative.studies.cesena.analysis.data import (
    load_opencivitas_summary,
)


@dataclass(frozen=True)
class PeerStandardNeedsSummary:
    peers_df: pd.DataFrame
    cesena_spesa_storica: float
    cesena_fabbisogno_standard: float
    cesena_gap_eur: float
    cesena_gap_pct: float
    cesena_service_level_0_10: int
    cesena_spending_level_0_10: int
    cesena_fl_virtuoso: bool
    cesena_quadrant: str


@dataclass(frozen=True)
class SocialChildcareSummary:
    social_df: pd.DataFrame
    cesena_childcare_coverage_pct: float
    cesena_deprivation_index: float
    cesena_social_service_coord: float
    cesena_associated_management: bool


def peer_standard_needs_comparison(processed_dir: Path | None = None) -> PeerStandardNeedsSummary:
    """Compare standard expenditure needs and service delivery across the 13 ER peers."""
    df = load_opencivitas_summary(processed_dir)
    peers = df[df["is_peer"]].copy().sort_values("tot_spesa_storica_eur_pc", ascending=True).reset_index(drop=True)

    ces_row = peers[peers["comune_id"] == "040007"].iloc[0]

    return PeerStandardNeedsSummary(
        peers_df=peers,
        cesena_spesa_storica=float(ces_row["tot_spesa_storica_eur_pc"]),
        cesena_fabbisogno_standard=float(ces_row["tot_fabbisogno_standard_eur_pc"]),
        cesena_gap_eur=float(ces_row["tot_diff_eur_pc"]),
        cesena_gap_pct=float(ces_row["tot_diff_pct"]),
        cesena_service_level_0_10=int(ces_row["tot_livello_servizi_0_10"]),
        cesena_spending_level_0_10=int(ces_row["tot_livello_spesa_0_10"]),
        cesena_fl_virtuoso=bool(ces_row["fl_virtuoso"]),
        cesena_quadrant=str(ces_row["quadrante_performance"]),
    )


def comprensorio_standard_needs_comparison(processed_dir: Path | None = None) -> pd.DataFrame:
    """Compare standard needs and service delivery across the 17 Comprensorio Cesenate municipalities."""
    df = load_opencivitas_summary(processed_dir)
    comprensorio = df[df["is_belt"] | (df["comune_id"] == "040007")].copy()
    return comprensorio.sort_values("tot_diff_pct", ascending=True).reset_index(drop=True)


def social_and_childcare_benchmarks(processed_dir: Path | None = None) -> SocialChildcareSummary:
    """Analyze social service provision, childcare coverage, and deprivation across ER peers."""
    df = load_opencivitas_summary(processed_dir)
    peers = df[df["is_peer"]].copy()
    ces_row = peers[peers["comune_id"] == "040007"].iloc[0]

    return SocialChildcareSummary(
        social_df=peers,
        cesena_childcare_coverage_pct=float(ces_row["copertura_nidi_pct"]),
        cesena_deprivation_index=float(ces_row["deprivazione_indice"]),
        cesena_social_service_coord=float(ces_row["soc_coord_servizi"]),
        cesena_associated_management=bool(ces_row["gestione_associata_sociale"] == 1.0),
    )
