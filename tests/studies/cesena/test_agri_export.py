"""Unit tests for agricultural land specialization and provincial export analysis (`studies.cesena.analysis.agri_export`)."""
from pathlib import Path

import pytest

from cifresignificative.studies.cesena.analysis.agri_export import (
    agricultural_land_specialization,
    agroindustrial_cluster_integration,
    provincial_export_comparison,
)

PROCESSED_DIR = Path("data/processed/cesena")


def test_provincial_export_comparison() -> None:
    """Verify provincial export comparison and Forlì-Cesena metrics."""
    if not (PROCESSED_DIR / "coeweb_provinces_export.parquet").exists():
        pytest.skip("Processed Coeweb provinces table not yet built")

    res = provincial_export_comparison()
    assert res.forli_cesena_export_eur > 4.5e9
    assert res.forli_cesena_balance_eur > 2.0e9
    assert res.forli_cesena_coverage_pct > 175.0
    assert res.forli_cesena_rank_export == 7
    assert res.forli_cesena_rank_balance == 5
    assert len(res.provinces_df) == 9


def test_agricultural_land_specialization() -> None:
    """Verify agricultural land use specialization in fruit orchards and farm fragmentation."""
    if not (PROCESSED_DIR / "agri_sat_land_use_provinces.parquet").exists():
        pytest.skip("Processed agricultural tables not yet built")

    res = agricultural_land_specialization()
    # Forlì-Cesena fruit acreage
    assert 8500.0 < res.forli_cesena_frutta_ha < 8800.0  # 8,621.64 ha
    assert 9.5 < res.forli_cesena_frutta_share_sau_pct < 11.0  # 10.2% of SAU
    assert 15.0 < res.forli_cesena_frutta_share_er_pct < 17.5  # 16.06% of ER fruit
    assert res.forli_cesena_frutta_rank_er == 3

    # Romagna (Ravenna + Forlì-Cesena) accounts for ~47% of ER fruit orchards
    assert 45.0 < res.romagna_frutta_share_er_pct < 50.0

    # Smallholder fragmentation
    assert 16000.0 < res.forli_cesena_under10_ha < 17500.0
    assert 18.5 < res.forli_cesena_under10_share_sau_pct < 21.5


def test_agroindustrial_cluster_integration() -> None:
    """Verify integration synthesis connecting land, processing, and trade."""
    if not (PROCESSED_DIR / "coeweb_commodities_export.parquet").exists():
        pytest.skip("Processed commodities not yet built")

    res = agroindustrial_cluster_integration()
    assert res.export_summary is not None
    assert res.agricultural_summary is not None
    assert not res.agrifood_commodities.empty
    assert res.frutta_ha > 8500.0
    assert res.provincial_export_eur > 4.5e9
    assert res.provincial_surplus_eur > 2.0e9
