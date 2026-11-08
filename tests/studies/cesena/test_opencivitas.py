"""Unit tests for OpenCivitas standard needs analysis (`studies.cesena.analysis.opencivitas`)."""
from pathlib import Path

import pytest

from cifresignificative.studies.cesena.analysis.opencivitas import (
    comprensorio_standard_needs_comparison,
    peer_standard_needs_comparison,
    social_and_childcare_benchmarks,
)

PROCESSED_DIR = Path("data/processed/cesena")


def test_peer_standard_needs_comparison() -> None:
    """Verify peer comparison of standard needs and service delivery."""
    if not (PROCESSED_DIR / "opencivitas_municipal_summary.parquet").exists():
        pytest.skip("Processed OpenCivitas summary not yet built")

    res = peer_standard_needs_comparison()
    assert len(res.peers_df) == 13
    assert 810.0 < res.cesena_spesa_storica < 830.0
    assert 860.0 < res.cesena_fabbisogno_standard < 880.0
    assert res.cesena_gap_eur < 0.0
    assert res.cesena_service_level_0_10 == 10
    assert res.cesena_fl_virtuoso is True
    assert res.cesena_quadrant == "virtuoso"


def test_comprensorio_standard_needs_comparison() -> None:
    """Verify standard needs comparison across the Comprensorio Cesenate."""
    if not (PROCESSED_DIR / "opencivitas_municipal_summary.parquet").exists():
        pytest.skip("Processed OpenCivitas summary not yet built")

    df = comprensorio_standard_needs_comparison()
    assert len(df) == 17
    assert "040007" in df["comune_id"].values
    assert set(df["belt_group"].unique()).issuperset({"unione_valle_savio", "rubicone_mare", "via_emilia_west"})


def test_social_and_childcare_benchmarks() -> None:
    """Verify social service provision, childcare coverage, and deprivation metrics."""
    if not (PROCESSED_DIR / "opencivitas_municipal_summary.parquet").exists():
        pytest.skip("Processed OpenCivitas summary not yet built")

    res = social_and_childcare_benchmarks()
    assert 40.0 < res.cesena_childcare_coverage_pct < 44.0  # 41.89%
    assert 0.5 < res.cesena_deprivation_index < 2.0  # 1.33
    assert res.cesena_associated_management is True
    assert res.cesena_social_service_coord == 5.0  # Top score
