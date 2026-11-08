"""Unit tests for ISTAT FRAME-SBS Territoriale municipal source loader (core.sources.istat_frame_sbs)."""

import numpy as np
import pandas as pd
import pytest

from cifresignificative.core.geo import ALL_COMUNI_IDS, BELT_COMUNI_IDS
from cifresignificative.core.sources.istat_frame_sbs import (
    PROCESSED_DIR,
    RAW_DIR,
    build_frame_sbs_panel,
    build_frame_sbs_summary,
)

pytestmark = pytest.mark.skipif(not (RAW_DIR.exists() and any(RAW_DIR.iterdir())) and not (PROCESSED_DIR / "frame_sbs_municipal.parquet").exists(), reason="FRAME-SBS raw or processed files missing")


@pytest.fixture(scope="module")
def frame_panel() -> pd.DataFrame:
    path = PROCESSED_DIR / "frame_sbs_municipal.parquet"
    if not path.exists():
        return build_frame_sbs_panel()
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def frame_summary() -> pd.DataFrame:
    path = PROCESSED_DIR / "frame_sbs_summary.parquet"
    if not path.exists():
        return build_frame_sbs_summary()
    return pd.read_parquet(path)


def test_frame_sbs_dimensions_and_coverage(frame_panel: pd.DataFrame) -> None:
    """Verify that all 29 target municipalities across 7 years and 3 sectors are present."""
    assert len(frame_panel) == 29 * 7 * 3  # 609 rows

    years = set(frame_panel["year"].unique())
    assert years == set(range(2017, 2024))

    sectors = set(frame_panel["macro_sector"].unique())
    assert sectors == {"totale", "industria", "servizi"}

    comuni = set(frame_panel["comune_id"].unique())
    expected_comuni = set(ALL_COMUNI_IDS) | set(BELT_COMUNI_IDS)
    assert comuni == expected_comuni


def test_frame_sbs_mathematical_identities(frame_panel: pd.DataFrame) -> None:
    """Verify economic and mathematical identities across municipalities."""
    # Active workers must be >= salaried employees
    valid_emp = frame_panel.dropna(subset=["addetti", "dipendenti"])
    assert (valid_emp["addetti"] >= valid_emp["dipendenti"]).all()

    # Total economy addetti should approximately equal industry + services
    piv_add = frame_panel.pivot(index=["comune_id", "year"], columns="macro_sector", values="addetti")
    diff = piv_add["totale"] - (piv_add["industria"] + piv_add["servizi"])
    # Small differences can occur due to unclassified ATECO or rounding in ISTAT tables, but diff should be small
    assert (diff.abs() / piv_add["totale"]).mean() < 0.05

    # Ratio va_per_addetto_eur should match valore_aggiunto_eur / addetti within rounding tolerance
    valid_va = frame_panel.dropna(subset=["valore_aggiunto_eur", "addetti", "va_per_addetto_eur"])
    computed_va_add = valid_va["valore_aggiunto_eur"] / valid_va["addetti"]
    np.testing.assert_allclose(computed_va_add, valid_va["va_per_addetto_eur"], rtol=0.02)


def test_frame_sbs_summary_cesena_benchmarks(frame_summary: pd.DataFrame) -> None:
    """Verify summary table structure and Cesena values."""
    assert len(frame_summary) == 7 * 3  # 21 rows (7 years x 3 sectors)
    assert "cesena_va_per_addetto_eur" in frame_summary.columns
    assert "core_peer_va_per_addetto_eur" in frame_summary.columns
    assert "rank_va_per_addetto_er" in frame_summary.columns

    # In Cesena 2023, total VA per worker should be approx 58,946 EUR
    c23_tot = frame_summary[(frame_summary["year"] == 2023) & (frame_summary["macro_sector"] == "totale")].iloc[0]
    assert 55_000 < c23_tot["cesena_va_per_addetto_eur"] < 65_000
    assert 25_000 < c23_tot["cesena_retrib_per_dipendente_eur"] < 32_000

    # In Cesena 2023, industrial VA per worker should exceed 80,000 EUR
    c23_ind = frame_summary[(frame_summary["year"] == 2023) & (frame_summary["macro_sector"] == "industria")].iloc[0]
    assert c23_ind["cesena_va_per_addetto_eur"] > 80_000
