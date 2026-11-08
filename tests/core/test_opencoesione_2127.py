"""Unit tests for OpenCoesione 2021-2027 cycle extraction (`core.sources.opencoesione_municipal`)."""

from pathlib import Path

import pytest

from cifresignificative.core.geo import CESENA
from cifresignificative.core.sources import opencoesione_municipal

pytestmark = pytest.mark.skipif(not Path("data/raw/opencoesione/localizzazioni.zip").exists(), reason="OpenCoesione files not downloaded")


def test_load_municipal_projects_cycle_options():
    """Verify load_municipal_projects supports cycle parameter ('2014-2020', '2021-2027', 'all')."""
    # 2014-2020 cycle
    p14 = opencoesione_municipal.load_municipal_projects(cycle="2014-2020")
    assert not p14.empty
    assert (p14["dataset"] == "opencoesione_1420").all()
    ces_14 = p14[p14["comune_id"] == CESENA]
    assert len(ces_14) == 511

    # 2021-2027 cycle
    p21 = opencoesione_municipal.load_municipal_projects(cycle="2021-2027")
    assert not p21.empty
    assert (p21["dataset"] == "opencoesione_2127").all()
    ces_21 = p21[p21["comune_id"] == CESENA]
    assert len(ces_21) == 229

    # 'all' cycle combines both
    pall = opencoesione_municipal.load_municipal_projects(cycle="all")
    assert len(pall) == len(p14) + len(p21)
    ces_all = pall[pall["comune_id"] == CESENA]
    assert len(ces_all) == 511 + 229


def test_2021_2027_projects_columns_and_data_integrity():
    """Verify that 2021-2027 projects contain required financial and date columns without NaNs in critical amounts."""
    p21 = opencoesione_municipal.load_municipal_projects(cycle="2021-2027")
    required_cols = [
        "project_id", "cup", "comune_id", "n_comuni", "dataset", "fund_source", "theme_code",
        "allocated_eur", "disbursed_eur", "status", "lag_days",
    ]
    for c in required_cols:
        assert c in p21.columns

    # Amounts must be non-negative
    assert (p21["allocated_eur"] >= 0).all()
    assert (p21["disbursed_eur"] >= 0).all()

    # Multi-municipality division
    assert (p21["n_comuni"] >= 1).all()
