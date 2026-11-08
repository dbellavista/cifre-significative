"""Tests for Housing Affordability & Suburbanization analysis (`studies.cesena.analysis.housing`)."""

from pathlib import Path

import pytest

from cifresignificative.core.geo import CESENA
from cifresignificative.studies.cesena.analysis.data import (
    load_omi_municipal,
    load_omi_transactions,
    load_omi_zones,
)
from cifresignificative.studies.cesena.analysis.housing import (
    comprensorio_housing_gradient,
    compute_housing_affordability,
    peer_housing_comparison,
)

needs_processed = pytest.mark.skipif(not Path("data/processed/cesena/irpef_summary.parquet").exists(), reason="cesena data not built")


@needs_processed
def test_data_loaders_exist():
    muni = load_omi_municipal()
    assert len(muni) == 29
    zones = load_omi_zones()
    assert len(zones) > 0
    trans = load_omi_transactions()
    assert len(trans) > 0


@needs_processed
def test_compute_housing_affordability():
    df = compute_housing_affordability()
    assert len(df) == 29

    required_cols = [
        "comune_id",
        "comune_name",
        "belt_group",
        "compr_mid_eur_m2",
        "loc_mid_eur_m2_mo",
        "mean_income_eur",
        "price_to_income_ratio_100m2",
        "annual_rent_80m2_eur",
        "rent_burden_pct_80m2",
        "annual_rent_savings_vs_cesena_eur",
        "purchase_savings_100m2_vs_cesena_eur",
        "net_annual_suburban_incentive_eur",
        "pop_growth_08_25_pct",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col}"

    ces = df[df["is_cesena"]].iloc[0]
    assert ces["comune_id"] == CESENA
    assert 4.0 < ces["price_to_income_ratio_100m2"] < 8.0
    assert 15.0 < ces["rent_burden_pct_80m2"] < 25.0
    assert ces["purchase_savings_100m2_vs_cesena_eur"] == 0.0


@needs_processed
def test_comprensorio_housing_gradient():
    grad = comprensorio_housing_gradient()
    assert len(grad) == 4
    rings = set(grad["belt_group"])
    assert rings == {"cesena_core", "rubicone_mare", "via_emilia_west", "unione_valle_savio"}

    # Cesena core should have higher purchase price and rents than the hill belt
    ces_row = grad[grad["belt_group"] == "cesena_core"].iloc[0]
    savio_row = grad[grad["belt_group"] == "unione_valle_savio"].iloc[0]
    rubicone_row = grad[grad["belt_group"] == "rubicone_mare"].iloc[0]

    assert ces_row["compr_mid_median_eur_m2"] > savio_row["compr_mid_median_eur_m2"]
    assert ces_row["loc_mid_median_eur_mo"] > savio_row["loc_mid_median_eur_mo"]
    assert savio_row["purchase_savings_100m2_median_eur"] > 30000.0
    assert rubicone_row["annual_rent_savings_80m2_median_eur"] > 400.0


@needs_processed
def test_peer_housing_comparison():
    res = peer_housing_comparison()
    assert len(res) == 4
    metrics = set(res["metric"])
    assert "compr_mid_eur_m2" in metrics
    assert "loc_mid_eur_m2_mo" in metrics
    assert "price_to_income_ratio_100m2" in metrics
    assert "gross_rental_yield_pct" in metrics

    p_row = res[res["metric"] == "compr_mid_eur_m2"].iloc[0]
    assert 1 <= p_row["cesena_rank_1_to_13"] <= 13
