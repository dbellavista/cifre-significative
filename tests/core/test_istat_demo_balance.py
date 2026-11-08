"""Tests for the population balance in `core.sources.istat_demo`."""
import pandas as pd
import pytest

from cifresignificative.core.sources.istat_demo import add_balance_components


def test_balance_components_close_the_population_change():
    """Change to next January = natural balance + internal + international migration + registry adjustment."""
    df = pd.DataFrame([
        {"comune_id": "040007", "year": 2019, "pop_total": 1000.0, "births": 10.0, "deaths": 20.0, "inflow_total": 30.0, "outflow_total": 10.0, "inflow_abroad": 8.0, "outflow_abroad": 3.0},
        {"comune_id": "040007", "year": 2020, "pop_total": 1005.0, "births": 10.0, "deaths": 20.0, "inflow_total": 30.0, "outflow_total": 10.0, "inflow_abroad": 8.0, "outflow_abroad": 3.0},
    ])
    out = add_balance_components(df)
    row = out.iloc[0]
    assert row["net_migration_internal"] == 20.0
    assert row["net_migration_international"] == 5.0
    assert row["net_migration"] == 25.0
    assert row["registry_adjustment"] == pytest.approx(5.0 - (-10.0) - 25.0)
    assert row["natural_balance"] + row["net_migration"] + row["registry_adjustment"] == pytest.approx(row["pop_change_next"])
