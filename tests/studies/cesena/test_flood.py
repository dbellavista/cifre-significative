"""Tests for May 2023 Romagna flood shock & resilience analysis (`studies.cesena.analysis.flood`)."""

from pathlib import Path

import pytest

from cifresignificative.core.geo import ALL_COMUNI_IDS, CESENA, COMPRENSORIO_COMUNI_IDS
from cifresignificative.studies.cesena.analysis.data import load_flood_shock
from cifresignificative.studies.cesena.analysis.flood import (
    FLOOD_TIERS,
    build_flood_shock_panel,
    compare_flood_tiers,
)

needs_processed = pytest.mark.skipif(not Path("data/processed/cesena/demography.parquet").exists(), reason="cesena data not built")


def test_flood_tier_classification():
    assert FLOOD_TIERS["040007"] == "whole_territory"  # Cesena
    assert FLOOD_TIERS["040012"] == "whole_territory"  # Forlì
    assert FLOOD_TIERS["039010"] == "whole_territory"  # Faenza
    assert FLOOD_TIERS["039014"] == "whole_territory"  # Ravenna
    assert FLOOD_TIERS["037032"] == "part_territory" # Imola
    assert FLOOD_TIERS["037006"] == "part_territory" # Bologna
    assert FLOOD_TIERS["099014"] == "not_listed" # Rimini
    assert FLOOD_TIERS["035033"] == "not_listed" # Reggio Emilia
    assert FLOOD_TIERS["034027"] == "not_listed" # Parma
    assert FLOOD_TIERS["036023"] == "not_listed" # Modena
    assert FLOOD_TIERS["033032"] == "not_listed" # Piacenza
    assert FLOOD_TIERS["036005"] == "not_listed" # Carpi
    assert FLOOD_TIERS["038008"] == "not_listed" # Ferrara


@needs_processed
def test_build_flood_shock_panel_structure(tmp_path):
    df = build_flood_shock_panel(out_dir=tmp_path)
    assert len(df) == len(ALL_COMUNI_IDS)
    assert set(df["comune_id"]) == set(ALL_COMUNI_IDS)

    required_cols = [
        "comune_id",
        "comune_name",
        "flood_tier",
        "flood_tier_label",
        "is_cesena",
        "pop_2022",
        "pop_2025",
        "pop_growth_22_25_pct",
        "net_migr_rate_2022",
        "net_migr_rate_2023",
        "net_migr_rate_2024",
        "real_mean_income_2022",
        "real_mean_income_2023",
        "income_growth_22_23_pct",
        "taxpayer_growth_22_23_pct",
        "capitale_cassa_pc_2022",
        "capitale_cassa_pc_2024",
        "capitale_cassa_growth_22_24_pct",
        "employees_2022",
        "employees_2024",
        "emp_growth_22_24_pct",
        "local_units_2022",
        "local_units_2024",
        "units_growth_23_24_pct",
        "real_va_growth_22_23_pct",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col}"

    ces = df[df["is_cesena"]].iloc[0]
    assert ces["comune_id"] == CESENA
    assert ces["flood_tier"] == "whole_territory"
    assert ces["emp_growth_22_24_pct"] > 5.0
    assert ces["capitale_cassa_growth_22_24_pct"] > 100.0


@needs_processed
def test_compare_flood_tiers():
    panel = load_flood_shock()
    tiers = compare_flood_tiers(panel)
    assert len(tiers) == 4
    assert set(tiers["flood_tier"]) == {
        "cesena_focal",
        "whole_territory",
        "part_territory",
        "not_listed",
    }

    # Severely impacted tier experienced a massive capital reconstruction cash surge
    sev_row = tiers[tiers["flood_tier"] == "whole_territory"].iloc[0]
    ctrl_row = tiers[tiers["flood_tier"] == "not_listed"].iloc[0]
    assert sev_row["capitale_cassa_growth_22_24_pct_median"] > ctrl_row["capitale_cassa_growth_22_24_pct_median"]
    assert sev_row["capitale_cassa_pc_2024_median"] > 350.0  # EUR / pc in 2024


@needs_processed
def test_build_flood_shock_panel_with_belt_comuni(tmp_path):
    df_belt = build_flood_shock_panel(out_dir=tmp_path, comuni_ids=COMPRENSORIO_COMUNI_IDS[:5])
    assert len(df_belt) == 5
    # Allegato 1 was checked only for the 13 benchmark municipalities
    for _, row in df_belt.iterrows():
        assert row["flood_tier"] == FLOOD_TIERS.get(row["comune_id"], "not_checked")
