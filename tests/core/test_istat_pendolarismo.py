"""Unit tests for ISTAT 15th Census commuting matrices loader (`core.sources.istat_pendolarismo`)."""
from pathlib import Path

import pandas as pd
import pytest

from cifresignificative.core.sources import istat_pendolarismo

RAW_DIR = Path("data/raw/istat_pendolarismo")
PROCESSED_DIR = Path("data/processed/cesena")


@pytest.mark.skipif(not (RAW_DIR / "matrici_pendolarismo_2011.zip").exists(), reason="Raw commuting ZIP not present")
def test_pendolarismo_build_output_structure(tmp_path: Path) -> None:
    """Test that build generates the three parquet files with expected schemas."""
    paths = istat_pendolarismo.build(raw_dir=RAW_DIR, out_dir=tmp_path)
    assert "commuting_municipal_summary" in paths
    assert "commuting_od_matrix" in paths
    assert "commuting_belt_corridors" in paths

    df_sum = pd.read_parquet(paths["commuting_municipal_summary"])
    assert len(df_sum) == 29
    assert "comune_id" in df_sum.columns
    assert "jobs_local" in df_sum.columns
    assert "resident_workers" in df_sum.columns
    assert "job_self_containment_rate" in df_sum.columns

    df_od = pd.read_parquet(paths["commuting_od_matrix"])
    assert len(df_od) > 1000
    assert {"origin_id", "dest_id", "purpose", "persons"}.issubset(df_od.columns)

    df_corr = pd.read_parquet(paths["commuting_belt_corridors"])
    assert len(df_corr) >= 20
    assert {"partner_id", "inbound_to_cesena", "outbound_from_cesena"}.issubset(df_corr.columns)


def test_commuting_accounting_identities() -> None:
    """Verify exact demographic-employment accounting identities on the generated summary."""
    p_sum = PROCESSED_DIR / "commuting_municipal_summary.parquet"
    if not p_sum.exists():
        pytest.skip("Processed commuting summary not yet built")

    df_sum = pd.read_parquet(p_sum)
    for _, row in df_sum.iterrows():
        # Identity 1: Local jobs = internal + inbound
        assert row["jobs_local"] == row["internal_workers"] + row["inbound_workers"]
        # Identity 2: Resident workers = internal + outbound
        assert row["resident_workers"] == row["internal_workers"] + row["outbound_workers"]
        # Identity 3: Net inflow = inbound - outbound
        assert row["net_commuter_inflow"] == row["inbound_workers"] - row["outbound_workers"]
        # Identity 4: Rates are bounded between 0 and 1
        assert 0.0 <= row["job_self_containment_rate"] <= 1.0
        assert 0.0 <= row["labor_retention_rate"] <= 1.0
        # Modal shares sum to approximately 1.0
        mode_sum = row["pct_car"] + row["pct_active"] + row["pct_transit"] + row["pct_twowheel"]
        assert 0.85 <= mode_sum <= 1.05


def test_cesena_commuting_known_values() -> None:
    """Verify Cesena specific benchmark values from the 2011 census."""
    p_sum = PROCESSED_DIR / "commuting_municipal_summary.parquet"
    if not p_sum.exists():
        pytest.skip("Processed commuting summary not yet built")

    df_sum = pd.read_parquet(p_sum)
    ces = df_sum[df_sum["comune_id"] == "040007"].iloc[0]

    assert ces["jobs_local"] == 42437
    assert ces["resident_workers"] == 37840
    assert ces["internal_workers"] == 29174
    assert ces["inbound_workers"] == 13263
    assert ces["outbound_workers"] == 8666
    assert ces["net_commuter_inflow"] == 4597
    assert 0.68 <= ces["job_self_containment_rate"] <= 0.69
    assert 0.76 <= ces["labor_retention_rate"] <= 0.78
