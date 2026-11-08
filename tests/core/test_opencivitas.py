"""Unit tests for SOSE / MEF OpenCivitas loader (`core.sources.opencivitas`)."""
from pathlib import Path

import pandas as pd
import pytest

from cifresignificative.core.sources import opencivitas

RAW_DIR = Path("data/raw/opencivitas")
PROCESSED_DIR = Path("data/processed/cesena")


@pytest.mark.skipif(not (RAW_DIR / "Metadati_Enti_2023_IT.xlsx").exists(), reason="Raw OpenCivitas metadata not present")
def test_opencivitas_build_all(tmp_path: Path) -> None:
    """Test that build_all generates summary and detailed parquet files."""
    dfs = opencivitas.build_all(raw_dir=RAW_DIR, processed_dir=tmp_path)
    assert "summary" in dfs
    assert "detailed" in dfs

    assert (tmp_path / "opencivitas_municipal_summary.parquet").exists()
    assert (tmp_path / "opencivitas_indicators_detailed.parquet").exists()


def test_opencivitas_summary_accounting_and_metrics() -> None:
    """Verify OpenCivitas summary metrics, accounting identities, and Cesena benchmarks."""
    p_sum = PROCESSED_DIR / "opencivitas_municipal_summary.parquet"
    if not p_sum.exists():
        pytest.skip("Processed OpenCivitas summary not yet built")

    df = pd.read_parquet(p_sum)
    assert len(df) == 29
    assert {"comune_id", "comune", "tot_spesa_storica_eur_pc", "tot_fabbisogno_standard_eur_pc", "tot_diff_eur_pc", "tot_livello_servizi_0_10"}.issubset(df.columns)

    for _, row in df.iterrows():
        # Identity: tot_diff_eur_pc = tot_spesa_storica_eur_pc - tot_fabbisogno_standard_eur_pc
        expected_diff = round(row["tot_spesa_storica_eur_pc"] - row["tot_fabbisogno_standard_eur_pc"], 2)
        assert abs(row["tot_diff_eur_pc"] - expected_diff) < 0.05
        # Service level is 0..10
        assert 0 <= row["tot_livello_servizi_0_10"] <= 10

    # Verify Cesena (040007)
    ces = df[df["comune_id"] == "040007"].iloc[0]
    assert 810.0 < ces["tot_spesa_storica_eur_pc"] < 830.0  # 819.97 EUR/pc
    assert 860.0 < ces["tot_fabbisogno_standard_eur_pc"] < 880.0  # 869.13 EUR/pc
    assert ces["tot_diff_eur_pc"] < 0.0  # -49.16 EUR/pc lower than standard need
    assert ces["tot_livello_servizi_0_10"] == 10  # Maximum service level
    assert ces["fl_virtuoso"] is True or ces["fl_virtuoso"] == 1
    assert ces["quadrante_performance"] == "virtuoso"
    assert 40.0 < ces["copertura_nidi_pct"] < 44.0  # 41.89% childcare coverage


def test_opencivitas_detailed_structure() -> None:
    """Verify structure and coverage of detailed indicators."""
    p_det = PROCESSED_DIR / "opencivitas_indicators_detailed.parquet"
    if not p_det.exists():
        pytest.skip("Processed OpenCivitas detailed indicators not yet built")

    df = pd.read_parquet(p_det)
    assert len(df) > 5000
    assert {"comune_id", "comune", "servizio", "indicatore_codice", "valore_num"}.issubset(df.columns)
    assert set(df["comune_id"].unique()).issuperset({"040007", "040012", "037006"})
