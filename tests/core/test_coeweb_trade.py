"""Unit tests for ISTAT / Coeweb / Regione Emilia-Romagna foreign trade loader (`core.sources.coeweb_trade`)."""
from pathlib import Path

import pandas as pd
import pytest

from cifresignificative.core.sources import coeweb_trade

RAW_DIR = Path("data/raw/coeweb_trade")
PROCESSED_DIR = Path("data/processed/cesena")


@pytest.mark.skipif(not (RAW_DIR / "import_export_emilia_romagna.zip").exists(), reason="Raw Coeweb ZIP not present")
def test_coeweb_build_all(tmp_path: Path) -> None:
    """Test that build_all generates all 4 trade tables with valid schemas."""
    dfs = coeweb_trade.build_all(raw_dir=RAW_DIR, processed_dir=tmp_path)
    assert "provinces" in dfs
    assert "time_series" in dfs
    assert "commodities" in dfs
    assert "countries" in dfs

    # Check parquet output files exist on disk
    assert (tmp_path / "coeweb_provinces_export.parquet").exists()
    assert (tmp_path / "coeweb_emr_time_series.parquet").exists()
    assert (tmp_path / "coeweb_commodities_export.parquet").exists()
    assert (tmp_path / "coeweb_countries_balance.parquet").exists()


def test_coeweb_provinces_accounting_and_metrics() -> None:
    """Verify provincial trade accounting identities, ranks, and Forlì-Cesena values."""
    p_prov = PROCESSED_DIR / "coeweb_provinces_export.parquet"
    if not p_prov.exists():
        pytest.skip("Processed Coeweb provinces table not yet built")

    df = pd.read_parquet(p_prov)
    assert len(df) == 9
    assert {"provincia", "nuts3", "export_eur", "import_eur", "trade_balance_eur", "coverage_ratio_pct"}.issubset(df.columns)

    for _, row in df.iterrows():
        # Identity: balance = export - import (within rounding tolerance)
        assert abs(row["trade_balance_eur"] - (row["export_eur"] - row["import_eur"])) < 1.0
        # Coverage ratio: export / import * 100
        expected_cov = round(row["export_eur"] / row["import_eur"] * 100, 2)
        assert abs(row["coverage_ratio_pct"] - expected_cov) < 0.05
        # Ranks are 1..9
        assert 1 <= row["rank_export"] <= 9
        assert 1 <= row["rank_balance"] <= 9

    # Check Forlì-Cesena (ITH58)
    fc = df[df["provincia"] == "Forlì-Cesena"].iloc[0]
    assert fc["nuts3"] == "ITH58"
    assert fc["prov_sigla"] == "FC"
    assert 4.5e9 < fc["export_eur"] < 4.8e9  # ~4.63 Billion EUR
    assert 1.9e9 < fc["trade_balance_eur"] < 2.2e9  # ~+2.04 Billion EUR
    assert 175.0 < fc["coverage_ratio_pct"] < 182.0  # ~178.85%


def test_coeweb_time_series_monotonicity() -> None:
    """Verify annual time series spans 1991 to 2025 and maintains chronological ordering."""
    p_ts = PROCESSED_DIR / "coeweb_emr_time_series.parquet"
    if not p_ts.exists():
        pytest.skip("Processed Coeweb time series not yet built")

    df = pd.read_parquet(p_ts)
    assert len(df) == 35
    assert df["year"].min() == 1991
    assert df["year"].max() == 2025
    assert df["year"].is_monotonic_increasing

    # Total 2025 export > 80 Billion EUR
    y2025 = df[df["year"] == 2025].iloc[0]
    assert y2025["export_eur"] > 80e9
    assert y2025["trade_balance_eur"] > 30e9


def test_coeweb_commodities_agrifood() -> None:
    """Verify presence of key agrifood and machinery categories in top commodities."""
    p_comm = PROCESSED_DIR / "coeweb_commodities_export.parquet"
    if not p_comm.exists():
        pytest.skip("Processed Coeweb commodities not yet built")

    df = pd.read_parquet(p_comm)
    codes = set(df["code"].dropna())
    # Agri-food categories
    assert "CA101" in codes  # Carne lavorata
    assert "CA108" in codes  # Altri prodotti alimentari
    assert "CK283" in codes  # Macchine per l'agricoltura
