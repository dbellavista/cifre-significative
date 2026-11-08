"""Unit tests for ISTAT 7th Agricultural Census loader (`core.sources.istat_agricoltura`)."""
from pathlib import Path

import pandas as pd
import pytest

from cifresignificative.core.sources import istat_agricoltura

RAW_DIR = Path("data/raw/istat_agricoltura")
PROCESSED_DIR = Path("data/processed/cesena")


@pytest.mark.skipif(not (RAW_DIR / "sat_utilizzo_provincia_2020.csv").exists(), reason="Raw agri CSV not present")
def test_agricoltura_build_all(tmp_path: Path) -> None:
    """Test that build_all generates land use and size class tables."""
    dfs = istat_agricoltura.build_all(raw_dir=RAW_DIR, processed_dir=tmp_path)
    assert "land_use" in dfs
    assert "size_classes" in dfs

    assert (tmp_path / "agri_sat_land_use_provinces.parquet").exists()
    assert (tmp_path / "agri_sau_size_classes_provinces.parquet").exists()


def test_agricoltura_land_use_identities() -> None:
    """Verify agricultural land use categories and accounting bounds."""
    p_sat = PROCESSED_DIR / "agri_sat_land_use_provinces.parquet"
    if not p_sat.exists():
        pytest.skip("Processed agricultural land use table not yet built")

    df = pd.read_parquet(p_sat).set_index("category")
    assert "Frutta" in df.index
    assert "Vite" in df.index
    assert "Superficie agricola utilizzata" in df.index
    assert "Superficie agricola totale" in df.index

    # For each province, SAU <= SAT and Frutta + Vite <= Legnose agrarie
    provinces = ["Piacenza", "Parma", "Reggio Emilia", "Modena", "Bologna", "Ferrara", "Ravenna", "Forlì-Cesena", "Rimini"]
    for p in provinces:
        sau = df.loc["Superficie agricola utilizzata", p]
        sat = df.loc["Superficie agricola totale", p]
        frutta = df.loc["Frutta", p]
        vite = df.loc["Vite", p]
        legnose = df.loc["Legnose agrarie", p]

        assert sau <= sat
        assert frutta + vite <= legnose * 1.01  # allow minor rounding/other woody crops

    # Check Forlì-Cesena (ITH58) values
    fc_frutta = df.loc["Frutta", "Forlì-Cesena"]
    assert 8500.0 < fc_frutta < 8800.0  # 8,621.64 ha


def test_agricoltura_size_classes() -> None:
    """Verify farm size class distribution and totals."""
    p_sau = PROCESSED_DIR / "agri_sau_size_classes_provinces.parquet"
    if not p_sau.exists():
        pytest.skip("Processed farm size class table not yet built")

    df = pd.read_parquet(p_sau).set_index("size_class")
    assert "< 10 ha" in df.index
    assert ">= 100 ha" in df.index
    assert "Totale" in df.index

    # Check Forlì-Cesena smallholder acreage (< 10 ha)
    fc_under10 = df.loc["< 10 ha", "Forlì-Cesena"]
    assert 16000.0 < fc_under10 < 17500.0  # 16,872 ha
