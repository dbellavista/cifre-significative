"""Unit tests for the MUR USTAT university enrolments and graduates source loader."""
from pathlib import Path

import pandas as pd
import pytest

from cifresignificative.core.sources import mur_ustat


@pytest.fixture
def mock_mur_raw_dir(tmp_path: Path) -> Path:
    """Create mock MUR USTAT CSV files in a temporary directory."""
    # Mock iscritti 2019-2025
    isc_2019 = pd.DataFrame({
        "AnnoA": ["2024/2025", "2024/2025", "2024/2025"],
        "AteneoCOD": ["3701", "3701", "3701"],
        "AteneoNOME": ["Bologna", "Bologna", "Bologna"],
        "ClasseNUMERO": ["L-31", "L-8", "L-24"],
        "CorsoNOME": ["Ingegneria Informatica", "Ingegneria Elettronica", "Psicologia"],
        "SedeP": ["40", "40", "40"],
        "SedeC": ["40007", "40007", "40007"],
        "GruppoCODICE": [10, 12, 6],
        "Sesso": ["M", "M", "F"],
        "Iscritti": [600, 1400, 1000],
    })
    isc_2019.to_csv(tmp_path / "iscritti_2019_2025.csv", sep=";", index=False, encoding="latin1")

    # Mock iscritti 2016-2018
    isc_2016 = pd.DataFrame({
        "AnnoA": ["2018/2019"],
        "AteneoCOD": ["3701"],
        "AteneoNOME": ["Bologna"],
        "ClasseNUMERO": ["L-31"],
        "CorsoNOME": ["Ingegneria Informatica"],
        "SedeP": ["40"],
        "SedeC": ["40007"],
        "GruppoCODICE": [10],
        "Sesso": ["M"],
        "Isc": [500],
    })
    isc_2016.to_csv(tmp_path / "iscritti_2016_2018.csv", sep=";", index=False, encoding="latin1")

    # Mock laureati 2018-2025
    lau_2018 = pd.DataFrame({
        "ANNO_LAUREA": [2024],
        "COD_Ateneo": [3701],
        "NomeOperativo": ["Bologna"],
        "NUMERO": ["L-31"],
        "NOME_CORSO": ["Ingegneria Informatica"],
        "COMUNE": ["CESENA"],
        "ID_PROVINCIA_ISTAT": [40],
        "ID_SESSO": ["M"],
        "laureati": [120],
    })
    lau_2018.to_csv(tmp_path / "laureati_2018_2025.csv", sep=";", index=False, encoding="latin1")

    return tmp_path


def test_load_enrolments_mock(mock_mur_raw_dir: Path):
    df = mur_ustat.load_enrolments(comuni_ids=["040007"], raw_dir=mock_mur_raw_dir)
    assert not df.empty
    assert len(df) == 4
    assert set(df["comune_id"]) == {"040007"}
    assert set(df["calendar_year"]) == {2018, 2024}
    assert (df[df["macro_group"] == "stem_ict"]["students"].sum()) == 1100


def test_load_graduates_mock(mock_mur_raw_dir: Path):
    df = mur_ustat.load_graduates(comuni_ids=["040007"], raw_dir=mock_mur_raw_dir)
    assert not df.empty
    assert len(df) == 1
    assert df.iloc[0]["comune_id"] == "040007"
    assert df.iloc[0]["graduates"] == 120


def test_summarize_university_poles_mock(mock_mur_raw_dir: Path):
    summary = mur_ustat.summarize_university_poles(comuni_ids=["040007"], raw_dir=mock_mur_raw_dir)
    assert not summary.empty
    row_24 = summary[summary["calendar_year"] == 2024].iloc[0]
    assert row_24["students_total"] == 3000
    assert row_24["students_stem"] == 2000  # 600 ICT (group 10) + 1400 Eng (group 12)
    assert row_24["stem_share"] == pytest.approx(2000 / 3000)
    assert row_24["ict_share"] == pytest.approx(600 / 3000)
    assert row_24["graduates_total"] == 120


def test_processed_tables_exist_if_built():
    """If real processed tables exist, verify structural integrity and Cesena presence."""
    p = Path("data/processed/cesena/mur_university_summary.parquet")
    if not p.exists():
        pytest.skip("Processed university summary parquet not built yet.")
    df = pd.read_parquet(p)
    assert not df.empty
    assert "040007" in set(df["comune_id"])
    ces = df[df["comune_id"] == "040007"]
    assert len(ces) >= 10
    # Cesena STEM share should be substantially above zero
    assert ces["stem_share"].mean() > 0.50
