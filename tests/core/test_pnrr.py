"""Unit tests for the PNRR public procurement source loader."""
from pathlib import Path

import pandas as pd
import pytest

from cifresignificative.core.sources import pnrr


@pytest.fixture
def mock_pnrr_raw_dir(tmp_path: Path) -> Path:
    """Create mock PNRR CSV files in a temporary directory."""
    # Mock PNRR_Gare.csv
    gare = pd.DataFrame({
        "\ufeffCodice Univoco Submisura": ["M1C1I1.01.00", "M2C4I4.01.00"],
        "Descrizione Submisura": ["Infrastrutture digitali", "Infrastrutture idriche"],
        "Descrizione Procedura di Aggiudicazione": ["AFFIDAMENTO DIRETTO", "PROCEDURA APERTA"],
        "CUP": ["C57H210001", "C57H210002"],
        "Codice Locale Progetto": ["PRJ1", "PRJ2"],
        "CIG": ["CIG111", "CIG222"],
        "CIG Accordo Quadro": ["", ""],
        "Codice Procedura Utente": ["", ""],
        "Modalità di Realizzazione": ["APPALTO", "APPALTO"],
        "Codice Interno PDA": ["PDA1", "PDA2"],
        "Oggetto Principale del Contratto": ["SERVIZI", "LAVORI"],
        "Oggetto Gara": ["Comune di Cesena digitalizzazione", "Comune di Forlì acquedotto"],
        "Data Pubblicazione del CIG": ["01/01/2023", "01/02/2023"],
        "Codice Motivo Assenza CIG": ["000", "000"],
        "Descrizione Motivo Assenza CIG": ["", ""],
        "Importo Complessivo Gara": ["100000,00", "500000,00"],
        "Importo Aggiudicazione": ["80000,00", "450000,00"],
        "Data Aggiudicazione Definitiva": ["15/01/2023", "01/04/2023"],
        "Data di Estrazione": ["13/06/2026", "13/06/2026"],
    })
    gare.to_csv(tmp_path / "PNRR_Gare.csv", sep=";", index=False, encoding="utf-8")

    # Mock PNRR_Aggiudicatari_Gare.csv
    agg = pd.DataFrame({
        "Codice Univoco Submisura": ["M1C1I1.01.00", "M2C4I4.01.00"],
        "Descrizione Submisura": ["Infrastrutture digitali", "Infrastrutture idriche"],
        "CUP": ["C57H210001", "C57H210002"],
        "Codice Locale Progetto": ["PRJ1", "PRJ2"],
        "CIG": ["CIG111", "CIG222"],
        "Codice interno PDA": ["PDA1", "PDA2"],
        "Codice Fiscale/P.IVA": ["01234567890", "09876543210"],
        "Denominazione Aggiudicatario": ["Tech Corp SpA", "Costruzioni Srl"],
        "Ruolo Soggetto": ["8888888898", "8888888898"],
        "Descrizione Ruolo Soggetto": ["IMPRESA SINGOLA", "IMPRESA SINGOLA"],
        "Forma Giuridica Aggiudicatario": ["10", "11"],
        "Descrizione Forma Giuridica Aggiudicatario": ["SpA", "Srl"],
        "Codice ATECO Aggiudicatario": ["62.01.00", "41.20.00"],
        "Codice Procedura Utente": ["", ""],
        "Data di Estrazione": ["13/06/2026", "13/06/2026"],
    })
    agg.to_csv(tmp_path / "PNRR_Aggiudicatari_Gare.csv", sep=";", index=False, encoding="utf-8")

    return tmp_path


def test_load_municipal_tenders_mock(mock_pnrr_raw_dir: Path):
    df = pnrr.load_municipal_tenders(comuni_ids=["040007"], raw_dir=mock_pnrr_raw_dir)
    assert not df.empty
    assert len(df) == 1
    assert df.iloc[0]["comune_id"] == "040007"
    assert df.iloc[0]["cig"] == "CIG111"
    assert df.iloc[0]["importo_gara"] == 100000.0
    assert df.iloc[0]["importo_agg"] == 80000.0
    assert df.iloc[0]["award_lag_days"] == 14
    assert df.iloc[0]["rebate_pct"] == pytest.approx(0.20)
    assert bool(df.iloc[0]["is_direct"]) is True


def test_load_municipal_contractors_mock(mock_pnrr_raw_dir: Path):
    df = pnrr.load_municipal_contractors(cigs=["CIG111"], raw_dir=mock_pnrr_raw_dir)
    assert not df.empty
    assert len(df) == 1
    assert df.iloc[0]["cig"] == "CIG111"
    assert df.iloc[0]["contractor_name"] == "Tech Corp SpA"
    assert df.iloc[0]["ateco"] == "62.01.00"


def test_summarize_municipal_procurement_mock(mock_pnrr_raw_dir: Path):
    summary = pnrr.summarize_municipal_procurement(raw_dir=mock_pnrr_raw_dir)
    assert not summary.empty
    assert len(summary) == 2  # Cesena and Forlì
    ces = summary[summary["comune_id"] == "040007"].iloc[0]
    assert ces["n_tenders"] == 1
    assert ces["total_tendered_eur"] == 100000.0
    assert ces["total_awarded_eur"] == 80000.0
    assert ces["direct_award_share"] == 1.0


def test_processed_tables_exist_if_built():
    """If real processed tables exist, verify structural integrity and Cesena presence."""
    p = Path("data/processed/cesena/pnrr_municipal_summary.parquet")
    if not p.exists():
        pytest.skip("Processed PNRR summary parquet not built yet.")
    df = pd.read_parquet(p)
    assert not df.empty
    assert "040007" in set(df["comune_id"])
    ces = df[df["comune_id"] == "040007"].iloc[0]
    assert ces["n_tenders"] > 50
    assert ces["total_awarded_eur"] > 1e7
