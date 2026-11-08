"""Unit tests for MEF RGS BDAP SIOPE cash expenditure and consolidated perimeter (`core.sources.siope`)."""
import numpy as np
import pandas as pd
import pytest

from cifresignificative.core.geo import CESENA
from cifresignificative.core.sources import siope


def test_siope_entity_mappings():
    """Verify target municipalities and unioni mappings are complete."""
    assert CESENA in siope.MUNI_ENTITIES
    assert siope.MUNI_ENTITIES[CESENA] == "COMUNE DI CESENA"
    assert "unione_valle_savio" in siope.UNIONI_ENTITIES
    unione_name, focal_id = siope.UNIONI_ENTITIES["unione_valle_savio"]
    assert unione_name == "UNIONE DEI COMUNI VALLE DEL SAVIO"
    assert focal_id == CESENA
    assert len(siope.MUNI_ENTITIES) == 13


def test_parse_annual_summary_aggregates_titles_and_macroaggregates():
    """Verify parse_annual_summary aggregates titles U1, U2, U3 and macroaggregates U101, U103, U104 correctly."""
    sample_raw = pd.DataFrame([
        # Cesena
        {"Descrizione Ente BDAP": "COMUNE DI CESENA", "Codice Gestionale Enti Locali": "U1010101001", "Importo cumulato": 1_000_000.0},
        {"Descrizione Ente BDAP": "COMUNE DI CESENA", "Codice Gestionale Enti Locali": "U1030101001", "Importo cumulato": 2_000_000.0},
        {"Descrizione Ente BDAP": "COMUNE DI CESENA", "Codice Gestionale Enti Locali": "U1040101001", "Importo cumulato": 500_000.0},
        {"Descrizione Ente BDAP": "COMUNE DI CESENA", "Codice Gestionale Enti Locali": "U2020101001", "Importo cumulato": 1_500_000.0},
        {"Descrizione Ente BDAP": "COMUNE DI CESENA", "Codice Gestionale Enti Locali": "U3010103002", "Importo cumulato": 70_000.0},  # financial assets, not a repayment
        {"Descrizione Ente BDAP": "COMUNE DI CESENA", "Codice Gestionale Enti Locali": "U4030104999", "Importo cumulato": 300_000.0},
        # Unione Valle del Savio
        {"Descrizione Ente BDAP": "UNIONE DEI COMUNI VALLE DEL SAVIO", "Codice Gestionale Enti Locali": "U1010101001", "Importo cumulato": 400_000.0},
        {"Descrizione Ente BDAP": "UNIONE DEI COMUNI VALLE DEL SAVIO", "Codice Gestionale Enti Locali": "U1030101001", "Importo cumulato": 600_000.0},
        {"Descrizione Ente BDAP": "UNIONE DEI COMUNI VALLE DEL SAVIO", "Codice Gestionale Enti Locali": "U2020101001", "Importo cumulato": 200_000.0},
    ])

    summary = siope.parse_annual_summary(sample_raw, year=2024)
    assert len(summary) == 2

    ces = summary[summary["comune_id"] == CESENA].iloc[0]
    assert ces["entity_type"] == "comune"
    assert ces["spese_correnti_cassa_eur"] == 3_500_000.0  # 1M + 2M + 0.5M
    assert ces["spesa_personale_cassa_eur"] == 1_000_000.0
    assert ces["spesa_beni_servizi_cassa_eur"] == 2_000_000.0
    assert ces["spese_capitale_cassa_eur"] == 1_500_000.0
    assert ces["rimborso_prestiti_cassa_eur"] == 300_000.0  # Titolo 4: loan repayment
    assert ces["incremento_att_fin_cassa_eur"] == 70_000.0  # Titolo 3: not a repayment
    assert ces["spese_totali_cassa_eur"] == 5_370_000.0

    uv = summary[summary["entity_name"] == "UNIONE DEI COMUNI VALLE DEL SAVIO"].iloc[0]
    assert uv["entity_type"] == "unione"
    assert uv["focal_comune_id"] == CESENA
    assert uv["spese_correnti_cassa_eur"] == 1_000_000.0
    assert uv["spesa_personale_cassa_eur"] == 400_000.0
    assert uv["spese_capitale_cassa_eur"] == 200_000.0


def test_build_consolidated_panel_adds_pro_rata_unione_spending():
    """Verify build_consolidated_panel attributes pro-rata Unione spending to the focal municipality."""
    siope_df = pd.DataFrame([
        {
            "year": 2024, "entity_name": "COMUNE DI CESENA", "entity_type": "comune", "comune_id": CESENA,
            "unione_key": None, "focal_comune_id": CESENA,
            "spese_correnti_cassa_eur": 80_000_000.0, "spese_capitale_cassa_eur": 20_000_000.0,
            "rimborso_prestiti_cassa_eur": 2_000_000.0, "spese_totali_cassa_eur": 102_000_000.0,
            "spesa_personale_cassa_eur": 22_000_000.0, "spesa_beni_servizi_cassa_eur": 40_000_000.0,
            "trasferimenti_correnti_cassa_eur": 10_000_000.0, "interessi_passivi_cassa_eur": 1_000_000.0,
            "personale_share_pct": 27.5, "servizi_share_pct": 50.0,
        },
        {
            "year": 2024, "entity_name": "UNIONE DEI COMUNI VALLE DEL SAVIO", "entity_type": "unione", "comune_id": None,
            "unione_key": "unione_valle_savio", "focal_comune_id": CESENA,
            "spese_correnti_cassa_eur": 20_000_000.0, "spese_capitale_cassa_eur": 2_000_000.0,
            "rimborso_prestiti_cassa_eur": 0.0, "spese_totali_cassa_eur": 22_000_000.0,
            "spesa_personale_cassa_eur": 6_000_000.0, "spesa_beni_servizi_cassa_eur": 12_000_000.0,
            "trasferimenti_correnti_cassa_eur": 2_000_000.0, "interessi_passivi_cassa_eur": 0.0,
            "personale_share_pct": 30.0, "servizi_share_pct": 60.0,
        },
    ])
    demo_df = pd.DataFrame([
        {"comune_id": CESENA, "year": 2024, "pop_total": 96_000.0},
    ])

    cons = siope.build_consolidated_panel(siope_df, demo_df=demo_df)
    assert len(cons) == 1
    row = cons.iloc[0]
    weight = 0.834  # Cesena's weight in Unione Valle Savio
    expected_u_correnti = 20_000_000.0 * weight
    assert row["unione_correnti_share_eur"] == pytest.approx(expected_u_correnti)
    assert row["consolidated_correnti_cassa_eur"] == pytest.approx(80_000_000.0 + expected_u_correnti)
    assert row["direct_correnti_pc"] == pytest.approx(80_000_000.0 / 96_000.0)
    assert row["consolidated_correnti_pc"] == pytest.approx((80_000_000.0 + expected_u_correnti) / 96_000.0)
    assert row["consolidated_correnti_pc"] > row["direct_correnti_pc"]


def _entity(name: str, etype: str, cid: str | None, correnti: float, capitale: float, **extra: float) -> dict:
    return {
        "year": 2024, "entity_name": name, "entity_type": etype, "comune_id": cid, "unione_key": None if etype == "comune" else "unione_valle_savio", "focal_comune_id": CESENA,
        "spese_correnti_cassa_eur": correnti, "spese_capitale_cassa_eur": capitale, "rimborso_prestiti_cassa_eur": 0.0, "spese_totali_cassa_eur": correnti + capitale,
        "spesa_personale_cassa_eur": 0.0, "spesa_beni_servizi_cassa_eur": 0.0, "trasferimenti_correnti_cassa_eur": 0.0, "interessi_passivi_cassa_eur": 0.0, **extra,
    }


def test_consolidation_nets_transfers_inside_the_perimeter():
    """A euro the Comune transfers to its Unione is spent once: consolidated spending must not count it on both sides."""
    siope_df = pd.DataFrame([
        _entity("COMUNE DI CESENA", "comune", CESENA, 80_000_000.0, 20_000_000.0, trasf_correnti_a_unioni_eur=10_000_000.0, trasf_capitale_a_unioni_eur=1_000_000.0),
        _entity("UNIONE DEI COMUNI VALLE DEL SAVIO", "unione", None, 20_000_000.0, 2_000_000.0, trasf_correnti_a_comuni_eur=1_000_000.0, trasf_capitale_a_comuni_eur=500_000.0),
    ])
    row = siope.build_consolidated_panel(siope_df).iloc[0]
    assert row["consolidated_correnti_cassa_eur"] == pytest.approx(80_000_000.0 - 10_000_000.0 + 0.834 * 19_000_000.0)
    assert row["consolidated_capitale_cassa_eur"] == pytest.approx(20_000_000.0 - 1_000_000.0 + 0.834 * 1_500_000.0)


def test_old_siope_codes_give_missing_macroaggregates_not_zero():
    """The 2016 files use old codes such as `U1101`: personnel cannot be read from them and must not appear as zero."""
    raw = pd.DataFrame([{"Descrizione Ente BDAP": "COMUNE DI CESENA", "Codice Gestionale Enti Locali": "U1101", "Importo cumulato": 1_000_000.0}])
    row = siope.parse_annual_summary(raw, year=2016).iloc[0]
    assert row["spese_correnti_cassa_eur"] == 1_000_000.0
    assert pd.isna(row["spesa_personale_cassa_eur"])


def test_old_2016_codes_read_titolo_3_as_loan_repayment_and_hide_macroaggregates():
    """In the old SIOPE codes (2016) Titolo 3 is the repayment of loans and the macroaggregates cannot be read."""
    raw = pd.DataFrame([
        {"Descrizione Ente BDAP": "COMUNE DI CESENA", "Codice Gestionale Enti Locali": "U1101", "Importo cumulato": 900.0},
        {"Descrizione Ente BDAP": "COMUNE DI CESENA", "Codice Gestionale Enti Locali": "U3101", "Importo cumulato": 40.0},
    ])
    row = siope.parse_annual_summary(raw, year=2016).iloc[0]
    assert row["rimborso_prestiti_cassa_eur"] == 40.0
    assert np.isnan(row["incremento_att_fin_cassa_eur"]) and np.isnan(row["spesa_personale_cassa_eur"])


def test_parse_revenue_summary_taxes_and_current_revenue():
    raw = pd.DataFrame([
        {"Descrizione Ente BDAP": "COMUNE DI CESENA", "Codice Gestionale Enti Locali": c, "Importo cumulato": v}
        for c, v in (("E1010106001", 100.0), ("E1010106002", 10.0), ("E1010108002", 1.0), ("E1010116001", 30.0), ("E1010151001", 20.0), ("E1010161001", 5.0),
                     ("E1030101001", 40.0), ("E2010101001", 60.0), ("E3010201001", 25.0), ("E4020101001", 999.0))
    ])
    r = siope.parse_revenue_summary(raw, year=2023).iloc[0]
    assert r["imu_ici_cassa_eur"] == 111.0 and r["addizionale_irpef_cassa_eur"] == 30.0 and r["tari_tarsu_cassa_eur"] == 25.0
    assert r["fondi_perequativi_cassa_eur"] == 40.0
    assert r["entrate_tributarie_cassa_eur"] == 206.0  # E1: 100+10+1+30+20+5+40
    assert r["entrate_correnti_cassa_eur"] == 206.0 + 60.0 + 25.0  # capital revenue (E4) is not current


def test_cash_rigidity_is_personnel_interest_and_repayment_over_current_revenue():
    spesa = pd.DataFrame([
        {"year": 2023, "entity_name": "A", "spesa_personale_cassa_eur": 30.0, "interessi_passivi_cassa_eur": 5.0, "rimborso_prestiti_cassa_eur": 15.0},
        {"year": 2016, "entity_name": "A", "spesa_personale_cassa_eur": np.nan, "interessi_passivi_cassa_eur": np.nan, "rimborso_prestiti_cassa_eur": 15.0},
    ])
    rev = pd.DataFrame([{"year": 2023, "entity_name": "A", "entrate_correnti_cassa_eur": 200.0, "tari_tarsu_cassa_eur": 50.0}, {"year": 2016, "entity_name": "A", "entrate_correnti_cassa_eur": 200.0, "tari_tarsu_cassa_eur": np.nan}])
    out = siope.add_cash_rigidity(spesa, rev).set_index("year")
    assert out.loc[2023, "rigidity_cash_pct"] == 25.0
    assert abs(out.loc[2023, "rigidity_cash_ex_tari_pct"] - 50 / 150 * 100) < 1e-9  # waste tax out of the denominator
    assert np.isnan(out.loc[2016, "rigidity_cash_pct"])  # 2016 macroaggregates unreadable: missing, not 7.5
