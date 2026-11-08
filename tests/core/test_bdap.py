"""Unit tests for `core.sources.bdap` (D.P.R. 194/1996 + D.Lgs. 118/2011 harmonized municipal budgets)."""
import io
import zipfile
from pathlib import Path

import numpy as np

from cifresignificative.core.sources import bdap


def _make_mock_storico_zip(tmp_path: Path) -> Path:
    zp = tmp_path / "040007_CESENA--2080320060.zip"
    q2_t1 = (
        '"VOCI";"ACCERTAMENTI";"RISCOSSIONI IN C/C COMPETENZA";"RISCOSSIONI IN C/RESIDUI"\n'
        '"Categoria 1 - Imposte";"40.000.000,00";"36.000.000,00";"3.000.000,00"\n'
        '"- I.M.U. fattispecie diversa da abitazione principale";"22.000.000,00";"22.000.000,00";"0,00"\n'
        '"Addizionale IRPEF";"6.000.000,00";"3.000.000,00";"2.500.000,00"\n'
        '"Tassa sui rifiuti (Tari) - art. 1 co.639 L. 147/13";"12.000.000,00";"10.000.000,00";"1.000.000,00"\n'
        '"TOTALE ENTRATE TRIBUTARIE";"60.000.000,00";"50.000.000,00";"6.000.000,00"\n'
    )
    q2_t2 = (
        '"VOCI";"ACCERTAMENTI";"RISCOSSIONI IN C/C COMPETENZA";"RISCOSSIONI IN C/RESIDUI"\n'
        '"TOTALE ENTRATE DERIVANTI DA CONTRIBUTI E TRASFERIMENTI CORRENTI";"5.000.000,00";"4.000.000,00";"500.000,00"\n'
    )
    q2_t3 = (
        '"VOCI";"ACCERTAMENTI";"RISCOSSIONI IN C/C COMPETENZA";"RISCOSSIONI IN C/RESIDUI"\n'
        '"TOTALE ENTRATE EXTRATRIBUTARIE";"20.000.000,00";"16.000.000,00";"2.000.000,00"\n'
    )
    q2_t4 = (
        '"VOCI";"ACCERTAMENTI";"RISCOSSIONI IN C/C COMPETENZA";"RISCOSSIONI IN C/RESIDUI"\n'
        '"Categoria 1 - Alienazione di beni patrimoniali (2)";"2.000.000,00";"1.500.000,00";"100.000,00"\n'
        '"Categoria 2 - Trasferimenti di capitali dallo Stato";"3.000.000,00";"2.500.000,00";"200.000,00"\n'
        '"Categoria 3 - Trasferimenti di capitali dalla Regione";"4.000.000,00";"3.000.000,00";"300.000,00"\n'
        '"TOTALE ENTRATE DERIVANTI DA ALIENAZIONE, TRASFERIMENTI DI CAPITALI E DA RISCOSSIONI DI CREDITI";"15.000.000,00";"12.000.000,00";"1.000.000,00"\n'
    )
    q2_t5 = (
        '"VOCI";"ACCERTAMENTI";"RISCOSSIONI IN C/C COMPETENZA";"RISCOSSIONI IN C/RESIDUI"\n'
        '"Categoria 3 - Assunzioni di mutui e prestiti di cui:";"1.800.000,00";"1.000.000,00";"0,00"\n'
        '"TOTALE ENTRATE DERIVANTI DA ACCENSIONE DI PRESTITI";"1.800.000,00";"1.000.000,00";"0,00"\n'
    )
    q3 = (
        '"VOCI";"Impegni";"Pagamenti in c/competenza";"Pagamenti in c/residui"\n'
        '"TITOLO I - SPESE CORRENTI al netto della spesa per alimentazione del fondo do solidarietà comunale (FSC)";"78.000.000,00";"65.000.000,00";"10.000.000,00"\n'
        '"Titolo II - Spese in C/Capitale";"16.000.000,00";"5.000.000,00";"8.000.000,00"\n'
        '"Titolo III - Spese per rimborso di prestiti";"5.000.000,00";"5.000.000,00";"0,00"\n'
        '"Totale Generale delle Spese";"110.000.000,00";"85.000.000,00";"18.000.000,00"\n'
    )
    q4 = (
        '"FUNZIONI E SERVIZI / INTERVENTI CORRENTI";"Personale";"Interessi passivi e oneri finanziari diversi";"Totale"\n'
        '"Totale";"20.000.000,00";"2.000.000,00";"78.000.000,00"\n'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("csv/CESENA--2080320060/2014/Consuntivo/02/quadro-2-entrate-titolo-i-entrate-tributarie.csv", q2_t1)
        z.writestr("csv/CESENA--2080320060/2014/Consuntivo/02/quadro-2-entrate-titolo-ii-entrate-derivanti-da-contributi-e-trasferimenti-correnti.csv", q2_t2)
        z.writestr("csv/CESENA--2080320060/2014/Consuntivo/02/quadro-2-entrate-titolo-iii-entrate-extratributarie.csv", q2_t3)
        z.writestr("csv/CESENA--2080320060/2014/Consuntivo/02/quadro-2-entrate-titolo-iv-entrate-derivanti-da-alienazione.csv", q2_t4)
        z.writestr("csv/CESENA--2080320060/2014/Consuntivo/02/quadro-2-entrate-titolo-v-entrate-derivanti-da-accensione.csv", q2_t5)
        z.writestr("csv/CESENA--2080320060/2014/Consuntivo/03/quadro-3-riepilogo-generale-delle-spese.csv", q3)
        z.writestr("csv/CESENA--2080320060/2014/Consuntivo/04/quadro-4-a-impegni.csv", q4)
    zp.write_bytes(buf.getvalue())
    return zp


def test_parse_storico_zip_and_break_flag(tmp_path: Path):
    zp = _make_mock_storico_zip(tmp_path)
    rows = bdap.parse_storico_zip(zp, "040007", years=(2014,))
    assert len(rows) == 1
    r = rows[0]
    assert r["regime"] == "dpr194"
    assert r["flag"] == "break_dlgs118"
    assert np.isclose(r["entrate_correnti_comp_eur"], 85_000_000.0)
    assert np.isclose(r["spesa_personale_comp_eur"], 20_000_000.0)
    assert np.isclose(r["imu_ici_comp_eur"], 22_000_000.0)
    assert np.isclose(r["alienazioni_comp_eur"], 2_000_000.0)
    assert np.isclose(r["contributi_investimenti_comp_eur"], 7_000_000.0)
    assert np.isclose(r["accensione_mutui_comp_eur"], 1_800_000.0)


def test_parse_armonizzati_handles_parent_nan_pitfall():
    """Verify that summing child Programs recovers capital expenditure when the parent Mission node has `NaN`."""
    arm_bundle = {
        "2021_consuntivo": {
            "entrate_competenza": [
                {"slug": "ccox-quadro-2-11", "values": [{"2021": {"abs": 100_000_000.0}}]},
                {
                    "slug": "ccox-quadro-2-1",
                    "values": [{"2021": {"abs": 35_000_000.0}}],
                    "children": [
                        {"slug": "ccox-quadro-2-1-20", "values": [{"2021": {"abs": 30_000_000.0}}]},
                        {"slug": "ccox-quadro-2-1-30", "values": [{"2021": {"abs": 5_000_000.0}}]},
                    ],
                },
                {
                    "slug": "ccox-quadro-2-2",
                    "values": [{"2021": {"abs": 60_000_000.0}}],
                    "children": [{"slug": "ccox-quadro-2-2-41", "values": [{"2021": {"abs": 45_000_000.0}}]}],
                },
                {"slug": "ccox-quadro-2-3", "values": [{"2021": {"abs": 10_000_000.0}}]},
                {"slug": "ccox-quadro-2-4", "values": [{"2021": {"abs": 15_000_000.0}}]},
                {
                    "slug": "ccox-quadro-2-5",
                    "values": [{"2021": {"abs": 12_000_000.0}}],
                    "children": [
                        {"slug": "ccox-quadro-2-5-230", "values": [{"2021": {"abs": 7_000_000.0}}]},
                        {"slug": "ccox-quadro-2-5-250", "values": [{"2021": {"abs": 3_000_000.0}}]},
                    ],
                },
                {
                    "slug": "ccox-quadro-2-7",
                    "values": [{"2021": {"abs": 4_000_000.0}}],
                    "children": [{"slug": "ccox-quadro-2-7-340", "values": [{"2021": {"abs": 3_500_000.0}}]}],
                },
            ],
            "entrate_cassa": [
                {"slug": "ccox-quadro-2-11", "values": [{"2021": {"abs": 95_000_000.0}}]},
                {"slug": "ccox-quadro-2-2", "values": [{"2021": {"abs": 54_000_000.0}}]},
                {"slug": "ccox-quadro-2-3", "values": [{"2021": {"abs": 9_500_000.0}}]},
                {"slug": "ccox-quadro-2-4", "values": [{"2021": {"abs": 13_000_000.0}}]},
                {"slug": "ccox-quadro-2-5", "values": [{"2021": {"abs": 11_000_000.0}}]},
            ],
            "spese_competenza": [
                {"slug": "ccox-quadro-4-26", "values": [{"2021": {"abs": 92_000_000.0}}]},
                {
                    "slug": "ccox-quadro-4-5",  # M04 Education with parent NaN in investimenti_value
                    "values": [{"2021": {"abs": 12_000_000.0}}],
                    "composition": {"correnti_value": 10_000_000.0, "investimenti_value": float("nan"), "altro_value": 0.0},
                    "children": [
                        {
                            "slug": "ccox-quadro-4-5-1",
                            "values": [{"2021": {"abs": 7_000_000.0}}],
                            "composition": {"correnti_value": 5_000_000.0, "investimenti_value": 2_000_000.0, "altro_value": 0.0},
                        },
                        {
                            "slug": "ccox-quadro-4-5-2",
                            "values": [{"2021": {"abs": 5_000_000.0}}],
                            "composition": {"correnti_value": 5_000_000.0, "investimenti_value": float("nan"), "altro_value": 0.0},
                        },
                    ],
                },
            ],
            "spese_cassa": [],
        }
    }
    b_rows, m_rows = bdap.parse_armonizzati_bundle(arm_bundle, "040007")
    assert len(b_rows) == 1
    assert np.isclose(b_rows[0]["spese_correnti_comp_eur"], 10_000_000.0)
    assert np.isclose(b_rows[0]["spese_capitale_comp_eur"], 2_000_000.0)
    assert np.isclose(b_rows[0]["contributi_investimenti_comp_eur"], 7_000_000.0)
    assert np.isclose(b_rows[0]["alienazioni_comp_eur"], 3_000_000.0)
    assert np.isclose(b_rows[0]["accensione_mutui_comp_eur"], 3_500_000.0)
    assert np.isclose(b_rows[0]["fpv_capitale_comp_eur"], 30_000_000.0)
    assert np.isclose(b_rows[0]["avanzo_amministrazione_comp_eur"], 5_000_000.0)
    m04 = next(m for m in m_rows if m["mission_code"] == "M04")
    assert np.isclose(m04["capitale_comp_eur"], 2_000_000.0)
