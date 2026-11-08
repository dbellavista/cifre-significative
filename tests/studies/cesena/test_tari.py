"""Waste-tax sheet parser and the ex-TARI accrual ratio (`studies.cesena.analysis.tari`)."""
import pandas as pd
import pytest

from cifresignificative.studies.cesena.analysis import tari

TEXT = """Altro testo 2019 1.000,00
Scheda Tassa sui Rifiuti (TARI) – (somme accertate in bilancio)
 Anno         Numero                  Gettito di              Recupero                             NOTE
 2022     -      Domestiche          15.270.420,12             1.192.521,00
          45.565
 2023             ///                     ///                  1.040.596,00         prelievo sui rifiuti TARI è stato
 2024                                                          1.052.961,26
"""


def test_parse_tari_sheet_reads_levy_and_recovery_and_treats_slashes_as_no_levy():
    t = tari.parse_tari_sheet(TEXT).set_index("year")
    assert t.loc[2022, "levy_eur"] == pytest.approx(15270420.12) and t.loc[2022, "evasion_recovery_eur"] == 1192521
    assert t.loc[2023, "levy_eur"] == 0 and t.loc[2024, "evasion_recovery_eur"] == pytest.approx(1052961.26) and 2019 not in t.index


def test_parse_tari_sheet_empty_without_the_sheet():
    assert tari.parse_tari_sheet("nothing").empty


def test_add_ex_tari_removes_the_levy_only_where_it_exists():
    acc = pd.DataFrame({"year": [2022, 2023], "entrate_correnti_acc_eur": [100.0, 80.0], "spesa_personale_imp_eur": [20.0, 20.0], "interessi_passivi_imp_eur": [2.0, 2.0], "rimborso_prestiti_imp_eur": [2.0, 2.0]})
    out = tari.add_ex_tari(acc, pd.DataFrame({"year": [2022, 2023], "levy_eur": [20.0, 0.0]})).set_index("year")
    assert out.loc[2022, "rigidity_accrual_ex_tari_pct"] == pytest.approx(30) and out.loc[2023, "rigidity_accrual_ex_tari_pct"] == pytest.approx(30)
    assert tari.add_ex_tari(acc, pd.DataFrame({"year": [2023], "levy_eur": [0.0]})).set_index("year").loc[2022].isna()["rigidity_accrual_ex_tari_pct"]
