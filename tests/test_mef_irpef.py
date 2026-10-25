"""MEF IRPEF return statistics: parser quirks on small synthetic files, plus consistency checks on the processed table (skipped without it)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.sources import mef_irpef as M

PROCESSED = Path("data/processed/mef_irpef.parquet")

HEAD = "﻿Analisi statistiche - Dichiarazioni 2015 - Anno d'imposta 2014\nAnno presentazione:;2015\nData ultimo aggiornamento:;Marzo 2016\n\n"


def _write(tmp_path, name, text, enc="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(enc))
    return p


def test_number_format_and_suppression():
    assert M.num("1.234.567") == 1234567 and M.num(" -1.970.890") == -1970890 and M.num("0,37") == 0.37 and np.isnan(M.num("***"))


def test_canon_merges_formatting_variants():
    names = ["Reddito di spettanza dell' imprenditore in contabilita' ordinaria (*)", "Reddito di spettanza dell'imprenditore in contabilità ordinaria*", "Reddito di spettanza dell'imprenditore in contabilita'ordinaria (1)"]
    assert len({M.canon(n) for n in names}) == 1 and M.canon("Imposa netta") == "Imposta netta"


def test_class_bounds():
    assert M.class_bounds("minore di -1.000") == (-np.inf, -1000) and M.class_bounds("da 6.000 a 7.500") == (6000, 7500) and M.class_bounds("oltre 300.000") == (300000, np.inf)
    assert M.class_bounds("zero") == (0, 0) and M.class_bounds("da -1.000 a 0") == (-1000, 0)


def test_parse_file_handles_lost_byte_encoding_footnotes_and_totals(tmp_path):
    body = ("Classi di reddito complessivo in euro;Codice;Numero contribuenti;Reddito da pensione - Frequenza;Reddito da pensione - Ammontare;Reddito da pensione - Media;\n"
            "minore di -1.000;minore di -1000;10;5;100;20,00;\nda 0 a 1.000;0-1000;20;***;***;***;\nTOTALE;0;30;5;100;20,00;\n"
            "Ammontare e media in migliaia di euro\n(*) Comprensivo dei valori nulli\n")
    p = _write(tmp_path, "2015AAPFTOT0202.csv", HEAD + body)
    d = M.parse_file(p)
    assert set(d.tax_year) == {2014} and set(d.filing_year) == {2015} and set(d.group) == {"TOT"} and "migliaia" in d.notes.iloc[0]
    assert d[d.is_total & (d.measure == "freq") & (d.variable == "taxpayers")].value.iloc[0] == 30
    assert d[(d.class_label == "da 0 a 1.000") & (d.variable == "Reddito da pensione")].suppressed.all()
    # a file where the 0xA0 byte of "à" was dropped (lone 0xC3 before a space)
    raw = (HEAD + body.replace("Reddito da pensione", "Reddito in contabilità ordinaria")).encode("utf-8").replace("à".encode(), b"\xc3")
    p2 = tmp_path / "2015AAPFTOT0203.csv"
    p2.write_bytes(raw)
    assert M.parse_file(p2).variable.iloc[0] == "taxpayers" and "Reddito in contabilita' ordinaria" in set(M.parse_file(p2).variable)


def test_jump_flags_marks_a_one_year_spike():
    rows = [{"code": "X", "variable": "v", "class_label": "c", "tax_year": y, "measure": "amount_keur", "is_total": False, "value": 100.0 * 1000 if y != 2013 else 9e6} for y in range(2011, 2016)]
    d = pd.DataFrame(rows)
    assert list(d.tax_year[M.jump_flags(d)]) == [2013]


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed data missing")
def test_processed_table_is_consistent():
    d = pd.read_parquet(PROCESSED)
    assert d.tax_year.min() == 2008 and d.tax_year.max() == 2024 and (d.tax_year == d.filing_year - 1).all() and not d.duplicated(["code", "tax_year", "class_label", "variable_raw", "measure"]).any()
    tot = d[d.is_total & (d.measure == "freq") & (d.variable == "taxpayers") & (d.code == "AAPFTOT0106")].set_index("tax_year").value
    assert tot.loc[2014] == 40716548 and 40e6 < tot.min() and tot.max() < 44e6
    q = M.qa(d)
    ok = q[(q.suppressed == 0) & q.total.notna()]
    assert ((ok.class_sum_ratio - 1).abs() < 0.005).all()  # class rows add up to the TOTALE row wherever nothing is suppressed
    j = d[d.jump]
    assert ((j.tax_year == 2017) & (j.class_label == "minore di -1.000")).sum() > 50  # the 2017 lowest-class anomaly is flagged


AE_HEAD = ("﻿Analisi statistiche - Dichiarazioni 2015 - Anno d'imposta 2014\nAnno presentazione:;2015\nModello:;Persone fisiche in base al reddito prevalente\n"
           "Tematica:;Variabili tipiche della specie quadro RE\nClassificazione:;Classi di volume d'affari in euro\nData ultimo aggiornamento:;Marzo 2016\n\n")
AE_COLS = "Classi di volume d'affari in euro;Codice;Numero contribuenti;Totale compensi - Frequenza;Totale compensi - Ammontare;Totale compensi - Media;Totale spese - Frequenza;Totale spese - Ammontare;Totale spese - Media;\n"


def test_ae_table_with_second_total_row_and_group(tmp_path):
    rows = "da 0 a 2.500;03;10;10;100;10,00;9;20;2,22;\nTOTALE;0;10;10;100;10,00;9;20;2,22;\nTOTALE;0;***;***;***;***;***;***;***;\n"  # the second TOTALE row carries other items, here suppressed
    d = M.parse_file(_write(tmp_path, "2015AETPREA0406.csv", AE_HEAD + AE_COLS + rows))
    assert set(d.group) == {"PPRO"} and set(d.model) == {"AE"} and d.topic.iloc[0].endswith("quadro RE") and d.dimension.iloc[0] == "turnover"
    t = d[d.is_total & (d.measure == "amount_keur")].set_index("variable").value
    assert not d.duplicated(["variable", "measure", "class_label"]).any() and t["Totale compensi"] == 100 and t["Totale spese"] == 20  # the filled cell is kept


def test_ae_table_without_data_rows_is_empty_not_an_error(tmp_path):
    d = M.parse_file(_write(tmp_path, "2015AETPREA0406.csv", AE_HEAD + AE_COLS + "TOTALE;\n\nAmmontare e media in migliaia di euro\n"))
    assert d.empty and d.value.dtype == float


def test_professional_costs_bands_and_cost_share():
    from cifresignificative.analysis import irpef as IR
    def row(year, lo, hi, var, meas, v):
        return {"model": "AE", "group": "PPRO", "code": "AETPREA0406", "tax_year": year, "variable": var, "measure": meas, "is_total": False, "class_lo": lo, "class_hi": hi, "value": v}
    rows = []
    for lo, hi, fees, exp in [(0.0, 2500.0, 100.0, 10.0), (2500.0, 5000.0, 100.0, 30.0), (60000.0, 75000.0, 400.0, 100.0), (75000.0, 100000.0, 600.0, 150.0)]:
        rows += [row(2020, lo, hi, "Totale compensi", "amount_keur", fees), row(2020, lo, hi, "Totale compensi", "freq", 5.0), row(2020, lo, hi, "Totale spese", "amount_keur", exp), row(2020, lo, hi, "Totale spese", "freq", 4.0)]
    c = IR.professional_costs(pd.DataFrame(rows)).set_index("band")
    assert c.loc["0-25k", "cost_share"] == pytest.approx(0.2) and c.loc["0-25k", "filers"] == 10 and c.loc["60-75k", "cost_share"] == pytest.approx(0.25) and c.loc["75-100k", "fees_keur"] == 600
    bad = pd.DataFrame([row(2020, 20000.0, 30000.0, "Totale compensi", "amount_keur", 1.0), row(2020, 20000.0, 30000.0, "Totale compensi", "freq", 1.0), row(2020, 20000.0, 30000.0, "Totale spese", "amount_keur", 1.0), row(2020, 20000.0, 30000.0, "Totale spese", "freq", 1.0)])
    with pytest.raises(AssertionError):
        IR.professional_costs(bad)  # a class that straddles the 25,000 band edge
