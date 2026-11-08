"""Rendiconto CSV parser (`studies.cesena.analysis.rendiconto`) on synthetic files shaped like the portal's."""
from pathlib import Path

import pytest

from cifresignificative.studies.cesena.analysis import rendiconto as r

ENTRATE = [["RENDICONTO 2024 - ENTRATE", "", "", ""], ["", "", "", ""], ["Titolo", "Tipologia", "Accertamenti", "Riscossioni"],
           ["Fondo pluriennale vincolato", "", "100", "0"],
           ["1-Entrate correnti", "TIPOLOGIA 101 - Imposte", "600,5", "500"], ["1-Entrate correnti", "TIPOLOGIA 102 - Sanita", "0", "0"],
           ["3-Entrate extratributarie", "TIPOLOGIA 100 - Vendita", "399,5", "300"],
           ["4-Entrate in conto capitale", "TIPOLOGIA 200 - Contributi", "5000", "4000"]]
SPESE = [["RENDICONTO 2024 - SPESE"], [""], ["Titolo", "Macroaggregato", "Mission A Impegnato", "A FPV", "A Pagato", "Totale Impegnato", "Totale FPV", "Totale Pagato"],
         ["1-Spese correnti", "MACROAGGR. 1 - Redditi", "1", "0", "1", "200", "10", "190"], ["1-Spese correnti", "MACROAGGR. 7 - Interessi", "0", "0", "0", "20", "0", "20"],
         ["1-Spese correnti", "MACROAGGR. 10 - Altre", "0", "0", "0", "300", "0", "300"], ["TOTALE TITOLO 1 - Spese correnti", "", "1", "0", "1", "520", "10", "510"],
         ["TOTALE TITOLO 4 - Rimborso di prestiti", "", "0", "0", "0", "30", "0", "30"]]


@pytest.mark.parametrize(("raw", "value"), [("1.234,56", 1234.56), ("1234,56", 1234.56), ("0", 0.0), ("", 0.0), ("12", 12.0)])
def test_num_reads_both_italian_formats(raw, value):
    assert r.num(raw) == pytest.approx(value)


def test_accrual_row_rigidity_uses_current_revenue_only():
    row = r.accrual_row(ENTRATE, SPESE)
    assert row is not None and row["year"] == 2024
    assert row["entrate_correnti_acc_eur"] == pytest.approx(1000.0)  # capital revenue (5000) and FPV excluded
    assert row["spesa_personale_imp_eur"] == 200 and row["interessi_passivi_imp_eur"] == 20 and row["rimborso_prestiti_imp_eur"] == 30
    assert row["spese_correnti_imp_eur"] == 520  # personnel macroaggregate 10 is not caught by the macroaggregate 1 prefix
    assert row["rigidity_accrual_pct"] == pytest.approx(25.0)


def test_old_layout_without_tipologia_is_skipped():
    old = [["Titolo", "Categoria", "Accertamenti", "Riscossioni"], ["1-Entrate tributarie", "CATEGORIA 1 - Imposte", "1.000,00", "900,00"]]
    assert r.accrual_row(old, SPESE) is None


def test_load_accruals_pairs_files_and_skips_unreadable(tmp_path: Path):
    d = tmp_path / "files" / "1"
    d.mkdir(parents=True)
    for name, rows in (("9_ENTRATERENDICONTO2024C.csv", ENTRATE), ("10_SPESERENDICONTO2024C.csv", SPESE)):
        (d / name).write_text("\n".join(";".join(x) for x in rows), encoding="latin1")
    lone = tmp_path / "files" / "2"
    lone.mkdir()
    (lone / "11_ENTRATERENDICONTO2023.csv").write_text("RENDICONTO 2023 - ENTRATE;;;\n", encoding="latin1")  # no spese sibling
    out = r.load_accruals(tmp_path, tmp_path / "no_archive")
    assert out["year"].tolist() == [2024] and out["rigidity_accrual_pct"].iloc[0] == pytest.approx(25.0)
    assert r.load_accruals(tmp_path / "missing", tmp_path / "no_archive").empty


OLD_ENTRATE = [["ENTRATE", "", "", ""], ["DATI DI RENDICONTO ANNO: 2021", "", "", ""], ["TITOLO/TIPOLOGIA", "DENOMINAZIONE", "COMPETENZA", "CASSA"], ["", "Fondo pluriennale vincolato", "000000000100,00", ""],
               ["10000", "TOTALE TITOLO 1: Entrate tributarie", "000000000600,50", "000000000500,00"], ["20000", "TOTALE TITOLO 2: Trasferimenti", "000000000000,00", "000000000000,00"],
               ["30000", "TOTALE TITOLO 3: Extratributarie", "000000000399,50", "000000000300,00"], ["40000", "TOTALE TITOLO 4: Conto capitale", "000000005000,00", "000000004000,00"]]
OLD_SPESE = [["SPESA", "", "", ""], ["DATI DI RENDICONTO ANNO: 2021", "", "", ""], ["", "TITOLI E MACROAGGREGATI DI SPESA", "Servizi generali", "", "", "Giustizia", "", ""], ["", "", "Competenza", "FPV", "Cassa", "Competenza", "FPV", "Cassa"],
             ["", "TITOLO 1", "Spese correnti", "", "", "", "", ""], ["101", "Redditi da lavoro dipendente", "000000000001,00", "0", "1", "000000000200,00", "000000000010,00", "000000000190,00"],
             ["107", "Interessi passivi", "0", "0", "0", "000000000020,00", "0", "000000000020,00"], ["100", "Totale TITOLO 1", "1", "0", "1", "000000000520,00", "000000000010,00", "000000000510,00"],
             ["", "TITOLO 4", "Rimborso di prestiti", "", "", "", "", ""], ["400", "Totale TITOLO 4", "0", "0", "0", "000000000030,00", "0", "000000000030,00"]]


def test_archive_row_reads_zero_padded_old_layout():
    row = r.archive_row(OLD_ENTRATE, OLD_SPESE)
    assert row is not None and row["year"] == 2021
    assert row["entrate_correnti_acc_eur"] == pytest.approx(1000.0)  # Titoli 1-3 only, the FPV line and Titolo 4 revenue are out
    assert row["spesa_personale_imp_eur"] == 200 and row["interessi_passivi_imp_eur"] == 20 and row["rimborso_prestiti_imp_eur"] == 30 and row["spese_correnti_imp_eur"] == 520
    assert row["rigidity_accrual_pct"] == pytest.approx(25.0)


def test_archive_row_none_when_a_total_is_missing():
    assert r.archive_row([x for x in OLD_ENTRATE if x[0] != "20000"], OLD_SPESE) is None
    assert r.archive_row(OLD_ENTRATE, [x for x in OLD_SPESE if x[0] != "107"]) is None
    assert r.archive_row(OLD_ENTRATE, [x for x in OLD_SPESE if not x[1].startswith("Totale TITOLO 4")]) is None


def test_load_accruals_joins_archive_years_before_the_portal_ones(tmp_path: Path):
    a = tmp_path / "archive" / "files" / "bilancioconsuntivo2021"
    a.mkdir(parents=True)
    for name, rows in (("Entrate_del_rendiconto_2021.csv", OLD_ENTRATE), ("Spese_del_rendiconto_2021.csv", OLD_SPESE)):
        (a / name).write_text("\n".join(";".join(x) for x in rows), encoding="latin1")
    assert r.load_archive_accruals(tmp_path / "archive")["year"].tolist() == [2021]
    assert r.load_accruals(tmp_path / "none", tmp_path / "archive")["year"].tolist() == [2021]
