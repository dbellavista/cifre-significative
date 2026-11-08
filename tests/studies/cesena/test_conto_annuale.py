"""Conto annuale summary parser (`studies.cesena.analysis.conto_annuale`) on synthetic `pdftotext -layout` text."""
from pathlib import Path

import pytest

from cifresignificative.studies.cesena.analysis import conto_annuale as ca

TEXT = """Tipo Rilevazione : CONSUNTIVAZIONE SPESE                       Anno : 2024
DIRIGENTI                                    8        9       9     7,5       9     9,01     665.588   799.053   922.594    0   599   74.528
Totale                                     516      506     540  478,65  494,29   502,27  15.310.770  15.323.719  16.190.799  505.925  24.933  96.276
                                           Tabella 14                      7.791.327    7.592.723   7.654.196
                       Totale costo annuo del lavoro(Tab.12+13+14)       23.102.097   22.916.442  23.844.995
"""


def test_parse_summary_reads_three_years_and_italian_numbers():
    t = ca.parse_summary(TEXT)
    assert t["year"].tolist() == [2022, 2023, 2024] and t["heads"].tolist() == [516, 506, 540]
    assert t["fte"].tolist() == pytest.approx([478.65, 494.29, 502.27]) and t["pay_eur"].iloc[0] == 15310770
    assert t["labour_cost_eur"].tolist() == [23102097, 22916442, 23844995] and t["arrears_eur"].iloc[2] == 96276


def test_parse_summary_empty_when_layout_missing():
    assert ca.parse_summary("nothing here").empty


def test_load_prefers_the_later_survey(tmp_path: Path, monkeypatch):
    d = tmp_path / "files" / "1"
    d.mkdir(parents=True)
    for n in ("a_CONTOANNUALE2023.pdf", "b_contoannuale2024.pdf"):
        (d / n).write_bytes(b"x")
    older = TEXT.replace("2024", "2023").replace("23.102.097", "1").replace("516 ", "999 ")
    monkeypatch.setattr(ca, "pdf_text", lambda p: older if "2023" in p.name else TEXT)
    out = ca.load_conto_annuale(tmp_path)
    assert out["year"].tolist() == [2021, 2022, 2023, 2024] and out.set_index("year").loc[2022, "heads"] == 516


DETAIL = """Tipo Rilevazione : CONSUNTIVAZIONE SPESE                       Anno : 2024
        Personale a tempo indeterminato (Tab.1) - Spese medie pro-capite annue in euro
     DIRIGENTI          7,5          9          9,01     88.785     88.717    94.169     47.489     47.853    49.184     41.295     40.864    44.984        0          66         6.994         0           0         1.282
  CATEGORIA D        121,72        0           0       35.464       0         0        27.931       0         0        7.533        0         0         1.192         0          0            2           0          0
  ISTRUTTORI                   0            256,12           257,37           0             29.073          30.033             0            24.392          24.373             0            4.681           5.659              0        23          47
1. Le Spese medie annue per ciascuna Categoria
                               Personale Flessibile (Tab. 2A) - Dati riepilogativi dell'ultimo triennio
CD - CATEGORIA D               6              0              0
TOTALE                         65             62             40
                               Personale Flessibile (Tab. 2 e SI1) - Dati riepilogativi dell'ultimo triennio
Personale a tempo                39,25            45,48            40,76       Retribuzioni, codice P015        1.165.162      1.184.311   1.143.122   valore medio                     29.686         26.040      28.045
Lavoratori Interinali             0,1              1,73            5,08        Retribuzioni, codice               5.512          42.579    101.953     valore medio                     55.120         24.612      20.069
N. contratti co.co.co (SI1)        1                0                -        Oneri per co.co.co.                2.407            0          -         valore medio riferito ai          2.407          -         -
           Contrattazione Integrativa - Dati riepilogativi dell'ultimo triennio
    PERSONALE NON DIRIGENTE                  Fondo risorse decentrate            2.544.455          2.611.448          2.666.372         2.544.455            2.611.448           2.666.371
    PERSONALE NON DIRIGENTE              Posizioni organizzative (bilancio)       177.797               -                  -              176.698                 -                   -
             DIRIGENTI                Risorse per la retribuzione di posizione    379.261            379.261           453.899            372.993             379.261              453.899
                                                                        Tabella di Riconciliazione
"""


def test_parse_flexible_ignores_codes_and_reads_blank_cells():
    t = ca.parse_flexible(DETAIL).set_index(["kind", "year"])
    assert t.loc[("fixed_term", 2024), "cost_eur"] == 1143122 and t.loc[("fixed_term", 2023), "units"] == pytest.approx(45.48)
    assert t.loc[("agency", 2022), "avg_cost_eur"] == 55120 and t.loc[("cococo", 2024), "cost_eur"] == 0
    assert t.loc[("fixed_term_heads", 2023), "units"] == 62  # the TOTALE row of Tab. 2A, not the CD category row


def test_parse_levels_drops_levels_absent_from_the_year_and_labels_the_regime():
    t = ca.parse_levels(DETAIL).set_index(["level", "year"])
    assert ("cat_D", 2023) not in t.index and t.loc[("cat_D", 2022), "avg_total_eur"] == 35464 and t.loc[("cat_D", 2022), "regime"] == "categorie"
    assert t.loc[("area_istruttori", 2024), "avg_fixed_eur"] == 24373 and t.loc[("area_istruttori", 2024), "regime"] == "aree"
    assert t.loc[("dirigenti", 2024), "avg_accessory_eur"] == 44984  # DIRIGENTI is not the row `DIRIGENTI E ALTE...`


def test_parse_funds_reads_blank_as_zero_and_both_columns():
    t = ca.parse_funds(DETAIL).set_index(["fund", "year"])
    assert t.loc[("decentralised_fund", 2024), "resources_eur"] == 2666372 and t.loc[("decentralised_fund", 2024), "used_eur"] == 2666371
    assert t.loc[("po", 2023), "resources_eur"] == 0 and t.loc[("dirigenti", 2022), "used_eur"] == 372993
