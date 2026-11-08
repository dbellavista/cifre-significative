"""Controlled-company risk report parser (`studies.cesena.analysis.risk_reports`) on synthetic `pdftotext -layout` text."""
import pytest

from cifresignificative.studies.cesena.analysis import risk_reports as rr

TEXT = """                                                 Filiera
Soglie di allarme da bilancio    Energie per Ortofrutticola                                         Romagna
2025                               la Citta   Romagnola       ATR     Techne   Unica   LEPIDA spa    acque
Gestione operativa della società
negativa per tre esercizi
consecutivi                          NO           NO           NO      NO       NO        NO          NO
Perdite di esercizio cumulate
patrimonio netto in misura
superiore al 15%                        NO         NO          NO      NO       NO        NO          NO
La relazione redatta dalla società
continuità aziendale                    NO         NO          NO      NO       NO        NO          SI
Indice di struttura finanziaria,
superiore del 20%                      1,47       0,62        1,10     4,56    1,01       1,42       1,10
Indice di disponibilità finanziaria,
inferiore a 1                          2,43       0,38        3,36     1,33    1,42       1,73       2,70
Il peso degli oneri finanziari
sul fatturato e' superiore al 5%       0,00%     3,23%        1,45%   0,02%    0,91%     0,18%       6,00%
"""


def test_parse_risk_names_columns_by_position_and_applies_label_thresholds():
    t = rr.parse_risk(TEXT).set_index("company")
    assert t.index.tolist() == ["Energie per la Città", "Filiera Ortofrutticola Romagnola", "ATR", "Techne", "Unica Reti", "Lepida", "Romagna Acque"]
    assert t["year"].eq(2025).all() and t.loc["Filiera Ortofrutticola Romagnola", "liquidity_index"] == pytest.approx(0.38)
    assert t.loc["Filiera Ortofrutticola Romagnola", ["structure_alarm", "liquidity_alarm"]].all() and not t.loc["ATR", ["structure_alarm", "liquidity_alarm"]].any()
    assert t.loc["Romagna Acque", "going_concern_doubt"] and t.loc["Romagna Acque", "charges_alarm"] and not t.loc["Lepida", "going_concern_doubt"]


def test_parse_risk_empty_when_a_row_is_missing():
    assert rr.parse_risk(TEXT.replace("inferiore a 1                          2,43", "inferiore a 1")).empty
