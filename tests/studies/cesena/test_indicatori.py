"""Indicator-PDF text parser (`studies.cesena.analysis.indicatori`) on a synthetic page shaped like `pdftotext -layout` output."""
from pathlib import Path

import pytest

from cifresignificative.studies.cesena.analysis import indicatori as i

TEXT = """ 1 Rigidità strutturale di bilancio
1.1 Incidenza spese rigide (ripiano disavanzo, personale    [ripiano disavanzo + Impegni (Macroaggregati 1.1 "Redditi" +                    29,896
    e debito) su entrate correnti                           IRAP – FPV) / (Accertamenti primi tre titoli)
 4.4 Spesa di personale procapite                             Impegni (Macroaggregato 1.1) / popolazione                                    245,012
 9.5 Indicatore annuale di tempestività dei pagamenti         Giorni effettivi                                                              -16,620
 2.1 Incidenza degli accertamenti                             Totale accertamenti / Stanziamenti iniziali 1.1 e 2.2                         109,467
 2.1 Duplicate line                                           x                                                                              1,000
Piano degli indicatori di bilancio  137
"""


def test_parse_indicators_takes_last_value_of_first_line_and_signs():
    d = i.parse_indicators(TEXT)
    assert d == {"1.1": 29.896, "4.4": 245.012, "9.5": -16.62, "2.1": 109.467}


def test_load_indicators_empty_without_pdfs(tmp_path: Path):
    assert i.load_indicators(tmp_path, tmp_path).empty


@pytest.mark.parametrize("name", ["Pianoindicatorierisultatiattesi-Rendiconto2024.pdf"])
def test_name_pattern_gives_the_year(name):
    m = i.NAME.search(name)
    assert m is not None and m[1] == "2024"
