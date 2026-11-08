"""Half-year report parser (`studies.cesena.analysis.semestrale`) on synthetic `pdftotext -layout` text (pages split by form feed)."""
import pytest

from cifresignificative.studies.cesena.analysis import semestrale as sm

PAGE1 = """                  REPORT SOCIETA' PARTECIPATE AL 30/06/2026

                                   ENERGIE PER LA
                                   CITTA' S.p.A. (bilancio al    Cesena Fiera S.p.A.        TECHNE S.c.a.r.l.
                                                    30/6)        (situazione al 30/6)       (bilancio al 30/6)
1. Situazione Economico-patrimoniale:
1.a Conto Economico
Ricavi delle vendite                       1.524.241             8.095.192                     219.043
Valore Aggiunto                              649.697             1.516.008                    -216.874
Imposte                                      -24.121                -6.544                    -22.001
RISULTATO NETTO                               57.831             1.186.756                      6.306
differenza                                         0                     0                          0
"""
PAGE2 = """                  REPORT SOCIETA' PARTECIPATE AL 30/06/2026
                                              Situazione Contabile
                                 ENERGIE PER
                                 LA CITTA' S.p.A.     Cesena Fiera S.p.A.        TECHNE S.c.a.r.l.
RISULTATO NETTO                       57.831              1.186.756                     6.500
Posizione Finanziaria Netta (PFN)  1.189.612              3.268.700                    97.220
Debiti vs/banche (c/c passivi-mutui -altri
                                           0                238.917                         0
finanz.bancari)
PFN / EBITDA                         42,374                 2,766                    -0,151
"""


def test_parse_semestrale_reads_both_sheets_by_company_column():
    t = sm.parse_semestrale(PAGE1 + "\f" + PAGE2).set_index("company")
    assert t.index.tolist() == ["Energie per la Città", "Cesena Fiera", "Techne"] and t["year"].eq(2026).all()
    assert t.loc["Techne", "value_added_eur"] == -216874 and t.loc["Cesena Fiera", "sales_eur"] == 8095192
    assert t.loc["Techne", "net_result_eur"] == 6306 and t.loc["Techne", "net_result_sheet2_eur"] == 6500  # the two sheets can disagree: both are kept
    assert t.loc["Energie per la Città", "pfn_eur"] == 1189612 and t.loc["Techne", "pfn_over_ebitda"] == pytest.approx(-0.151)


def test_parse_semestrale_skips_wrapped_labels_and_needs_a_year_and_an_income_sheet():
    assert "Debiti" not in "".join(sm.parse_semestrale(PAGE1 + "\f" + PAGE2).columns)
    assert sm.parse_semestrale(PAGE1.replace("30/06/2026", "")).empty
    assert sm.parse_semestrale(PAGE2).empty
