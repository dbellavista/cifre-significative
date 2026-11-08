"""DUP staffing-plan parsers (`studies.cesena.analysis.fabbisogni`) on synthetic `pdftotext -layout` text."""
import pandas as pd
import pytest

from cifresignificative.studies.cesena.analysis import fabbisogni as fb

LIMITS = """LIMITI SPESA DEL PERSONALE (art. 1, comma 557, legge n. 296/2006)
                               Consuntivo        Bilancio 2024        Bilancio          Bilancio          Bilancio
                                  2023             assestato            2025             2026              2027
          - impegni personale (macro 01)        21.701.609,33     23.905.139,49    22.913.337,00     22.815.387,00     22.815.387,00
          - Tirocini (cap. 41150/03)                  0,00          3.000,00        3.000,00          3.000,00          3.000,00
                         Totale spesa personale    19.434.575,39   20.449.985,69   19.570.337,53   19.760.496,53   19.800.496,53
                     Personale trasferito ad Unione    € 3.024.850   € 3.024.850   € 3.024.850   € 3.024.850   € 3.024.850
                         Totale spesa personale    22.459.425,39   23.474.835,69   22.595.187,53   22.785.346,53   22.825.346,53
"""
STAFF = """   Staff al Segretario generale                                 0  3  7  0  10  10  0
                       Personale a tempo indeterminato        1          98         277         145           6        527       481      46                                          100,00
                         personale a tempo determinato                     3         2           13           3         21        0        0        21        3       6     0
organico del personale in servizio a tempo indeterminato alla data del 01/07/2025 al quale
"""


def test_parse_limits_orders_years_when_the_header_wraps_and_reads_euro_signs():
    t = fb.parse_limits(LIMITS).set_index("year")
    assert t.index.tolist() == [2023, 2024, 2025, 2026, 2027] and t["outturn"].tolist() == [True, False, False, False, False]
    assert t.loc[2023, "total_eur"] == pytest.approx(22459425.39) and t.loc[2024, "municipal_eur"] == pytest.approx(20449985.69)
    assert t["unione_eur"].eq(3024850).all() and t.loc[2024, "commitments_eur"] == pytest.approx(23905139.49)


def test_parse_limits_empty_when_a_row_is_missing():
    assert fb.parse_limits(LIMITS.replace("Personale trasferito", "x").rsplit("Totale spesa personale", 1)[0]).empty


def test_parse_staff_checks_the_sums():
    s = fb.parse_staff(STAFF)
    assert s is not None
    assert s["ref_date"] == "2025-07-01" and s["heads"] == 527 and s["full_time"] == 481 and s["fixed_term"] == 21
    assert fb.parse_staff(STAFF.replace("527", "530")) is None


def test_parse_retirements():
    t = fb.parse_retirements("ANNO CESSAZIONI ECONOMIA\n 2026 7 208.847,25\n 2027 6 218.528,09\n TOTALE 13 427.375,34\n")
    assert t["year"].tolist() == [2026, 2027] and t["saving_eur"].iloc[0] == pytest.approx(208847.25)


def test_plan_vs_outturn_takes_the_last_plan_and_first_outturn():
    rows = [("a", 0, 2023, False, 100.0), ("b", 1, 2023, False, 80.0), ("c", 2, 2023, True, 60.0), ("d", 3, 2023, True, 61.0), ("c", 2, 2024, False, 50.0)]
    lim = pd.DataFrame(rows, columns=["source", "order", "year", "outturn", "total_eur"])
    t = fb.plan_vs_outturn(lim)
    assert t["year"].tolist() == [2023] and t["plan_eur"].iloc[0] == 80 and t["gap_pct"].iloc[0] == pytest.approx(-25)
