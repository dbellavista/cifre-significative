"""Consolidated-accounts parser (`studies.cesena.analysis.consolidato`) on synthetic `pdftotext -layout` text, and the accounting identities on the real files when present."""
import math
import re
from pathlib import Path

import pytest

from cifresignificative.studies.cesena.analysis import consolidato as cs
from cifresignificative.studies.cesena.analysis.rendiconto import ARCHIVE_DIR

TEXT = """Comune di Cesena

                      BILANCIO CONSOLIDATO

                           2021

PERIMETRO
  ENERGIE PER LA CITTA'                   100,00%             Integrale
  TECHNE                                   50,00%          Proporzionale

                                 Conto economico               Capogruppo           Impatto dei valori consolidati       Bilancio consolidato
  Valore della produzione                                           € 90.000                       € 52.000                       € 142.000
  Risultato della gestione operativa                                 -€ 11.800                        € 1.600                       -€ 10.200
  Risultato di esercizio                                            € 4.900                       -€ 2.200                         € 2.700
  Crediti                                                            € 10                            € 20                            € 30
  Totale Crediti verso partecipanti                                  €0                              €0                              €0

   Conto economico                              Bilancio consolidato 2021 Bilancio consolidato 2020   Differenza
 Valore della produzione                                                         € 142.000             € 137.900          € 4.100
Risultato di esercizio comprensivo della quota di pertinenza di terzi                € 2.700                € 1.400         € 1.300
Risultato d'esercizio di pertinenza di terzi                                        -€ 240                  -€ 41          -€ 199
"""


def test_parse_statements_reads_group_and_prior_year_tables():
    t = cs.parse_statements(TEXT).set_index(["year", "item"])
    assert t.loc[(2021, "output"), ["capogruppo_eur", "impact_eur", "consolidated_eur"]].tolist() == [90000, 52000, 142000]
    assert t.loc[(2021, "operating_result"), "consolidated_eur"] == -10200 and t.loc[(2021, "net_result"), "capogruppo_eur"] == 4900
    assert t.loc[(2020, "output"), "consolidated_eur"] == 137900 and t.loc[(2020, "output"), "source"] == "prior-year column"
    assert t.loc[(2020, "net_result"), "consolidated_eur"] == 1400 and math.isnan(t.loc[(2020, "net_result"), "capogruppo_eur"])
    assert (2021, "receivables") in t.index and len(t) == 6  # the minorities' result and the credits-to-members line are not mistaken for a total


def test_parse_statements_empty_without_year_or_table():
    assert cs.parse_statements(TEXT.replace("BILANCIO CONSOLIDATO", "")).empty
    assert cs.parse_statements("BILANCIO CONSOLIDATO 2021\nnothing here").empty


def test_parse_perimeter_reads_share_and_method():
    p = cs.parse_perimeter(TEXT).set_index("body")
    assert p.loc["ENERGIE PER LA CITTA'", ["share_pct", "method"]].tolist() == [100.0, "Integrale"] and p.loc["TECHNE", "method"] == "Proporzionale"
    assert p["year"].eq(2021).all()


@pytest.mark.skipif(not (ARCHIVE_DIR / "files").exists(), reason="old-site files not downloaded")
def test_real_accounts_close_their_identities():
    st, per = cs.load_consolidated()
    assert {2018, 2019, 2020, 2021} <= set(st["year"]) and {2018, 2021} <= set(per["year"])
    for y, g in st.groupby("year"):
        v = g.set_index("item")["consolidated_eur"]
        assert v["output"] - v["costs"] == pytest.approx(v["operating_result"])
        assert v[["operating_result", "financial_result", "financial_adjustments", "extraordinary_result"]].sum() - v["taxes"] == pytest.approx(v["net_result"])
        assert v["fixed_assets"] + v["current_assets"] + v["prepayments"] == pytest.approx(v["total_assets"]) == pytest.approx(v["total_liabilities"]), y
    grp = st[st["source"] == "group table"]
    assert (grp["capogruppo_eur"] + grp["impact_eur"] - grp["consolidated_eur"]).abs().max() < 1


OCR = Path("data/raw/ocr/Bilancio_consolidato_2019.json")


@pytest.mark.skipif(not (OCR.exists() and (ARCHIVE_DIR / "files").exists()), reason="OCR cache or old-site files missing")
def test_ocr_2019_prior_year_column_agrees_with_the_2018_report():
    from cifresignificative.core.sources import ocr

    f = next(p for p in cs.consolidated_files(ARCHIVE_DIR) if "2019" in p.stem)
    both = cs.parse_ocr_statements(ocr.ocr_file(f, "it"), 2019).set_index(["year", "item"])["consolidated_eur"]
    st, _ = cs.load_consolidated(ARCHIVE_DIR)
    ref = st[st["year"] == 2018].set_index("item")["consolidated_eur"]
    assert len(ref) >= 20 and (both.loc[2018].reindex(ref.index) - ref).abs().max() < 1
    assert both[(2019, "net_result")] == -8174355


def test_parse_ocr_statements_rejects_a_misread_total():
    def tok(text, y, x=0):
        return {"text": text, "confidence": 1.0, "x1": x, "y1": y, "x2": x + 50, "y2": y + 20}
    rows = [("totale componenti positivi della gestione", "100.000", "90.000")]
    page = [tok(t, 30 * i, 100 * j) for i, r in enumerate(rows) for j, t in enumerate(r)]
    with pytest.raises(ValueError, match="do not close"):
        cs.parse_ocr_statements([page], 2019)


def test_review_file_round_trip_and_identities(tmp_path):
    def tok(text, y, x):
        return {"text": text, "confidence": 1.0, "x1": x, "y1": y, "x2": x + 50, "y2": y + 20}
    data = [("totale dell'attivo", "100.000", "90.000"), ("totale del passivo", "100.000", "80.000")]
    page = [tok(t, 40 * i, 100 * j) for i, r in enumerate(data) for j, t in enumerate(r)]
    reading = cs.read_ocr_statements([page], 2020)
    assert reading.values["total_assets"] == [100000, 90000] and ("assets = liabilities", 1) in reading.failures and not reading.ok
    path = cs.write_review(Path("x.pdf"), reading, tmp_path / "r.txt")
    assert "FAILED assets = liabilities for 2019: off by 10,000" in path.read_text() and "CHECK" in path.read_text()
    path.write_text(re.sub(r"(total_liabilities\s+\|\s+100000 \|\s+)80000", r"\g<1>90000", path.read_text()))
    assert cs.load_review(path)["total_liabilities"] == [100000, 90000] and cs.read_ocr_statements([page], 2020, cs.load_review(path)).values["total_liabilities"][1] == 90000
    assert cs.load_review(tmp_path / "missing.txt") == {}
    path.write_text("total_assets | ? | 1.077.413.797 |\n")
    assert cs.load_review(path)["total_assets"] == [None, 1077413797]


@pytest.mark.skipif(not (OCR.exists() and (ARCHIVE_DIR / "files").exists()), reason="OCR cache or old-site files missing")
def test_two_scans_agree_on_2019_and_the_2020_scan_on_the_2021_report():
    from cifresignificative.core.sources import ocr

    f19, f20 = (next(p for p in cs.consolidated_files(ARCHIVE_DIR) if str(y) in p.stem) for y in (2019, 2020))
    a, b = cs.read_ocr_statements(ocr.ocr_file(f19, "it"), 2019), cs.read_ocr_statements(ocr.ocr_file(f20, "it"), 2020)
    assert [it for it in cs.OCR_ITEMS if None not in (a.values[it][0], b.values[it][1]) and a.values[it][0] != b.values[it][1]] == []  # two independent readings of the same year
    assert a.ok and b.suspects() <= {("total_assets", 1)}  # the only unreadable token of the 2020 scan: fixed by the hand-checked review file
    assert cs.read_ocr_statements(ocr.ocr_file(f20, "it"), 2020, cs.load_review(cs.review_path(f20))).ok
    st, _ = cs.load_consolidated(ARCHIVE_DIR)
    ref = st[(st["year"] == 2020) & (st["source"] == "prior-year column")].set_index("item")["consolidated_eur"]
    assert len(ref) >= 20 and all(abs(ref[it] - b.values[it][0]) < 1 for it in ref.index if it in b.values)
