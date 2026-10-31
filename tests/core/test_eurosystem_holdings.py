"""Eurosystem purchases loader: block parsing, number formats and the check against the ECB's cumulative column."""
import pandas as pd
import pytest

from cifresignificative.core.sources import eurosystem_holdings as eh

PSPP = """Breakdown of debt securities under the Public Sector Purchase Programme,,,,
Monthly net purchases,,,,
,31/03/2015,30/04/2015,30/05/2015,cumulative net purchases
Italy,100,"-2,000",50,"-1,850"
Total,300,200,100,600
In EUR millions. ,,,,
,,,,
WAM of PSPP portfolio holdings,,,,
,31/03/2015,30/04/2015,30/05/2015,
Italy,9.06,8.4,,
Remaining Weighted Average Maturity (WAM) in years ,,,,
"""


def test_parse_blocks_and_formats(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text(PSPP)
    d, cum = eh.parse(p, "PSPP")
    npur = d[(d.measure == "net_purchases") & (d.country == "Italy")].set_index("period").value
    assert npur.tolist() == [100, -2000, 50] and npur.index[0] == pd.Timestamp("2015-03-31")
    assert cum["Italy"] == -1850 and cum["Total"] == 600
    w = d[d.measure == "wam_portfolio"]
    assert w.value.tolist() == [9.06, 8.4] and set(w.country) == {"Italy"}  # the empty cell is dropped, the footnote is not a country


def test_mon_yy_headers():
    assert eh._dates(["Feb-24", "Mar-20"]).tolist() == [pd.Timestamp("2024-02-29"), pd.Timestamp("2020-03-31")]


@pytest.mark.skipif(not (eh.DIR / eh.FILES["PSPP"]).exists(), reason="ECB files not present")
def test_real_files_match_ecb_cumulative():
    c = eh.cumulative_check()
    assert c["diff"].abs().max() <= 10  # rounding of the monthly figures only
    d = eh.load()
    it = d[(d.country == "Italy") & (d.measure == "stock")].groupby("programme").value.last()
    assert 270_000 < it["PSPP"] < 285_000 and 215_000 < it["PEPP"] < 225_000
    assert d.period.max() >= pd.Timestamp("2026-09-30") and set(d.programme) == {"PSPP", "PEPP"}
