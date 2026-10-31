"""Draft budgetary plans of the euro-area members: parser identity check and the committed table."""
import pandas as pd
import pytest

from cifresignificative.core.sources import dbp_euro as DB
from cifresignificative.studies.italy_productivity.analysis import budget_xc as BX


def test_parse_plan_finds_the_government_row_by_identity(tmp_path, monkeypatch):
    text = ("1. Net lending/net borrowing vis-a-vis the rest of the world   B.9   5,0   4.0   3.5\n"
            "2. Net lending/net borrowing of the private sector            B.9   7.0   6.0   5.0\n"
            "3. Net lending/net borrowing of general government            B.9  -2.0  -2.0  -1.5\n"
            "1. General government   S.13   -2.0   -1.5   -1.0\n")
    monkeypatch.setattr(DB.DT, "_text", lambda pdf: text)
    v, ok = DB.parse_plan(tmp_path / "x.pdf")
    assert v == -1.5 and ok
    monkeypatch.setattr(DB.DT, "_text", lambda pdf: text.replace("-1.5\n1.", "-0.5\n1.").replace("private sector", "corporations"))  # the rows no longer add up, no label fallback
    with pytest.raises(ValueError):
        DB.parse_plan(tmp_path / "x.pdf")


def test_committed_plans_and_italy_check():
    d = pd.read_csv(DB.CSV)
    assert len(d) == len(pd.read_csv(DB.URLS)) and d.evidence.notna().all()
    v = d[d.evidence.str.startswith("identity")]
    it = v[v.iso3 == "ITA"].set_index("budget_year").target_pct_gdp
    ref = pd.read_csv("data/external/deficit_targets.csv").set_index("budget_year").target_pct_gdp
    assert len(v) >= 40 and all(abs(it[y] - ref[y]) < 0.051 for y in it.index if y != 2023)  # 2023: the October plan, the reference holds the November update


def test_dbp_rank_table_orders_italy():
    S = pd.DataFrame({"ITA": [2.0, 2.0], "AAA": [0.0, 1.0], "BBB": [-1.0, 0.0]}, index=[2021, 2022]).reindex(range(2017, 2026))
    r = BX.dbp_rank_table(S, S, windows={"2021-23": [2021, 2022, 2023]})
    assert (r["rank"] == 1).all() and (r.n == 3).all()
