"""ENEA Superbonus investments by region (committed table) against regional income."""
import pandas as pd
import pytest

from cifresignificative.analysis import superbonus_regions as SR
from cifresignificative.sources import enea_regions as ER


def test_parse_table_and_committed_totals():
    text = "Tabella 4.3 - Investimenti per lavori realizzati (€)\n\n  Italia   10,0   20,0   30,0   40,0   100,0\n" + "\n".join(f"  Reg{i}   1,0   1,0   1,0   1,0   4,0" for i in range(20)) + "\n"
    with pytest.raises(ValueError):  # the regions must add up to the national total
        ER.parse_table(text.replace("Reg0   1,0   1,0   1,0   1,0   4,0", "Reg0   1,0   1,0   1,0   1,0   9999,0"), "Tabella 4.3")
    d = ER.load()
    assert set(d.snapshot.dt.strftime("%Y-%m-%d")) == {"2023-12-31", "2025-12-31"} and (d.groupby("snapshot").size() == 21).all()
    tot = d[d.region == "Italia"].set_index("snapshot").total
    assert tot.loc["2025-12-31"] == pytest.approx(118007895683) and tot.loc["2023-12-31"] == pytest.approx(94341114283.38)
    for snap, g in d.groupby("snapshot"):
        assert g[g.region != "Italia"].total.sum() == pytest.approx(g[g.region == "Italia"].total.iloc[0], abs=100)  # 20 regions add to Italy


def test_zone_shares_and_rank_test_on_committed_data():
    reg = pd.read_parquet("data/processed/regional.parquet") if __import__("pathlib").Path("data/processed/regional.parquet").exists() else None
    if reg is None:
        pytest.skip("processed regional data missing")
    t = SR.region_table(ER.load(), reg, "2023-12-31")
    z = SR.zone_shares(t)
    assert len(t) == 20 and abs(z.share_of_investment.sum() - 1) < 1e-9 and z.loc["North", "ratio_of_shares"] > 1 > z.loc["South and Islands", "ratio_of_shares"]  # the North got more than its population share
    r = SR.rank_test(t, n_perm=2000)
    assert 0.3 < r["rho"] < 0.7 and r["n"] == 20 and r["p"] < 0.1


def test_rebase_on_monthly_series():
    from cifresignificative.analysis import superbonus as SB
    from cifresignificative.sources import enea_monthly as EM
    f = SB.yearly_flows_monthly(EM.load())
    a = SB.yearly_flows(SB.load())
    assert list(f.index) == [2021, 2022, 2023] and (f.deductions_accrued_eur > 0).all()
    assert 0.97 < f.deductions_accrued_eur.sum() / a.deductions_accrued_eur.sum() < 1.06  # monthly reports are later extractions, a few % above the annual reports
