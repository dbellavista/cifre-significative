"""SHSS loader: pivot, derived bank-to-government series, and the key layout."""
import pandas as pd

from cifresignificative.core.sources import ecb_shss


def test_key_has_18_positions():
    assert len(ecb_shss.KEY.format(holder="S122", issuer="S1").split(".")) == 18


def test_load_derives_banks_government(tmp_path):
    rows = [("it_banks_domestic_total", 700.0), ("it_banks_in_fin", 250.0), ("it_banks_in_households", 0.0), ("it_fin_corps_gov", 1200.0)]
    p = tmp_path / "s.csv"
    pd.DataFrame([{"KEY": "k", "TIME_PERIOD": q, "OBS_VALUE": v + i, "series": s} for i, q in enumerate(["2026-Q1", "2026-Q2"]) for s, v in rows]).to_csv(p, index=False)
    df = ecb_shss.load(p)
    d = df[df.series == ecb_shss.DERIVED].set_index("period").value
    assert d.tolist() == [450.0, 449.0]  # (700+i) - (250+i) - (0+i)... = 450 - i
    assert df.period.max() == pd.Timestamp("2026-06-30") and set(df.source) == {"ecb:SHSS"}
