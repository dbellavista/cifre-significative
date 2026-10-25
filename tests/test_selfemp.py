"""Self-employment: exact shift-share identity on synthetic data, flag handling, and a data-dependent check on the processed table."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.analysis import selfemp as SE

PROCESSED = Path("data/processed/selfemp.parquet")


def _frame(seed=1):
    rng = np.random.default_rng(seed)
    idx = ["ITA", *[f"C{i:02d}" for i in range(8)]]
    emp = pd.DataFrame(rng.uniform(50, 500, (9, 3)), index=idx, columns=["A", "B", "C"])
    rate = pd.DataFrame(rng.uniform(0.05, 0.4, (9, 3)), index=idx, columns=["A", "B", "C"])
    return pd.concat({"self": emp * rate, "emp": emp}, axis=1)


def test_decomposition_is_exact_and_ranked():
    F = _frame()
    d = SE.decompose(F)
    assert d["composition"] + d["within_sector"] == pytest.approx(d["share"] - d["pooled_others"])
    D = SE.decompose_all(F)
    assert len(D) == 9 and SE.rank_in(D.within_sector)["n_countries"] == 9


def test_decomposition_attributes_a_planted_rate_gap_to_within_sector():
    F = _frame()
    F.loc["ITA", "self"] = (F.loc["ITA", "emp"] * 0.6).to_numpy()  # same sector weights as drawn, but very high rates
    d = SE.decompose(F)
    assert d["within_sector"] > 5 * abs(d["composition"])
    assert SE.rank_in(SE.decompose_all(F).within_sector)["rank_from_top"] == 1


def test_share_wide_drops_break_flags():
    rows = [{"table": "selfemp_share_status", "iso3": "ITA", "year": y, "sex": "T", "wstatus": w, "value": v, "flag": f}
            for y, f in [(2010, None), (2011, "b")] for w, v in [("SELF", 20.0), ("SELF_NS", 15.0), ("SELF_S", 5.0)]]
    W = SE.share_wide(pd.DataFrame(rows))
    assert list(W.index.get_level_values("year")) == [2010] and W.SELF_NS.iloc[0] == 15.0


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed data missing")
def test_processed_selfemployment_is_consistent():
    se = SE.load_selfemp()
    S = SE.share_wide(se)
    assert (S.SELF_S + S.SELF_NS - S.SELF).abs().dropna().max() < 0.3  # employers + own-account = self-employed (rounding)
    it = S.xs("ITA", level="iso3").SELF
    assert 15 < it.loc[2015] < 25
    F = SE.sector_frame(se, 2019)
    assert "ITA" in F.index and abs(SE.decompose(F)["share"] - 21) < 2
