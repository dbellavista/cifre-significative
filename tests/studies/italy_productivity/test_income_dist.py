"""Income distribution: placebo ranking on synthetic data, redistribution arithmetic, and a data-dependent check on the processed table."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.studies.italy_productivity.analysis import income_dist as ID

PROCESSED = Path("data/processed/income_dist.parquet")


def test_placebo_gap_ranks_the_outlier_first():
    x = pd.Series(np.linspace(4, 5, 26), index=[f"C{i:02d}" for i in range(26)])
    x["ITA"] = 9.0
    r = ID.placebo_gap(x)
    assert r["rank_from_top"] == 1 and r["gap"] > 4 and r["p_abs"] < 0.05 and r["n_countries"] == 27


def test_placebo_gap_middle_country_is_not_extreme():
    x = pd.Series(np.linspace(4, 5, 27), index=[f"C{i:02d}" for i in range(13)] + ["ITA"] + [f"D{i:02d}" for i in range(13)])
    assert ID.placebo_gap(x)["p_abs"] > 0.5


def test_wide_derives_redistribution_and_drops_breaks():
    rows = []
    for tbl, v in [("gini_disp", 30.0), ("gini_pre_pens_incl", 50.0), ("gini_pre_pens_excl", 40.0)]:
        for yr, flag in [(2010, None), (2011, "b")]:
            rows.append({"table": tbl, "quant_inc": None, "iso3": "ITA", "year": yr, "value": v, "flag": flag})
    W = ID.wide(pd.DataFrame(rows))
    assert list(W.index.get_level_values("year")) == [2010]
    assert W.redistribution_incl.iloc[0] == pytest.approx(40.0) and W.redistribution_excl.iloc[0] == pytest.approx(25.0)


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed data missing")
def test_processed_income_table_is_consistent():
    W = ID.wide(ID.load_income())
    T = ID.window_table(W, (2015, 2019))
    assert (T[ID.QUINTILES].sum(axis=1).dropna().sub(100).abs() < 0.6).all()  # quintile shares sum to 100 (rounding)
    assert 25 < T.loc["ITA", "gini_disp"] < 36 and T.loc["ITA", "s80_s20"] > 4.5
