"""Superbonus use by region against the change of the notebook 34 outcomes."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.analysis import superbonus_outcomes as SO
from cifresignificative.analysis import superbonus_regions as SR
from cifresignificative.sources import enea_regions as ER

REG = Path("data/processed/regional.parquet")


def test_partial_rank_removes_a_shared_driver():
    rng = np.random.default_rng(1)
    z = pd.Series(rng.normal(size=60)); x = z + 0.3 * rng.normal(size=60); y = z + 0.3 * rng.normal(size=60)
    assert np.corrcoef(x.rank(), y.rank())[0, 1] > 0.8
    assert abs(SO.partial_rank(x, y, pd.DataFrame({"z": z}), n_perm=500)["rho"]) < 0.4


def test_link_on_committed_data():
    if not REG.exists():
        pytest.skip("processed regional data missing")
    reg = pd.read_parquet(REG)
    t = SR.region_table(ER.load(), reg, "2023-12-31")
    ch = SO.outcome_changes(SO.outcome_levels(t, reg))
    assert len(ch) == 20 and not ch.isna().any().any()
    # Trentino-Alto Adige aggregates two NUTS2 regions: its employment rate lies between theirs
    lv = SO.outcome_levels(t, reg)
    taa = lv[(lv.region == "Trentino-Alto Adige") & (lv.indicator == "employment_rate_20_64") & (lv.year == 2019)].value.iloc[0]
    e = reg[(reg.indicator == "nuts2_employment_rate_20_64") & (reg.year == 2019) & reg.geo.isin(["ITH1", "ITH2"])].value
    assert e.min() <= taa <= e.max()
    out = SO.link(t, ch, n_perm=2000)
    assert len(out) == 12 and out.primary.sum() == 1 and out.q.between(0, 1).all() and (out.q >= out.p - 1e-12).all()
    pl = SO.placebo(t, ch, n_shuffle=60, n_perm=300)
    assert pl["share_with_discovery"] < 0.25
