"""Policy dates of the winter 2022-23 and their timing against the ENEA monthly flows."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.core.sources import enea_monthly as EM
from cifresignificative.studies.italy_productivity.analysis import (
    superbonus_timing as ST,
)


def test_policy_dates_table():
    d = ST.load_dates()
    assert d.date.is_monotonic_increasing and set(d.evidence) <= {"verified", "partly verified"}
    assert {"2022-11-25", "2023-02-16", "2022-12-29"} <= set(d.date.dt.strftime("%Y-%m-%d"))
    for r in d[d.evidence == "verified"].itertuples():
        assert r.sources.count("http") >= 2  # verified = two independent sources


def test_contrast_and_rank():
    s = pd.Series(np.r_[np.zeros(10), np.ones(10)], index=pd.date_range("2022-01-31", periods=20, freq="ME"))
    assert ST.contrast(s, s.index[10]) == pytest.approx(1.0) and np.isnan(ST.contrast(s, s.index[2]))


def test_event_tests_on_committed_data():
    reg = Path("data/processed/regional.parquet")
    if not reg.exists():
        pytest.skip("processed regional data missing")
    p = ST.monthly_path(EM.load(), pd.read_parquet(reg))
    t = ST.event_tests(p)
    assert len(t) == 4 and t.q.between(0, 1).all() and t.p_rank.between(0, 1).all() and (t.n_cuts > 40).all()
    assert ST.onset(p)["gradient_onset"] is not None


def test_building_counts_on_committed_data():
    reg = Path("data/processed/regional.parquet")
    if not reg.exists():
        pytest.skip("processed regional data missing")
    r, t = ST.building_counts(EM.load())
    assert (r["all"].loc["2022-09-30"] > 3 * r["all"].loc["2022-10-31"]) and len(t) == 8 and t.p_rank.between(0, 1).all()  # the September 2022 single-family rush
