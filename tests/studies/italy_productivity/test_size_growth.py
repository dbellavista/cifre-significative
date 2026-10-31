"""Firm size against the level and the growth of productivity: identities on invented numbers, and the facts the post quotes when the processed data exist."""
import numpy as np
import pandas as pd
import pytest

from cifresignificative.studies.italy_productivity.analysis import size_class as SC
from cifresignificative.studies.italy_productivity.analysis import size_growth as SG
from cifresignificative.studies.italy_productivity.analysis.data import (
    PROCESSED,
    load_panel,
    load_sectors,
)


def _size(rows: dict[tuple[str, int], tuple[list[float], list[float]]], vintage: str = "new") -> pd.DataFrame:
    out = []
    for (iso3, year), (persons, va) in rows.items():
        for cl, p, v in zip(SC.CLASSES, persons, va, strict=True):
            out.append({"iso3": iso3, "year": year, "vintage": vintage, "nace_r2": SC.ECONOMY[vintage], "size_emp": cl, "persons": p, "va_meur": v, "enterprises": 1.0, "pli": 100.0, "flag": None})
        out.append({"iso3": iso3, "year": year, "vintage": vintage, "nace_r2": SC.ECONOMY[vintage], "size_emp": "TOTAL", "persons": sum(persons), "va_meur": sum(va), "enterprises": 5.0, "pli": 100.0, "flag": None})
    return pd.DataFrame(out)


def test_mix_change_is_the_log_ratio_at_end_year_productivities():
    p0, p1 = [60.0, 10, 10, 10, 10], [50.0, 10, 10, 10, 20]
    x1 = np.array([1.0, 2, 2, 2, 3])
    size = _size({("ITA", 2021): (p0, list(np.array(p0) * 2)), ("ITA", 2024): (p1, list(np.array(p1) * x1))})
    want = np.log((np.array(p1) / 100 * x1).sum() / (np.array(p0) / 100 * x1).sum()) * 100
    assert SG.mix_change(size, "new", 2021, 2024) == pytest.approx(want)


def test_mix_change_is_zero_when_the_mix_does_not_move():
    p = [60.0, 10, 10, 10, 10]
    size = _size({("ITA", 2021): (p, p), ("ITA", 2024): (p, list(np.array(p) * 3))})
    assert SG.mix_change(size, "new", 2021, 2024) == pytest.approx(0)


def test_micro_shares_use_the_published_total():
    size = _size({("ITA", 2021): ([40.0, 10, 10, 20, 20], [1.0] * 5)})
    assert SG.micro_shares(size, "new").loc["ITA", 2021] == pytest.approx(40)


def test_rank_link_finds_a_perfect_order_and_not_a_shuffled_one():
    x = pd.Series(np.arange(12.0), index=[f"c{i}" for i in range(12)])
    strong = SG.rank_link(x, x * 2, n_perm=999)
    assert strong["rho"] == pytest.approx(1) and strong["p"] < 0.01
    y = pd.Series([3.0, 9, 1, 11, 5, 7, 0, 10, 2, 8, 4, 6], index=x.index)
    assert SG.rank_link(x, y, n_perm=999)["p"] > 0.05


def test_partial_slope_recovers_a_planted_slope():
    idx = [f"c{i}" for i in range(30)]
    rng = np.random.default_rng(0)
    m = pd.Series(rng.uniform(15, 45, 30), index=idx)
    level = pd.Series(np.exp(rng.uniform(2.5, 4, 30)), index=idx)
    g = -0.05 * m - 1.0 * np.log(level) + 7
    r = SG.partial_slope(m, g, level, n_perm=199)
    assert r["slope_per_10"] == pytest.approx(-0.5) and r["p"] < 0.05


needs_data = pytest.mark.skipif(not (PROCESSED / "size_class.parquet").exists() or not (PROCESSED / "sectors.parquet").exists(), reason="processed data missing")


@needs_data
def test_italy_micro_share_falls_in_every_stretch_without_a_break():
    size = SC.load_size()
    old, new = SG.micro_shares(size, "old").loc["ITA"], SG.micro_shares(size, "new").loc["ITA"]
    assert old[2016] < old[2012] and old[2019] < old[2017] and new[2024] < new[2021]


@needs_data
def test_level_link_is_negative_and_growth_link_is_not_significant():
    r = SG.growth_link(SC.load_size(), load_sectors())
    links = r["links"]
    assert links.loc["level, all", "rho"] < -0.5 and links.loc["level, all", "p"] < 0.01
    assert links.loc["growth, all", "p"] > 0.1 and r["partial"].loc["all", "p"] > 0.05
    assert links.loc["growth, all", "n"] == 27


@needs_data
def test_coverage_scenarios_lower_the_mix_term():
    s = SG.coverage_sensitivity(SC.load_size(), load_sectors(), load_panel())["scenarios"].set_index("scenario")
    assert s.loc["as published", "total_vs_germany"] == pytest.approx(s.loc["as published", "mix_vs_germany"] + s.loc["as published", "within_vs_germany"])
    assert abs(s.loc["to Germany's ratio", "total_vs_germany"]) < abs(s.loc["to the median ratio", "total_vs_germany"]) < abs(s.loc["as published", "total_vs_germany"])
    assert s.mix_share.min() > 60  # the gap shrinks along with the term: its share does not fall
    assert s.loc["as published", "added_meur"] == 0
    assert abs(s.loc["to the median ratio", "mix_vs_germany"]) < abs(s.loc["as published", "mix_vs_germany"])
    assert abs(s.loc["to Germany's ratio", "mix_vs_germany"]) < abs(s.loc["to the median ratio", "mix_vs_germany"])
