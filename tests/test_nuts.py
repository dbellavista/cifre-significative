"""NUTS crosswalk: synthetic boundaries (offline) plus checks against the real GISCO files when they are downloaded."""
from pathlib import Path

import pytest
from shapely.geometry import box

from cifresignificative.clean import nuts

REAL = pytest.mark.skipif(not Path("data/raw/gisco_nuts").exists(), reason="GISCO boundaries not downloaded")


@pytest.fixture
def toy(monkeypatch):
    """2016: AA01 (0-10), AA02 (10-20), BB01 (30-40), CC01 (50-60), DD01 (70-80). 2021: AA01 kept; AA02 renamed AA03 and 1% of it is open sea; BB01 merged with a neighbour into BB05;
    CC01 divided into CC02 (0.8) and CC03 (0.2); DD01 is not in 2021 (and DD99 exists only in 2024)."""
    geo = {
        2016: {"AA01": box(0, 0, 10, 10), "AA02": box(10, 0, 20, 10), "BB01": box(30, 0, 40, 10), "CC01": box(50, 0, 60, 10), "DD01": box(70, 0, 80, 10)},
        2021: {"AA01": box(0, 0, 10, 10), "AA03": box(10, 0, 19.9, 10), "BB05": box(30, 0, 50, 10).difference(box(50, 0, 50, 10)), "CC02": box(50, 0, 58, 10), "CC03": box(58, 0, 60, 10)},
        2024: {"DD99": box(70, 0, 80, 10)},
    }
    monkeypatch.setattr(nuts, "regions", lambda year: geo.get(year, {}))
    return nuts.crosswalk(old_years=(2016,), later=2024).set_index("old")


def test_crosswalk_classes(toy):
    assert toy.kind["AA01"] == "same" and toy.kind["AA02"] == "nested" and toy.new["AA02"] == "AA03"  # a coastal sliver outside every 2021 region does not make it a split
    assert toy.kind["BB01"] == "nested" and toy.new["BB01"] == "BB05" and toy.share_of_new["BB01"] == pytest.approx(0.5)  # merged into a bigger region
    assert toy.kind["CC01"] == "split" and toy.new["CC01"] == "CC02"
    assert toy.kind["DD99"] == "later" and toy.new.isna()["DD99"]
    assert toy.kind["ITD1"] == "legacy" and toy.new["ITD1"] == "ITH1" and toy.kind["FI18"] == "split"


@REAL
def test_real_crosswalk_known_cases():
    cw = nuts.crosswalk().set_index("old")
    assert cw.loc["FR21", "new"] == "FRF2" and cw.loc["FR21", "kind"] == "nested"  # France 2016: codes changed, regions did not
    assert cw.loc["EL11", "new"] == "EL51" and cw.loc["SI01", "new"] == "SI03"
    assert cw.loc["HR04", "kind"] == "split" and cw.loc["HU10", "kind"] == "split" and cw.loc["PL12", "kind"] == "split"  # Zagreb, Budapest and Warsaw were separated
    assert cw.loc["PT19", "kind"] == "later" and cw.loc["ITC1", "kind"] == "same"


def _toy_cw():
    import pandas as pd
    return pd.DataFrame([("FR21", "FRF2", "nested", 2013), ("FR91", "FRY1", "nested", 2010), ("FRA1", "FRY1", "nested", 2013), ("HR04", "HR02", "split", 2016), ("PT19", None, "later", 2024), ("ITD1", "ITH1", "legacy", 2006)],
                        columns=["old", "new", "kind", "from_year"])


def test_harmonise_renames_drops_splits_and_prefers_native(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(nuts, "regions", lambda year: {"FRF2": None, "FRY1": None, "HR02": None, "ITH1": None})
    rows = [("FR21", 2010, 1.0), ("FRF2", 2010, 1.1), ("FRF2", 2018, 2.0), ("FR91", 2008, 3.0), ("FRA1", 2008, 3.5), ("FRA1", 2009, 4.0), ("HR04", 2010, 5.0), ("PT19", 2024, 6.0), ("ITD1", 2009, 7.0), ("FRXX", 2010, 8.0)]
    df = pd.DataFrame(rows, columns=["geo", "year", "value"]).assign(indicator="x", flag=None, country=lambda d: d.geo.str[:2])
    h = nuts.harmonise(df, _toy_cw()).set_index(["geo", "year"])
    assert set(h.index.get_level_values("geo")) == {"FRF2", "FRY1", "ITH1"}  # the split HR04, the post-2021 PT19 and extra-regio FRXX are not mapped
    assert h.loc[("FRF2", 2010), "value"] == 1.1 and h.loc[("FRF2", 2010), "geo_method"] == "native"  # both codes in the same year: the native one wins
    assert h.loc[("FRY1", 2008), "value"] == 3.5 and h.loc[("FRY1", 2008), "geo_old"] == "FRA1"  # two renamed codes: the newest classification wins
    assert h.loc[("FRY1", 2009), "geo_method"] == "renamed" and h.loc[("ITH1", 2009), "country"] == "IT"
    assert not h.index.duplicated().any()


def test_overlap_agreement_reports_relative_differences():
    import pandas as pd
    df = pd.DataFrame({"geo": ["FR21", "FRF2", "FR21", "FRF2"], "year": [2010, 2010, 2011, 2011], "value": [100.0, 101.0, 100.0, 100.0], "indicator": "x"})
    a = nuts.overlap_agreement(df, _toy_cw()).iloc[0]
    assert a.pairs == 2 and a["max"] == pytest.approx(1 / 101) and a["median"] == pytest.approx(1 / 202)
