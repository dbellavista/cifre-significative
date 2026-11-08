"""Tests for sub-national geography (`core.geo`) and price deflators (`core.deflators`)."""
import numpy as np
import pandas as pd

from cifresignificative.core import deflators, geo


def test_cesena_and_peers_registry():
    assert geo.CESENA == "040007"
    assert geo.COMUNI_BY_ID[geo.CESENA].name == "Cesena"
    assert geo.COMUNI_BY_ID[geo.CESENA].nuts3 == "ITH58"
    assert geo.COMUNI_BY_ID[geo.CESENA].catastale == "C573"
    assert geo.CESENA not in geo.CORE_PEER_IDS
    assert geo.CESENA not in geo.EXTENDED_PEER_IDS
    assert len(geo.CORE_PEER_IDS) == 6
    assert len(geo.EXTENDED_PEER_IDS) == 12
    assert set(geo.CORE_PEER_IDS) < set(geo.EXTENDED_PEER_IDS)


def test_from_prov_comune_and_formatting():
    assert geo.format_comune_id(40007) == "040007"
    assert geo.format_comune_id("40007.0") == "040007"
    prov = pd.Series(["040", "40", "000", "99"])
    com = pd.Series(["007", "12", "007", "14"])
    res = geo.from_prov_comune(prov, com)
    assert res.iloc[0] == "040007"
    assert res.iloc[1] == "040012"
    assert pd.isna(res.iloc[2])
    assert res.iloc[3] == "099014"
    assert list(geo.harmonise_nuts3(pd.Series(["ITD58", "ITH58"]))) == ["ITH58", "ITH58"]


def test_italy_deflator_and_deflate():
    d = deflators.italy_deflator(base_year=2020)
    assert np.isclose(d.loc[2020], 1.0)
    assert d.loc[2023] > 1.12  # post-2022 inflation surge
    assert d.loc[2010] < 0.95
    nom = pd.Series([100.0, 117.3])
    yrs = pd.Series([2020, 2023])
    real = deflators.deflate(nom, yrs, base_year=2020)
    assert np.isclose(real.iloc[0], 100.0)
    assert real.iloc[1] < 105.0
