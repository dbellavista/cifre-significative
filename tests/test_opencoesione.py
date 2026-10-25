"""OpenCoesione 2014-2020 by region: committed table and the link family."""
from pathlib import Path

import pandas as pd
import pytest

from cifresignificative.analysis import opencoesione as OCA
from cifresignificative.sources import opencoesione as OC

REG = Path("data/processed/regional.parquet")


def test_committed_region_table():
    d = pd.read_csv(OC.REGIONS_CSV)
    assert set(d.fund) == set(OC.FUNDS) and d.area.nunique() == 21 and set(d.area) == set(OC.NUTS2.values()) | set(OC.TAA.values())
    t = d.groupby("fund").total_public_meur.sum()
    assert 50_000 < t["FESR"] < 56_000 and 100_000 < t["FSC"] < 115_000  # the programme sizes of the 2014-2020 cycle
    assert (d.eu_meur + d.national_public_meur - d.total_public_meur).abs().max() < 1e-3  # the parts add up
    u = pd.read_csv(OC.UNALLOC_CSV)
    assert (u.unallocated_total_public_meur / u.total_public_meur < 0.2).all()  # most of the money has an Italian localisation


def test_predictors_and_family_on_committed_data():
    if not REG.exists():
        pytest.skip("processed regional data missing")
    reg = pd.read_parquet(REG)
    pred = OCA.region_predictors(pd.read_csv(OC.REGIONS_CSV), reg)
    assert len(pred) == 21 and pred.notna().all().all()
    z = OCA.zone_table(pred)
    assert (z.loc["South / North", ["oc_eu_pc_eur", "oc_cofin_pc_eur", "oc_fsc_pc_eur"]] > 2).all() and z.loc["South / North", "oc_payment_rate_pct"] < 1  # funds go South, payments lag there
