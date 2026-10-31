"""EU payments by region, the NUTS1 aggregation for Germany and Belgium, and the funds predictors (offline toy data; data-gated checks on the real tables)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.core.clean import nuts
from cifresignificative.core.sources import eqi, eu_payments
from cifresignificative.studies.italy_productivity.analysis import eu_funds as EF
from cifresignificative.studies.italy_productivity.analysis import nuts1 as N1


@pytest.fixture
def toy_regions(monkeypatch):
    monkeypatch.setattr(nuts, "regions", lambda year: {c: None for c in ("DE11", "DE12", "DE21", "BE10", "BE21", "BE22", "FR10")})
    rows = []
    for geo, pop, gdp, emp in (("DE11", 1000.0, 50000.0, 80.0), ("DE12", 3000.0, 90000.0, 70.0), ("DE21", 2000.0, 70000.0, 75.0), ("BE10", 1000.0, 60000.0, 60.0), ("BE21", 2000.0, 80000.0, 70.0), ("BE22", 1000.0, 30000.0, 66.0)):
        for ind, v in (("nuts2_population", pop), ("nuts2_gdp_mio_eur", gdp), ("nuts2_employment_rate_20_64", emp)):
            rows.append({"geo": geo, "year": 2010, "value": v, "flag": None, "indicator": ind, "theme": "t", "source": "s", "country": geo[:2]})
    return pd.DataFrame(rows)


def test_nuts1_sums_extensive_and_weights_rates_by_population(toy_regions):
    n = N1.to_nuts1(toy_regions, ("DE", "BE")).set_index(["geo", "indicator"]).value
    assert n[("DE1", "nuts2_population")] == 4000 and n[("DE1", "nuts2_gdp_mio_eur")] == 140000  # DE11 + DE12
    assert n[("DE1", "nuts2_employment_rate_20_64")] == pytest.approx((1000 * 80 + 3000 * 70) / 4000)  # population-weighted
    assert n[("BE2", "nuts2_employment_rate_20_64")] == pytest.approx((2000 * 70 + 1000 * 66) / 3000) and ("DE2", "nuts2_population") in n.index


def test_nuts1_needs_all_the_children(toy_regions):
    partial = toy_regions[~((toy_regions.geo == "DE12") & (toy_regions.indicator == "nuts2_gdp_mio_eur"))]
    n = N1.to_nuts1(partial, ("DE",)).set_index(["geo", "indicator"]).value
    assert ("DE1", "nuts2_gdp_mio_eur") not in n.index and ("DE1", "nuts2_population") in n.index  # a region missing one child is not aggregated for that indicator


def test_eqi_load_adds_nuts1_rows_only_for_the_listed_countries(tmp_path):
    d = pd.DataFrame({"region_code": ["ITC1", "DE1", "BE2", "FR1"], "nuts_level": [2, 1, 1, 1], "year": 2017, "EQI": [0.1, 0.2, 0.3, 0.4], "qualityp": 1.0, "impartialityp": 1.0, "corruptionp": 1.0, "EQI_me": 0.1})
    p = tmp_path / "e.csv"
    d.to_csv(p, index=False)
    assert set(eqi.load(p).geo) == {"ITC1"} and set(eqi.load(p, nuts1=("DE", "BE")).geo) == {"ITC1", "DE1", "BE2"}


def test_payments_are_summed_per_programme_and_translated(tmp_path, monkeypatch):
    monkeypatch.setattr(nuts, "regions", lambda year: {"DE11": None, "FR10": None})
    monkeypatch.setattr(nuts, "crosswalk", lambda *a, **k: pd.DataFrame({"old": ["FR21"], "new": ["FR10"], "kind": ["nested"], "from_year": [2013]}))
    d = pd.DataFrame({"nuts2_id": ["DE11", "DE11", "DE11", "FR21", "DE11"], "fund": ["ERDF", "ERDF", "ESF", "ERDF", "ERDF"], "programming_period": ["2007-2013", "2007-2013", "2007-2013", "2007-2013", "2000-2006"],
                      "eu_payment_annual": [100.0, 50.0, 30.0, 10.0, 999.0], "year": [2010, 2011, 2010, 2010, 2005]})
    p = tmp_path / "p.csv"
    d.to_csv(p, index=False)
    f = eu_payments.load_period("2007-2013", p)
    assert f.loc["DE11", "ERDF"] == 150 and f.loc["DE11", "cohesion"] == 180 and f.loc["FR10", "cohesion"] == 10  # the other programme is not counted; FR21 is renamed FR10
    assert f.loc["DE11", "all_funds"] == 180 and f.loc["DE11", "CF"] == 0


def test_predictors_per_head_and_share_of_gdp():
    reg = pd.DataFrame([{"geo": "DE11", "year": y, "indicator": i, "value": v} for y in (2007, 2013) for i, v in (("nuts2_population", 1000.0), ("nuts2_gdp_mio_eur", 100.0))])
    funds = pd.DataFrame({"cohesion": [500_000.0], "all_funds": [800_000.0]}, index=pd.Index(["DE11"], name="geo"))
    p = EF.predictors(reg, funds)
    assert p.loc["DE11", "funds_cohesion_per_head"] == pytest.approx(500.0) and p.loc["DE11", "funds_cohesion_pct_gdp"] == pytest.approx(0.25) and p.loc["DE11", "funds_all_per_head"] == pytest.approx(800.0)


@pytest.mark.skipif(not (Path("data/processed/regional_nuts2021.parquet").exists() and Path("data/raw/cohesion/historic_payments_nuts2.csv").exists()), reason="regional tables or payments not built")
def test_real_funds_and_nuts1_runs(tmp_path):
    from cifresignificative.studies.italy_productivity.analysis import experiments as ex
    f = ex.run_funds_all(out=tmp_path, n_perm=500, n_shuffle=2)
    assert len(f["f2_family"]) == 10 and f["f1_funds_by_region"].funds_cohesion_per_head.notna().sum() > 200 and f["f6_null"].discoveries.max() <= 1
    n = ex.run_nuts1_south(out=tmp_path, n_perm=300)["n1_family_comparison"]
    e = n[n.x.str.startswith("eqi")]
    assert (e.n_with_de_be > e.n_baseline).all() and np.sign(e.rho_baseline).eq(np.sign(e.rho_with_de_be)).mean() > 0.9
