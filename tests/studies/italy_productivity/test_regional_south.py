"""'Why the South' round: source parsers on small fixtures, the within-country permutation test (planted effect and calibrated null) and the Italy-only exploratory design."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.core.sources import cpt, eqi, istat_migration
from cifresignificative.studies.italy_productivity.analysis import regional_corr as rc
from cifresignificative.studies.italy_productivity.analysis import regional_south as rs

EXTRA = Path("data/processed/regional_extra.parquet")


def test_eqi_keeps_nuts2_rows_only(tmp_path):
    p = tmp_path / "eqi.csv"
    base = {"cname": "x", "name": "n", "EQIregion": 1, "NUTS0_code": "IT", "NUTS1_code": "", "NUTS2_code": "", "NUTS0_name": "", "NUTS1_name": "", "NUTS2_name": "", "EQI_me": 0.1, "qualityp": 0.2, "impartialityp": 0.3, "corruptionp": 0.4}
    pd.DataFrame([{**base, "region_code": "ITF3", "year": 2017, "nuts_level": 2, "EQI": -1.5}, {**base, "region_code": "DE1", "year": 2017, "nuts_level": 1, "EQI": 0.4}, {**base, "region_code": "IT", "year": 2017, "nuts_level": 0, "EQI": -0.5}]).to_csv(p, index=False)
    d = eqi.load(p)
    assert set(d.geo) == {"ITF3"} and set(d.indicator) == set(eqi.COLUMNS.values()) and d.loc[d.indicator.eq("nuts2_eqi"), "value"].item() == -1.5 and set(d.country) == {"IT"}


def test_migration_flows_drop_aggregates_and_net_sums_to_zero(tmp_path):
    rows = [("ITF3", "ITC4", 10), ("ITC4", "ITF3", 4), ("ITF3", "ITF3", 99), ("ITC", "ITF3", 7), ("ITFG", "ITC4", 5), ("ITC4", "ITI4", 3), ("ITI4", "ITC4", 1)]
    p = tmp_path / "m.csv"
    pd.DataFrame([{"REF_AREA": o, "TERRITORY_NEXT_RESID": d, "TIME_PERIOD": 2019, "CITIZENSHIP": "TOTAL", "SEX": 9, "OBS_VALUE": v, "OBS_STATUS": None} for o, d, v in rows]).to_csv(p, index=False)
    f = istat_migration.load_flows(p)
    assert len(f) == 5 and not f.origin.isin(["ITC", "ITFG"]).any()  # macro-area and aggregate codes dropped
    n = istat_migration.net_flows(f)
    assert n.net.sum() == 0 and n.set_index("geo").net["ITF3"] == 4 - 10  # moves within a region are not migration


def test_migration_translates_nuts2010_codes(tmp_path):
    p = tmp_path / "m.csv"
    pd.DataFrame([{"REF_AREA": o, "TERRITORY_NEXT_RESID": d, "TIME_PERIOD": 2019, "CITIZENSHIP": "TOTAL", "SEX": 9, "OBS_VALUE": 1, "OBS_STATUS": None} for o, d in [("ITD3", "ITE4"), ("ITE4", "ITD3"), ("ITD", "ITE4")]]).to_csv(p, index=False)
    f = istat_migration.load_flows(p)
    assert set(f.origin) == {"ITH3", "ITI4"} and set(f.destination) == {"ITH3", "ITI4"}  # Veneto and Lazio under the current codes; the macro-area row is gone


def test_cpt_links_and_load(tmp_path):
    html = '<a href="/media/abc/sp_spa_se_ts_cetot.csv">x</a> <a href="/media/def/other.csv">y</a>'
    assert cpt.find_links([html]) == {"sp_spa_se_ts_cetot.csv": cpt.HOST + "/media/abc/sp_spa_se_ts_cetot.csv", "other.csv": cpt.HOST + "/media/def/other.csv"}
    for f in cpt.FILES.values():
        (tmp_path / f).write_text("Regione per Dettaglio;Settore;Tipologia di Soggetti;Categoria Spese;Anno;S - Consolidato SPA\n19 - Sicilia;00010 - Sanità;Amministrazioni Regionali;X;2019;1,5\n22 - Provincia Autonoma di Bolzano;00010 - Sanità;Amministrazioni Locali;X;2019;2,25\n", encoding="utf-8-sig")
    d = cpt.load(tmp_path)
    assert set(d.geo) == {"ITG1", "ITH1"} and set(d.sector) == {"Sanità"} and d.value.max() == 2.25  # decimal commas, autonomous provinces mapped to their own NUTS2
    (tmp_path / cpt.FILES["total"]).write_text("Regione per Dettaglio;Settore;Tipologia di Soggetti;Categoria Spese;Anno;S - Consolidato SPA\n99 - Nowhere;00010 - Sanità;X;X;2019;1\n", encoding="utf-8-sig")
    with pytest.raises(ValueError):
        cpt.load(tmp_path)


def regions(n_countries=16, n_regions=12, effect=0.0, seed=0):
    """Country-demeaned x and y; countries differ a lot in dispersion (the same scale for x and y), which is what makes the asymptotic p too liberal."""
    rng = np.random.default_rng(seed)
    rows, idx = [], []
    for c in range(n_countries):
        scale = np.exp(rng.normal(scale=0.8))
        for r in range(n_regions):
            x = rng.normal() * scale
            rows.append({"x": x, "y": effect * x + rng.normal() * scale})
            idx.append(f"{chr(65 + c)}{chr(65 + c)}{r:02d}"[:4])
    X = pd.DataFrame(rows, index=pd.Index(idx, name="geo"))
    return rc.country_demean(X)


def test_perm_p_finds_planted_effect_and_null_rejects_about_five_percent():
    X = regions(effect=0.5)
    t = rs.pair_family(X, ["x"], ["y"], rc.country_of(X.index), n_perm=500)
    assert t.rho.iloc[0] > 0.3 and t.p.iloc[0] < 0.01 and t.loo_sign_flips.iloc[0] == 0
    hits = [rs.pair_family(X0, ["x"], ["y"], rc.country_of(X0.index), n_perm=300, seed=s).p.iloc[0] < 0.05 for s in range(60) for X0 in [regions(effect=0.0, seed=s + 10)]]
    assert np.mean(hits) < 0.15  # nominal 0.05 with 60 runs; the asymptotic p with these heteroscedastic countries is clearly higher


def test_screen_perm_keeps_planted_pair_and_reports_both_p_values():
    X = regions(effect=0.5).assign(z=lambda d: np.random.default_rng(5).normal(size=len(d)))
    t = rs.screen_perm(X, n_perm=500, n_min=20, exclusions=frozenset())
    xy = t[(t.x == "x") & (t.y == "y")].iloc[0]
    assert xy.significant and xy.significant_asymptotic and xy.p <= 1 and 0 < xy.p_asymptotic < 0.01
    assert len(t) == 3 and not t[(t.x == "z") | (t.y == "z")].significant.any() and {"q_asymptotic", "q_value"} <= set(t.columns)


def test_family_null_gives_about_zero_discoveries():
    X = regions(effect=0.5)
    n = rs.family_null(X, ["x"], ["y"], rc.country_of(X.index), n_shuffle=20, n_perm=300)
    assert n.discoveries.sum() == 0 and n.share_p_below_005.mean() < 0.2


def test_mechanical_pairs_are_excluded():
    X = regions(effect=0.5).rename(columns={"x": "sbs_wages_per_employee_keur", "y": "gdp_per_hour_eur"})
    assert rs.pair_family(X, ["sbs_wages_per_employee_keur"], ["gdp_per_hour_eur"]).empty


def italy_regions(effect_within=0.0, seed=0):
    geos = ["ITC1", "ITC2", "ITC3", "ITC4", "ITH1", "ITH2", "ITH3", "ITH4", "ITH5", "ITI1", "ITI2", "ITI3", "ITI4", "ITF1", "ITF2", "ITF3", "ITF4", "ITF5", "ITF6", "ITG1", "ITG2"]
    rng = np.random.default_rng(seed)
    zone = np.array([{"North": 2.0, "Centre": 1.0, "South and Islands": 0.0}[rs.zone_of(g)] for g in geos])
    x = zone + rng.normal(scale=0.6, size=len(geos))  # x and y share the North-South split ...
    y = zone + effect_within * (x - zone) + rng.normal(scale=0.6, size=len(geos))  # ... and optionally a within-zone link
    return pd.DataFrame({"x": x, "y": y}, index=pd.Index(geos, name="geo"))


def test_italy_exploratory_separates_zone_gradient_from_within_zone_link():
    gradient = rs.italy_exploratory(italy_regions(0.0), ["x"], ["y"], n_perm=500).iloc[0]
    assert gradient.rho > 0.5 and gradient.p_perm < 0.05  # the shared North-South gradient shows as a raw association ...
    assert gradient.p_perm_within_zone > 0.05  # ... but not once the three zones are held fixed
    within = rs.italy_exploratory(italy_regions(1.5), ["x"], ["y"], n_perm=500).iloc[0]
    assert within.rho_within_zone > 0.5 and within.p_perm_within_zone < 0.05


def test_zone_gaps_sign():
    T = italy_regions(0.0)
    g = rs.zone_gaps(T, ["x"]).iloc[0]
    assert g.gap_sd < -1 and g.north_mean > g.south_mean


@pytest.mark.skipif(not EXTRA.exists(), reason="needs data/processed/regional_extra.parquet")
def test_extra_table_has_eqi_for_every_italian_region_and_wave():
    d = pd.read_parquet(EXTRA)
    e = d[d.indicator.eq("nuts2_eqi")]
    assert e[e.geo.str.startswith("IT")].groupby("year").geo.nunique().eq(21).all() and set(e.year) == {2010, 2013, 2017, 2021, 2024}
    assert {"nuts2_sbs_local_units_per_1000_inhabitants", "nuts2_sbs_manufacturing_employment_share_pct", "nuts2_sbs_wages_per_employee_keur"} <= set(d.indicator)
