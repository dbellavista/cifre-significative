"""Tests for the review sensitivity tables (`studies.cesena.analysis.robustness`) and the synthetic leave-one-out."""
import numpy as np
import pandas as pd

from cifresignificative.core.geo import CESENA
from cifresignificative.studies.cesena.analysis.experiments import (
    extended_pool_sensitivity,
    synthetic_leave_one_out,
)
from cifresignificative.studies.cesena.analysis.robustness import (
    cohort_by_period,
    housing_gradient_test,
    income_composition,
    population_balance,
)

PEERS = ("040012", "099014", "039014", "037032")


def test_income_composition_separates_mean_from_aggregate():
    """A mean that rises because taxpayers shrink must not rank first on aggregate income."""
    rows = []
    for cid in (CESENA, *PEERS):
        for y in (2008, 2018, 2024):
            growing = cid == "040012"
            n = 100.0 * (1.5 if growing and y > 2008 else 1.0)
            mean = 20.0 * (1.1 if cid == CESENA and y > 2008 else 1.0)
            rows.append({"comune_id": cid, "year": y, "mean_income_real_eur": mean, "median_income_real_eur": mean, "taxpayers_pos": n, "income_pos_real_eur": n * mean, "irpef_taxbase_real_pc": mean})
    out = income_composition(pd.DataFrame(rows)).set_index(["measure", "t0", "t1"])
    assert out.loc[("mean_income", 2008, 2024), "rank"] == 1
    assert out.loc[("aggregate_income", 2008, 2024), "rank"] == 2


def test_population_balance_closes_for_every_area():
    rows = []
    for cid, grp in ((CESENA, "cesena_core"), ("040008", "rubicone_mare")):
        pop = 1000.0
        for y in range(2008, 2026):
            flows = {"natural_balance": -5.0, "net_migration_internal": 4.0, "net_migration_international": 3.0, "registry_adjustment": 1.0}
            rows.append({"comune_id": cid, "belt_group": grp, "year": y, "pop_total": pop, **flows})
            pop += sum(flows.values())
    out = population_balance(pd.DataFrame(rows))
    total = out["natural_balance"] + out["net_migration_internal"] + out["net_migration_international"] + out["registry_adjustment"]
    assert np.allclose(total, out["pop_change"])
    assert set(out["area"]) >= {"cesena", "belt_16", "rubicone_mare"}


def test_cohort_by_period_shows_a_rank_that_changes():
    rows = []
    for y in range(2012, 2025):
        rows.append({"comune_id": CESENA, "year": y, "net_cohort_25_44_per_1k": 0.0 if y <= 2016 else 20.0})
        for i, cid in enumerate(PEERS):
            rows.append({"comune_id": cid, "year": y, "net_cohort_25_44_per_1k": 10.0 + i})
    out = cohort_by_period(pd.DataFrame(rows)).set_index(["t0", "t1"])
    assert out.loc[(2012, 2016), "rank"] == 5
    assert out.loc[(2017, 2024), "rank"] == 1


def test_housing_gradient_detects_planted_sign():
    """Planted: dearer belt towns grow more, so the correlation must come out positive."""
    belt_ids = ["040008", "040015", "040016", "040018", "040041", "040045"]
    belt, omi = [], []
    for i, cid in enumerate(belt_ids):
        omi.append({"comune_id": cid, "compr_mid_eur_m2": 1000.0 + 100 * i})
        for y in range(2008, 2026):
            belt.append({"comune_id": cid, "year": y, "pop_total": 1000.0 * (1 + 0.01 * i * (y - 2008))})
    demo = []
    for i, cid in enumerate((CESENA, *PEERS)):
        omi.append({"comune_id": cid, "compr_mid_eur_m2": 1500.0 + 10 * i})
        demo.extend({"comune_id": cid, "year": y, "net_cohort_25_44_per_1k": float(i)} for y in range(2012, 2025))
    out = housing_gradient_test(pd.DataFrame(omi), pd.DataFrame(belt), pd.DataFrame(demo)).set_index("y")
    assert out.loc["pop_growth_2008_2025", "spearman"] > 0.99


def test_synthetic_leave_one_out_reports_each_used_donor():
    rows = []
    for y in range(2008, 2025):
        t = y - 2008
        vals = {"040012": 100 + t, "039014": 100 + 3 * t, "099014": 100 - t, "037032": 100 + 5 * t}
        vals[CESENA] = 0.5 * vals["040012"] + 0.5 * vals["039014"] + (5.0 if y > 2018 else 0.0)
        rows.extend({"comune_id": c, "year": y, "mean_income_real_eur": float(v)} for c, v in vals.items())
    out = synthetic_leave_one_out(pd.DataFrame(rows))
    assert (out["dropped"] == "none").sum() == 1
    assert len(out) >= 3
    assert np.isclose(out[out["dropped"] == "none"]["gap_end_pp"].iloc[0], 5.0, atol=0.05)


def test_extended_pool_sensitivity_counts_ranks_by_window():
    m = pd.DataFrame(
        {
            "prov": ["FC", "RA", "MI", "NA"],
            "regione": ["Emilia-Romagna", "Emilia-Romagna", "Lombardia", "Campania"],
            "taxpayers_total": [77000.0, 60000.0, 90000.0, 300000.0],
            "mean_income_growth_08_24_pct": [5.0, 1.0, 7.0, 9.0],
            "median_income_growth_08_24_pct": [5.0, 1.0, 7.0, 9.0],
        },
        index=["C573", "H199", "X1", "X2"],
    )
    out = extended_pool_sensitivity(m)
    sel = out[(out["metric"] == "mean_income_growth_08_24_pct") & (out["taxpayers_lo"] == 40000)].set_index("region")
    assert sel.loc["Emilia-Romagna", "rank"] == 1
    assert sel.loc["Italy", "rank"] == 2
