"""Tests for Pillar 1 (`studies.cesena.analysis.fiscal` and `studies.cesena.analysis.data`)."""
import numpy as np
import pandas as pd

from cifresignificative.core.geo import CESENA
from cifresignificative.studies.cesena.analysis.data import peer_median_ex_cesena
from cifresignificative.studies.cesena.analysis.fiscal import (
    capital_funding_breakdown,
    enrich_fiscal_panel,
)


def test_fiscal_panel_does_not_splice_across_2016_dlgs118_break():
    budgets = pd.DataFrame([
        {
            "comune_id": CESENA, "comune": "Cesena", "year": 2014, "regime": "dpr194", "budget_type": "consuntivo", "flag": "break_dlgs118",
            "spese_correnti_comp_eur": 80_000_000.0, "spese_capitale_comp_eur": 15_000_000.0, "entrate_tributarie_comp_eur": 60_000_000.0,
        },
        {
            "comune_id": CESENA, "comune": "Cesena", "year": 2015, "regime": "dpr194", "budget_type": "consuntivo", "flag": "break_dlgs118",
            "spese_correnti_comp_eur": 84_000_000.0, "spese_capitale_comp_eur": 16_000_000.0, "entrate_tributarie_comp_eur": 62_000_000.0,
        },
        {
            "comune_id": CESENA, "comune": "Cesena", "year": 2016, "regime": "dlgs118", "budget_type": "consuntivo", "flag": "",
            "spese_correnti_comp_eur": 76_000_000.0, "spese_capitale_comp_eur": 18_000_000.0, "entrate_tributarie_comp_eur": 64_000_000.0,
        },
        {
            "comune_id": CESENA, "comune": "Cesena", "year": 2017, "regime": "dlgs118", "budget_type": "consuntivo", "flag": "",
            "spese_correnti_comp_eur": 79_000_000.0, "spese_capitale_comp_eur": 20_000_000.0, "entrate_tributarie_comp_eur": 66_000_000.0,
        },
    ])
    demo = pd.DataFrame([
        {"comune_id": CESENA, "year": y, "pop_total": 97_000.0}
        for y in (2014, 2015, 2016, 2017)
    ])
    out = enrich_fiscal_panel(budgets, demo, base_year=2020)
    out = out.set_index("year")
    # Growth within 2014->2015 (dpr194) and within 2016->2017 (dlgs118) is valid, but 2015->2016 across the break MUST be NaN
    assert pd.notna(out.loc[2015, "dlog_spese_correnti_comp_real_pc"])
    assert pd.isna(out.loc[2016, "dlog_spese_correnti_comp_real_pc"])
    assert pd.notna(out.loc[2017, "dlog_spese_correnti_comp_real_pc"])


def test_peer_median_strictly_excludes_cesena():
    df = pd.DataFrame([
        {"comune_id": CESENA, "year": 2020, "val": 10_000.0},  # extreme outlier for Cesena
        {"comune_id": "040012", "year": 2020, "val": 100.0},   # Forlì
        {"comune_id": "099014", "year": 2020, "val": 200.0},   # Rimini
        {"comune_id": "039014", "year": 2020, "val": 300.0},   # Ravenna
    ])
    res = peer_median_ex_cesena(df, "val", core_only=True)
    assert len(res) == 1
    assert np.isclose(res.iloc[0]["peer_median"], 200.0)
    assert res.iloc[0]["n_peers"] == 3
    assert "Carpi" in res.iloc[0]["missing_peers"]


def test_capital_funding_breakdown_metrics():
    df = pd.DataFrame([
        {
            "comune_id": CESENA, "year": 2021, "regime": "dlgs118", "budget_type": "consuntivo",
            "spese_capitale_comp_real_pc": 136.0, "contributi_investimenti_comp_real_pc": 50.0,
            "alienazioni_comp_real_pc": 34.0, "accensione_mutui_comp_real_pc": 31.0,
            "fpv_capitale_comp_real_pc": 300.0, "avanzo_amministrazione_comp_real_pc": 105.0,
        },
        {
            "comune_id": "040012", "year": 2021, "regime": "dlgs118", "budget_type": "consuntivo",
            "spese_capitale_comp_real_pc": 170.0, "contributi_investimenti_comp_real_pc": 50.0,
            "alienazioni_comp_real_pc": 5.0, "accensione_mutui_comp_real_pc": 21.0,
            "fpv_capitale_comp_real_pc": 216.0, "avanzo_amministrazione_comp_real_pc": 158.0,
        },
    ])
    cf = capital_funding_breakdown(df, core_only=True)
    assert len(cf) == 1
    row = cf.iloc[0]
    assert row["year"] == 2021
    assert np.isclose(row["cesena_spese_capitale_pc"], 136.0)
    assert np.isclose(row["cesena_fpv_capitale_pc"], 300.0)
    assert np.isclose(row["peer_fpv_capitale_pc"], 216.0)
    assert np.isclose(row["cesena_alienazioni_pc"], 34.0)
    assert np.isclose(row["peer_alienazioni_pc"], 5.0)


def test_summarize_consolidated_benchmarks():
    from cifresignificative.studies.cesena.analysis.fiscal import (
        summarize_consolidated_benchmarks,
    )

    cons_df = pd.DataFrame([
        {
            "year": 2024, "comune_id": CESENA, "direct_correnti_pc": 839.0, "unione_correnti_pc": 185.0,
            "consolidated_correnti_pc": 1024.0, "direct_personale_pc": 235.0, "consolidated_personale_pc": 285.0,
            "direct_capitale_pc": 401.0, "consolidated_capitale_pc": 427.0,
        },
        {
            "year": 2024, "comune_id": "040012", "direct_correnti_pc": 980.0, "unione_correnti_pc": 20.0,
            "consolidated_correnti_pc": 1000.0, "direct_personale_pc": 265.0, "consolidated_personale_pc": 275.0,
            "direct_capitale_pc": 310.0, "consolidated_capitale_pc": 320.0,
        },
        {
            "year": 2024, "comune_id": "099014", "direct_correnti_pc": 1150.0, "unione_correnti_pc": 0.0,
            "consolidated_correnti_pc": 1150.0, "direct_personale_pc": 320.0, "consolidated_personale_pc": 320.0,
            "direct_capitale_pc": 350.0, "consolidated_capitale_pc": 350.0,
        },
    ])

    bench = summarize_consolidated_benchmarks(cons_df, focal_id=CESENA, year=2024, core_only=False)
    assert not bench.empty
    m_set = set(bench["metric"])
    assert "direct_correnti_pc" in m_set
    assert "consolidated_correnti_pc" in m_set
    c_row = bench[bench["metric"] == "consolidated_correnti_pc"].iloc[0]
    assert c_row["cesena"] == 1024.0


