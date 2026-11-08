"""Tests for Pillar 3 (`studies.cesena.analysis.economy` Shift-Share and firm-size decomposition)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.core.geo import CESENA
from cifresignificative.studies.cesena.analysis.economy import (
    compare_productivity_and_wages,
    firm_size_decomposition,
    shift_share,
    size_class_by_cluster,
    size_class_summary,
    summarize_frame_sbs_benchmarks,
)


def _make_synthetic_asia() -> pd.DataFrame:
    rows = []
    # Two sectors: ict_software (fast benchmark growth +50%) and construction_re (flat benchmark 0%)
    # Cesena specializes in ict_software and also outperforms within ict_software (+80%)
    data = [
        # (comune_id, year, spec, units, emp)
        (CESENA, 2012, "ict_software", 100.0, 1000.0),
        (CESENA, 2012, "construction_re", 200.0, 1000.0),
        (CESENA, 2024, "ict_software", 150.0, 1800.0),
        (CESENA, 2024, "construction_re", 180.0, 990.0),
        # Peer 1 (Forlì)
        ("040012", 2012, "ict_software", 50.0, 400.0),
        ("040012", 2012, "construction_re", 300.0, 1600.0),
        ("040012", 2024, "ict_software", 60.0, 600.0),
        ("040012", 2024, "construction_re", 300.0, 1600.0),
        # Peer 2 (Rimini) — also add an intermediate year 2018 with only one sector to test the two-year selection guard
        ("099014", 2012, "ict_software", 50.0, 400.0),
        ("099014", 2012, "construction_re", 300.0, 1600.0),
        ("099014", 2018, "ict_software", 55.0, 500.0),
        ("099014", 2024, "ict_software", 60.0, 600.0),
        ("099014", 2024, "construction_re", 300.0, 1600.0),
    ]
    for cid, yr, spec, u, e in data:
        rows.append({
            "comune_id": cid,
            "comune": cid,
            "year": yr,
            "ateco_2d": "62" if spec == "ict_software" else "41",
            "ateco_label": spec,
            "specialization": spec,
            "specialization_label": spec,
            "is_total": False,
            "local_units": u,
            "employees": e,
            "avg_firm_size": e / u,
        })
    return pd.DataFrame(rows)


def test_shift_share_identities_and_planted_sign():
    df = _make_synthetic_asia()
    tbl, summary = shift_share(df, t0=2012, t1=2024, focal_id=CESENA, level="specialization", metric="employees")
    assert len(tbl) == 2
    # Exact accounting identity: NS + IM + CS == delta_abs for every sector and in aggregate pp
    for _, r in tbl.iterrows():
        assert np.isclose(r["ns_abs"] + r["im_abs"] + r["cs_abs"], r["delta_abs"])
        assert np.isclose(r["ns_pp"] + r["im_pp"] + r["cs_pp"], r["total_contrib_pp"])
        assert np.isclose(r["em_comp_pp"] + r["em_alloc_pp"], r["cs_pp"])

    assert np.isclose(
        summary["ns_total_pp"] + summary["im_total_pp"] + summary["cs_total_pp"],
        summary["local_growth_total_pct"],
    )
    # Cesena over-weights fast-growing ICT (50% vs 20% in benchmark) -> positive Industry Mix
    assert summary["im_total_pp"] > 0
    # Cesena grows +80% in ICT vs +50% in benchmark -> positive Competitive Shift in ICT
    ict_row = tbl[tbl["sector"] == "ict_software"].iloc[0]
    assert ict_row["cs_pp"] > 0


def test_firm_size_decomposition_exact_bennet_identity():
    df = _make_synthetic_asia()
    res = firm_size_decomposition(df, t0=2012, t1=2024)
    for _, r in res.iterrows():
        assert np.isclose(r["mix_effect"] + r["within_scaling_effect"], r["delta_avg_size"])


def test_size_class_summary_and_cluster_identities():
    rows = []
    for cid, w50_12, w50_24 in [(CESENA, 100.0, 300.0), ("040012", 100.0, 140.0), ("099014", 100.0, 120.0)]:
        for yr, w09, w10, w50, w250 in [(2012, 400.0, 300.0, w50_12, 200.0), (2024, 360.0, 340.0, w50_24, 240.0)]:
            tot_e = w09 + w10 + w50 + w250
            for sc, u, e in [("W0_9", 100.0, w09), ("W10_49", 20.0, w10), ("W50_249", 5.0, w50), ("W_GE250", 1.0, w250), ("TOTAL", 126.0, tot_e)]:
                for is_tot, ateco, spec in [(True, "TOTAL", "TOTAL"), (False, "10", "agro_industrial")]:
                    rows.append({
                        "comune_id": cid,
                        "year": yr,
                        "ateco_2d": ateco,
                        "ateco_label": spec,
                        "specialization": spec,
                        "specialization_label": spec,
                        "size_class": sc,
                        "size_label": sc,
                        "is_total": is_tot,
                        "local_units": u,
                        "employees": e,
                        "avg_firm_size": e / u,
                    })
    sdf = pd.DataFrame(rows)
    class_df, muni_df = size_class_summary(sdf, t0=2012, t1=2024, focal_id=CESENA, core_only=True)
    assert len(class_df) == 4
    # Sum of size-class growth contributions equals total employment growth (%)
    ces_m = muni_df[muni_df["comune_id"] == CESENA].iloc[0]
    assert np.isclose(class_df["contrib_to_total_growth_pp"].sum(), ces_m["emp_growth_pct"])
    assert np.isclose(class_df["emp_share_t0_pct"].sum(), 100.0)
    assert np.isclose(class_df["emp_share_t1_pct"].sum(), 100.0)
    assert int(ces_m["rank_W50_249_contrib"]) == 1

    clust_df = size_class_by_cluster(sdf, t0=2012, t1=2024, focal_id=CESENA)
    assert len(clust_df) == 1
    r0 = clust_df.iloc[0]
    assert np.isclose(
        r0["delta_W0_9"] + r0["delta_W10_49"] + r0["delta_W50_249"] + r0["delta_W_GE250"],
        r0["delta_total"],
    )


@pytest.mark.skipif(not Path("data/processed/cesena/frame_sbs_municipal.parquet").exists(), reason="cesena data not built")
def test_frame_sbs_benchmarks_and_growth_ranking():
    bench_tot = summarize_frame_sbs_benchmarks(sector="totale")
    assert len(bench_tot) == 7
    # Verify Cesena columns
    assert "cesena_va_per_addetto_eur" in bench_tot.columns
    assert "core_peer_va_per_addetto_eur" in bench_tot.columns
    assert "rank_va_per_addetto_er" in bench_tot.columns

    # Verify 2023 values
    r23 = bench_tot[bench_tot["year"] == 2023].iloc[0]
    assert 55_000 < r23["cesena_va_per_addetto_eur"] < 65_000
    assert 25_000 < r23["cesena_retrib_per_dipendente_eur"] < 32_000

    # Industry benchmarks
    bench_ind = summarize_frame_sbs_benchmarks(sector="industria")
    r23_ind = bench_ind[bench_ind["year"] == 2023].iloc[0]
    assert r23_ind["cesena_va_per_addetto_eur"] > 80_000

    # Cumulative growth ranking
    growth_df = compare_productivity_and_wages(sector="totale", t0=2017, t1=2023)
    assert len(growth_df) == 13
    ces_row = growth_df[growth_df["comune_id"] == CESENA].iloc[0]
    # Cesena ranks #1 on total job growth between 2017 and 2023 (+14.0%)
    assert int(ces_row["rank_growth_addetti"]) == 1
    assert ces_row["growth_addetti_pct"] > 13.0
    # Positive real labor productivity growth
    assert ces_row["growth_real_vap_pct"] > 0


def _asia_frame(series: dict[tuple[str, str], dict[int, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"comune_id": cid, "year": y, "ateco_2d": sector, "ateco_label": sector, "specialization": "x", "is_total": False, "employees": v, "local_units": 10.0}
        for (cid, sector), vals in series.items() for y, v in vals.items()
    ])


def test_employment_anomalies_flags_single_year_dip_and_rebound():
    from cifresignificative.studies.cesena.analysis.economy import employment_anomalies

    df = _asia_frame({
        ("040007", "10"): {2021: 3400.0, 2022: 1500.0, 2023: 4900.0, 2024: 4250.0},
        ("040012", "10"): {2021: 900.0, 2022: 910.0, 2023: 950.0, 2024: 980.0},
    })
    out = employment_anomalies(df)
    assert ("10", 2022, "reversal") in set(zip(out["ateco_2d"], out["year"], out["kind"], strict=True))
    assert set(out["comune_id"]) == {"040007"}


def test_employment_sensitivity_removes_the_planted_sector():
    from cifresignificative.studies.cesena.analysis.economy import (
        employment_sensitivity,
    )

    series = {}
    for i, cid in enumerate(("040007", "040012", "099014", "039014")):
        series[(cid, "47")] = {2012: 1000.0, 2021: 1000.0 + i * 50, 2022: 1000.0 + i * 50, 2024: 1000.0 + i * 50}
        series[(cid, "10")] = {2012: 100.0, 2021: 100.0, 2022: 100.0, 2024: 2000.0 if cid == "040007" else 100.0}
    out = employment_sensitivity(_asia_frame(series)).set_index(["variant", "t1", "metric"])
    assert out.loc[("all_sectors", 2024, "employment_growth_pct"), "rank"] == 1
    assert out.loc[("without_food_10", 2024, "employment_growth_pct"), "rank"] == 4


def test_sector_size_swing_locates_the_planted_large_unit():
    from cifresignificative.studies.cesena.analysis.economy import sector_size_swing

    rows = []
    for y, big, small in ((2021, 2500.0, 240.0), (2022, 0.0, 238.0), (2023, 4000.0, 236.0)):
        rows += [
            {"comune_id": "040007", "year": y, "ateco_2d": "10", "size_class": "W_GE250", "employees": big, "local_units": 2.0 if big else 1.0},
            {"comune_id": "040007", "year": y, "ateco_2d": "10", "size_class": "W0_9", "employees": small, "local_units": 70.0},
            {"comune_id": "040007", "year": y, "ateco_2d": "81", "size_class": "W0_9", "employees": 10.0, "local_units": 1.0},
        ]
    out = sector_size_swing(pd.DataFrame(rows), ("10",), (2021, 2022, 2023)).set_index("size_class")
    assert out.loc["W_GE250", "delta_2022"] == -2500.0 and out.loc["W0_9", "delta_2022"] == -2.0
    assert out["delta_2022"].sum() == -2502.0
    assert abs(out.loc["W_GE250", "share_of_change_2022"] - 2500 / 2502) < 1e-12
    both = sector_size_swing(pd.DataFrame(rows), ("10", "81"), (2021, 2022)).set_index("size_class")
    assert both.loc["W0_9", "emp_2021"] == 250.0
