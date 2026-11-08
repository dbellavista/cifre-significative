"""Data loaders, benchmark helpers, and master panel builder for the `cesena` study (`studies.cesena.analysis.data`).

Builds and loads processed Parquet tables under `data/processed/cesena/`:
- `fiscal.parquet`: municipal balance sheets (`2008-2022` BDAP/OpenBilanci + `2008-2024` MEF Addizionale IRPEF), deflated to constant 2020 EUR per capita, with explicit `regime` (`"dpr194"` vs `"dlgs118"`) and `flag` (`"break_dlgs118"`).
- `missions.parquet`: harmonized municipal expenditure by Mission (`M01`..`M20`, `M50`, `2016-2022`).
- `projects.parquet`: OpenCoesione municipal investment projects across the 13 ER peer municipalities (`10,377` rows, `511` for Cesena).
- `asia_sectors.parquet`: ISTAT ASIA active local units and employees by 2-digit ATECO division and specialization cluster (`2012-2024`).
- `irpef_brackets.parquet` & `irpef_summary.parquet`: MEF municipal IRPEF declarations by bracket and income source (`2008-2024`).
- `demography.parquet`: ISTAT municipal population, age cohorts, aging index, and 25-44 cohort net attractiveness (`2008-2026`).
- `panel_comuni.parquet`: master annual municipality x year panel (`2008-2025`, 13 ER municipalities) combining all four pillars plus administrative cycle labels.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cifresignificative.core.geo import (
    BELT_COMUNI_IDS,
    CESENA,
    COMUNI_BY_ID,
    CORE_PEER_IDS,
    ER_COMUNI,
    EXTENDED_PEER_IDS,
)
from cifresignificative.core.sources import (
    agenzia_entrate_omi,
    bdap,
    coeweb_trade,
    istat_agricoltura,
    istat_asia,
    istat_demo,
    istat_frame_sbs,
    istat_pendolarismo,
    mef_comuni,
    mur_ustat,
    opencivitas,
    opencoesione_municipal,
    siope,
)
from cifresignificative.studies.cesena.analysis.consolidato import load_consolidated
from cifresignificative.studies.cesena.analysis.conto_annuale import (
    load_conto_annuale,
    load_flexible,
    load_funds,
    load_levels,
)
from cifresignificative.studies.cesena.analysis.fabbisogni import (
    load_limits,
    load_retirements,
    load_staff,
)
from cifresignificative.studies.cesena.analysis.fiscal import enrich_fiscal_panel
from cifresignificative.studies.cesena.analysis.indicatori import load_indicators
from cifresignificative.studies.cesena.analysis.investments import (
    annual_investment_flows,
    enrich_projects,
)
from cifresignificative.studies.cesena.analysis.outcomes import (
    build_comprensorio_panel,
    compute_demographic_outcomes,
    compute_irpef_distribution,
)
from cifresignificative.studies.cesena.analysis.rendiconto import (
    RAW_DIR as RENDICONTO_DIR,
)
from cifresignificative.studies.cesena.analysis.rendiconto import load_accruals
from cifresignificative.studies.cesena.analysis.risk_reports import load_risk
from cifresignificative.studies.cesena.analysis.semestrale import load_semestrale
from cifresignificative.studies.cesena.analysis.tari import add_ex_tari, load_tari

PROCESSED_DIR = Path("data/processed/cesena")
ADMIN_CSV = Path("data/external/cesena_administrations.csv")


def load_administrations(path: Path = ADMIN_CSV) -> pd.DataFrame:
    """Load `data/external/cesena_administrations.csv` with parsed dates."""
    df = pd.read_csv(path)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def administration_for_year(year: int) -> str:
    """Map calendar year to its predominant Cesena municipal administration cycle."""
    if year < 2009:
        return "Pre-2009"
    if year <= 2013:
        return "Lucchi I (2009-2014)"
    if year <= 2018:
        return "Lucchi II (2014-2019)"
    if year <= 2023:
        return "Lattuca I (2019-2024)"
    return "Lattuca II (2024+)"


def peer_median_ex_cesena(
    df: pd.DataFrame,
    value_col: str,
    group_cols: tuple[str, ...] = ("year",),
    core_only: bool = True,
    focal_id: str = CESENA,
) -> pd.DataFrame:
    """Compute the peer median, interquartile range (`p25`, `p75`), peer count, and missing peers strictly excluding `focal_id`."""
    target_peers = set(CORE_PEER_IDS if core_only else EXTENDED_PEER_IDS) - {focal_id}
    sub = df[df["comune_id"].isin(target_peers)].copy()
    g_cols = list(group_cols)

    rows: list[dict] = []
    for key, grp in sub.groupby(g_cols, sort=True):
        key_tup = (key,) if not isinstance(key, tuple) else key
        valid = grp.dropna(subset=[value_col])
        present_ids = set(valid["comune_id"].astype(str))
        missing_names = sorted(COMUNI_BY_ID[cid].name for cid in (target_peers - present_ids) if cid in COMUNI_BY_ID)
        vals = valid[value_col].to_numpy(dtype=float)
        row = dict(zip(g_cols, key_tup, strict=False))
        row.update({
            "peer_median": float(np.median(vals)) if len(vals) else np.nan,
            "peer_p25": float(np.percentile(vals, 25)) if len(vals) else np.nan,
            "peer_p75": float(np.percentile(vals, 75)) if len(vals) else np.nan,
            "peer_mean": float(np.mean(vals)) if len(vals) else np.nan,
            "n_peers": len(vals),
            "missing_peers": ", ".join(missing_names),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def cesena_vs_peers_series(
    df: pd.DataFrame,
    value_col: str,
    core_only: bool = True,
    focal_id: str = CESENA,
) -> pd.DataFrame:
    """Build an annual comparison table of `focal_id` vs `peer_median_ex_cesena` (both Core Peers and All ER Peers)."""
    ces = (
        df[df["comune_id"] == focal_id][["year", value_col]]
        .dropna(subset=[value_col])
        .rename(columns={value_col: "cesena"})
    )
    med_core = peer_median_ex_cesena(df, value_col, core_only=True, focal_id=focal_id).rename(
        columns={
            "peer_median": "core_peer_median",
            "peer_p25": "core_peer_p25",
            "peer_p75": "core_peer_p75",
            "n_peers": "n_core_peers",
            "missing_peers": "missing_core_peers",
        }
    )
    med_all = peer_median_ex_cesena(df, value_col, core_only=False, focal_id=focal_id)[
        ["year", "peer_median", "n_peers", "missing_peers"]
    ].rename(
        columns={
            "peer_median": "er_peer_median",
            "n_peers": "n_er_peers",
            "missing_peers": "missing_er_peers",
        }
    )
    out = ces.merge(med_core, on="year", how="outer").merge(med_all, on="year", how="outer").sort_values("year", ignore_index=True)
    bench_col = "core_peer_median" if core_only else "er_peer_median"
    out["gap_vs_benchmark"] = out["cesena"] - out[bench_col]
    return out


def build_cesena_all(out_dir: Path = PROCESSED_DIR) -> dict[str, Path]:
    """Build all processed Parquet datasets for the `cesena` study under `data/processed/cesena/`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1. Demography (2008-2026)
    demo_raw = istat_demo.load_all()
    demo_df = compute_demographic_outcomes(demo_raw)
    p_demo = out_dir / "demography.parquet"
    demo_df.to_parquet(p_demo, index=False)
    paths["demography"] = p_demo

    # 2. MEF IRPEF brackets & summary (2008-2024)
    brackets_df, irpef_sum_raw = mef_comuni.load_all()
    irpef_dist_df = compute_irpef_distribution(brackets_df, summary_df=irpef_sum_raw, base_year=2020)
    p_brackets = out_dir / "irpef_brackets.parquet"
    p_irpef = out_dir / "irpef_summary.parquet"
    brackets_df.to_parquet(p_brackets, index=False)
    irpef_dist_df.to_parquet(p_irpef, index=False)
    paths["irpef_brackets"] = p_brackets
    paths["irpef_summary"] = p_irpef

    # 3. ISTAT ASIA 2-digit sectors & employment size classes (2012-2024)
    asia_df = istat_asia.load_all()
    p_asia = out_dir / "asia_sectors.parquet"
    asia_df.to_parquet(p_asia, index=False)
    paths["asia_sectors"] = p_asia

    from cifresignificative.studies.cesena.analysis.economy import (
        employment_anomalies,
        employment_sensitivity,
    )

    employment_anomalies(asia_df).to_parquet(out_dir / "asia_anomalies.parquet", index=False)
    employment_sensitivity(asia_df).to_parquet(out_dir / "employment_sensitivity.parquet", index=False)
    paths["asia_anomalies"] = out_dir / "asia_anomalies.parquet"
    paths["employment_sensitivity"] = out_dir / "employment_sensitivity.parquet"

    if istat_asia.size_raw_path().exists():
        asia_size_df = istat_asia.load_size_classes()
        p_asia_size = out_dir / "asia_size_classes.parquet"
        asia_size_df.to_parquet(p_asia_size, index=False)
        paths["asia_size_classes"] = p_asia_size

    if istat_asia.areas_raw_path().exists():
        p_areas = out_dir / "asia_areas.parquet"
        istat_asia.load_areas().to_parquet(p_areas, index=False)
        paths["asia_areas"] = p_areas

    # 4. OpenCoesione municipal projects (2014-2020)
    proj_raw = opencoesione_municipal.load_municipal_projects(cycle="2014-2020")
    proj_df = enrich_projects(proj_raw, demo_df=demo_df, base_year=2020)
    p_proj = out_dir / "projects.parquet"
    proj_df.to_parquet(p_proj, index=False)
    paths["projects"] = p_proj

    # 4b. OpenCoesione 2021-2027 projects
    proj_2127_raw = opencoesione_municipal.load_municipal_projects(cycle="2021-2027")
    if not proj_2127_raw.empty:
        proj_2127_df = enrich_projects(proj_2127_raw, demo_df=demo_df, base_year=2020, fallback_year=None)
        p_proj_2127 = out_dir / "projects_2021_2027.parquet"
        proj_2127_df.to_parquet(p_proj_2127, index=False)
        paths["projects_2021_2027"] = p_proj_2127

    # 5. BDAP / OpenBilanci / Finanza Locale municipal budgets (2008-2022) + Missions (2016-2022)
    budgets_raw, missions_df = bdap.load_all()
    fiscal_df = enrich_fiscal_panel(budgets_raw, demo_df=demo_df, irpef_summary_df=irpef_sum_raw, base_year=2020)
    p_fiscal = out_dir / "fiscal.parquet"
    p_missions = out_dir / "missions.parquet"
    fiscal_df.to_parquet(p_fiscal, index=False)
    missions_df.to_parquet(p_missions, index=False)
    paths["fiscal"] = p_fiscal
    paths["missions"] = p_missions

    # 5c. MUR USTAT Higher Education & University Enrolments (2010-2026)
    if (mur_ustat.RAW_DIR / "iscritti_2019_2025.csv").exists():
        mur_ustat.build(out_dir)
        paths["mur_university_summary"] = out_dir / "mur_university_summary.parquet"
        paths["mur_university_enrolments"] = out_dir / "mur_university_enrolments.parquet"
        paths["mur_university_graduates"] = out_dir / "mur_university_graduates.parquet"

    # 5d. SIOPE Municipal & Unioni Cash Expenditure & Consolidated Perimeter (2016-2024)
    if siope.RAW_DIR.exists():
        siope_paths = siope.build(out_dir)
        paths.update(siope_paths)

    # 5d2. Cesena accruals 2022+ from the Rendiconto CSVs of the transparency portal (single-city anchor)
    if RENDICONTO_DIR.exists() and len(acc := load_accruals()):
        if len(tari := load_tari()):  # the waste-tax levy until 2022 (TARI), for the ex-TARI ratio
            acc = add_ex_tari(acc, tari)
        acc.to_parquet(out_dir / "rendiconto_accrual.parquet", index=False)
        paths["rendiconto_accrual"] = out_dir / "rendiconto_accrual.parquet"

    # 5d3. Cesena official budget indicators (rendiconto PDFs, 2023+)
    if RENDICONTO_DIR.exists() and len(ind := load_indicators()):
        ind.to_parquet(out_dir / "indicatori_rendiconto.parquet", index=False)
        paths["indicatori_rendiconto"] = out_dir / "indicatori_rendiconto.parquet"

    # 5d4. Cesena Conto annuale del personale summary 2020-2024 (permanent staff and labour cost)
    if RENDICONTO_DIR.exists() and len(ca := load_conto_annuale()):
        ca.to_parquet(out_dir / "conto_annuale.parquet", index=False)
        paths["conto_annuale"] = out_dir / "conto_annuale.parquet"
        for name, df in (("conto_annuale_flexible", load_flexible()), ("conto_annuale_levels", load_levels()), ("conto_annuale_funds", load_funds())):
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
            paths[name] = out_dir / f"{name}.parquet"

    # 5d5. staffing plan in the DUPs: personnel-spending limit table (plan vs outturn), staff in service, retirements forecast
    if RENDICONTO_DIR.exists() and len(lim := load_limits()):
        for name, df in (("fabbisogni_limits", lim), ("fabbisogni_staff", load_staff()), ("fabbisogni_retirements", load_retirements())):
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
            paths[name] = out_dir / f"{name}.parquet"

    # 5d6. controlled companies: crisis-risk alarms and ratios (annual accounts 2023 and 2025)
    if RENDICONTO_DIR.exists() and len(risk := load_risk()):
        risk.to_parquet(out_dir / "controlled_risk.parquet", index=False)
        paths["controlled_risk"] = out_dir / "controlled_risk.parquet"

    # 5d7. controlled companies: half-year income statement and net financial position (30 June 2024 and 2026)
    if RENDICONTO_DIR.exists() and len(half := load_semestrale()):
        half.to_parquet(out_dir / "controlled_halfyear.parquet", index=False)
        paths["controlled_halfyear"] = out_dir / "controlled_halfyear.parquet"

    # 5d8. consolidated accounts of the Comune group, 2018, 2019 (OCR cache), 2020 (prior-year column) and 2021 (old transparency site)
    if len((cons := load_consolidated())[0]):
        for name, df in (("consolidated_statements", cons[0]), ("consolidated_perimeter", cons[1])):
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
            paths[name] = out_dir / f"{name}.parquet"

    # 5e. ISTAT FRAME-SBS Territorial Economic Accounts (2017-2023)
    if istat_frame_sbs.RAW_DIR.exists():
        p_frame = istat_frame_sbs.build_frame_sbs_panel(out_dir=out_dir)
        istat_frame_sbs.build_frame_sbs_summary(p_frame, out_dir=out_dir)
        paths["frame_sbs_municipal"] = out_dir / "frame_sbs_municipal.parquet"
        paths["frame_sbs_summary"] = out_dir / "frame_sbs_summary.parquet"

    # 5f. Agenzia delle Entrate OMI Real Estate Quotations & Transactions (2011-2025)
    if agenzia_entrate_omi.RAW_DIR.exists():
        omi_paths = agenzia_entrate_omi.build(out_dir)
        paths.update(omi_paths)

    # 5g. ISTAT 15th Census Origin-Destination Commuting Matrices (2011)
    if istat_pendolarismo.RAW_DIR.exists():
        pendo_paths = istat_pendolarismo.build(raw_dir=istat_pendolarismo.RAW_DIR, out_dir=out_dir)
        paths.update(pendo_paths)

    # 5h. Coeweb / ISTAT Foreign Trade & Provincial Exports (1991-2025)
    if coeweb_trade.RAW_DIR.exists():
        coeweb_trade.build_all(raw_dir=coeweb_trade.RAW_DIR, processed_dir=out_dir)
        paths["coeweb_provinces"] = out_dir / "coeweb_provinces_export.parquet"
        paths["coeweb_time_series"] = out_dir / "coeweb_emr_time_series.parquet"
        paths["coeweb_commodities"] = out_dir / "coeweb_commodities_export.parquet"
        paths["coeweb_countries"] = out_dir / "coeweb_countries_balance.parquet"

    # 5i. ISTAT 7th Agricultural Census (2020) Land Use & Farm Size Classes
    if istat_agricoltura.RAW_DIR.exists():
        istat_agricoltura.build_all(raw_dir=istat_agricoltura.RAW_DIR, processed_dir=out_dir)
        paths["agri_sat_land_use"] = out_dir / "agri_sat_land_use_provinces.parquet"
        paths["agri_sau_size_classes"] = out_dir / "agri_sau_size_classes_provinces.parquet"

    # 5j. SOSE / MEF OpenCivitas Standard Expenditure Needs & Service Levels (2023)
    if opencivitas.RAW_DIR.exists():
        opencivitas.build_all(raw_dir=opencivitas.RAW_DIR, processed_dir=out_dir)
        paths["opencivitas_summary"] = out_dir / "opencivitas_municipal_summary.parquet"
        paths["opencivitas_detailed"] = out_dir / "opencivitas_indicators_detailed.parquet"

    # 6. Master municipality x year panel (2008-2025)
    grid = pd.MultiIndex.from_product(
        [[c.id for c in ER_COMUNI], range(2008, 2026)],
        names=["comune_id", "year"],
    ).to_frame().reset_index(drop=True)
    grid["comune"] = grid["comune_id"].map(lambda cid: COMUNI_BY_ID[cid].name)
    grid["nuts3"] = grid["comune_id"].map(lambda cid: COMUNI_BY_ID[cid].nuts3)
    grid["core_peer"] = grid["comune_id"].map(lambda cid: COMUNI_BY_ID[cid].core_peer)
    grid["administration"] = grid["year"].map(administration_for_year)

    # Merge demography
    demo_cols = [
        "comune_id", "year", "pop_total", "pop_0_14", "pop_15_64", "pop_25_44", "pop_65_plus",
        "aging_index", "dependency_ratio", "share_25_44_pct", "share_65_plus_pct", "pop_growth_pct",
        "natural_balance", "net_migration_total", "net_migration_internal", "net_migration_international", "registry_adjustment", "net_migr_rate_per_1k",
        "net_cohort_25_44", "net_cohort_25_44_per_1k", "net_cohort_25_44_3y_ma",
    ]
    panel = grid.merge(demo_df[[c for c in demo_cols if c in demo_df.columns]], on=["comune_id", "year"], how="left")

    # Merge IRPEF distribution & tax base
    irpef_cols = [
        "comune_id", "year", "taxpayers_total", "taxpayers_pos", "income_pos_real_eur",
        "mean_income_eur", "mean_income_real_eur", "median_income_eur", "median_income_real_eur",
        "gini_trap", "gini_pareto", "below_15k_payer_share_pct", "top_55k_income_share_pct", "top_120k_income_share_pct",
        "addizionale_comunale_eur", "addizionale_comunale_real_eur",
        "dipendente_share_pct", "pensione_share_pct", "autonomo_share_pct",
    ]
    panel = panel.merge(irpef_dist_df[[c for c in irpef_cols if c in irpef_dist_df.columns]], on=["comune_id", "year"], how="left")
    panel["irpef_taxbase_real_pc"] = panel["income_pos_real_eur"] / panel["pop_total"].replace(0, np.nan)
    panel["mef_addizionale_real_pc"] = panel["addizionale_comunale_real_eur"] / panel["pop_total"].replace(0, np.nan)

    # Merge ASIA total local units, employees, average firm size, and key cluster employment shares
    asia_tot = (
        asia_df[asia_df["is_total"].astype(bool)][["comune_id", "year", "local_units", "employees", "avg_firm_size"]]
        .drop_duplicates(["comune_id", "year"])
    )
    asia_clusters = (
        asia_df[~asia_df["is_total"].astype(bool)]
        .groupby(["comune_id", "year", "specialization"], as_index=False)["employees"]
        .sum()
        .pivot(index=["comune_id", "year"], columns="specialization", values="employees")
        .reset_index()
    )
    asia_clusters.columns.name = None
    panel = panel.merge(asia_tot, on=["comune_id", "year"], how="left")
    for cl in ("agro_industrial", "ict_software", "wellness_biomed", "construction_re", "tradable_manuf"):
        if cl in asia_clusters.columns:
            asia_clusters[f"emp_share_{cl}_pct"] = (
                asia_clusters[cl] / asia_clusters.set_index(["comune_id", "year"]).sum(axis=1).to_numpy(dtype=float) * 100.0
            )
    keep_cl = ["comune_id", "year"] + [c for c in asia_clusters.columns if c.startswith("emp_share_")]
    panel = panel.merge(asia_clusters[keep_cl], on=["comune_id", "year"], how="left")
    panel["employees_per_1k_pop"] = panel["employees"] / panel["pop_total"].replace(0, np.nan) * 1000.0
    panel["local_units_per_1k_pop"] = panel["local_units"] / panel["pop_total"].replace(0, np.nan) * 1000.0

    # Merge annual OpenCoesione project flows
    proj_flows = annual_investment_flows(proj_df, demo_df=demo_df, base_year=2020)
    panel = panel.merge(proj_flows, on=["comune_id", "year"], how="left")
    for col in ("projects_started_n", "projects_allocated_real_eur", "projects_disbursed_real_eur", "projects_allocated_real_pc", "projects_disbursed_real_pc"):
        if col in panel.columns:
            panel[col] = np.where((panel["year"] >= 2008) & (panel["year"] <= 2024), panel[col].fillna(0.0), panel[col])

    # Merge BDAP / OpenBilanci fiscal indicators (consuntivo rows for panel regressions; keep regime & flag)
    fisc_cons = fiscal_df[fiscal_df["budget_type"] == "consuntivo"].copy()
    fisc_cols = [
        "comune_id", "year", "regime", "flag",
        "spese_correnti_comp_real_pc", "spese_capitale_comp_real_pc", "spese_capitale_cassa_real_pc",
        "entrate_tributarie_comp_real_pc", "imposte_tasse_comp_real_pc", "imu_ici_comp_real_pc",
        "entrate_correnti_comp_real_pc", "capital_share_pct", "autonomia_finanziaria_pct",
        "rigidity_official_pct", "rigidity_struct_pct", "current_absorption_pct",
        "collection_rate_correnti_pct", "collection_rate_tributarie_pct", "collection_rate_proprie_pct",
    ]
    panel = panel.merge(fisc_cons[[c for c in fisc_cols if c in fisc_cons.columns]], on=["comune_id", "year"], how="left")

    # Merge MUR USTAT higher education indicators (2010-2025)
    if (out_dir / "mur_university_summary.parquet").exists():
        uni_s = pd.read_parquet(out_dir / "mur_university_summary.parquet")
        uni_cols = [
            "comune_id", "calendar_year", "students_total", "students_stem", "students_ict",
            "students_engineering", "students_psychology", "stem_share", "ict_share",
            "engineering_share", "psychology_share", "graduates_total",
        ]
        uni_s = uni_s[[c for c in uni_cols if c in uni_s.columns]].rename(columns={"calendar_year": "year"})
        panel = panel.merge(uni_s, on=["comune_id", "year"], how="left")
        panel["students_per_1k_pop"] = panel["students_total"] / panel["pop_total"].replace(0, np.nan) * 1000.0
        panel["graduates_per_1k_pop"] = panel["graduates_total"] / panel["pop_total"].replace(0, np.nan) * 1000.0

    # Stationarity transforms (annual log growth x100 and first differences) for Phase 3 econometric tests
    panel = panel.sort_values(["comune_id", "year"], ignore_index=True)
    for col in ("local_units", "employees", "avg_firm_size", "mean_income_real_eur", "median_income_real_eur", "irpef_taxbase_real_pc", "pop_total"):
        if col in panel.columns:
            prev = panel.groupby("comune_id")[col].shift(1)
            panel[f"dlog_{col}"] = np.log(panel[col].where(panel[col] > 0) / prev.where(prev > 0)) * 100.0
    for col in ("gini_pareto", "aging_index", "share_25_44_pct"):
        if col in panel.columns:
            panel[f"diff_{col}"] = panel.groupby("comune_id")[col].diff()

    p_panel = out_dir / "panel_comuni.parquet"
    panel.to_parquet(p_panel, index=False)
    paths["panel_comuni"] = p_panel

    # 7. Comprensorio Cesenate (Cesena + 16 belt municipalities, 2008-2025)
    if (istat_demo.RAW_DIR / "demo_belt_hist_2008_2019.csv").exists() and istat_asia.belt_raw_path().exists():
        demo_belt = istat_demo.load_belt_balance()
        asia_belt = istat_asia.load_belt_totals()
        b_belt, s_belt = mef_comuni.load_all(comuni_ids=set(BELT_COMUNI_IDS))
        irpef_belt = compute_irpef_distribution(b_belt, summary_df=s_belt, base_year=2020)
        belt_df = build_comprensorio_panel(panel, demo_belt, irpef_belt, asia_belt)
        p_belt = out_dir / "belt_comuni.parquet"
        belt_df.to_parquet(p_belt, index=False)
        paths["belt_comuni"] = p_belt

    # 8. Phase 3 experiment tables (administrative cycle scorecard, lagged Two-Way FE screen, placebo ranking)
    from cifresignificative.studies.cesena.analysis.experiments import (
        run_all_experiments,
    )

    run_all_experiments(panel, out_dir=out_dir)
    paths["cycle_scorecard"] = out_dir / "cycle_scorecard.parquet"
    paths["lagged_fe_results"] = out_dir / "lagged_fe_results.parquet"
    paths["placebo_ranking"] = out_dir / "placebo_ranking.parquet"
    paths["synthetic_cesena"] = out_dir / "synthetic_cesena.parquet"
    paths["synthetic_summary"] = out_dir / "synthetic_summary.parquet"
    paths["synthetic_in_space"] = out_dir / "synthetic_in_space.parquet"
    paths["synthetic_in_time"] = out_dir / "synthetic_in_time.parquet"
    if (out_dir / "extended_placebo_ranking.parquet").exists():
        paths["extended_placebo_ranking"] = out_dir / "extended_placebo_ranking.parquet"

    # 8b. Review sensitivity tables
    from cifresignificative.studies.cesena.analysis.robustness import (
        cohort_by_period,
        housing_gradient_test,
        income_composition,
        population_balance,
    )

    income_composition(panel).to_parquet(out_dir / "income_composition.parquet", index=False)
    cohort_by_period(demo_df).to_parquet(out_dir / "cohort_by_period.parquet", index=False)
    if (out_dir / "belt_comuni.parquet").exists():
        belt_saved = pd.read_parquet(out_dir / "belt_comuni.parquet")
        population_balance(belt_saved).to_parquet(out_dir / "population_balance.parquet", index=False)
        if (out_dir / "omi_quotations_municipal.parquet").exists():
            housing_gradient_test(pd.read_parquet(out_dir / "omi_quotations_municipal.parquet"), belt_saved, demo_df).to_parquet(out_dir / "housing_gradient_test.parquet", index=False)

    # 9. May 2023 Romagna flood shock & resilience panel
    if (out_dir / "siope_cash_panel.parquet").exists() and (out_dir / "frame_sbs_municipal.parquet").exists():
        from cifresignificative.studies.cesena.analysis.flood import (
            build_flood_shock_panel,
        )

        build_flood_shock_panel(out_dir=out_dir)
        paths["flood_shock_panel"] = out_dir / "flood_shock_panel.parquet"

    # 10. Every figure quoted in findings.md
    from cifresignificative.studies.cesena.analysis.claims import build_claims

    build_claims(out_dir)
    paths["claims"] = out_dir / "claims.parquet"

    return paths


def _load_parquet(name: str, out_dir: Path | None = None) -> pd.DataFrame:
    d = out_dir if out_dir is not None else PROCESSED_DIR
    p = d / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}; run `uv run python -m cifresignificative.studies.cesena.analysis.data`"
        )
    return pd.read_parquet(p)


def load_demography(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("demography", out_dir)


def load_irpef_brackets(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("irpef_brackets", out_dir)


def load_irpef_summary(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("irpef_summary", out_dir)


def load_asia(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("asia_sectors", out_dir)


def load_asia_size_classes(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("asia_size_classes", out_dir)


def load_projects(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("projects", out_dir)


def load_fiscal(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("fiscal", out_dir)


def load_missions(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("missions", out_dir)


def load_panel_comuni(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("panel_comuni", out_dir)


def load_belt_comuni(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("belt_comuni", out_dir)



def load_scorecard(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("cycle_scorecard", out_dir)


def load_lagged_fe(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("lagged_fe_results", out_dir)


def load_placebo_ranking(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("placebo_ranking", out_dir)


def load_extended_placebo_ranking(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("extended_placebo_ranking", out_dir)


def load_synthetic_cesena(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("synthetic_cesena", out_dir)


def load_synthetic_summary(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("synthetic_summary", out_dir)


def load_synthetic_in_space(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("synthetic_in_space", out_dir)


def load_synthetic_in_time(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("synthetic_in_time", out_dir)


def load_university_summary(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("mur_university_summary", out_dir)


def load_university_enrolments(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("mur_university_enrolments", out_dir)


def load_university_graduates(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("mur_university_graduates", out_dir)


def load_projects_2127(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("projects_2021_2027", out_dir)


def load_siope_panel(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("siope_cash_panel", out_dir)


def load_siope_consolidated(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("siope_consolidated", out_dir)


def load_frame_sbs_panel(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("frame_sbs_municipal", out_dir)


def load_frame_sbs_summary(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("frame_sbs_summary", out_dir)


def load_flood_shock(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("flood_shock_panel", out_dir)


def load_omi_municipal(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("omi_quotations_municipal", out_dir)


def load_omi_zones(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("omi_zones_cesena", out_dir)


def load_omi_transactions(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("omi_transactions_panel", out_dir)


def load_commuting_summary(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("commuting_municipal_summary", out_dir)


def load_commuting_od(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("commuting_od_matrix", out_dir)


def load_commuting_corridors(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return _load_parquet("commuting_belt_corridors", out_dir)


def load_coeweb_provinces(out_dir: Path | None = None) -> pd.DataFrame:
    return _load_parquet("coeweb_provinces_export", out_dir)


def load_coeweb_time_series(out_dir: Path | None = None) -> pd.DataFrame:
    return _load_parquet("coeweb_emr_time_series", out_dir)


def load_coeweb_commodities(out_dir: Path | None = None) -> pd.DataFrame:
    return _load_parquet("coeweb_commodities_export", out_dir)


def load_coeweb_countries(out_dir: Path | None = None) -> pd.DataFrame:
    return _load_parquet("coeweb_countries_balance", out_dir)


def load_agri_land_use(out_dir: Path | None = None) -> pd.DataFrame:
    return _load_parquet("agri_sat_land_use_provinces", out_dir)


def load_agri_size_classes(out_dir: Path | None = None) -> pd.DataFrame:
    return _load_parquet("agri_sau_size_classes_provinces", out_dir)


def load_opencivitas_summary(out_dir: Path | None = None) -> pd.DataFrame:
    return _load_parquet("opencivitas_municipal_summary", out_dir)


def load_opencivitas_detailed(out_dir: Path | None = None) -> pd.DataFrame:
    return _load_parquet("opencivitas_indicators_detailed", out_dir)







if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    built = build_cesena_all()
    for k, p in built.items():
        df = pd.read_parquet(p)
        print(f"{k:16s} -> {p} ({len(df)} rows)")
