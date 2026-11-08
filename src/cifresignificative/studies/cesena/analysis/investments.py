"""Pillar 2 — Public Investments, Cohesion/National Projects & Execution Capacity (`studies.cesena.analysis.investments`).

Implements:
1. **Deflated municipal project dataset (`enrich_projects`)**:
   - Deflates allocated public funding (`finanz_pub_eur`) and cumulative payments (`pagamenti_eur`) to constant 2020 EUR using the project start year. `fallback_year` fills missing start dates (2016, mid-cycle, for 2014-2020); with `None` those amounts stay missing, as in the open 2021-2027 cycle, whose rows all lack a start date and are compared in nominal euro.
   - Flags single-municipality (`n_comuni == 1`) vs multi-municipality (`n_comuni > 1`) projects so we can run a sensitivity check on equal-split allocation.
2. **Execution & lag summary by municipality and theme (`summarize_investments_by_comune`, `summarize_investments_by_theme`)**:
   - Total allocated vs disbursed funding per capita (`allocated_real_pc`, `disbursed_real_pc`).
   - Financial execution rate (`disbursement_rate_pct = sum(pagamenti) / sum(finanz_pub) * 100`).
   - Physical completion rate (`completion_rate_pct` of projects in `"Concluso"` or `"Liquidato"` state).
   - Implementation duration (`lag_days`, median and mean days from actual start to actual end for completed projects) and schedule overrun (`delay_days`, actual end minus planned end).
3. **Annual municipal project flow panel (`annual_investment_flows`)**:
   - Aggregates project starts (`projects_started_n`, `projects_allocated_real_eur`, `projects_disbursed_real_eur`) by `(comune_id, start_year)` to feed into the Phase 3 econometric evaluation.
"""
import numpy as np
import pandas as pd

from cifresignificative.core.deflators import deflate, italy_deflator
from cifresignificative.core.geo import CESENA, COMUNI_BY_ID


def enrich_projects(projects_df: pd.DataFrame, demo_df: pd.DataFrame | None = None, base_year: int = 2020, fallback_year: int | None = 2016) -> pd.DataFrame:
    """Attach constant-2020 EUR amounts, completion flags, and single-municipality indicators to OpenCoesione municipal projects."""
    out = projects_df.copy()
    defl = italy_deflator(base_year=base_year)
    ref_year = out["start_year"].fillna(fallback_year if fallback_year is not None else np.nan).clip(lower=int(defl.index.min()), upper=int(defl.index.max()))

    alloc_col = "allocated_eur" if "allocated_eur" in out.columns else "finanz_pub_eur"
    disb_col = "disbursed_eur" if "disbursed_eur" in out.columns else "pagamenti_eur"
    status_col = "status" if "status" in out.columns else "stato_progetto"

    out["allocated_real_eur"] = deflate(out[alloc_col], ref_year, base_year=base_year, defl=defl)
    out["disbursed_real_eur"] = deflate(out[disb_col], ref_year, base_year=base_year, defl=defl)
    out["finanz_pub_real_eur"] = out["allocated_real_eur"]
    out["pagamenti_real_eur"] = out["disbursed_real_eur"]
    out["is_single_comune"] = out["n_comuni"] == 1
    out["is_completed"] = out[status_col].astype(str).str.lower().isin(["concluso", "liquidato"])
    out["is_active_or_done"] = ~out[status_col].astype(str).str.lower().isin(["non_avviato", "non avviato", "annullato"])
    out["is_public_works"] = out["cup_nature"].astype(str).str.upper().str.contains("LAVORI PUBBLICI", na=False) if "cup_nature" in out.columns else False
    ben = out["beneficiary_name"].astype(str).str.upper() if "beneficiary_name" in out.columns else pd.Series("", index=out.index)
    act = out["actor_name"].astype(str).str.upper() if "actor_name" in out.columns else pd.Series("", index=out.index)
    out["is_comune_actor"] = ben.str.contains("COMUNE DI", na=False) | act.str.contains("COMUNE DI", na=False)

    if demo_df is not None and len(demo_df):
        pop_2020 = (
            demo_df[demo_df["year"] == base_year][["comune_id", "pop_total"]]
            .drop_duplicates("comune_id")
            .set_index("comune_id")["pop_total"]
        )
        out["pop_2020"] = out["comune_id"].map(pop_2020)
    return out


def summarize_investments_by_comune(
    projects_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    single_comune_only: bool = False,
    base_year: int = 2020,
) -> pd.DataFrame:
    """Summarize total public project allocations, disbursements, completion rates, and implementation lags across the 13 ER municipalities."""
    df = enrich_projects(projects_df, demo_df=demo_df, base_year=base_year)
    if single_comune_only:
        df = df[df["is_single_comune"]].copy()

    pop_map = (
        demo_df[demo_df["year"] == base_year]
        .set_index("comune_id")["pop_total"]
        .to_dict()
    )

    rows: list[dict] = []
    for cid, grp in df.groupby("comune_id"):
        cid_s = str(cid)
        pop = float(pop_map.get(cid_s, np.nan))
        alloc_real = float(grp["finanz_pub_real_eur"].sum())
        disb_real = float(grp["pagamenti_real_eur"].sum())
        n_proj = len(grp)
        done = grp[grp["is_completed"]]
        lags = grp["lag_days"].dropna()
        lags_pos = lags[lags >= 0]
        exec_lags = grp["exec_lag_days"].dropna() if "exec_lag_days" in grp.columns else pd.Series(dtype=float)
        exec_lags_pos = exec_lags[exec_lags >= 0]
        delays = grp["delay_days"].dropna()

        pw_grp = grp[grp["is_public_works"]]
        pw_lags = pw_grp["lag_days"].dropna()
        pw_lags_pos = pw_lags[pw_lags >= 0]
        pw_alloc = float(pw_grp["finanz_pub_real_eur"].sum())
        pw_disb = float(pw_grp["pagamenti_real_eur"].sum())

        rows.append({
            "comune_id": cid_s,
            "comune": COMUNI_BY_ID[cid_s].name if cid_s in COMUNI_BY_ID else cid_s,
            "core_peer": COMUNI_BY_ID[cid_s].core_peer if cid_s in COMUNI_BY_ID else False,
            "single_comune_only": single_comune_only,
            "n_projects": n_proj,
            "single_comune_share_pct": float(grp["is_single_comune"].mean() * 100.0),
            "allocated_real_m_eur": alloc_real / 1e6,
            "disbursed_real_m_eur": disb_real / 1e6,
            "allocated_real_pc": alloc_real / pop if pop > 0 else np.nan,
            "disbursed_real_pc": disb_real / pop if pop > 0 else np.nan,
            "disbursement_rate_pct": disb_real / alloc_real * 100.0 if alloc_real > 0 else np.nan,
            "completion_rate_pct": len(done) / n_proj * 100.0 if n_proj > 0 else np.nan,
            "median_lag_days": float(lags_pos.median()) if len(lags_pos) else np.nan,
            "mean_lag_days": float(lags_pos.mean()) if len(lags_pos) else np.nan,
            "median_exec_days": float(exec_lags_pos.median()) if len(exec_lags_pos) else np.nan,
            "median_delay_days": float(delays.median()) if len(delays) else np.nan,
            "n_public_works": len(pw_grp),
            "public_works_share_pct": len(pw_grp) / n_proj * 100.0 if n_proj > 0 else np.nan,
            "public_works_alloc_real_pc": pw_alloc / pop if pop > 0 else np.nan,
            "public_works_disb_rate_pct": pw_disb / pw_alloc * 100.0 if pw_alloc > 0 else np.nan,
            "public_works_median_lag_days": float(pw_lags_pos.median()) if len(pw_lags_pos) else np.nan,
            "comune_actor_share_pct": float(grp["is_comune_actor"].mean() * 100.0),
        })
    out = pd.DataFrame(rows).sort_values("allocated_real_pc", ascending=False, ignore_index=True)
    out["rank_allocated_pc"] = np.arange(1, len(out) + 1)
    return out


def summarize_investments_by_theme(
    projects_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    focal_id: str = CESENA,
    core_only: bool = True,
    base_year: int = 2020,
) -> pd.DataFrame:
    """Compare Cesena (`040007`) against the peer median across OpenCoesione synthetic policy themes (`theme_label` / `tema`)."""
    df = enrich_projects(projects_df, demo_df=demo_df, base_year=base_year)
    pop_map = demo_df[demo_df["year"] == base_year].set_index("comune_id")["pop_total"].to_dict()
    theme_col = "theme_label" if "theme_label" in df.columns else "tema"
    id_col = "project_id" if "project_id" in df.columns else "cod_locale_progetto"

    agg = (
        df.groupby(["comune_id", theme_col], as_index=False)
        .agg(
            n_projects=(id_col, "count"),
            alloc_real_eur=("finanz_pub_real_eur", "sum"),
            disb_real_eur=("pagamenti_real_eur", "sum"),
            completed_n=("is_completed", "sum"),
            median_lag_days=("lag_days", lambda s: float(s.dropna()[s.dropna() >= 0].median()) if len(s.dropna()[s.dropna() >= 0]) else np.nan),
        )
        .rename(columns={theme_col: "tema"})
    )
    agg["pop_2020"] = agg["comune_id"].map(pop_map)
    agg["alloc_real_pc"] = agg["alloc_real_eur"] / agg["pop_2020"]
    agg["disb_real_pc"] = agg["disb_real_eur"] / agg["pop_2020"]
    agg["disbursement_rate_pct"] = agg["disb_real_eur"] / agg["alloc_real_eur"].replace(0, np.nan) * 100.0
    agg["completion_rate_pct"] = agg["completed_n"] / agg["n_projects"].replace(0, np.nan) * 100.0

    focal = agg[agg["comune_id"] == focal_id].set_index("tema")
    peers = agg[agg["comune_id"] != focal_id].copy()
    if core_only:
        core_ids = {cid for cid, c in COMUNI_BY_ID.items() if c.core_peer and cid != focal_id}
        peers = peers[peers["comune_id"].isin(core_ids)]

    peer_med = peers.groupby("tema")[["alloc_real_pc", "disb_real_pc", "disbursement_rate_pct", "completion_rate_pct", "median_lag_days"]].median()
    tot_f_alloc = float(focal["alloc_real_eur"].sum()) or np.nan

    rows: list[dict] = []
    for tema in focal.index:
        pm = peer_med.loc[tema] if tema in peer_med.index else pd.Series(dtype=float)
        rows.append({
            "tema": str(tema),
            "cesena_n_projects": int(focal.loc[tema, "n_projects"]),
            "cesena_alloc_real_m_eur": float(focal.loc[tema, "alloc_real_eur"]) / 1e6,
            "cesena_disb_real_m_eur": float(focal.loc[tema, "disb_real_eur"]) / 1e6,
            "cesena_share_alloc_pct": float(focal.loc[tema, "alloc_real_eur"]) / tot_f_alloc * 100.0,
            "cesena_alloc_real_pc": float(focal.loc[tema, "alloc_real_pc"]),
            "peer_alloc_real_pc": float(pm.get("alloc_real_pc", np.nan)),
            "cesena_disbursement_rate_pct": float(focal.loc[tema, "disbursement_rate_pct"]),
            "peer_disbursement_rate_pct": float(pm.get("disbursement_rate_pct", np.nan)),
            "cesena_completion_rate_pct": float(focal.loc[tema, "completion_rate_pct"]),
            "peer_completion_rate_pct": float(pm.get("completion_rate_pct", np.nan)),
            "cesena_median_lag_days": float(focal.loc[tema, "median_lag_days"]),
            "peer_median_lag_days": float(pm.get("median_lag_days", np.nan)),
        })
    return pd.DataFrame(rows).sort_values("cesena_alloc_real_m_eur", ascending=False, ignore_index=True)


def annual_investment_flows(projects_df: pd.DataFrame, demo_df: pd.DataFrame, base_year: int = 2020) -> pd.DataFrame:
    """Aggregate project allocations and disbursements by `(comune_id, start_year)` in constant `base_year` EUR per capita."""
    df = enrich_projects(projects_df, demo_df=demo_df, base_year=base_year)
    id_col = "project_id" if "project_id" in df.columns else "cod_locale_progetto"
    valid = df[df["start_year"].notna() & (df["start_year"] >= 2008) & (df["start_year"] <= 2025)].copy()
    valid["year"] = valid["start_year"].astype(int)
    agg = (
        valid.groupby(["comune_id", "year"], as_index=False)
        .agg(
            projects_started_n=(id_col, "count"),
            projects_allocated_real_eur=("finanz_pub_real_eur", "sum"),
            projects_disbursed_real_eur=("pagamenti_real_eur", "sum"),
        )
    )
    agg = agg.merge(demo_df[["comune_id", "year", "pop_total"]], on=["comune_id", "year"], how="left")
    pop = agg["pop_total"].replace(0, np.nan)
    agg["projects_allocated_real_pc"] = agg["projects_allocated_real_eur"] / pop
    agg["projects_disbursed_real_pc"] = agg["projects_disbursed_real_eur"] / pop
    return agg.drop(columns=["pop_total"]).sort_values(["comune_id", "year"], ignore_index=True)


def summarize_opencoesione_2127_benchmarks(
    projects_2127_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    focal_id: str = CESENA,
    core_only: bool = True,
    ref_year: int = 2024,
) -> pd.DataFrame:
    """Benchmark OpenCoesione 2021-2027 project allocations (nominal euro; the cycle is still open), disbursement and completion rates against peers."""
    pop_map = (
        demo_df[demo_df["year"] == ref_year][["comune_id", "pop_total"]]
        .drop_duplicates("comune_id")
        .set_index("comune_id")["pop_total"]
    )
    df = projects_2127_df.copy()

    records = []
    for cid, grp in df.groupby("comune_id"):
        cid_s = str(cid)
        pop = float(pop_map.get(cid_s, np.nan))
        # Nominal euro: the open cycle has no start dates to deflate with.
        alloc = float(grp["allocated_eur"].sum())
        disb = float(grp["disbursed_eur"].sum())
        n_proj = len(grp)
        status_col = "status" if "status" in grp.columns else "stato_progetto"
        is_done = grp[status_col].astype(str).str.lower().isin(["concluso", "liquidato"])

        records.append({
            "comune_id": cid_s,
            "pop": pop,
            "n_projects": n_proj,
            "allocated_total_eur": alloc,
            "disbursed_total_eur": disb,
            "allocated_pc": alloc / pop if pop > 0 else np.nan,
            "disbursed_pc": disb / pop if pop > 0 else np.nan,
            "disbursement_rate_pct": (disb / alloc * 100.0) if alloc > 0 else np.nan,
            "completion_rate_pct": float(is_done.mean() * 100.0),
        })

    muni_df = pd.DataFrame(records)
    focal = muni_df[muni_df["comune_id"] == focal_id]
    if focal.empty:
        return pd.DataFrame()
    f_row = focal.iloc[0]

    peers = muni_df[muni_df["comune_id"] != focal_id].copy()
    if core_only:
        core_ids = {cid for cid, c in COMUNI_BY_ID.items() if c.core_peer and cid != focal_id}
        peers = peers[peers["comune_id"].isin(core_ids)]

    metrics = [
        ("n_projects", "Number of 2021-2027 Cohesion Projects", "{:.0f}"),
        ("allocated_total_eur", "Total 2021-2027 Public Allocation (nominal EUR)", "{:,.0f}"),
        ("allocated_pc", "Allocated Cohesion Funds per Capita (nominal EUR/pc)", "{:,.1f}"),
        ("disbursed_pc", "Disbursed Cohesion Funds per Capita (nominal EUR/pc)", "{:,.1f}"),
        ("disbursement_rate_pct", "Financial Disbursement Rate (%)", "{:.1f}%"),
        ("completion_rate_pct", "Physical Completion/Liquidation Rate (%)", "{:.1f}%"),
    ]

    rows = []
    for m_col, m_label, fmt in metrics:
        f_val = float(f_row.get(m_col, np.nan))
        p_med = float(peers[m_col].median()) if m_col in peers.columns and len(peers[m_col].dropna()) else np.nan
        gap = f_val - p_med
        gap_pct = (f_val / p_med - 1.0) * 100.0 if p_med and p_med != 0 else np.nan
        rows.append({
            "metric": m_col,
            "label": m_label,
            "cesena": f_val,
            "peer_median": p_med,
            "gap": gap,
            "gap_pct": gap_pct,
            "formatted_cesena": fmt.format(f_val) if not np.isnan(f_val) else "N/A",
            "formatted_peer": fmt.format(p_med) if not np.isnan(p_med) else "N/A",
        })

    return pd.DataFrame(rows)


