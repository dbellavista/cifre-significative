r"""Pillar 4 — Socio-Economic & Demographic Outcomes (`studies.cesena.analysis.outcomes`).

Computes:
1. **Municipal IRPEF income distribution metrics (`compute_irpef_distribution`)**:
   - Real mean and bracket-interpolated median income (`mean_income_real_eur`, `median_income_real_eur` in constant 2020 EUR).
   - Lorenz-curve Gini inequality index across the 8 MEF brackets (`gini_trap` trapezoidal lower bound and `gini_pareto` with Pareto top-tail correction above EUR 120,000).
   - Income concentration and composition shares (`top_120k_income_share_pct`, `top_55k_income_share_pct`, `below_15k_payer_share_pct`, `dipendente_share_pct`, `pensione_share_pct`, `autonomo_share_pct`).
2. **Demographic structure & 25-44 working-age cohort attractiveness (`compute_demographic_outcomes`)**:
   - Old-age index (*Indice di vecchiaia*, `pop_65_plus / pop_0_14 * 100`) and dependency ratio.
   - Natural balance rate and overall net migration rate per 1,000 inhabitants.
   - 25-44 working-age cohort net retention/attractiveness rate (`net_cohort_25_44_per_1k`, tracking 1-year cohort survival $\text{Pop}(25..44, t+1) - \text{Pop}(24..43, t)$ per 1,000 cohort members).
"""
import numpy as np
import pandas as pd

from cifresignificative.core.deflators import deflate
from cifresignificative.core.geo import (
    BELT_BY_ID,
    BELT_GROUP_LABELS,
    CESENA,
    COMUNI_BY_ID,
)


def bracket_gini_and_median(sub: pd.DataFrame) -> dict[str, float]:
    """Compute Lorenz trapezoidal Gini, Pareto-adjusted Gini, interpolated median, and bracket shares for one `(comune_id, tax_year)` slice."""
    pos = sub[(sub["class_order"] >= 1) & (sub["taxpayers"] > 0) & (sub["income_eur"] >= 0)].sort_values("class_order")
    if pos.empty:
        return {
            "taxpayers_pos": np.nan,
            "income_pos_eur": np.nan,
            "mean_income_eur": np.nan,
            "median_income_eur": np.nan,
            "gini_trap": np.nan,
            "gini_pareto": np.nan,
            "below_15k_payer_share_pct": np.nan,
            "top_55k_income_share_pct": np.nan,
            "top_120k_income_share_pct": np.nan,
        }

    n_arr = pos["taxpayers"].to_numpy(dtype=float)
    y_arr = pos["income_eur"].to_numpy(dtype=float)
    n_tot = float(n_arr.sum())
    y_tot = float(y_arr.sum())
    if n_tot <= 0 or y_tot <= 0:
        return {
            "taxpayers_pos": n_tot,
            "income_pos_eur": y_tot,
            "mean_income_eur": np.nan,
            "median_income_eur": np.nan,
            "gini_trap": np.nan,
            "gini_pareto": np.nan,
            "below_15k_payer_share_pct": np.nan,
            "top_55k_income_share_pct": np.nan,
            "top_120k_income_share_pct": np.nan,
        }

    # Sort brackets by mean bracket income (monotonicity guard)
    p_shares = n_arr / n_tot
    l_shares = y_arr / y_tot
    p_cum = np.concatenate([[0.0], np.cumsum(p_shares)])
    l_cum = np.concatenate([[0.0], np.cumsum(l_shares)])

    # Trapezoidal Gini: 1 - sum_k (p_k - p_{k-1}) * (L_k + L_{k-1})
    gini_trap = float(1.0 - np.sum((p_cum[1:] - p_cum[:-1]) * (l_cum[1:] + l_cum[:-1])))

    # Pareto top-bracket correction for open top bracket (> 120,000 EUR)
    gini_pareto = gini_trap
    top_row = pos[pos["class_order"] == 7]
    if len(top_row):
        n_top = float(top_row["taxpayers"].iloc[0])
        y_top = float(top_row["income_eur"].iloc[0])
        lo_top = float(top_row["class_lo"].iloc[0])
        if n_top > 0 and y_top > n_top * lo_top and lo_top > 0:
            mean_top = y_top / n_top
            alpha = mean_top / (mean_top - lo_top)
            if alpha > 1.0:
                g_top = 1.0 / (2.0 * alpha - 1.0)
                gini_pareto = float(gini_trap + (n_top / n_tot) * (y_top / y_tot) * g_top)

    # Linear interpolation of median across bounded brackets
    half_n = 0.5 * n_tot
    cum_n = 0.0
    median_val = np.nan
    for _, r in pos.iterrows():
        nk = float(r["taxpayers"])
        lo = float(r["class_lo"]) if np.isfinite(r["class_lo"]) else 0.0
        hi = float(r["class_hi"])
        if cum_n + nk >= half_n:
            if np.isfinite(hi) and nk > 0:
                median_val = lo + ((half_n - cum_n) / nk) * (hi - lo)
            else:
                median_val = float(r["income_eur"]) / nk if nk > 0 else lo
            break
        cum_n += nk

    below_15k = float(pos[pos["class_order"].isin([1, 2])]["taxpayers"].sum() / n_tot * 100.0)
    top_55k = float(pos[pos["class_order"] >= 5]["income_eur"].sum() / y_tot * 100.0)
    top_120k = float(pos[pos["class_order"] == 7]["income_eur"].sum() / y_tot * 100.0)

    return {
        "taxpayers_pos": n_tot,
        "income_pos_eur": y_tot,
        "mean_income_eur": y_tot / n_tot,
        "median_income_eur": float(median_val),
        "gini_trap": gini_trap,
        "gini_pareto": gini_pareto,
        "below_15k_payer_share_pct": below_15k,
        "top_55k_income_share_pct": top_55k,
        "top_120k_income_share_pct": top_120k,
    }


def compute_irpef_distribution(
    brackets_df: pd.DataFrame,
    summary_df: pd.DataFrame | None = None,
    base_year: int = 2020,
) -> pd.DataFrame:
    """Compute municipal IRPEF income, inequality, and composition series in nominal and constant `base_year` EUR."""
    rows: list[dict] = []
    all_by_id = {**BELT_BY_ID, **COMUNI_BY_ID}
    for key, grp in brackets_df.groupby(["comune_id", "tax_year"], sort=True):
        cid, ty = key if isinstance(key, tuple) else (key, 0)
        m = bracket_gini_and_median(grp)
        rows.append({
            "comune_id": str(cid),
            "comune": all_by_id[str(cid)].name if str(cid) in all_by_id else str(cid),
            "year": int(ty),
            **m,
        })
    out = pd.DataFrame(rows)
    if summary_df is not None and len(summary_df):
        s = summary_df.rename(columns={"tax_year": "year"}).copy()
        keep_cols = [
            c
            for c in (
                "comune_id",
                "year",
                "taxpayers_total",
                "reddito_complessivo_eur",
                "reddito_imponibile_eur",
                "imposta_netta_eur",
                "addizionale_comunale_eur",
                "addizionale_comunale_freq",
                "reddito_dipendente_eur",
                "reddito_pensione_eur",
                "reddito_autonomo_eur",
                "reddito_fabbricati_eur",
            )
            if c in s.columns
        ]
        out = out.merge(s[keep_cols], on=["comune_id", "year"], how="left")
        # MEF publishes the total only from 2023; earlier the positive brackets give the same figure within 0.01%
        rc = out["reddito_complessivo_eur"].replace(0, np.nan).fillna(out["income_pos_eur"])
        if "reddito_dipendente_eur" in out.columns:
            out["dipendente_share_pct"] = out["reddito_dipendente_eur"] / rc * 100.0
        if "reddito_pensione_eur" in out.columns:
            out["pensione_share_pct"] = out["reddito_pensione_eur"] / rc * 100.0
        if "reddito_autonomo_eur" in out.columns:
            out["autonomo_share_pct"] = out["reddito_autonomo_eur"] / rc * 100.0

    # Deflate nominal euro columns to constant 2020 EUR
    out["mean_income_real_eur"] = deflate(out["mean_income_eur"], out["year"], base_year=base_year)
    out["median_income_real_eur"] = deflate(out["median_income_eur"], out["year"], base_year=base_year)
    out["income_pos_real_eur"] = deflate(out["income_pos_eur"], out["year"], base_year=base_year)
    if "addizionale_comunale_eur" in out.columns:
        out["addizionale_comunale_real_eur"] = deflate(out["addizionale_comunale_eur"], out["year"], base_year=base_year)
    return out.sort_values(["comune_id", "year"], ignore_index=True)


def compute_demographic_outcomes(demo_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich municipal demographic panel with rate indicators and 3-year smoothed cohort attractiveness."""
    out = demo_df.sort_values(["comune_id", "year"]).copy()
    pop = out["pop_total"].replace(0, np.nan)
    out["share_0_14_pct"] = out["pop_0_14"] / pop * 100.0
    out["share_15_64_pct"] = out["pop_15_64"] / pop * 100.0
    out["share_25_44_pct"] = out["pop_25_44"] / pop * 100.0
    out["share_65_plus_pct"] = out["pop_65_plus"] / pop * 100.0
    out["pop_growth_pct"] = out.groupby("comune_id")["pop_total"].pct_change() * 100.0
    out["net_cohort_25_44_3y_ma"] = (
        out.groupby("comune_id")["net_cohort_25_44_per_1k"]
        .transform(lambda s: s.rolling(3, min_periods=2).mean())
    )
    return out.reset_index(drop=True)


def compare_cesena_outcomes_to_peers(panel_df: pd.DataFrame, core_only: bool = True) -> pd.DataFrame:
    """Compare Cesena (`040007`) against the peer median (excluding Cesena) on key socio-economic and demographic outcomes."""
    metrics = [
        "mean_income_real_eur",
        "median_income_real_eur",
        "gini_pareto",
        "aging_index",
        "share_25_44_pct",
        "net_cohort_25_44_per_1k",
    ]
    avail = [m for m in metrics if m in panel_df.columns]
    ces = panel_df[panel_df["comune_id"] == CESENA][["year", *avail]].set_index("year")
    peers = panel_df[panel_df["comune_id"] != CESENA].copy()
    if core_only and "core_peer" in peers.columns:
        peers = peers[peers["core_peer"].astype(bool)]
    med = peers.groupby("year")[avail].median()
    rows: list[dict] = []
    for y in sorted(ces.index.intersection(med.index)):
        row: dict[str, object] = {"year": int(y)}
        for m in avail:
            c_val = float(ces.loc[y, m]) if pd.notna(ces.loc[y, m]) else np.nan
            p_val = float(med.loc[y, m]) if pd.notna(med.loc[y, m]) else np.nan
            row[f"cesena_{m}"] = c_val
            row[f"peer_median_{m}"] = p_val
            row[f"gap_{m}"] = c_val - p_val
        rows.append(row)
    return pd.DataFrame(rows)


def build_comprensorio_panel(
    panel_er: pd.DataFrame,
    demo_belt: pd.DataFrame,
    irpef_belt: pd.DataFrame,
    asia_belt: pd.DataFrame,
) -> pd.DataFrame:
    """Combine Cesena (`040007`) with the 16 surrounding municipalities of the *Comprensorio Cesenate* (`2008-2025`)."""
    ces = panel_er[panel_er["comune_id"] == CESENA].copy()
    ces["belt_group"] = "cesena_core"
    ces["belt_label"] = BELT_GROUP_LABELS["cesena_core"]
    ces["net_migration"] = ces["net_migration_total"]

    belt = (
        demo_belt[demo_belt["year"] <= 2025]
        .merge(
            irpef_belt[
                [
                    "comune_id", "year", "taxpayers_total", "taxpayers_pos", "income_pos_real_eur",
                    "mean_income_real_eur", "median_income_real_eur", "gini_pareto",
                    "below_15k_payer_share_pct", "top_55k_income_share_pct", "top_120k_income_share_pct",
                    "addizionale_comunale_real_eur",
                ]
            ],
            on=["comune_id", "year"],
            how="left",
        )
        .merge(
            asia_belt[["comune_id", "year", "local_units", "employees", "avg_firm_size"]],
            on=["comune_id", "year"],
            how="left",
        )
    )
    belt["belt_label"] = belt["belt_group"].map(lambda g: BELT_GROUP_LABELS.get(str(g), str(g)))
    belt["employees_per_1k_pop"] = belt["employees"] / belt["pop_total"].replace(0, np.nan) * 1000.0
    belt["local_units_per_1k_pop"] = belt["local_units"] / belt["pop_total"].replace(0, np.nan) * 1000.0

    keep = [
        "comune_id", "comune", "belt_group", "belt_label", "year",
        "pop_total", "births", "deaths", "natural_balance", "natural_balance_per_1000",
        "inflow_total", "outflow_total", "net_migration", "net_migration_internal", "net_migration_international", "registry_adjustment", "net_migr_rate_per_1k",
        "taxpayers_total", "taxpayers_pos", "income_pos_real_eur",
        "mean_income_real_eur", "median_income_real_eur", "gini_pareto",
        "below_15k_payer_share_pct", "top_55k_income_share_pct", "top_120k_income_share_pct",
        "addizionale_comunale_real_eur",
        "local_units", "employees", "avg_firm_size", "employees_per_1k_pop", "local_units_per_1k_pop",
    ]
    for c in keep:
        if c not in ces.columns:
            ces[c] = np.nan
        if c not in belt.columns:
            belt[c] = np.nan
    out = pd.concat([ces[keep], belt[keep]], ignore_index=True)
    return out.sort_values(["belt_group", "comune_id", "year"], ignore_index=True)


def summarize_comprensorio_spillovers(comp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(by_group_df, by_muni_df)` comparing population, net migration, ASIA employment, and real IRPEF income across the *Comprensorio Cesenate*."""
    group_order = ("cesena_core", "rubicone_mare", "via_emilia_west", "unione_valle_savio", "COMMUTER_BELT_16", "COMPRENSORIO_17")
    group_rows: list[dict] = []

    def _agg_slice(sub: pd.DataFrame, g_key: str, g_label: str) -> dict[str, object]:
        p08 = float(sub[sub["year"] == 2008]["pop_total"].sum())
        p12 = float(sub[sub["year"] == 2012]["pop_total"].sum())
        p24 = float(sub[sub["year"] == 2024]["pop_total"].sum())
        p25 = float(sub[sub["year"] == 2025]["pop_total"].sum())
        mig_08_24 = float(sub[sub["year"].between(2008, 2024)]["net_migration"].sum())
        nat_08_24 = float(sub[sub["year"].between(2008, 2024)]["natural_balance"].sum())
        mig_int_08_24 = float(sub[sub["year"].between(2008, 2024)]["net_migration_internal"].sum()) if "net_migration_internal" in sub.columns else np.nan
        mig_abr_08_24 = float(sub[sub["year"].between(2008, 2024)]["net_migration_international"].sum()) if "net_migration_international" in sub.columns else np.nan
        adj_08_24 = float(sub[sub["year"].between(2008, 2024)]["registry_adjustment"].sum()) if "registry_adjustment" in sub.columns else np.nan
        e12 = float(sub[sub["year"] == 2012]["employees"].sum())
        e24 = float(sub[sub["year"] == 2024]["employees"].sum())
        u12 = float(sub[sub["year"] == 2012]["local_units"].sum())
        u24 = float(sub[sub["year"] == 2024]["local_units"].sum())
        y08_tot = float(sub[sub["year"] == 2008]["income_pos_real_eur"].sum())
        n08_tot = float(sub[sub["year"] == 2008]["taxpayers_pos"].sum())
        y24_tot = float(sub[sub["year"] == 2024]["income_pos_real_eur"].sum())
        n24_tot = float(sub[sub["year"] == 2024]["taxpayers_pos"].sum())
        add24_tot = float(sub[sub["year"] == 2024]["addizionale_comunale_real_eur"].sum()) if "addizionale_comunale_real_eur" in sub.columns else np.nan
        inc08 = y08_tot / n08_tot if n08_tot > 0 else np.nan
        inc24 = y24_tot / n24_tot if n24_tot > 0 else np.nan
        eff_add_rate = add24_tot / y24_tot * 100.0 if y24_tot > 0 and pd.notna(add24_tot) else np.nan
        add_pc = add24_tot / p24 if p24 > 0 and pd.notna(add24_tot) else np.nan
        return {
            "belt_group": g_key,
            "belt_label": g_label,
            "n_comuni": int(sub["comune_id"].nunique()),
            "pop_2008": p08,
            "pop_2012": p12,
            "pop_2024": p24,
            "pop_2025": p25,
            "pop_delta_08_25": p25 - p08,
            "pop_growth_08_25_pct": (p25 / p08 - 1.0) * 100.0 if p08 > 0 else np.nan,
            "pop_delta_12_24": p24 - p12,
            "pop_growth_12_24_pct": (p24 / p12 - 1.0) * 100.0 if p12 > 0 else np.nan,
            "net_migration_08_24": mig_08_24,
            "natural_balance_08_24": nat_08_24,
            "net_migration_internal_08_24": mig_int_08_24,
            "net_migration_international_08_24": mig_abr_08_24,
            "registry_adjustment_08_24": adj_08_24,
            "emp_2012": e12,
            "emp_2024": e24,
            "emp_delta_12_24": e24 - e12,
            "emp_growth_12_24_pct": (e24 / e12 - 1.0) * 100.0 if e12 > 0 else np.nan,
            "units_2012": u12,
            "units_2024": u24,
            "emp_per_1k_pop_2012": e12 / p12 * 1000.0 if p12 > 0 else np.nan,
            "emp_per_1k_pop_2024": e24 / p24 * 1000.0 if p24 > 0 else np.nan,
            "mean_income_real_2008": inc08,
            "mean_income_real_2024": inc24,
            "real_income_growth_08_24_pct": (inc24 / inc08 - 1.0) * 100.0 if inc08 > 0 else np.nan,
            "effective_addizionale_rate_2024_pct": eff_add_rate,
            "addizionale_real_pc_2024": add_pc,
        }

    for g_key in ("cesena_core", "rubicone_mare", "via_emilia_west", "unione_valle_savio"):
        sub = comp_df[comp_df["belt_group"] == g_key]
        if len(sub):
            group_rows.append(_agg_slice(sub, g_key, BELT_GROUP_LABELS.get(g_key, g_key)))
    belt_only = comp_df[comp_df["comune_id"] != CESENA]
    if len(belt_only):
        group_rows.append(_agg_slice(belt_only, "COMMUTER_BELT_16", "All 16 surrounding belt municipalities"))
    group_rows.append(_agg_slice(comp_df, "COMPRENSORIO_17", "Full Comprensorio Cesenate (Cesena + 16 belt comuni)"))
    by_group_df = pd.DataFrame(group_rows)
    by_group_df["_ord"] = by_group_df["belt_group"].map(lambda k: group_order.index(k) if k in group_order else 99)
    by_group_df = by_group_df.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)

    muni_rows: list[dict] = []
    for cid, sub in comp_df.groupby("comune_id", sort=True):
        r = _agg_slice(sub, str(sub["belt_group"].iloc[0]), str(sub["belt_label"].iloc[0]))
        r["comune_id"] = str(cid)
        r["comune"] = str(sub["comune"].iloc[0])
        muni_rows.append(r)
    by_muni_df = pd.DataFrame(muni_rows).sort_values("pop_growth_08_25_pct", ascending=False, ignore_index=True)
    return by_group_df, by_muni_df


def summarize_university_benchmarks(
    uni_summary_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    focal_id: str = CESENA,
    core_only: bool = True,
    year: int = 2024,
) -> pd.DataFrame:
    """Benchmark university higher education metrics (students, STEM/ICT specialisation, graduates) for focal municipality against peers."""
    pop_map = (
        demo_df[demo_df["year"] == year][["comune_id", "pop_total"]]
        .drop_duplicates("comune_id")
        .set_index("comune_id")["pop_total"]
    )
    df = uni_summary_df[uni_summary_df["calendar_year"] == year].copy()
    if df.empty:
        # Fall back to latest available year
        latest = uni_summary_df["calendar_year"].max()
        df = uni_summary_df[uni_summary_df["calendar_year"] == latest].copy()

    df["pop"] = df["comune_id"].map(pop_map)
    df["students_per_1k_hab"] = df["students_total"] / df["pop"].replace(0, np.nan) * 1000.0
    df["graduates_per_1k_hab"] = df["graduates_total"] / df["pop"].replace(0, np.nan) * 1000.0

    focal = df[df["comune_id"] == focal_id]
    if focal.empty:
        return pd.DataFrame()

    f_row = focal.iloc[0]
    peers = df[df["comune_id"] != focal_id].copy()
    if core_only:
        core_ids = {cid for cid, c in COMUNI_BY_ID.items() if c.core_peer and cid != focal_id}
        peers = peers[peers["comune_id"].isin(core_ids)]

    metrics = [
        ("students_total", "Total University Students Enrolled", "{:,.0f}"),
        ("students_per_1k_hab", "University Students per 1,000 Inhabitants", "{:.1f}"),
        ("stem_share", "STEM Student Share (% of total enrolments)", "{:.1%}"),
        ("ict_share", "Computer Science / ICT Share (% of enrolments)", "{:.1%}"),
        ("engineering_share", "Engineering Share (% of enrolments)", "{:.1%}"),
        ("psychology_share", "Psychology Share (% of enrolments)", "{:.1%}"),
        ("graduates_total", "Total Annual University Graduates", "{:,.0f}"),
        ("graduates_per_1k_hab", "Graduates per 1,000 Inhabitants", "{:.1f}"),
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


