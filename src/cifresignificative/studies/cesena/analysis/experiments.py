r"""Phase 3 — Comparative Benchmarking, Administrative Cycles & Lagged Panel Evaluation (`studies.cesena.analysis.experiments`).

Implements:
1. **Administrative Cycle Scorecard (`administrative_cycle_scorecard`)**:
   - Summarizes Cesena (`040007`), the 6 Core Peers median, and the 12 ER Peers median across the four municipal administrative cycles (`Lucchi I 2009-2013`, `Lucchi II 2014-2018`, `Lattuca I 2019-2023`, `Lattuca II 2024+`), keeping pre-2016 and post-2016 accounting regimes explicitly separated where applicable.
2. **Two-Way Fixed Effects Lagged Panel Screen (`lagged_fe_screen`)**:
   - Tests whether municipal capital expenditure (`spese_capitale_comp_real_pc`, `spese_capitale_cassa_real_pc`, `capital_share_pct`), cohesion/national project disbursements (`projects_disbursed_real_pc`, `projects_allocated_real_pc`), and fiscal health (`collection_rate_correnti_pct`, `rigidity_official_pct`) at year $t$ are associated with subsequent ($t+1 \dots t+4$) private-sector local unit growth (`dlog_local_units`), employment growth (`dlog_employees`), real IRPEF tax-base growth (`dlog_irpef_taxbase_real_pc`), real mean income growth (`dlog_mean_income_real_eur`), and 25-44 cohort net attractiveness (`net_cohort_25_44_per_1k`).
   - Uses Two-Way Fixed Effects (municipality FE + year FE, absorbing the 2016 D.Lgs. 118/2011 level shift and common macroeconomic shocks), entity-clustered standard errors, within-year permutation p-values (`perm_p`), and Benjamini-Hochberg FDR (`q_bh`) across the entire grid of tests.
3. **Shuffled-Municipality Placebo Null (`shuffled_null_check`)**:
   - Permutes treatment trajectories across municipalities/years and verifies ~0 false discoveries at $q < 0.05$.
4. **Cesena Placebo Ranking (`cesena_placebo_ranking`)**:
   - Ranks Cesena (`040007`) among all 13 Emilia-Romagna municipalities on cumulative multi-year changes and level gaps, reporting exact empirical rank ($1 \dots 13$) and two-sided rank proportion.
"""
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from cifresignificative.core.deflators import deflate
from cifresignificative.core.geo import CESENA, COMUNI_BY_ID
from cifresignificative.core.sources import mef_comuni
from cifresignificative.studies.cesena.analysis.data import PROCESSED_DIR
from cifresignificative.studies.cesena.analysis.outcomes import bracket_gini_and_median

CYCLE_ORDER: tuple[str, ...] = (
    "Lucchi I (2009-2014)",
    "Lucchi II (2014-2019)",
    "Lattuca I (2019-2024)",
    "Lattuca II (2024+)",
)

PREDICTORS: tuple[tuple[str, str, str], ...] = (
    ("spese_capitale_comp_real_pc", "Capital spending commitments per capita (2020 EUR)", "all"),
    ("spese_capitale_cassa_real_pc", "Capital spending cash payments per capita (2020 EUR)", "all"),
    ("capital_share_pct", "Capital share of primary spending (%)", "all"),
    ("projects_allocated_real_pc", "Public project allocations per capita (2020 EUR)", "all"),
    ("projects_disbursed_real_pc", "Public project disbursements per capita (2020 EUR)", "all"),
    ("collection_rate_correnti_pct", "Current revenue collection capacity (%)", "dlgs118"),
    ("rigidity_struct_pct", "Structural budget rigidity proxy (%)", "dlgs118"),
)

OUTCOMES: tuple[tuple[str, str], ...] = (
    ("dlog_local_units", "Active local units growth (log pts x100)"),
    ("dlog_employees", "Private employment growth (log pts x100)"),
    ("dlog_irpef_taxbase_real_pc", "Real IRPEF tax base per capita growth (log pts x100)"),
    ("dlog_mean_income_real_eur", "Real mean IRPEF income growth (log pts x100)"),
    ("net_cohort_25_44_per_1k", "25-44 cohort net attractiveness (per 1,000)"),
)


def bh_fdr(pvals: np.ndarray | pd.Series) -> np.ndarray:
    """Benjamini-Hochberg false discovery rate q-values (preserving NaNs)."""
    arr = np.asarray(pvals, dtype=float)
    q = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return q
    p = arr[mask]
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0.0, 1.0)
    inv = np.empty_like(order)
    inv[order] = np.arange(m)
    q[mask] = ranked[inv]
    return q


def _twoway_fe_demean(df: pd.DataFrame, cols: list[str], entity_col: str = "comune_id", time_col: str = "year") -> pd.DataFrame:
    """Exact two-way within transformation (alternating projections for unbalanced panels, exact in 1 step for balanced)."""
    out = df[[entity_col, time_col, *cols]].dropna().copy()
    if len(out) < 6:
        return out
    for c in cols:
        v = out[c].to_numpy(dtype=float)
        for _ in range(8):
            v = v - out.groupby(entity_col)[c].transform("mean").to_numpy(dtype=float)
            out[c] = v
            v = v - out.groupby(time_col)[c].transform("mean").to_numpy(dtype=float)
            out[c] = v
    return out


def twoway_fe_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    entity_col: str = "comune_id",
    time_col: str = "year",
    n_perm: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Fit Two-Way Fixed Effects regression of `y_col` on standardized `x_col` with entity-clustered SE and within-year permutation p."""
    sub = df[[entity_col, time_col, x_col, y_col]].dropna().copy()
    n_obs = len(sub)
    n_entities = int(sub[entity_col].nunique())
    n_years = int(sub[time_col].nunique())
    if n_obs < 15 or n_entities < 5 or n_years < 3:
        return {
            "n_obs": n_obs, "n_comuni": n_entities, "n_years": n_years,
            "beta": np.nan, "beta_std": np.nan, "se_cluster": np.nan,
            "t_stat": np.nan, "p_cluster": np.nan, "p_perm": np.nan, "r2_within": np.nan,
        }

    dm = _twoway_fe_demean(sub, [x_col, y_col], entity_col=entity_col, time_col=time_col)
    x_tilde = dm[x_col].to_numpy(dtype=float)
    y_tilde = dm[y_col].to_numpy(dtype=float)
    ssx = float(np.sum(x_tilde**2))
    ssy = float(np.sum(y_tilde**2))
    if ssx <= 1e-12 or ssy <= 1e-12:
        return {
            "n_obs": n_obs, "n_comuni": n_entities, "n_years": n_years,
            "beta": 0.0, "beta_std": 0.0, "se_cluster": np.nan,
            "t_stat": 0.0, "p_cluster": 1.0, "p_perm": 1.0, "r2_within": 0.0,
        }

    beta = float(np.sum(x_tilde * y_tilde) / ssx)
    resid = y_tilde - beta * x_tilde
    r2_within = float(1.0 - np.sum(resid**2) / ssy)

    # Standardized beta (SD of within-demeaned x vs SD of within-demeaned y)
    sd_x = float(np.std(x_tilde, ddof=1))
    sd_y = float(np.std(y_tilde, ddof=1))
    beta_std = beta * (sd_x / sd_y) if sd_y > 0 else 0.0

    # Arellano / Liang-Zeger entity-clustered variance estimator with small-sample adjustment
    ent_codes = dm[entity_col].to_numpy()
    unique_ents = np.unique(ent_codes)
    g = len(unique_ents)
    k_df = max(1, n_obs - n_entities - n_years + 1)
    meat = 0.0
    for e in unique_ents:
        m = ent_codes == e
        score_e = float(np.sum(x_tilde[m] * resid[m]))
        meat += score_e**2
    adj = (g / max(1, g - 1)) * ((n_obs - 1) / k_df)
    var_beta = adj * meat / (ssx**2)
    se_cluster = float(np.sqrt(max(var_beta, 1e-18)))
    t_stat = beta / se_cluster
    p_cluster = float(2.0 * stats.t.sf(abs(t_stat), df=max(1, g - 1)))

    # Within-year permutation test (permuting municipality assignments consistently or within year)
    rng = np.random.default_rng(seed)
    obs_abs_r = abs(float(np.sum(x_tilde * y_tilde)))
    yr_codes = dm[time_col].to_numpy()
    perm_sum = np.zeros(n_perm)
    for y in np.unique(yr_codes):
        idx = yr_codes == y
        perm_sum += rng.permuted(np.tile(x_tilde[idx], (n_perm, 1)), axis=1) @ y_tilde[idx]
    exceed = int((np.abs(perm_sum) >= obs_abs_r - 1e-12).sum())
    p_perm = float((exceed + 1) / (n_perm + 1))

    return {
        "n_obs": n_obs,
        "n_comuni": n_entities,
        "n_years": n_years,
        "beta": beta,
        "beta_std": beta_std,
        "se_cluster": se_cluster,
        "t_stat": t_stat,
        "p_cluster": p_cluster,
        "p_perm": p_perm,
        "r2_within": r2_within,
    }


def lagged_fe_screen(
    panel_df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 4),
    predictors: tuple[tuple[str, str, str], ...] = PREDICTORS,
    outcomes: tuple[tuple[str, str], ...] = OUTCOMES,
    n_perm: int = 5000,
    seed: int = 42,
    regime_fe: bool = False,
) -> pd.DataFrame:
    """Screen lagged predictor-outcome pairs ($t \\to t+k$) with Two-Way Fixed Effects and BH-FDR control.

    The outcome lead is taken before any regime filter, so a predictor known only to 2021 still sees its outcomes to 2024.
    `regime_fe` also removes municipality x accounting-regime means, so predictors spanning the 2016 break are compared only within a regime.
    The smallest attainable permutation p is `1 / (n_perm + 1)`: too few permutations would decide the BH result by themselves.
    """
    df_sorted = panel_df.sort_values(["comune_id", "year"]).copy()
    rows: list[dict] = []

    for x_col, x_lbl, reg_scope in predictors:
        if x_col not in df_sorted.columns:
            continue
        for y_col, y_lbl in outcomes:
            if y_col not in df_sorted.columns:
                continue
            for k in lags:
                work = df_sorted.copy()
                # Lead the outcome by k years within each municipality: y_{c, t+k} on x_{c, t}
                work["y_lead"] = work.groupby("comune_id")[y_col].shift(-k)
                if reg_scope == "dlgs118" and "regime" in work.columns:
                    work = work[work["regime"] == "dlgs118"].copy()
                if regime_fe and "regime" in work.columns:
                    work = work.dropna(subset=[x_col, "y_lead", "regime"]).copy()
                    for col in (x_col, "y_lead"):
                        work[col] = work[col] - work.groupby(["comune_id", "regime"])[col].transform("mean")
                res = twoway_fe_regression(
                    work, x_col=x_col, y_col="y_lead", n_perm=n_perm, seed=seed + k * 17
                )
                rows.append({
                    "predictor": x_col,
                    "predictor_label": x_lbl,
                    "outcome": y_col,
                    "outcome_label": y_lbl,
                    "regime_scope": reg_scope,
                    "lag_years": k,
                    **res,
                })

    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh_cluster"] = bh_fdr(out["p_cluster"])
        out["q_bh_perm"] = bh_fdr(out["p_perm"])
        out["significant_fdr05"] = (out["q_bh_perm"] < 0.05) & (out["q_bh_cluster"] < 0.05)
    return out.sort_values(["p_perm", "p_cluster"], ignore_index=True)


def shuffled_null_check(
    panel_df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 4),
    seed: int = 12345,
    n_rep: int = 20,
    n_perm: int = 2000,
) -> dict[str, object]:
    """Run `lagged_fe_screen` on `n_rep` panels with shuffled predictors; about 0 discoveries per run is the expected result, and `n_rep` keeps one lucky seed from deciding it."""
    rng = np.random.default_rng(seed)
    n_tests = 0
    disc_perm: list[int] = []
    disc_clust: list[int] = []
    min_q: list[float] = []
    min_p: list[float] = []
    for rep in range(n_rep):
        shuf = panel_df.sort_values(["comune_id", "year"]).copy()
        for x_col, _, _ in PREDICTORS:
            if x_col in shuf.columns:
                vals = shuf[x_col].to_numpy().copy()
                rng.shuffle(vals)
                shuf[x_col] = vals
        res = lagged_fe_screen(shuf, lags=lags, n_perm=n_perm, seed=seed + rep)
        n_tests = int(res["p_perm"].notna().sum()) if len(res) else 0
        disc_perm.append(int((res["q_bh_perm"] < 0.05).sum()) if len(res) else 0)
        disc_clust.append(int((res["q_bh_cluster"] < 0.05).sum()) if len(res) else 0)
        min_q.append(float(res["q_bh_perm"].min()) if n_tests else np.nan)
        min_p.append(float(res["p_perm"].min()) if n_tests else np.nan)
    return {
        "n_tests": n_tests,
        "n_rep": n_rep,
        "n_discoveries_q_perm_05": int(sum(disc_perm)),
        "n_discoveries_q_cluster_05": int(sum(disc_clust)),
        "share_runs_with_discovery": float(np.mean([d > 0 for d in disc_perm])),
        "min_q_perm": float(np.nanmin(min_q)),
        "min_p_perm": float(np.nanmin(min_p)),
    }


def administrative_cycle_scorecard(panel_df: pd.DataFrame, focal_id: str = CESENA) -> pd.DataFrame:
    """Summarize key fiscal, investment, economic, and demographic indicators across Cesena's administrative cycles vs peers."""
    metrics: list[tuple[str, str]] = [
        ("spese_correnti_comp_real_pc", "Current spending per capita (2020 EUR)"),
        ("spese_capitale_comp_real_pc", "Capital spending per capita (2020 EUR)"),
        ("capital_share_pct", "Capital share of primary spending (%)"),
        ("imposte_tasse_comp_real_pc", "Own taxes & fees per capita (2020 EUR)"),
        ("mef_addizionale_real_pc", "MEF IRPEF surcharge per capita (2020 EUR)"),
        ("rigidity_official_pct", "Official rigidity index (%)"),
        ("collection_rate_correnti_pct", "Current revenue collection rate (%)"),
        ("projects_allocated_real_pc", "Cohesion/national project allocations per capita (2020 EUR)"),
        ("local_units_per_1k_pop", "Active local units per 1,000 inhabitants"),
        ("employees_per_1k_pop", "Private employees per 1,000 inhabitants"),
        ("avg_firm_size", "Average unit size (employees / local unit)"),
        ("mean_income_real_eur", "Real mean IRPEF income (2020 EUR)"),
        ("median_income_real_eur", "Real median IRPEF income (2020 EUR)"),
        ("gini_pareto", "Local IRPEF Gini index (Pareto-adjusted)"),
        ("aging_index", "Old-age dependency / aging index (65+ / 0-14 x100)"),
        ("net_cohort_25_44_per_1k", "25-44 working-age cohort net attractiveness (per 1,000)"),
    ]
    rows: list[dict] = []
    for cyc in CYCLE_ORDER:
        sub = panel_df[panel_df["administration"] == cyc]
        if sub.empty:
            continue
        muni_means = sub.groupby("comune_id")[[m for m, _ in metrics if m in sub.columns]].mean()
        if focal_id not in muni_means.index:
            continue
        peers_all = muni_means.drop(index=focal_id)
        core_ids = [cid for cid in peers_all.index if COMUNI_BY_ID.get(str(cid), None) and COMUNI_BY_ID[str(cid)].core_peer]
        peers_core = peers_all.loc[core_ids]

        for col, lbl in metrics:
            if col not in muni_means.columns:
                continue
            c_val = float(muni_means.loc[focal_id, col])
            p_core = float(peers_core[col].dropna().median()) if len(peers_core[col].dropna()) else np.nan
            p_er = float(peers_all[col].dropna().median()) if len(peers_all[col].dropna()) else np.nan
            if np.isnan(c_val):
                continue
            all_valid = muni_means[col].dropna()
            rank_desc = int((all_valid > c_val).sum() + 1)
            rows.append({
                "administration": cyc,
                "metric": col,
                "metric_label": lbl,
                "cesena": c_val,
                "core_peer_median": p_core,
                "er_peer_median": p_er,
                "gap_vs_core": c_val - p_core,
                "gap_vs_er": c_val - p_er,
                "rank_1_to_13": rank_desc,
                "n_comuni": len(all_valid),
            })
    return pd.DataFrame(rows)


def cesena_placebo_ranking(panel_df: pd.DataFrame, focal_id: str = CESENA) -> pd.DataFrame:
    """Placebo rank of Cesena (`040007`) among all 13 Emilia-Romagna municipalities on key long-run trajectories."""
    specs: list[tuple[str, str, int, int, str]] = [
        ("employees", "Private employment cumulative growth (%)", 2012, 2024, "pct_change"),
        ("local_units", "Active local units cumulative change (%)", 2012, 2024, "pct_change"),
        ("avg_firm_size", "Average firm size change (employees/unit)", 2012, 2024, "diff"),
        ("mean_income_real_eur", "Real mean IRPEF income cumulative growth (%)", 2008, 2024, "pct_change"),
        ("median_income_real_eur", "Real median IRPEF income cumulative growth (%)", 2008, 2024, "pct_change"),
        ("gini_pareto", "Local Gini index change (pts)", 2008, 2024, "diff"),
        ("pop_total", "Resident population change (%)", 2008, 2025, "pct_change"),
        ("aging_index", "Old-age index change (pts)", 2008, 2025, "diff"),
        ("net_cohort_25_44_per_1k", "Mean 25-44 cohort net attractiveness (per 1k/yr)", 2012, 2024, "mean"),
        ("spese_capitale_comp_real_pc", "Mean harmonized capital spending per capita (2020 EUR)", 2016, 2021, "mean"),
        ("collection_rate_correnti_pct", "Mean harmonized current collection rate (%)", 2016, 2021, "mean"),
    ]
    rows: list[dict] = []
    for col, lbl, t0, t1, mode in specs:
        if col not in panel_df.columns:
            continue
        vals: dict[str, float] = {}
        for cid, grp in panel_df.groupby("comune_id"):
            g = grp.set_index("year")[col].dropna()
            if mode == "mean":
                sub = g[(g.index >= t0) & (g.index <= t1)]
                if len(sub):
                    vals[str(cid)] = float(sub.mean())
            elif t0 in g.index and t1 in g.index:
                v0, v1 = float(g.loc[t0]), float(g.loc[t1])
                if mode == "pct_change" and v0 != 0:
                    vals[str(cid)] = (v1 - v0) / abs(v0) * 100.0
                elif mode == "diff":
                    vals[str(cid)] = v1 - v0
        if focal_id not in vals or len(vals) < 5:
            continue
        ser = pd.Series(vals)
        c_val = float(ser.loc[focal_id])
        peers = ser.drop(index=focal_id)
        rank_desc = int((ser > c_val).sum() + 1)
        n_tot = len(ser)
        # Two-sided empirical rank proportion among n_tot municipalities
        p_placebo = float(min(1.0, 2.0 * min(rank_desc, n_tot - rank_desc + 1) / n_tot))
        rows.append({
            "metric": col,
            "label": lbl,
            "period": f"{t0}-{t1}",
            "mode": mode,
            "cesena_value": c_val,
            "peer_median_ex_cesena": float(peers.median()),
            "peer_min": float(peers.min()),
            "peer_max": float(peers.max()),
            "gap_vs_peer_median": c_val - float(peers.median()),
            "rank_1_to_13": rank_desc,
            "n_comuni": n_tot,
            "placebo_two_sided_prop": p_placebo,
        })
    return pd.DataFrame(rows)


def run_all_experiments(panel_df: pd.DataFrame, out_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Run and save all Phase 3 tables (`scorecard`, `lagged_fe`, `placebo_ranking`, `extended_placebo`, `synthetic_cesena`) under `data/processed/cesena/`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scorecard = administrative_cycle_scorecard(panel_df)
    lagged_fe = lagged_fe_screen(panel_df)
    lagged_fe_regime = lagged_fe_screen(panel_df, regime_fe=True)
    ranking = cesena_placebo_ranking(panel_df)
    synth_series, synth_summary = synthetic_cesena(panel_df)
    in_space_df, in_time_df = synthetic_control_placebos(panel_df)
    loo_df = synthetic_leave_one_out(panel_df)

    scorecard.to_parquet(out_dir / "cycle_scorecard.parquet", index=False)
    lagged_fe.to_parquet(out_dir / "lagged_fe_results.parquet", index=False)
    lagged_fe_regime.to_parquet(out_dir / "lagged_fe_regime_results.parquet", index=False)
    ranking.to_parquet(out_dir / "placebo_ranking.parquet", index=False)
    synth_series.to_parquet(out_dir / "synthetic_cesena.parquet", index=False)
    synth_summary.to_parquet(out_dir / "synthetic_summary.parquet", index=False)
    in_space_df.to_parquet(out_dir / "synthetic_in_space.parquet", index=False)
    in_time_df.to_parquet(out_dir / "synthetic_in_time.parquet", index=False)
    loo_df.to_parquet(out_dir / "synthetic_leave_one_out.parquet", index=False)
    pd.DataFrame([shuffled_null_check(panel_df)]).to_parquet(out_dir / "shuffled_null.parquet", index=False)

    out: dict[str, pd.DataFrame] = {
        "scorecard": scorecard,
        "lagged_fe": lagged_fe,
        "lagged_fe_regime": lagged_fe_regime,
        "ranking": ranking,
        "synthetic_cesena": synth_series,
        "synthetic_summary": synth_summary,
        "synthetic_in_space": in_space_df,
        "synthetic_in_time": in_time_df,
        "synthetic_leave_one_out": loo_df,
    }
    if (mef_comuni.RAW_DIR / "irpef_comuni_2008.zip").exists() and (mef_comuni.RAW_DIR / "irpef_comuni_2024.zip").exists():
        universe = _mef_universe()
        ext_ranking = extended_mef_placebo_ranking(m=universe)
        ext_ranking.to_parquet(out_dir / "extended_placebo_ranking.parquet", index=False)
        out["extended_ranking"] = ext_ranking
        ext_sens = extended_pool_sensitivity(universe)
        ext_sens.to_parquet(out_dir / "extended_pool_sensitivity.parquet", index=False)
        out["extended_sensitivity"] = ext_sens
    return out


NORTH_CENTRE_REGIONS: frozenset[str] = frozenset({
    "piemonte", "valle d'aosta", "lombardia", "veneto",
    "trentino-alto adige", "trentino alto adige",
    "friuli-venezia giulia", "friuli venezia giulia",
    "liguria", "emilia-romagna", "emilia romagna",
    "toscana", "umbria", "marche",
})
ER_PROVINCES: frozenset[str] = frozenset({"PC", "PR", "RE", "MO", "BO", "FE", "RA", "FC", "RN"})


def _parse_mef_universe_year(year: int, raw_dir: Path = mef_comuni.RAW_DIR, target_catastali: set[str] | None = None, min_taxpayers: float = 10000.0) -> pd.DataFrame:
    """Parse one MEF municipal ZIP across all municipalities meeting `min_taxpayers` (or in `target_catastali`)."""
    df = mef_comuni.read_year_frame(mef_comuni.zip_path(year, raw_dir))
    contrib_col = next(c for c in df.columns if "numero contribuenti" in c.lower())
    prov_col = next(c for c in df.columns if "sigla provincia" in c.lower())
    reg_col = next(c for c in df.columns if "regione" in c.lower())
    name_col = next(c for c in df.columns if "denominazione" in c.lower())
    add_eur_col = next((c for c in df.columns if re.match(r"^Addizionale comunale dovuta\s*-\s*Ammontare", c, re.IGNORECASE)), None)

    df["taxpayers_total"] = df[contrib_col].map(mef_comuni._to_num)
    if target_catastali is not None:
        df = df[df["catastale"].isin(target_catastali)].copy()
    else:
        df = df[df["taxpayers_total"] >= min_taxpayers].copy()

    rows: list[dict] = []
    for _, r in df.iterrows():
        bdf = pd.DataFrame(mef_comuni.parse_brackets(r, df.columns))
        st = bracket_gini_and_median(bdf)
        rows.append({
            "catastale": str(r["catastale"]),
            "comune": str(r[name_col]).strip(),
            "prov": str(r[prov_col]).strip(),
            "regione": str(r[reg_col]).strip(),
            "year": year,
            "taxpayers_total": float(r["taxpayers_total"]),
            "addizionale_eur": mef_comuni._to_num(r[add_eur_col]) if add_eur_col else np.nan,
            **st,
        })
    out = pd.DataFrame(rows)
    out["mean_income_real_eur"] = deflate(out["mean_income_eur"], out["year"], base_year=2020)
    out["median_income_real_eur"] = deflate(out["median_income_eur"], out["year"], base_year=2020)
    out["addizionale_real_per_payer"] = deflate(out["addizionale_eur"], out["year"], base_year=2020) / out["taxpayers_pos"].replace(0, np.nan)
    return out


def _mef_universe(raw_dir: Path = mef_comuni.RAW_DIR) -> pd.DataFrame:
    """Municipalities with 10k+ taxpayers in 2024, joined to 2008, with growth columns. Size is measured in 2024, so a pool conditions on later growth to a small degree."""
    y24 = _parse_mef_universe_year(2024, raw_dir=raw_dir, min_taxpayers=10000.0)
    y08 = _parse_mef_universe_year(2008, raw_dir=raw_dir, target_catastali=set(y24["catastale"]))
    m = y24.set_index("catastale")[
        ["comune", "prov", "regione", "taxpayers_total", "mean_income_real_eur", "median_income_real_eur", "gini_pareto", "addizionale_real_per_payer"]
    ].join(
        y08.set_index("catastale")[["mean_income_real_eur", "median_income_real_eur", "gini_pareto"]],
        rsuffix="_2008",
        how="inner",
    )
    m["mean_income_growth_08_24_pct"] = (m["mean_income_real_eur"] / m["mean_income_real_eur_2008"] - 1.0) * 100.0
    m["median_income_growth_08_24_pct"] = (m["median_income_real_eur"] / m["median_income_real_eur_2008"] - 1.0) * 100.0
    m["gini_change_08_24_pts"] = m["gini_pareto"] - m["gini_pareto_2008"]
    return m


def extended_pool_sensitivity(m: pd.DataFrame) -> pd.DataFrame:
    """Rank of Cesena on mean and median income growth across region and size windows: a result that needs one window is not a result."""
    regions = {
        "Emilia-Romagna": m["prov"].isin(ER_PROVINCES),
        "North and Centre": m["regione"].str.lower().isin(NORTH_CENTRE_REGIONS),
        "Italy": pd.Series(True, index=m.index),
    }
    windows = ((10000, 400000), (20000, 200000), (30000, 150000), (40000, 120000), (50000, 100000))
    rows: list[dict] = []
    for region, mask in regions.items():
        for lo, hi in windows:
            sub = m[mask & m["taxpayers_total"].between(lo, hi)]
            for col in ("mean_income_growth_08_24_pct", "median_income_growth_08_24_pct"):
                ser = sub[col].dropna()
                if "C573" not in ser.index:
                    continue
                rows.append({"region": region, "taxpayers_lo": lo, "taxpayers_hi": hi, "metric": col, "rank": int((ser > ser.loc["C573"]).sum() + 1), "n": len(ser), "peer_median": float(ser.drop(index="C573").median())})
    return pd.DataFrame(rows)


TAXPAYER_CLASSES: tuple[float, ...] = (0, 2000, 5000, 10000, 20000, 40000, np.inf)


def mef_region_universe(provinces: frozenset[str] = ER_PROVINCES, raw_dir: Path = mef_comuni.RAW_DIR) -> pd.DataFrame:
    """Every municipality of `provinces`, of any size, in 2024 joined to 2008 by cadastral code, with real mean income growth. Municipalities created by a merger after 2008 have no 2008 row and drop out (`attrs['dropped']`)."""
    frame = mef_comuni.read_year_frame(mef_comuni.zip_path(2024, raw_dir))
    prov_col = next(c for c in frame.columns if "sigla provincia" in c.lower())
    codes = set(frame.loc[frame[prov_col].astype(str).str.strip().isin(provinces), "catastale"].astype(str))
    y24 = _parse_mef_universe_year(2024, raw_dir=raw_dir, target_catastali=codes)
    y08 = _parse_mef_universe_year(2008, raw_dir=raw_dir, target_catastali=codes)
    m = y24.set_index("catastale")[["comune", "prov", "taxpayers_total", "mean_income_real_eur"]].join(y08.set_index("catastale")[["mean_income_real_eur"]], rsuffix="_2008", how="inner")
    m["mean_income_growth_08_24_pct"] = (m["mean_income_real_eur"] / m["mean_income_real_eur_2008"] - 1.0) * 100.0
    m.attrs["dropped"] = sorted(set(y24["comune"]) - set(m["comune"]))
    return m


def size_class_growth(m: pd.DataFrame, focal: str = "C573", bins: tuple[float, ...] = TAXPAYER_CLASSES, col: str = "mean_income_growth_08_24_pct", top: int = 3) -> pd.DataFrame:
    """Growth by 2024 taxpayer class with the focal municipality left out: count, median, spread, how many grew more than the focal one and the `top` of them by name.

    A mean over few taxpayers moves with a handful of incomes, so small classes spread more and fill both tails: a falling rank in a pool that adds small municipalities is expected even with no difference in level.
    """
    focal_value = float(m.loc[focal, col])
    others = m.drop(index=focal)
    rows: list[dict] = []
    for lo, hi in itertools.pairwise(bins):
        g = others[(others["taxpayers_total"] >= lo) & (others["taxpayers_total"] < hi)]
        if g.empty:
            continue
        above = g[g[col] > focal_value].sort_values(col, ascending=False)
        rows.append({"lo": lo, "hi": hi, "n": len(g), "median": float(g[col].median()), "sd": float(g[col].std()),
                     "n_above": len(above), "top_above": [str(c).title() for c in above["comune"].head(top)], "focal_value": focal_value})
    return pd.DataFrame(rows)


def extended_mef_placebo_ranking(raw_dir: Path = mef_comuni.RAW_DIR, m: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank Cesena (`C573`) across 4 overlapping MEF pools on 4 metrics. All 16 rank proportions are reported with BH q-values: they are orderings chosen after Cesena's rank among 13 was known, not independent tests. Main pool, declared at the review: `NC_midsize_40k_120k`."""
    m = _mef_universe(raw_dir) if m is None else m

    pools = [
        ("ER_15k_payers", "Emilia-Romagna municipalities >=15k taxpayers", m[m["prov"].isin(ER_PROVINCES) & (m["taxpayers_total"] >= 15000)]),
        ("ER_10k_payers", "Emilia-Romagna municipalities >=10k taxpayers", m[m["prov"].isin(ER_PROVINCES) & (m["taxpayers_total"] >= 10000)]),
        ("NC_midsize_40k_120k", "Northern & Central Italian cities (40k-120k taxpayers)", m[m["regione"].str.lower().isin(NORTH_CENTRE_REGIONS) & m["taxpayers_total"].between(40000, 120000)]),
        ("IT_midsize_40k_120k", "All Italian mid-sized cities (40k-120k taxpayers)", m[m["taxpayers_total"].between(40000, 120000)]),
    ]
    metrics = [
        ("mean_income_growth_08_24_pct", "Real mean IRPEF income growth 2008-2024 (%)", True),
        ("median_income_growth_08_24_pct", "Real median IRPEF income growth 2008-2024 (%)", True),
        ("addizionale_real_per_payer", "Lowest IRPEF surcharge per taxpayer 2024 (2020 EUR)", False),
        ("gini_change_08_24_pts", "Pareto Gini change 2008-2024 (pts, lower = more equal)", False),
    ]
    rows: list[dict] = []
    for p_key, p_lbl, sub in pools:
        if "C573" not in sub.index:
            continue
        n_tot = len(sub)
        for col, m_lbl, high_is_rank1 in metrics:
            ser = sub[col].dropna()
            n_val = len(ser)
            c_val = float(ser.loc["C573"])
            peers = ser.drop(index="C573")
            rank_pos = int((ser > c_val).sum() + 1) if high_is_rank1 else int((ser < c_val).sum() + 1)
            p_two = float(min(1.0, 2.0 * min(rank_pos, n_val - rank_pos + 1) / n_val))
            rows.append({
                "pool_key": p_key,
                "pool_label": p_lbl,
                "metric": col,
                "metric_label": m_lbl,
                "cesena_value": c_val,
                "peer_median_ex_cesena": float(peers.median()),
                "gap_vs_peer_median": c_val - float(peers.median()),
                "rank": rank_pos,
                "n_comuni": n_val,
                "placebo_two_sided_p": p_two,
                "below_05": p_two < 0.05,
            })
            _ = n_tot
    out = pd.DataFrame(rows)
    out["is_main_pool"] = out["pool_key"] == "NC_midsize_40k_120k"
    out["q_bh"] = bh_fdr(out["placebo_two_sided_p"])
    return out


def _fit_convex_weights(xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """Convex donor weights (`w >= 0`, `sum(w) = 1`) minimising the pre-period squared gap, with a small ridge for ties."""
    n_d = xp.shape[1]
    res = minimize(
        lambda w: float(np.mean((yp - xp @ w) ** 2)) + 1e-6 * float(np.sum(w**2)),
        x0=np.ones(n_d) / n_d,
        bounds=[(0.0, 1.0)] * n_d,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
    )
    w = np.clip(np.asarray(res.x, dtype=float), 0.0, 1.0)
    return w / np.sum(w)


SYNTH_SPECS: tuple[tuple[str, str, int, int], ...] = (
    ("employees", "Private employment cumulative growth (%)", 2012, 2024),
    ("avg_firm_size", "Average unit size cumulative growth (%)", 2012, 2024),
    ("mean_income_real_eur", "Real mean IRPEF income cumulative growth (%)", 2008, 2024),
    ("median_income_real_eur", "Real median IRPEF income cumulative growth (%)", 2008, 2024),
)


def synthetic_cesena(
    panel_df: pd.DataFrame,
    focal_id: str = CESENA,
    t_pre_end: int = 2018,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct a convex Synthetic Control for Cesena (`w >= 0`, `sum(w) = 1`) matched on pre-2019 trajectories (`t0..t_pre_end`)."""
    series_rows: list[dict] = []
    sum_rows: list[dict] = []

    for col, lbl, t0, t1 in SYNTH_SPECS:
        if col not in panel_df.columns:
            continue
        piv = (
            panel_df[(panel_df["year"] >= t0) & (panel_df["year"] <= t1)]
            .pivot(index="year", columns="comune_id", values=col)
            .dropna(axis=1)
        )
        if focal_id not in piv.columns or len(piv.columns) < 4:
            continue
        piv_pct = (piv / piv.loc[t0] - 1.0) * 100.0
        donors = [str(c) for c in piv_pct.columns if str(c) != focal_id]
        pre_years = [int(y) for y in piv_pct.index if int(y) <= t_pre_end]
        y_pre = piv_pct.loc[pre_years, focal_id].to_numpy(dtype=float)
        x_pre = piv_pct.loc[pre_years, donors].to_numpy(dtype=float)
        w = _fit_convex_weights(x_pre, y_pre)
        synth = piv_pct[donors].to_numpy(dtype=float) @ w
        peer_med = piv_pct[donors].median(axis=1).to_numpy(dtype=float)

        for idx_y, y in enumerate(piv_pct.index):
            c_v = float(piv_pct.loc[y, focal_id])
            s_v = float(synth[idx_y])
            m_v = float(peer_med[idx_y])
            series_rows.append({
                "metric": col,
                "label": lbl,
                "year": int(y),
                "cesena_cum_pct": c_v,
                "synthetic_cesena_cum_pct": s_v,
                "peer_median_cum_pct": m_v,
                "gap_vs_synthetic_pp": c_v - s_v,
                "gap_vs_peer_median_pp": c_v - m_v,
            })

        pre_mask = np.array([int(y) <= t_pre_end for y in piv_pct.index])
        pre_rmse = float(np.sqrt(np.mean((piv_pct[focal_id].to_numpy(dtype=float)[pre_mask] - synth[pre_mask]) ** 2)))
        top_donors = ", ".join(
            f"{COMUNI_BY_ID[d].name if d in COMUNI_BY_ID else d} ({wt:.2f})"
            for d, wt in sorted(zip(donors, w, strict=False), key=lambda kv: kv[1], reverse=True)
            if wt >= 0.03
        )
        sum_rows.append({
            "metric": col,
            "label": lbl,
            "pre_period": f"{t0}-{t_pre_end}",
            "eval_year": t1,
            "pre_rmse_pp": pre_rmse,
            "cesena_end_pct": float(piv_pct.loc[t1, focal_id]),
            "synthetic_end_pct": float(synth[-1]),
            "peer_median_end_pct": float(peer_med[-1]),
            "gap_vs_synthetic_pp": float(piv_pct.loc[t1, focal_id] - synth[-1]),
            "gap_vs_peer_median_pp": float(piv_pct.loc[t1, focal_id] - peer_med[-1]),
            "donor_weights": top_donors,
        })

    return pd.DataFrame(series_rows), pd.DataFrame(sum_rows)


def synthetic_control_placebos(
    panel_df: pd.DataFrame,
    focal_id: str = CESENA,
    t_pre_end: int = 2018,
    pseudo_t_end: int = 2014,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run donor-pool in-space permutations and in-time pseudo-treatment placebos for Synthetic Control evaluation."""
    in_space_rows: list[dict] = []
    in_time_rows: list[dict] = []

    for col, lbl, t0, t1 in SYNTH_SPECS:
        if col not in panel_df.columns:
            continue
        piv = (
            panel_df[(panel_df["year"] >= t0) & (panel_df["year"] <= t1)]
            .pivot(index="year", columns="comune_id", values=col)
            .dropna(axis=1)
        )
        if focal_id not in piv.columns or len(piv.columns) < 4:
            continue
        piv_pct = (piv / piv.loc[t0] - 1.0) * 100.0
        all_units = [str(c) for c in piv_pct.columns]
        n_tot = len(all_units)

        # 1. In-space permutations across all units in the sample
        unit_stats: list[dict] = []
        pre_years = [int(y) for y in piv_pct.index if int(y) <= t_pre_end]
        idx_pre = np.array([int(y) <= t_pre_end for y in piv_pct.index])
        idx_post = np.array([int(y) > t_pre_end for y in piv_pct.index])

        for uid in all_units:
            donors = [d for d in all_units if d != uid]
            y_pre = piv_pct.loc[pre_years, uid].to_numpy(dtype=float)
            x_pre = piv_pct.loc[pre_years, donors].to_numpy(dtype=float)
            w = _fit_convex_weights(x_pre, y_pre)
            synth = piv_pct[donors].to_numpy(dtype=float) @ w
            actual = piv_pct[uid].to_numpy(dtype=float)
            gap = actual - synth

            pre_rmspe = float(np.sqrt(np.mean(gap[idx_pre] ** 2)))
            post_rmspe = float(np.sqrt(np.mean(gap[idx_post] ** 2))) if any(idx_post) else 0.0
            ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else 0.0
            end_gap = float(gap[-1])

            unit_stats.append({
                "unit_id": uid,
                "unit_name": COMUNI_BY_ID[uid].name if uid in COMUNI_BY_ID else uid,
                "is_cesena": uid == focal_id,
                "pre_rmspe": pre_rmspe,
                "post_rmspe": post_rmspe,
                "ratio": ratio,
                "end_gap": end_gap,
            })

        df_u = pd.DataFrame(unit_stats)
        for r in unit_stats:
            r_gap = int((df_u["end_gap"] > r["end_gap"]).sum() + 1)
            r_ratio = int((df_u["ratio"] > r["ratio"]).sum() + 1)
            in_space_rows.append({
                "metric": col,
                "label": lbl,
                "unit_id": r["unit_id"],
                "unit_name": r["unit_name"],
                "is_cesena": r["is_cesena"],
                "pre_rmspe": r["pre_rmspe"],
                "post_rmspe": r["post_rmspe"],
                "ratio_post_pre_rmspe": r["ratio"],
                "gap_end_pp": r["end_gap"],
                "rank_gap_desc": r_gap,
                "rank_ratio_desc": r_ratio,
                "n_units": n_tot,
                "perm_p_gap": float(r_gap / n_tot),
                "perm_p_ratio": float(r_ratio / n_tot),
            })

        # 2. In-time placebo for Cesena with pseudo_t_end
        if t0 <= pseudo_t_end < t_pre_end:
            donors_ces = [d for d in all_units if d != focal_id]
            pre_ps_yrs = [int(y) for y in piv_pct.index if int(y) <= pseudo_t_end]
            yp_ps = piv_pct.loc[pre_ps_yrs, focal_id].to_numpy(dtype=float)
            xp_ps = piv_pct.loc[pre_ps_yrs, donors_ces].to_numpy(dtype=float)
            w_ps = _fit_convex_weights(xp_ps, yp_ps)
            synth_ps = piv_pct[donors_ces].to_numpy(dtype=float) @ w_ps
            actual_ces = piv_pct[focal_id].to_numpy(dtype=float)
            gap_ps = actual_ces - synth_ps

            m_pre_ps = np.array([int(y) <= pseudo_t_end for y in piv_pct.index])
            m_ps_post = np.array([(int(y) > pseudo_t_end and int(y) <= t_pre_end) for y in piv_pct.index])
            m_true_post = np.array([int(y) > t_pre_end for y in piv_pct.index])

            in_time_rows.append({
                "metric": col,
                "label": lbl,
                "pseudo_year": pseudo_t_end,
                "pre_rmse_pp": float(np.sqrt(np.mean(gap_ps[m_pre_ps] ** 2))),
                "pseudo_post_mean_gap_pp": float(np.mean(gap_ps[m_ps_post])) if any(m_ps_post) else 0.0,
                "true_post_mean_gap_pp": float(np.mean(gap_ps[m_true_post])) if any(m_true_post) else 0.0,
                "end_gap_pp": float(gap_ps[-1]),
            })

    return pd.DataFrame(in_space_rows), pd.DataFrame(in_time_rows)


def _fit_end_gap(pct: pd.DataFrame, names: list[str], actual: np.ndarray, pre: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Weights, end gap and pre-period RMSE of a synthetic control on the donors in `names`."""
    x = pct[names].to_numpy(dtype=float)
    w = _fit_convex_weights(x[pre], actual[pre])
    gap = actual - x @ w
    return w, float(gap[-1]), float(np.sqrt(np.mean(gap[pre] ** 2)))


def synthetic_leave_one_out(panel_df: pd.DataFrame, focal_id: str = CESENA, t_pre_end: int = 2018) -> pd.DataFrame:
    """Refit the synthetic control dropping each donor with weight >= 3% in turn: a gap resting on one donor is fragile."""
    rows: list[dict] = []
    for col, lbl, t0, t1 in SYNTH_SPECS:
        if col not in panel_df.columns:
            continue
        piv = panel_df[panel_df["year"].between(t0, t1)].pivot(index="year", columns="comune_id", values=col).dropna(axis=1)
        if focal_id not in piv.columns or len(piv.columns) < 4:
            continue
        pct = (piv / piv.loc[t0] - 1.0) * 100.0
        donors = [str(c) for c in pct.columns if str(c) != focal_id]
        pre = pct.index <= t_pre_end
        actual = pct[focal_id].to_numpy(dtype=float)

        w_full, gap_full, rmse_full = _fit_end_gap(pct, donors, actual, pre)
        rows.append({"metric": col, "label": lbl, "dropped": "none", "weight_dropped": 0.0, "gap_end_pp": gap_full, "pre_rmse_pp": rmse_full})
        for d, wt in zip(donors, w_full, strict=True):
            if wt >= 0.03:
                _, gap, rmse = _fit_end_gap(pct, [x for x in donors if x != d], actual, pre)
                rows.append({"metric": col, "label": lbl, "dropped": COMUNI_BY_ID[d].name if d in COMUNI_BY_ID else d, "weight_dropped": float(wt), "gap_end_pp": gap, "pre_rmse_pp": rmse})
    return pd.DataFrame(rows)
