"""Pillar 1 — Municipal Fiscal Policy & Resource Allocation (`studies.cesena.analysis.fiscal`).

Implements:
1. **Per-capita real fiscal panel (`enrich_fiscal_panel`)**:
   - Joins BDAP/OpenBilanci municipal budgets (`2008-2022`) with MEF municipal IRPEF surcharge (`2008-2024`), ISTAT population (`2008-2025`), and the Italian CPI/HICP deflator (`base_year=2020`).
   - Computes real per-capita current expenditure (`spese_correnti_real_pc`), capital expenditure (`spese_capitale_real_pc`), own tax revenues (`entrate_tributarie_real_pc`, `imposte_tasse_real_pc`), IMU/ICI per capita (`imu_ici_real_pc`), and Addizionale Comunale IRPEF per capita (`addizionale_irpef_real_pc` and `mef_addizionale_real_pc`).
   - Preserves the D.Lgs. 118/2011 accounting harmonization break (`regime = "dpr194"` with `flag = "break_dlgs118"` for `<=2015` vs `regime = "dlgs118"` for `>=2016`) and never computes growth rates across the 2015->2016 boundary.
2. **Harmonized expenditure by Mission (`mission_allocation_summary`)**:
   - Summarizes `2016-2021` consuntivo (plus `2022` preventivo) current and capital spending per capita across the harmonized Missions (`M01`..`M20`, `M50`) for Cesena (`040007`) vs the peer median (excluding Cesena).
3. **Regime-aware growth & peer ranking (`compare_fiscal_regimes`)**:
   - Evaluates Cesena's rank and gap against peers separately within the pre-harmonization (`2008-2015`) and harmonized (`2016-2021`) consuntivo regimes.
"""
import numpy as np
import pandas as pd

from cifresignificative.core.deflators import deflate
from cifresignificative.core.geo import CESENA, COMUNI_BY_ID

NOMINAL_EUR_COLS: tuple[str, ...] = (
    "entrate_tributarie_comp_eur",
    "entrate_tributarie_cassa_eur",
    "imposte_tasse_comp_eur",
    "imposte_tasse_cassa_eur",
    "fondo_perequativo_comp_eur",
    "imu_ici_comp_eur",
    "addizionale_irpef_comp_eur",
    "tari_tarsu_comp_eur",
    "trasferimenti_correnti_comp_eur",
    "entrate_extratributarie_comp_eur",
    "entrate_correnti_comp_eur",
    "entrate_correnti_cassa_eur",
    "entrate_capitale_comp_eur",
    "contributi_investimenti_comp_eur",
    "contributi_investimenti_cassa_eur",
    "alienazioni_comp_eur",
    "alienazioni_cassa_eur",
    "accensione_prestiti_comp_eur",
    "accensione_mutui_comp_eur",
    "accensione_mutui_cassa_eur",
    "fpv_corrente_comp_eur",
    "fpv_capitale_comp_eur",
    "avanzo_amministrazione_comp_eur",
    "entrate_totali_comp_eur",
    "spese_correnti_comp_eur",
    "spese_correnti_cassa_eur",
    "spese_capitale_comp_eur",
    "spese_capitale_cassa_eur",
    "rimborso_prestiti_comp_eur",
    "spese_totali_comp_eur",
    "spesa_personale_comp_eur",
    "interessi_passivi_comp_eur",
    "missione_01_correnti_comp_eur",
    "missione_50_debito_comp_eur",
)


def enrich_fiscal_panel(
    budgets_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    irpef_summary_df: pd.DataFrame | None = None,
    base_year: int = 2020,
) -> pd.DataFrame:
    """Merge municipal budgets with population and MEF Addizionale IRPEF, deflating all nominal series to constant `base_year` EUR per capita."""
    pop = demo_df[["comune_id", "year", "pop_total"]].copy()
    out = budgets_df.merge(pop, on=["comune_id", "year"], how="left")

    if irpef_summary_df is not None and len(irpef_summary_df):
        mef = irpef_summary_df.rename(columns={"tax_year": "year"})[
            ["comune_id", "year", "addizionale_comunale_eur", "addizionale_comunale_freq", "reddito_imponibile_eur"]
        ].copy()
        out = out.merge(mef, on=["comune_id", "year"], how="left")
        out["effective_surcharge_rate_pct"] = (
            out["addizionale_comunale_eur"] / out["reddito_imponibile_eur"].replace(0, np.nan) * 100.0
        )

    p_denom = out["pop_total"].replace(0, np.nan)
    for col in NOMINAL_EUR_COLS:
        if col not in out.columns:
            continue
        stem = col.removesuffix("_eur")
        real_s = deflate(out[col], out["year"], base_year=base_year)
        out[f"{stem}_real_eur"] = real_s
        out[f"{stem}_real_pc"] = real_s / p_denom

    if "addizionale_comunale_eur" in out.columns:
        out["mef_addizionale_real_eur"] = deflate(out["addizionale_comunale_eur"], out["year"], base_year=base_year)
        out["mef_addizionale_real_pc"] = out["mef_addizionale_real_eur"] / p_denom

    out["core_peer"] = out["comune_id"].map(lambda cid: COMUNI_BY_ID[str(cid)].core_peer if str(cid) in COMUNI_BY_ID else False)

    # Regime-safe annual growth rates (explicitly NaN across the 2015 -> 2016 D.Lgs. 118/2011 break and on 2022 preventivo)
    out = out.sort_values(["comune_id", "year"], ignore_index=True)
    for m_col in ("spese_correnti_comp_real_pc", "spese_capitale_comp_real_pc", "entrate_tributarie_comp_real_pc"):
        if m_col not in out.columns:
            continue
        prev_val = out.groupby(["comune_id", "regime", "budget_type"])[m_col].shift(1)
        out[f"dlog_{m_col}"] = np.log(out[m_col].where(out[m_col] > 0) / prev_val.where(prev_val > 0)) * 100.0
    return out


def mission_allocation_summary(
    missions_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    years: tuple[int, ...] = (2016, 2017, 2018, 2019, 2020, 2021),
    focal_id: str = CESENA,
    core_only: bool = True,
    base_year: int = 2020,
) -> pd.DataFrame:
    """Compare average real per-capita current and capital spending by harmonized Mission (`2016-2021` consuntivo) for Cesena vs peer median."""
    sub = missions_df[(missions_df["year"].isin(years)) & (missions_df["budget_type"] == "consuntivo")].copy()
    sub = sub.merge(demo_df[["comune_id", "year", "pop_total"]], on=["comune_id", "year"], how="left")
    pop = sub["pop_total"].replace(0, np.nan)

    for c in ("correnti_comp_eur", "capitale_comp_eur", "totale_comp_eur", "correnti_cassa_eur", "capitale_cassa_eur"):
        stem = c.removesuffix("_eur")
        sub[f"{stem}_real_pc"] = deflate(sub[c], sub["year"], base_year=base_year) / pop

    # Exclude technical Mission 99 (Servizi per conto terzi) and Mission 60 (Anticipazioni) from policy allocation shares
    policy = sub[~sub["mission_code"].isin(["M60", "M99"])].copy()
    muni_mean = (
        policy.groupby(["comune_id", "mission_code", "mission_label"], as_index=False)[
            ["correnti_comp_real_pc", "capitale_comp_real_pc", "totale_comp_real_pc"]
        ]
        .mean()
    )

    focal = muni_mean[muni_mean["comune_id"] == focal_id].set_index("mission_code")
    peers = muni_mean[muni_mean["comune_id"] != focal_id].copy()
    if core_only:
        core_ids = {cid for cid, c in COMUNI_BY_ID.items() if c.core_peer and cid != focal_id}
        peers = peers[peers["comune_id"].isin(core_ids)]

    peer_med = peers.groupby("mission_code")[["correnti_comp_real_pc", "capitale_comp_real_pc", "totale_comp_real_pc"]].median()

    tot_f_corr = float(focal["correnti_comp_real_pc"].sum()) or np.nan
    tot_f_cap = float(focal["capitale_comp_real_pc"].sum()) or np.nan

    rows: list[dict] = []
    for mc in sorted(focal.index.intersection(peer_med.index)):
        fc_corr = float(focal.loc[mc, "correnti_comp_real_pc"])
        fc_cap = float(focal.loc[mc, "capitale_comp_real_pc"])
        fc_tot = float(focal.loc[mc, "totale_comp_real_pc"])
        pm_corr = float(peer_med.loc[mc, "correnti_comp_real_pc"])
        pm_cap = float(peer_med.loc[mc, "capitale_comp_real_pc"])
        pm_tot = float(peer_med.loc[mc, "totale_comp_real_pc"])
        rows.append({
            "mission_code": mc,
            "mission_label": str(focal.loc[mc, "mission_label"]),
            "cesena_correnti_real_pc": fc_corr,
            "peer_correnti_real_pc": pm_corr,
            "gap_correnti_real_pc": fc_corr - pm_corr,
            "cesena_capitale_real_pc": fc_cap,
            "peer_capitale_real_pc": pm_cap,
            "gap_capitale_real_pc": fc_cap - pm_cap,
            "cesena_totale_real_pc": fc_tot,
            "peer_totale_real_pc": pm_tot,
            "gap_totale_real_pc": fc_tot - pm_tot,
            "cesena_correnti_share_pct": fc_corr / tot_f_corr * 100.0,
            "cesena_capitale_share_pct": fc_cap / tot_f_cap * 100.0,
        })
    return pd.DataFrame(rows).sort_values("cesena_totale_real_pc", ascending=False, ignore_index=True)


def compare_fiscal_regimes(fiscal_df: pd.DataFrame, focal_id: str = CESENA, core_only: bool = True) -> pd.DataFrame:
    """Summarize Cesena vs peer median separately in pre-harmonization (`2008-2015`) and harmonized (`2016-2021`) consuntivo regimes."""
    cons = fiscal_df[fiscal_df["budget_type"] == "consuntivo"].copy()
    metrics = [
        "spese_correnti_comp_real_pc",
        "spese_capitale_comp_real_pc",
        "capital_share_pct",
        "contributi_investimenti_comp_real_pc",
        "alienazioni_comp_real_pc",
        "accensione_mutui_comp_real_pc",
        "fpv_capitale_comp_real_pc",
        "avanzo_amministrazione_comp_real_pc",
        "entrate_tributarie_comp_real_pc",
        "imposte_tasse_comp_real_pc",
        "imu_ici_comp_real_pc",
        "mef_addizionale_real_pc",
        "rigidity_official_pct",
        "rigidity_struct_pct",
        "collection_rate_correnti_pct",
        "collection_rate_proprie_pct",
    ]
    avail = [m for m in metrics if m in cons.columns]
    rows: list[dict] = []
    for reg, reg_lbl in [("dpr194", "D.P.R. 194/1996 (2008-2015)"), ("dlgs118", "D.Lgs. 118/2011 (2016-2021)")]:
        sub = cons[cons["regime"] == reg]
        if sub.empty:
            continue
        muni_avg = sub.groupby("comune_id")[avail].mean()
        if focal_id not in muni_avg.index:
            continue
        peers = muni_avg.drop(index=focal_id)
        if core_only:
            core_ids = [cid for cid in peers.index if COMUNI_BY_ID.get(str(cid), None) and COMUNI_BY_ID[str(cid)].core_peer]
            peers = peers.loc[core_ids]
        for m in avail:
            c_val = float(muni_avg.loc[focal_id, m])
            p_ser = peers[m].dropna()
            if np.isnan(c_val) or p_ser.empty:
                continue
            p_med = float(p_ser.median())
            # Rank 1 = highest value among (Cesena + all 12 ER peers)
            all_ser = muni_avg[m].dropna()
            rank_desc = int((all_ser > c_val).sum() + 1)
            rows.append({
                "regime": reg,
                "regime_label": reg_lbl,
                "metric": m,
                "cesena_mean": c_val,
                "peer_median": p_med,
                "gap_vs_peers": c_val - p_med,
                "gap_pct": (c_val - p_med) / abs(p_med) * 100.0 if p_med != 0 else np.nan,
                "rank_1_to_n": rank_desc,
                "n_comuni": len(all_ser),
            })
    return pd.DataFrame(rows)


def capital_funding_breakdown(
    fiscal_df: pd.DataFrame,
    focal_id: str = CESENA,
    core_only: bool = True,
) -> pd.DataFrame:
    """Annual breakdown of capital expenditure vs funding channels (grants, asset sales, borrowing, capital FPV, surplus)."""
    cons = fiscal_df[fiscal_df["budget_type"] == "consuntivo"].copy()
    cols = [
        "spese_capitale_comp_real_pc",
        "contributi_investimenti_comp_real_pc",
        "alienazioni_comp_real_pc",
        "accensione_mutui_comp_real_pc",
        "fpv_capitale_comp_real_pc",
        "avanzo_amministrazione_comp_real_pc",
    ]
    focal = cons[cons["comune_id"] == focal_id].set_index("year")
    peers = cons[cons["comune_id"] != focal_id].copy()
    if core_only:
        core_ids = [cid for cid, c in COMUNI_BY_ID.items() if c.core_peer and cid != focal_id]
        peers = peers[peers["comune_id"].isin(core_ids)]
    peer_med = peers.groupby("year")[cols].median()

    rows: list[dict] = []
    for y in sorted(focal.index.intersection(peer_med.index)):
        reg = str(focal.loc[y, "regime"]) if "regime" in focal.columns else ""
        row = {"year": int(y), "regime": reg}
        for c in cols:
            stem = c.removesuffix("_comp_real_pc").removesuffix("_real_pc")
            row[f"cesena_{stem}_pc"] = float(focal.loc[y, c])
            row[f"peer_{stem}_pc"] = float(peer_med.loc[y, c])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_consolidated_benchmarks(
    cons_df: pd.DataFrame,
    focal_id: str = CESENA,
    year: int = 2024,
    core_only: bool = True,
) -> pd.DataFrame:
    """Benchmark direct municipal cash expenditure vs consolidated (Comune + Unione) per-capita spending across peers.

    Evaluates whether Cesena's lower apparent direct spending on current services and personnel
    represents net cost efficiency or institutional delegation to *Unione dei Comuni Valle del Savio*.
    """
    sub = cons_df[cons_df["year"] == year].copy()
    if sub.empty:
        return pd.DataFrame()

    metrics = [
        ("direct_correnti_pc", "Direct Current Spending (EUR/pc)"),
        ("unione_correnti_pc", "Unione Delegated Spending (EUR/pc)"),
        ("consolidated_correnti_pc", "Consolidated Current Spending (EUR/pc)"),
        ("direct_personale_pc", "Direct Personnel Spending (EUR/pc)"),
        ("consolidated_personale_pc", "Consolidated Personnel Spending (EUR/pc)"),
        ("direct_capitale_pc", "Direct Capital Investment (EUR/pc)"),
        ("consolidated_capitale_pc", "Consolidated Capital Investment (EUR/pc)"),
    ]

    focal_row = sub[sub["comune_id"] == focal_id]
    if focal_row.empty:
        return pd.DataFrame()
    focal = focal_row.iloc[0]

    peers = sub[sub["comune_id"] != focal_id].copy()
    if core_only:
        core_ids = [cid for cid, c in COMUNI_BY_ID.items() if c.core_peer and cid != focal_id]
        peers = peers[peers["comune_id"].isin(core_ids)]

    rows: list[dict] = []
    for col, lbl in metrics:
        if col not in sub.columns:
            continue
        c_val = float(focal[col])
        p_ser = peers[col].dropna()
        if p_ser.empty:
            continue
        p_med = float(p_ser.median())
        all_ser = sub[col].dropna()
        rank_desc = int((all_ser > c_val).sum() + 1)
        rows.append({
            "year": year,
            "metric": col,
            "metric_label": lbl,
            "cesena": c_val,
            "peer_median": p_med,
            "gap_vs_peers": c_val - p_med,
            "gap_pct": (c_val - p_med) / abs(p_med) * 100.0 if p_med != 0 else np.nan,
            "rank_desc": rank_desc,
            "n_comuni": len(all_ser),
        })
    return pd.DataFrame(rows)

