r"""Pillar 3 — Local Real Economy, Specialization & Shift-Share Decomposition (`studies.cesena.analysis.economy`).

Implements:
1. **Classical & Esteban-Marquillas Shift-Share Decomposition (`shift_share`)**:
   - Decomposes employment (`employees`) or active local unit (`local_units`) growth between `t0` and `t1` into:
     - **Benchmark Share (`ns_abs`, `ns_pp`)**: growth that would have occurred if the municipality grew at the aggregate peer benchmark rate $G_B$.
     - **Industry Mix (`im_abs`, `im_pp`)**: structural component from specializing in sectors that grew faster/slower than $G_B$ in the peer benchmark $\sum_i E_{i,c}^0 (G_{i,B} - G_B)$.
     - **Local Competitive Shift (`cs_abs`, `cs_pp`)**: idiosyncratic local outperformance within each sector $\sum_i E_{i,c}^0 (g_{i,c} - G_{i,B})$.
     - **Esteban-Marquillas Homothetic Competitive & Allocation Effects (`em_comp_pp`, `em_alloc_pp`)**.
   - **Pitfall guard**: selects only `{t0, t1}` before pivoting/filtering (never dropping a sector across all years) and strictly excludes the focal municipality from the benchmark aggregate.
2. **Location Quotients (`location_quotients`)**:
   - $LQ_{i,c,t} = (E_{i,c,t} / E_{c,t}) / (E_{i,B,t} / E_{B,t})$ across the 7 Cesena specialization clusters and 2-digit ATECO divisions.
3. **Average Firm Size Trajectory & Shift-Share (`firm_size_decomposition`)**:
   - Decomposes the change in municipal average unit size (`employees / local_units`) into sector-mix change vs. within-sector firm scaling.
4. **Register breaks and sensitivity (`employment_anomalies`, `employment_sensitivity`)**:
   - ASIA counts persons employed by local unit, so one large unit can move a municipal sector by thousands in a year. The screen flags such moves; the sensitivity table shows what each headline result is without the affected sectors and at other end years.
"""
import itertools

import numpy as np
import pandas as pd

from cifresignificative.core.geo import CESENA, COMUNI_BY_ID, CORE_PEER_IDS
from cifresignificative.core.sources.istat_asia import (
    SIZE_CLASS_LABELS,
    SIZE_CLASS_ORDER,
    SPECIALIZATION_LABELS,
)


def _prepare_two_year_sector_panel(
    asia_df: pd.DataFrame,
    t0: int,
    t1: int,
    level: str = "specialization",
    metric: str = "employees",
) -> pd.DataFrame:
    """Filter `asia_df` to `{t0, t1}` and non-total 2-digit sectors before aggregating to `level` (`specialization` or `ateco_2d`)."""
    sub = asia_df[(asia_df["year"].isin([t0, t1])) & (~asia_df["is_total"].astype(bool))].copy()
    if level == "specialization":
        grp_col = "specialization"
        lbl_map = SPECIALIZATION_LABELS
    else:
        grp_col = "ateco_2d"
        lbl_map = dict(zip(sub["ateco_2d"], sub["ateco_label"], strict=False))

    agg = (
        sub.groupby(["comune_id", "year", grp_col], as_index=False)[metric]
        .sum()
        .rename(columns={grp_col: "sector"})
    )
    piv = (
        agg.pivot(index=["comune_id", "sector"], columns="year", values=metric)
        .reindex(columns=[t0, t1])
        .fillna(0.0)
        .reset_index()
        .rename(columns={t0: "v0", t1: "v1"})
    )
    piv["sector_label"] = piv["sector"].map(lambda s: lbl_map.get(str(s), str(s)))
    return piv


def shift_share(
    asia_df: pd.DataFrame,
    t0: int = 2012,
    t1: int = 2024,
    focal_id: str = CESENA,
    level: str = "specialization",
    metric: str = "employees",
    core_only: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Compute sector-by-sector Shift-Share decomposition for `focal_id` between `t0` and `t1` against the peer benchmark (ex `focal_id`)."""
    piv = _prepare_two_year_sector_panel(asia_df, t0=t0, t1=t1, level=level, metric=metric)

    focal = piv[piv["comune_id"] == focal_id].set_index("sector")
    peers = piv[piv["comune_id"] != focal_id].copy()
    if core_only:
        core_ids = {cid for cid, c in COMUNI_BY_ID.items() if c.core_peer and cid != focal_id}
        peers = peers[peers["comune_id"].isin(core_ids)]

    peer_ids_used = sorted(peers["comune_id"].unique())
    bench = peers.groupby("sector")[["v0", "v1"]].sum()

    common_sectors = sorted(set(focal.index).intersection(bench.index))
    focal = focal.loc[common_sectors]
    bench = bench.loc[common_sectors]

    e_c0 = float(focal["v0"].sum())
    e_c1 = float(focal["v1"].sum())
    e_b0 = float(bench["v0"].sum())
    e_b1 = float(bench["v1"].sum())
    if e_c0 <= 0 or e_b0 <= 0:
        raise ValueError(f"Non-positive base {metric} in {t0} for focal={focal_id} ({e_c0}) or benchmark ({e_b0})")

    g_b_tot = (e_b1 - e_b0) / e_b0
    g_c_tot = (e_c1 - e_c0) / e_c0

    rows: list[dict] = []
    for sec in common_sectors:
        c0 = float(focal.loc[sec, "v0"])
        c1 = float(focal.loc[sec, "v1"])
        b0 = float(bench.loc[sec, "v0"])
        b1 = float(bench.loc[sec, "v1"])
        lbl = str(focal.loc[sec, "sector_label"])

        if b0 <= 0 and c0 <= 0:
            continue
        g_ib = (b1 - b0) / b0 if b0 > 0 else g_b_tot
        g_ic = (c1 - c0) / c0 if c0 > 0 else 0.0

        ns_abs = c0 * g_b_tot
        im_abs = c0 * (g_ib - g_b_tot)
        cs_abs = (c1 - c0) - ns_abs - im_abs  # exact identity even when c0 == 0 and c1 > 0

        # Esteban-Marquillas homothetic employment E*_{i,c} = E_{c,0} * (E_{i,B,0} / E_{B,0})
        homothetic_c0 = e_c0 * (b0 / e_b0)
        em_comp_abs = homothetic_c0 * (g_ic - g_ib)
        em_alloc_abs = cs_abs - em_comp_abs

        rows.append({
            "sector": sec,
            "sector_label": lbl,
            "v0": c0,
            "v1": c1,
            "delta_abs": c1 - c0,
            "share_t0_pct": c0 / e_c0 * 100.0,
            "share_t1_pct": c1 / e_c1 * 100.0 if e_c1 > 0 else np.nan,
            "bench_share_t0_pct": b0 / e_b0 * 100.0,
            "local_growth_pct": g_ic * 100.0,
            "bench_growth_pct": g_ib * 100.0,
            "ns_abs": ns_abs,
            "im_abs": im_abs,
            "cs_abs": cs_abs,
            "ns_pp": ns_abs / e_c0 * 100.0,
            "im_pp": im_abs / e_c0 * 100.0,
            "cs_pp": cs_abs / e_c0 * 100.0,
            "total_contrib_pp": (c1 - c0) / e_c0 * 100.0,
            "em_comp_pp": em_comp_abs / e_c0 * 100.0,
            "em_alloc_pp": em_alloc_abs / e_c0 * 100.0,
        })

    df_out = pd.DataFrame(rows).sort_values("total_contrib_pp", ascending=False, ignore_index=True)
    summary = {
        "focal_id": focal_id,
        "focal_name": COMUNI_BY_ID[focal_id].name if focal_id in COMUNI_BY_ID else focal_id,
        "t0": t0,
        "t1": t1,
        "metric": metric,
        "level": level,
        "n_peers": len(peer_ids_used),
        "peer_ids": peer_ids_used,
        "v0_total": e_c0,
        "v1_total": e_c1,
        "delta_total_abs": e_c1 - e_c0,
        "local_growth_total_pct": g_c_tot * 100.0,
        "bench_growth_total_pct": g_b_tot * 100.0,
        "ns_total_pp": float(df_out["ns_pp"].sum()),
        "im_total_pp": float(df_out["im_pp"].sum()),
        "cs_total_pp": float(df_out["cs_pp"].sum()),
        "active_gap_pp": float(df_out["im_pp"].sum() + df_out["cs_pp"].sum()),
    }
    return df_out, summary


def shift_share_all_municipalities(
    asia_df: pd.DataFrame,
    t0: int = 2012,
    t1: int = 2024,
    level: str = "specialization",
    metric: str = "employees",
) -> pd.DataFrame:
    """Run Shift-Share decomposition for every municipality in `asia_df` against the leave-one-out benchmark of the remaining ER peers."""
    summaries: list[dict] = []
    for cid in sorted(asia_df["comune_id"].unique()):
        _, s = shift_share(asia_df, t0=t0, t1=t1, focal_id=str(cid), level=level, metric=metric, core_only=False)
        summaries.append(s)
    df = pd.DataFrame(summaries).sort_values("cs_total_pp", ascending=False, ignore_index=True)
    df["cs_rank"] = np.arange(1, len(df) + 1)
    return df


def location_quotients(
    asia_df: pd.DataFrame,
    focal_id: str = CESENA,
    level: str = "specialization",
    metric: str = "employees",
) -> pd.DataFrame:
    """Compute annual Location Quotients ($LQ_{i,c,t}$) for `focal_id` relative to the leave-one-out ER peer pool."""
    sub = asia_df[~asia_df["is_total"].astype(bool)].copy()
    grp_col = "specialization" if level == "specialization" else "ateco_2d"
    agg = sub.groupby(["comune_id", "year", grp_col], as_index=False)[metric].sum()

    focal = agg[agg["comune_id"] == focal_id].rename(columns={metric: "focal_val"})
    bench = (
        agg[agg["comune_id"] != focal_id]
        .groupby(["year", grp_col], as_index=False)[metric]
        .sum()
        .rename(columns={metric: "bench_val"})
    )
    m = focal.merge(bench, on=["year", grp_col], how="inner")
    tot_f = m.groupby("year")["focal_val"].transform("sum").replace(0, np.nan)
    tot_b = m.groupby("year")["bench_val"].transform("sum").replace(0, np.nan)
    m["focal_share_pct"] = m["focal_val"] / tot_f * 100.0
    m["bench_share_pct"] = m["bench_val"] / tot_b * 100.0
    m["lq"] = m["focal_share_pct"] / m["bench_share_pct"].replace(0, np.nan)
    if level == "specialization":
        m["sector_label"] = m[grp_col].map(lambda s: SPECIALIZATION_LABELS.get(str(s), str(s)))
    return m.sort_values(["year", "lq"], ascending=[True, False], ignore_index=True)


def firm_size_decomposition(
    asia_df: pd.DataFrame,
    t0: int = 2012,
    t1: int = 2024,
) -> pd.DataFrame:
    """Decompose municipal change in average unit size (`employees / local_units`) between `t0` and `t1` into sector-unit-mix vs within-sector scaling."""
    sub = asia_df[(asia_df["year"].isin([t0, t1])) & (~asia_df["is_total"].astype(bool))].copy()
    agg = sub.groupby(["comune_id", "year", "specialization"], as_index=False)[["local_units", "employees"]].sum()

    rows: list[dict] = []
    for cid, grp in agg.groupby("comune_id"):
        g0 = grp[grp["year"] == t0].set_index("specialization")
        g1 = grp[grp["year"] == t1].set_index("specialization")
        common = sorted(set(g0.index).intersection(g1.index))
        if not common:
            continue
        u0_tot = float(g0.loc[common, "local_units"].sum())
        e0_tot = float(g0.loc[common, "employees"].sum())
        u1_tot = float(g1.loc[common, "local_units"].sum())
        e1_tot = float(g1.loc[common, "employees"].sum())
        if u0_tot <= 0 or u1_tot <= 0:
            continue

        size0 = e0_tot / u0_tot
        size1 = e1_tot / u1_tot

        # Exact Bennet / Baily-Bartelsman decomposition of delta(E/U) = sum_s bar_size_s * delta(w_s) + sum_s bar_w_s * delta(size_s)
        mix_effect = 0.0
        within_effect = 0.0
        for s in common:
            u0, e0 = float(g0.loc[s, "local_units"]), float(g0.loc[s, "employees"])
            u1, e1 = float(g1.loc[s, "local_units"]), float(g1.loc[s, "employees"])
            w0, w1 = u0 / u0_tot, u1 / u1_tot
            s0 = e0 / u0 if u0 > 0 else 0.0
            s1 = e1 / u1 if u1 > 0 else 0.0
            mix_effect += 0.5 * (s0 + s1) * (w1 - w0)
            within_effect += 0.5 * (w0 + w1) * (s1 - s0)

        rows.append({
            "comune_id": str(cid),
            "comune": COMUNI_BY_ID[str(cid)].name if str(cid) in COMUNI_BY_ID else str(cid),
            "t0": t0,
            "t1": t1,
            "units_t0": u0_tot,
            "units_t1": u1_tot,
            "units_growth_pct": (u1_tot - u0_tot) / u0_tot * 100.0,
            "employees_t0": e0_tot,
            "employees_t1": e1_tot,
            "employees_growth_pct": (e1_tot - e0_tot) / e0_tot * 100.0,
            "avg_size_t0": size0,
            "avg_size_t1": size1,
            "delta_avg_size": size1 - size0,
            "mix_effect": mix_effect,
            "within_scaling_effect": within_effect,
        })
    return pd.DataFrame(rows).sort_values("delta_avg_size", ascending=False, ignore_index=True)


def size_class_summary(
    size_df: pd.DataFrame,
    t0: int = 2012,
    t1: int = 2024,
    focal_id: str = CESENA,
    core_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare employment and local-unit dynamics across the 4 ASIA size classes (`W0_9`, `W10_49`, `W50_249`, `W_GE250`) between `t0` and `t1`.

    Returns:
    - `class_df`: Cesena vs peer benchmark (excluding `focal_id`) by size class (`t0` vs `t1` local units, employees, shares, and contributions to total employment growth).
    - `muni_df`: Cross-municipality comparison of micro-firm (`0-9`), small (`10-49`), medium (`50-249`), and large (`250+`) employment shares and growth contributions across all 13 ER municipalities.
    """
    classes = [sc for sc in SIZE_CLASS_ORDER if sc != "TOTAL"]
    tot = size_df[(size_df["year"].isin([t0, t1])) & (size_df["is_total"].astype(bool)) & (size_df["size_class"].isin(classes))].copy()

    # 1. Cross-municipality size-class share and growth contribution table
    piv_e = (
        tot.pivot_table(index=["comune_id", "size_class"], columns="year", values="employees", aggfunc="sum")
        .reindex(columns=[t0, t1])
        .fillna(0.0)
    )
    piv_u = (
        tot.pivot_table(index=["comune_id", "size_class"], columns="year", values="local_units", aggfunc="sum")
        .reindex(columns=[t0, t1])
        .fillna(0.0)
    )

    muni_rows: list[dict] = []
    for cid in sorted(tot["comune_id"].unique()):
        cid_s = str(cid)
        if cid_s not in piv_e.index.get_level_values(0):
            continue
        e_sub = piv_e.loc[cid_s].reindex(classes).fillna(0.0)
        u_sub = piv_u.loc[cid_s].reindex(classes).fillna(0.0)
        e0_tot = float(e_sub[t0].sum())
        e1_tot = float(e_sub[t1].sum())
        if e0_tot <= 0 or e1_tot <= 0:
            continue
        row: dict[str, object] = {
            "comune_id": cid_s,
            "comune": COMUNI_BY_ID[cid_s].name if cid_s in COMUNI_BY_ID else cid_s,
            "core_peer": COMUNI_BY_ID[cid_s].core_peer if cid_s in COMUNI_BY_ID else False,
            "emp_t0": e0_tot,
            "emp_t1": e1_tot,
            "emp_growth_pct": (e1_tot - e0_tot) / e0_tot * 100.0,
        }
        for sc in classes:
            e0_sc = float(e_sub.loc[sc, t0])
            e1_sc = float(e_sub.loc[sc, t1])
            u0_sc = float(u_sub.loc[sc, t0])
            u1_sc = float(u_sub.loc[sc, t1])
            row[f"share_{sc}_t0_pct"] = e0_sc / e0_tot * 100.0
            row[f"share_{sc}_t1_pct"] = e1_sc / e1_tot * 100.0
            row[f"delta_share_{sc}_pp"] = (e1_sc / e1_tot - e0_sc / e0_tot) * 100.0
            row[f"contrib_{sc}_pp"] = (e1_sc - e0_sc) / e0_tot * 100.0
            row[f"delta_units_{sc}"] = u1_sc - u0_sc
        row["sme_50_249_and_large_contrib_pp"] = (
            float(e_sub.loc["W50_249", t1] - e_sub.loc["W50_249", t0] + e_sub.loc["W_GE250", t1] - e_sub.loc["W_GE250", t0])
            / e0_tot
            * 100.0
        )
        muni_rows.append(row)
    muni_df = pd.DataFrame(muni_rows).sort_values("contrib_W50_249_pp", ascending=False, ignore_index=True)
    muni_df["rank_W50_249_contrib"] = np.arange(1, len(muni_df) + 1)

    # 2. Detailed size-class table for focal_id vs peer pool (excluding focal_id)
    focal_e = piv_e.loc[focal_id].reindex(classes).fillna(0.0)
    focal_u = piv_u.loc[focal_id].reindex(classes).fillna(0.0)

    peer_ids = [cid for cid in piv_e.index.get_level_values(0).unique() if str(cid) != focal_id]
    if core_only:
        peer_ids = [cid for cid in peer_ids if COMUNI_BY_ID.get(str(cid), None) and COMUNI_BY_ID[str(cid)].core_peer]

    bench_e = piv_e.loc[peer_ids].groupby("size_class")[[t0, t1]].sum().reindex(classes).fillna(0.0)
    bench_u = piv_u.loc[peer_ids].groupby("size_class")[[t0, t1]].sum().reindex(classes).fillna(0.0)

    fe0_tot, fe1_tot = float(focal_e[t0].sum()), float(focal_e[t1].sum())
    be0_tot, be1_tot = float(bench_e[t0].sum()), float(bench_e[t1].sum())

    class_rows: list[dict] = []
    for sc in classes:
        u0, u1 = float(focal_u.loc[sc, t0]), float(focal_u.loc[sc, t1])
        e0, e1 = float(focal_e.loc[sc, t0]), float(focal_e.loc[sc, t1])
        be0, be1 = float(bench_e.loc[sc, t0]), float(bench_e.loc[sc, t1])
        bu0, bu1 = float(bench_u.loc[sc, t0]), float(bench_u.loc[sc, t1])
        class_rows.append({
            "size_class": sc,
            "size_label": SIZE_CLASS_LABELS.get(sc, sc),
            "units_t0": u0,
            "units_t1": u1,
            "delta_units": u1 - u0,
            "units_growth_pct": (u1 - u0) / u0 * 100.0 if u0 > 0 else np.nan,
            "emp_t0": e0,
            "emp_t1": e1,
            "delta_emp": e1 - e0,
            "emp_growth_pct": (e1 - e0) / e0 * 100.0 if e0 > 0 else np.nan,
            "bench_emp_growth_pct": (be1 - be0) / be0 * 100.0 if be0 > 0 else np.nan,
            "bench_units_growth_pct": (bu1 - bu0) / bu0 * 100.0 if bu0 > 0 else np.nan,
            "emp_share_t0_pct": e0 / fe0_tot * 100.0,
            "emp_share_t1_pct": e1 / fe1_tot * 100.0,
            "bench_emp_share_t0_pct": be0 / be0_tot * 100.0,
            "bench_emp_share_t1_pct": be1 / be1_tot * 100.0,
            "contrib_to_total_growth_pp": (e1 - e0) / fe0_tot * 100.0,
            "bench_contrib_to_total_growth_pp": (be1 - be0) / be0_tot * 100.0,
        })
    class_df = pd.DataFrame(class_rows)
    return class_df, muni_df


def size_class_by_cluster(
    size_df: pd.DataFrame,
    t0: int = 2012,
    t1: int = 2024,
    focal_id: str = CESENA,
) -> pd.DataFrame:
    """Break down `focal_id`'s employment change between `t0` and `t1` across specialization clusters x size classes."""
    classes = [sc for sc in SIZE_CLASS_ORDER if sc != "TOTAL"]
    sub = size_df[
        (size_df["comune_id"] == focal_id)
        & (size_df["year"].isin([t0, t1]))
        & (~size_df["is_total"].astype(bool))
        & (size_df["size_class"].isin(classes))
    ].copy()
    # Compute t0, t1, and delta by cluster x size_class
    sub_y = sub.groupby(["specialization", "specialization_label", "year", "size_class"], as_index=False)["employees"].sum()
    p0 = (
        sub_y[sub_y["year"] == t0]
        .pivot(index=["specialization", "specialization_label"], columns="size_class", values="employees")
        .reindex(columns=classes)
        .fillna(0.0)
    )
    p1 = (
        sub_y[sub_y["year"] == t1]
        .pivot(index=["specialization", "specialization_label"], columns="size_class", values="employees")
        .reindex(columns=classes)
        .fillna(0.0)
    )
    common = p1.index.union(p0.index)
    p0 = p0.reindex(common).fillna(0.0)
    p1 = p1.reindex(common).fillna(0.0)
    delta = p1 - p0
    out = pd.DataFrame(index=common)
    for sc in classes:
        out[f"emp_{sc}_{t1}"] = p1[sc]
        out[f"delta_{sc}"] = delta[sc]
    out["emp_total_t0"] = p0.sum(axis=1)
    out["emp_total_t1"] = p1.sum(axis=1)
    out["delta_total"] = delta.sum(axis=1)
    return out.reset_index().sort_values("delta_total", ascending=False, ignore_index=True)


def sector_size_swing(
    size_df: pd.DataFrame,
    ateco: tuple[str, ...] = ("10",),
    years: tuple[int, ...] = (2021, 2022, 2023, 2024),
    focal_id: str = CESENA,
) -> pd.DataFrame:
    """Employees and local units of `focal_id` in the 2-digit divisions `ateco` (summed) by size class and year, with the year-on-year employee change.

    A swing confined to the 250+ class (a unit appearing or vanishing) points to one large unit; a broad one to the sector. `share_of_change_<y>` is the class's part of the total change into year `y`.
    """
    classes = [sc for sc in SIZE_CLASS_ORDER if sc != "TOTAL"]
    sub = size_df[
        (size_df["comune_id"] == focal_id) & size_df["ateco_2d"].isin(ateco) & size_df["year"].isin(years) & size_df["size_class"].isin(classes)
    ]
    out = pd.DataFrame(index=pd.Index(classes, name="size_class"))
    for col, tag in (("employees", "emp"), ("local_units", "lu")):
        p = sub.groupby(["size_class", "year"])[col].sum().unstack("year").reindex(index=classes, columns=list(years)).fillna(0.0)
        for y in years:
            out[f"{tag}_{y}"] = p[y]
    for a, b in itertools.pairwise(years):
        out[f"delta_{b}"] = out[f"emp_{b}"] - out[f"emp_{a}"]
        tot = out[f"delta_{b}"].sum()
        out[f"share_of_change_{b}"] = out[f"delta_{b}"] / tot if tot else np.nan
    return out.reset_index()


def summarize_frame_sbs_benchmarks(
    panel: pd.DataFrame | None = None,
    sector: str = "totale",
) -> pd.DataFrame:
    """Extract annual benchmark table comparing Cesena's Value Added per worker and wages against Core and All ER Peers."""
    if panel is None:
        from cifresignificative.studies.cesena.analysis.data import load_frame_sbs_panel
        panel = load_frame_sbs_panel()

    sec_df = panel[(panel["macro_sector"] == sector) & (panel["is_er_benchmark"])].copy()
    rows = []
    for yr, y_df in sec_df.groupby("year"):
        ces = y_df[y_df["is_cesena"]]
        if ces.empty:
            continue
        c = ces.iloc[0]
        core_peers = y_df[y_df["is_core_peer"]]

        va_add_rank = int((y_df["va_per_addetto_eur"].rank(ascending=False, method="min")).loc[c.name])
        retrib_dip_rank = int((y_df["retribuzione_per_dipendente_eur"].rank(ascending=False, method="min")).loc[c.name])

        rows.append({
            "year": int(yr),
            "macro_sector": sector,
            "cesena_addetti": float(c["addetti"]),
            "cesena_va_per_addetto_eur": float(c["va_per_addetto_eur"]),
            "cesena_va_per_addetto_real_eur": float(c["va_per_addetto_real_eur"]),
            "core_peer_va_per_addetto_eur": float(core_peers["va_per_addetto_eur"].median()),
            "gap_va_per_addetto_core_pct": float((c["va_per_addetto_eur"] / core_peers["va_per_addetto_eur"].median() - 1.0) * 100.0),
            "rank_va_per_addetto_er": va_add_rank,
            "cesena_retrib_per_dipendente_eur": float(c["retribuzione_per_dipendente_eur"]),
            "cesena_retrib_per_dipendente_real_eur": float(c["retribuzione_per_dipendente_real_eur"]),
            "core_peer_retrib_per_dipendente_eur": float(core_peers["retribuzione_per_dipendente_eur"].median()),
            "gap_retrib_per_dipendente_core_pct": float((c["retribuzione_per_dipendente_eur"] / core_peers["retribuzione_per_dipendente_eur"].median() - 1.0) * 100.0),
            "rank_retrib_per_dipendente_er": retrib_dip_rank,
            "cesena_va_su_fatturato_pct": float(c["va_su_fatturato_pct"]) if pd.notna(c["va_su_fatturato_pct"]) else None,
            "cesena_retrib_su_va_pct": float(c["retribuzioni_su_va_pct"]) if pd.notna(c["retribuzioni_su_va_pct"]) else None,
            "n_peers": len(y_df),
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def compare_productivity_and_wages(
    panel: pd.DataFrame | None = None,
    sector: str = "totale",
    t0: int = 2017,
    t1: int = 2023,
) -> pd.DataFrame:
    """Compute cumulative growth in real Value Added, employment, VA per worker, and wages across ER benchmark municipalities."""
    if panel is None:
        from cifresignificative.studies.cesena.analysis.data import load_frame_sbs_panel
        panel = load_frame_sbs_panel()

    sub = panel[(panel["macro_sector"] == sector) & (panel["is_er_benchmark"]) & (panel["year"].isin([t0, t1]))].copy()
    piv_va = sub.pivot(index="comune_id", columns="year", values="valore_aggiunto_real_eur")
    piv_vap = sub.pivot(index="comune_id", columns="year", values="va_per_addetto_real_eur")
    piv_retrib = sub.pivot(index="comune_id", columns="year", values="retribuzione_per_dipendente_real_eur")
    piv_add = sub.pivot(index="comune_id", columns="year", values="addetti")

    out = pd.DataFrame(index=piv_va.index)
    out["comune_nome"] = sub.groupby("comune_id")["comune_nome"].first()
    out["is_cesena"] = out.index == CESENA
    out["is_core_peer"] = out.index.isin(CORE_PEER_IDS)
    out["growth_addetti_pct"] = (piv_add[t1] / piv_add[t0] - 1.0) * 100.0
    out["growth_real_va_pct"] = (piv_va[t1] / piv_va[t0] - 1.0) * 100.0
    out["growth_real_vap_pct"] = (piv_vap[t1] / piv_vap[t0] - 1.0) * 100.0
    out["growth_real_wage_pct"] = (piv_retrib[t1] / piv_retrib[t0] - 1.0) * 100.0

    out["rank_growth_addetti"] = out["growth_addetti_pct"].rank(ascending=False, method="min").astype(int)
    out["rank_growth_real_va"] = out["growth_real_va_pct"].rank(ascending=False, method="min").astype(int)
    out["rank_growth_real_vap"] = out["growth_real_vap_pct"].rank(ascending=False, method="min").astype(int)
    out["rank_growth_real_wage"] = out["growth_real_wage_pct"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("growth_real_va_pct", ascending=False).reset_index()


def employment_anomalies(
    asia_df: pd.DataFrame,
    min_jobs: float = 300.0,
    min_share: float = 0.20,
    reversal_share: float = 0.50,
    step_jobs: float = 1000.0,
) -> pd.DataFrame:
    """Flag one-year sector moves too large to read as ordinary growth (rule fixed before looking at corrected results).

    A row is flagged when the yearly change is at least `min_jobs` employees and `min_share` of the previous level, and either the next year undoes at least `reversal_share` of it (`reversal`) or the change is at least `step_jobs` (`step`).
    The screen finds candidates for a register or single-unit effect; it does not establish the cause.
    """
    sec = asia_df[~asia_df["is_total"].astype(bool)]
    wide = sec.pivot_table(index=["comune_id", "ateco_2d"], columns="year", values="employees", aggfunc="sum").fillna(0.0)
    years = sorted(wide.columns)
    rows: list[dict] = []
    for (cid, sector), vals in wide.iterrows():
        for prev, cur, nxt in zip(years[:-1], years[1:], [*years[2:], None], strict=True):
            delta = float(vals[cur] - vals[prev])
            if abs(delta) < min_jobs or vals[prev] <= 0 or abs(delta) / float(vals[prev]) < min_share:
                continue
            undone = -float(vals[nxt] - vals[cur]) / delta if nxt is not None else np.nan
            kind = "reversal" if np.isfinite(undone) and undone >= reversal_share else ("step" if abs(delta) >= step_jobs else "")
            if kind:
                rows.append({
                    "comune_id": str(cid),
                    "comune": COMUNI_BY_ID[str(cid)].name if str(cid) in COMUNI_BY_ID else str(cid),
                    "ateco_2d": str(sector),
                    "year": int(cur),
                    "employees_prev": float(vals[prev]),
                    "employees": float(vals[cur]),
                    "change": delta,
                    "share_undone_next_year": undone,
                    "kind": kind,
                })
    return pd.DataFrame(rows, columns=["comune_id", "comune", "ateco_2d", "year", "employees_prev", "employees", "change", "share_undone_next_year", "kind"])


SENSITIVITY_VARIANTS: dict[str, tuple[str, ...]] = {
    "all_sectors": (),
    "without_food_10": ("10",),
    "without_food_agency_cleaning": ("10", "78", "81"),
}


def employment_sensitivity(
    asia_df: pd.DataFrame,
    t0: int = 2012,
    end_years: tuple[int, ...] = (2021, 2022, 2024),
    focal_id: str = CESENA,
) -> pd.DataFrame:
    """Value, rank (1 = highest) and peer median of `focal_id` on employment growth, competitive shift and unit-size change, by excluded sectors and end year."""
    rows: list[dict] = []
    for variant, excluded in SENSITIVITY_VARIANTS.items():
        sub = asia_df[~asia_df["ateco_2d"].isin(excluded)]
        sec = sub[~sub["is_total"].astype(bool)]
        emp = sec.groupby(["comune_id", "year"])["employees"].sum().unstack()
        units = sec.groupby(["comune_id", "year"])["local_units"].sum().unstack()
        for t1 in end_years:
            cs = shift_share_all_municipalities(sub, t0=t0, t1=t1, level="ateco_2d").set_index("focal_id")["cs_total_pp"]
            metrics = {
                "employment_growth_pct": (emp[t1] / emp[t0] - 1.0) * 100.0,
                "competitive_shift_pp": cs,
                "unit_size_change": emp[t1] / units[t1] - emp[t0] / units[t0],
            }
            for metric, ser in metrics.items():
                ser = ser.dropna()
                rows.append({
                    "variant": variant,
                    "t0": t0,
                    "t1": t1,
                    "metric": metric,
                    "cesena": float(ser.loc[focal_id]),
                    "rank": int((ser > ser.loc[focal_id]).sum() + 1),
                    "n": len(ser),
                    "peer_median": float(ser.drop(index=focal_id).median()),
                })
    return pd.DataFrame(rows)
