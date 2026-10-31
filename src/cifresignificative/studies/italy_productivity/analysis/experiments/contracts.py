"""Contract timing and the real wage loss across Istat agreement groups."""
from pathlib import Path

import pandas as pd

from cifresignificative.core.sources import cnel_contracts as CN
from cifresignificative.core.sources import cnel_workers as CW
from cifresignificative.core.sources import conto_annuale as CA
from cifresignificative.core.sources import istat_contracts as IC
from cifresignificative.studies.italy_productivity.analysis import contracts as CT

from .common import OUT


def run_contracts_all(out: Path = OUT, n_perm: int = 20000, n_shuffle: int = 200) -> dict[str, pd.DataFrame]:
    """Contract timing and the real wage loss of the 2021-23 inflation shock across Istat's agreement groups (`analysis.contracts`): C1 leaf groups (staleness, standstill, real change to 2024 and 2025), C2 the primary test on the parent
    groups (mean of the sub-sectors, x shuffled within the private and public branch, BH over staleness and standstill against the 2024 and 2025 outcomes), C3 the same on the leaves (branch shuffle) and C4 within parents (the conservative
    null), C5 private groups only, C6 step thresholds of 0.15% and 0.5%, C7 the calm-period placebo (December 2016 to 2019), C8 the shuffled-outcome null, C9 the waiting-time indicators by sector, C10 the aggregates and C11 persistence."""
    from cifresignificative.studies.italy_productivity.analysis.hf import load_highfreq
    wages, groups = IC.load_wages(), IC.load_groups()
    dec = CT.december_inflation(load_highfreq())
    d24, d25 = CT.window_table(wages, groups, dec, 2021, 2024), CT.window_table(wages, groups, dec, 2021, 2025)
    leaves = d24.join(d25[["real_change", "nominal_change"]].rename(columns=lambda c: f"{c}_25"))
    clusters = CT.cluster_table(d24).join(CT.cluster_table(d25)[["real_change"]].rename(columns={"real_change": "real_change_25"}))
    ys = ("real_change", "real_change_25")
    placebo = CT.cluster_table(CT.window_table(wages, groups, dec, *CT.PLACEBO))
    thresholds = pd.concat([CT.family(CT.cluster_table(CT.window_table(wages, groups, dec, 2021, 2024, step=st)), ("real_change",), n_perm=n_perm).assign(step=st) for st in (0.0015, 0.0025, 0.005)], ignore_index=True)
    results = {"c1_leaf_groups": leaves.reset_index(), "c2_cluster_family": CT.family(clusters, ys, n_perm=n_perm), "c3_leaf_family_branch": CT.family(leaves, ys, n_perm=n_perm),
               "c4_leaf_family_within_parent": CT.family(leaves, ys, by="parent", n_perm=n_perm), "c5_cluster_private": CT.family(clusters[clusters.branch == "private"], ("real_change",), by="branch", n_perm=n_perm, n_min=15),
               "c6_cluster_thresholds": thresholds, "c7_placebo_2016_19": CT.family(placebo, ("real_change",), n_perm=n_perm), "c8_null": CT.shuffle_null(clusters, ys, n_shuffle=n_shuffle),
               "c9_tension": CT.tension_table(IC.load_tension()), "c10_aggregates": pd.concat([CT.aggregates_table(wages, groups, dec, 2021, e).assign(end=e) for e in (2024, 2025)]).reset_index(),
               "c11_persistence": CT.persistence(clusters.assign(real_change=clusters.real_change)), "c12_clusters": clusters.reset_index()}
    cyc, cmap = CN.primary_cycles(CN.load()), CN.load_map()
    timing = CT.renewal_timing(cyc, cmap, 2021, CN.ASOF)
    xs = (*CT.XS_CNEL, *CT.XS)
    leaves_v = CT.with_renewals(leaves, timing)
    leaves_a = CT.with_renewals(leaves, timing, ("verified", "partly verified"))
    cl_v, cl_a = (CT.cluster_table(t, extra=(*CT.XS_CNEL, "months_to_renewal")) for t in (leaves_v, leaves_a))
    for cl, t in ((cl_v, leaves_v), (cl_a, leaves_a)):
        cl["real_change_25"] = CT.cluster_table(t.assign(real_change=t.real_change_25)).real_change
    p_t = CT.renewal_timing(cyc, cmap, CT.PLACEBO[0], CN.ASOF)
    p_cl = CT.cluster_table(CT.with_renewals(CT.window_table(wages, groups, dec, *CT.PLACEBO), p_t), extra=CT.XS_CNEL)
    results |= {"c13_renewal_leaves": timing.reset_index(), "c14_cnel_cluster_family": CT.family(cl_v, ys, CT.XS_CNEL, n_perm=n_perm, n_min=15), "c15_proxy_same_clusters": CT.family(cl_v, ys, CT.XS, n_perm=n_perm, n_min=15),
                "c16_cnel_leaf_family": CT.family(leaves_v, ys, CT.XS_CNEL, n_perm=n_perm, n_min=15), "c17_cnel_cluster_with_partly": CT.family(cl_a, ys, xs, n_perm=n_perm, n_min=15), "c18_proxy_agreement": CT.proxy_agreement(leaves_a),
                "c19_cnel_placebo_2016_19": CT.family(p_cl, ("real_change",), CT.XS_CNEL, n_perm=n_perm, n_min=15), "c20_cnel_null": CT.shuffle_null(cl_v, ys, CT.XS_CNEL, n_shuffle=n_shuffle)}
    hf = load_highfreq()
    decs = {k: CT.december_inflation(hf, indicator=v) for k, v in CT.DEFLATOR_INDICATORS.items()}
    defl = CT.deflator_of(groups)
    cl_sector = defl.reindex(leaves.index).groupby(leaves.parent).first().rename("sector")  # a parent's sector is its leaves' (the parent code itself can be a bare aggregate, as Z2520 for agriculture)
    sec = leaves.assign(real_change=CT.sector_real_change(leaves, defl, decs, 2021, 2024), real_change_25=CT.sector_real_change(leaves, defl, decs, 2021, 2025, "nominal_change_25"))
    sec_cl = CT.cluster_table(sec, extra=("real_change_25",))
    sec_v = CT.with_renewals(sec, timing)
    sec_v_cl = CT.cluster_table(sec_v, extra=(*CT.XS_CNEL, "real_change_25"))
    results |= {"c21_sector_deflator_proxy": CT.family(sec_cl, ys, CT.XS, n_perm=n_perm), "c22_sector_deflator_cnel": CT.family(sec_v_cl, ys, CT.XS_CNEL, n_perm=n_perm, n_min=15),
                "c24_proxy_within_sector": CT.family(clusters.join(cl_sector), ys, CT.XS, by="sector", n_perm=n_perm), "c25_cnel_within_sector": CT.family(cl_v.join(cl_sector), ys, CT.XS_CNEL, by="sector", n_perm=n_perm, n_min=15),
                "c26_timing_by_sector": clusters.join(cl_sector).join(cl_v[[*CT.XS_CNEL, "months_to_renewal"]]).groupby("sector").agg(n=("real_change", "size"), staleness=("staleness_months", "mean"), months_since_renewal=("months_since_renewal", "mean"), months_expired=("months_expired", "mean"), real_change_national=("real_change", "mean")).reset_index(),
                "c23_deflators": pd.DataFrame([{"component": k, "start": 2021, "end": e, "cumulative_pct": CT.cumulative_inflation(v, 2021, e)} for k, v in decs.items() for e in (2024, 2025)])}
    # employee weights (CNEL-INPS 2025 counts of the mapped agreement codes; public agreements have none, so the weighted tests are private-sector tests)
    lw = CT.leaf_weights(cmap, CW.load())
    pw = CT.parent_weights(leaves, lw).rename_axis("group")
    cl_n = leaves.groupby("parent").size()
    results |= {"c27_weighted_summary": pd.concat([CT.weighted_summary(leaves, lw, y).assign(outcome=y) for y in ys]), "c28_weighted_proxy": CT.weighted_family(clusters, pw, ys, CT.XS, n_perm=n_perm),
                "c29_unweighted_same_rows": CT.family(clusters[pw.reindex(clusters.index).notna()], ys, CT.XS, n_perm=n_perm, n_min=15), "c30_weighted_cnel": CT.weighted_family(cl_v, pw, ys, CT.XS_CNEL, n_perm=n_perm),
                "c31_weights": pd.DataFrame({"employees": pw, "n_leaves": cl_n}).join(clusters[["real_change", "branch"]]).rename_axis("group").reset_index(),
                "c32_weighted_leaf_proxy": CT.weighted_family(leaves, lw, ys, CT.XS, n_perm=n_perm), "c33_weighted_null": CT.weighted_null(clusters, pw, ys, CT.XS, n_shuffle=n_shuffle)}
    # all groups: the private CNEL-INPS counts plus the public Conto annuale heads (31 December 2021)
    law = CT.all_leaf_weights(lw, CT.public_leaf_weights(CA.load_map(), CA.load(year=2021)))
    paw = CT.parent_weights(leaves, law).rename_axis("group")
    results |= {"c34_weighted_summary_all": pd.concat([CT.weighted_summary(leaves, law, y).assign(outcome=y) for y in ys]), "c35_weights_all": pd.DataFrame({"employees": paw, "n_leaves": cl_n}).join(clusters[["real_change", "branch"]]).rename_axis("group").reset_index(),
                "c36_weighted_proxy_all": CT.weighted_family(clusters, paw, ys, CT.XS, n_perm=n_perm), "c37_unweighted_same_rows_all": CT.family(clusters[paw.reindex(clusters.index).notna()], ys, CT.XS, n_perm=n_perm, n_min=15),
                "c38_weighted_cnel_all": CT.weighted_family(cl_v, paw, ys, CT.XS_CNEL, n_perm=n_perm), "c39_weighted_leaf_proxy_all": CT.weighted_family(leaves, law, ys, CT.XS, n_perm=n_perm),
                "c40_weighted_null_all": CT.weighted_null(clusters, paw, ys, CT.XS, n_shuffle=n_shuffle)}
    # variants of the weighting: Conto annuale year (start or end of the window), special-statute regions and Bolzano/Trento schools added, parent outcomes as unweighted or employee-weighted means of the leaves
    rows = []
    for year in (2021, 2024):
        for ext in (False, True):
            wv = CT.all_leaf_weights(lw, CT.public_leaf_weights(CA.load_map(), CA.load(year=year), extended=ext))
            tags = {"year": year, "extended": ext}
            pwv = CT.parent_weights(leaves, wv)
            rows.append(CT.weighted_family(clusters, pwv.rename_axis("group"), ys, CT.XS, n_perm=n_perm).assign(**tags, parent_mean="unweighted", level="parent", family="proxy"))
            wc, wc_v = CT.weighted_cluster_table(leaves, wv), CT.weighted_cluster_table(leaves_v, wv, extra=CT.XS_CNEL)
            rows += [CT.weighted_family(wc, wc.employees, ys, CT.XS, n_perm=n_perm).assign(**tags, parent_mean="weighted", level="parent", family="proxy"),
                     CT.weighted_family(wc_v, wc_v.employees, ys, CT.XS_CNEL, n_perm=n_perm).assign(**tags, parent_mean="weighted", level="parent", family="cnel"),
                     CT.weighted_family(leaves, wv, ys, CT.XS, n_perm=n_perm).assign(**tags, parent_mean="", level="leaf", family="proxy")]
    results["c41_weighted_variants"] = pd.concat(rows, ignore_index=True)
    wc21 = CT.weighted_cluster_table(leaves, law)
    results["c42_parent_weighted_means"] = wc21.join(clusters[["real_change"]].rename(columns={"real_change": "real_change_unweighted_mean"})).reset_index()
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results
