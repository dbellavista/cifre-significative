"""NUTS 2021 crosswalk, spatial dependence, EU cohesion funds and NUTS1 experiments."""
from pathlib import Path

import numpy as np
import pandas as pd

from cifresignificative.core.clean import nuts
from cifresignificative.core.sources import eqi as EQI
from cifresignificative.core.sources import eu_payments as EP
from cifresignificative.studies.italy_productivity.analysis import corr
from cifresignificative.studies.italy_productivity.analysis import eu_funds as EF
from cifresignificative.studies.italy_productivity.analysis import nuts1 as N1
from cifresignificative.studies.italy_productivity.analysis import regional_corr as rc
from cifresignificative.studies.italy_productivity.analysis import regional_south as rs
from cifresignificative.studies.italy_productivity.analysis import spatial as SP
from cifresignificative.studies.italy_productivity.analysis.data import (
    PROCESSED,
    load_regional,
    load_regional_extra,
)

from .common import OUT, WINDOWS


def _nuts2021_matrix() -> pd.DataFrame:
    """Region x year indicator table on the NUTS 2021 geography (harmonised `regional` plus `regional_extra`), the counterpart of the matrix `run_south_all` builds."""
    reg, ex = pd.read_parquet(PROCESSED / "regional_nuts2021.parquet"), pd.read_parquet(PROCESSED / "regional_extra_nuts2021.parquet")
    return rc.regional_matrix(pd.concat([reg, ex], ignore_index=True), drop_flag_chars="u")


def run_nuts_all(out: Path = OUT, n_perm: int = 2000) -> dict[str, pd.DataFrame]:
    """NUTS harmonisation round: N1 regions per window before and after, N2 agreement of old and new codes where Eurostat publishes both, N3 the crosswalk of the codes in use, N4 the "why the South" EU-wide family (`rs.eu_wide`)
    on the original and on the NUTS 2021 geography side by side (sensitivity of notebook 34 to the code mix)."""
    regional, extra = load_regional(), load_regional_extra()
    reg21 = pd.read_parquet(PROCESSED / "regional_nuts2021.parquet")
    cw = nuts.crosswalk()
    used = set(regional.geo) | set(extra.geo)
    inds = sorted(regional.indicator.unique())
    n1 = nuts.coverage(regional, reg21, {"early 2005-07": (2005, 2007), "recent 2015-19": (2015, 2019), "late 2021-23": (2021, 2023)}, inds)
    n2 = nuts.overlap_agreement(regional, cw)
    n3 = cw[cw.old.isin(used) & cw.kind.ne("same")].reset_index(drop=True)
    Ro, Rh = rc.regional_matrix(pd.concat([regional, extra], ignore_index=True), drop_flag_chars="u"), _nuts2021_matrix()
    frames = []
    for geography, R in (("original", Ro), ("nuts2021", Rh)):
        for ic in (False, True):
            frames.append(rs.eu_wide(R, income_control=ic, n_perm=n_perm).assign(geography=geography, income_control=ic))
    f = pd.concat(frames, ignore_index=True)
    key = ["x", "y", "wave", "income_control"]
    n4 = f[f.geography.eq("original")][[*key, "n", "rho", "q_value", "significant"]].merge(f[f.geography.eq("nuts2021")][[*key, "n", "rho", "q_value", "significant"]], on=key, suffixes=("_original", "_nuts2021"), how="outer")
    results = {"n1_coverage": n1, "n2_overlap": n2, "n3_crosswalk": n3, "n4_south_geography": n4}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_spatial_all(out: Path = OUT, n_perm: int = 2000, n_sim: int = 100) -> dict[str, pd.DataFrame]:
    """Spatial-dependence round on the NUTS 2021 geography (`analysis.spatial`): SP1 Moran's I of every column before and after filtering, SP2 the "why the South" EU-wide family plain and filtered for every EQI wave
    (plain and beyond income), SP3 Italy's rank among the countries on filtered data, SP4 calibration with simulated spatially smooth x (plain against filtered pipeline)."""
    R = _nuts2021_matrix()
    nb = SP.Neighbours(sorted(R.index.get_level_values(0).unique()))
    moran, fam, placebo = [], [], []
    for w in rs.WAVES:
        for ic in (False, True):
            X = rs.within_country(rs.wave_table(R, w), ic)
            t, d = SP.compare_family(X, nb, rs.EU_WIDE, rs.OUTCOMES, n_perm=n_perm)
            fam.append(t.assign(wave=w, income_control=ic, primary=(w == rs.PRIMARY_WAVE)))
            moran.append(d.assign(wave=w, income_control=ic))
            if w == rs.PRIMARY_WAVE and not ic:
                placebo.append(SP.italy_placebo_filtered(X, nb, rs.EU_WIDE, rs.OUTCOMES).assign(wave=w))
    X0 = rs.within_country(rs.wave_table(R, rs.PRIMARY_WAVE), False)
    sp4 = pd.concat([SP.spatial_null(X0, nb, rs.EU_WIDE, rs.OUTCOMES, n_sim=n_sim, rho=r) for r in (0.4, 0.8)], ignore_index=True)  # weak and strong autocorrelation (the EQI columns have Moran's I about 0.3-0.5)
    results = {"sp1_moran": pd.concat(moran, ignore_index=True), "sp2_family": pd.concat(fam, ignore_index=True), "sp3_italy_rank": pd.concat(placebo, ignore_index=True), "sp4_null": sp4}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def _screen_tables(R: pd.DataFrame, Rb: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The two within-country tables of the regional screens (R1 levels 2015-19, R2 changes between 2005-09 and 2015-19), as in `run_regional_screen`."""
    return {"levels": rc.within_country_table(R, "level", WINDOWS["recent"]), "changes": rc.within_country_table(Rb, "change", WINDOWS["recent"], WINDOWS["early"])}


def run_spatial_screens(out: Path = OUT, n_perm: int = 20000, n_sim: int = 50, n_perm_null: int = 500, seed: int = 0) -> dict[str, pd.DataFrame]:
    """The notebook 25 pair screens (levels and changes, within-country permutation p, BH over all pairs) on the original region codes, on the NUTS 2021 geography, and on the NUTS 2021 geography after spatial filtering of every column
    (SP5 pair table, SP6 counts, SP7 the null of the whole screen: columns shuffled within countries, and independent spatially smooth noise with autocorrelation 0.4 and 0.8, plain against filtered; BH discoveries per screen)."""
    regional = load_regional()
    reg21 = pd.read_parquet(PROCESSED / "regional_nuts2021.parquet")
    orig = _screen_tables(rc.regional_matrix(regional, drop_flag_chars="u"), rc.regional_matrix(regional, drop_flag_chars="ubp"))
    R21, Rb21 = rc.regional_matrix(reg21, drop_flag_chars="u"), rc.regional_matrix(reg21, drop_flag_chars="ubp")
    new = _screen_tables(R21, Rb21)
    nb = SP.Neighbours(sorted(set(R21.index.get_level_values(0)) | set(Rb21.index.get_level_values(0))))
    pairs, counts = [], []
    for name in ("levels", "changes"):
        a, b = rs.screen_perm(orig[name], n_perm), rs.screen_perm(new[name], n_perm)
        XF, diag = nb.filter_table(new[name])
        c = rs.screen_perm(XF, n_perm)
        keep = ["x", "y", "n", "rho", "p", "q_value", "significant"]
        t = b[keep].merge(c[keep], on=["x", "y"], suffixes=("", "_filtered"), how="outer")
        t["same_sign"] = np.sign(t.rho) == np.sign(t.rho_filtered)
        pairs.append(t.assign(screen=name))
        counts.append({"screen": name, "pairs_original": len(a), "significant_original": int(a.significant.sum()), "pairs_nuts2021": len(b), "significant_nuts2021": int(b.significant.sum()),
                       "pairs_filtered": len(c), "significant_filtered": int(c.significant.sum()), "both_nuts2021_and_filtered": int((t.significant & t.significant_filtered).sum()), "sign_changed": int((~t.same_sign.fillna(True)).sum()),
                       "columns_with_positive_autocorrelation": int((diag.moran_z > 1.645).sum()), "columns_after_filter": int((diag.moran_z_filtered > 1.645).sum()), "columns": len(diag)})
    rng = np.random.default_rng(seed)
    rows = []
    for name in ("levels", "changes"):
        X = new[name]
        g = rc.country_of(X.index)
        for k in range(n_sim):
            variants = {"shuffled within countries": corr.shuffle_within(X, g, rng)}
            for rho in (0.4, 0.8):
                S = {}
                for c in X:
                    sr = X[c].dropna()
                    sr = sr[sr.index.isin(nb.pos)]
                    if len(sr) >= 20:
                        S[c] = pd.Series(SP.smooth_noise(sr.to_numpy(), nb.sub(np.array([nb.pos[i] for i in sr.index]))[0], rho, rng), index=sr.index)
                variants[f"smooth noise {rho}"] = rc.country_demean(pd.DataFrame(S).reindex(X.index))
            for label, V in variants.items():
                VF, _ = nb.filter_table(V)
                for pipe, M in (("plain", V), ("filtered", VF)):
                    t = rs.screen_perm(M, n_perm_null, seed=k)
                    rows.append({"screen": name, "null": label, "pipeline": pipe, "sim": k, "discoveries": int(t.significant.sum()), "share_p_below_005": float((t.p < 0.05).mean()), "n_tests": len(t)})
    results = {"sp5_screen_pairs": pd.concat(pairs, ignore_index=True), "sp6_screen_counts": pd.DataFrame(counts), "sp7_screen_null": pd.DataFrame(rows)}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_funds_all(out: Path = OUT, n_perm: int = 20000, n_shuffle: int = 100) -> dict[str, pd.DataFrame]:
    """EU cohesion payments (2007-2013 programme) against the change of regional outcomes between 2005-07 and 2015-19, within countries and beyond the initial income (`analysis.eu_funds`): F1 the descriptive table of the
    support by region, F2 the pre-specified family (two funds measures x five outcomes, within-country permutation p, BH), F3 the same plain and after spatial filtering, F4 variants (all funds with the EAFRD, the pre-2004 member states
    only, no income control, a 2005-09 baseline), F5 Italy's slope against the other countries' (plain and filtered), F6 the shuffled null."""
    reg21 = pd.read_parquet(PROCESSED / "regional_nuts2021.parquet")
    funds = EP.load_period()
    pred = EF.predictors(reg21, funds)
    W = EF.build_table(reg21, funds)
    g = rc.country_of(W.index)
    nb = SP.Neighbours(sorted(set(reg21.geo)))
    fam, _ = SP.compare_family(W, nb, EF.XS, EF.OUTCOMES, n_perm=n_perm)
    plain = rs.pair_family(W, EF.XS, EF.OUTCOMES, g, n_perm=n_perm)
    variants = []
    for name, tab in (("all funds (with the EAFRD)", W), ("pre-2004 member states only", W[W.index.str[:2].isin(EF.EU15)]), ("no income control", EF.build_table(reg21, funds, income_control=False)), ("baseline 2005-09", EF.build_table(reg21, funds, early=(2005, 2009)))):
        xs = ["funds_all_per_head"] if name.startswith("all funds") else EF.XS
        variants.append(rs.pair_family(tab, xs, EF.OUTCOMES, rc.country_of(tab.index), n_perm=n_perm).assign(variant=name))
    desc = pred.assign(country=pred.index.str[:2]).reset_index()
    results = {"f1_funds_by_region": desc, "f2_family": plain, "f3_family_filtered": fam, "f4_variants": pd.concat(variants, ignore_index=True),
               "f5_italy_rank": SP.italy_placebo_filtered(W, nb, EF.XS, EF.OUTCOMES), "f6_null": rs.family_null(W, EF.XS, EF.OUTCOMES, g, n_shuffle=n_shuffle)}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_nuts1_south(out: Path = OUT, n_perm: int = 2000) -> dict[str, pd.DataFrame]:
    """The EU-wide "why the South" family (`regional_south.eu_wide`, EQI against the six regional gaps) with Germany (16 Laender) and Belgium (3 regions) added at NUTS1, where the EQI exists only at NUTS1 (`analysis.nuts1`: outcomes aggregated
    from NUTS2 by sums and population weights). The two countries enter only at NUTS1 (their NUTS2 rows and structural-business predictors are left out), and spatial filtering is not applied (it needs NUTS1 boundaries). N1 compares the family
    without and with them (all waves, plain and beyond income), N2 Italy's rank among the countries without and with them, N3 the within-country rank correlations of Germany and Belgium themselves."""
    reg21, extra21 = pd.read_parquet(PROCESSED / "regional_nuts2021.parquet"), pd.read_parquet(PROCESSED / "regional_extra_nuts2021.parquet")
    de_be = ("DE", "BE")
    R0 = rc.regional_matrix(pd.concat([reg21, extra21], ignore_index=True), drop_flag_chars="u")
    eqi1 = EQI.load(nuts1=de_be)
    eqi1 = eqi1[eqi1.geo.str.len().eq(3)]
    alt = pd.concat([reg21[~reg21.country.isin(de_be)], N1.to_nuts1(reg21, de_be), extra21[~extra21.country.isin(de_be)], eqi1], ignore_index=True)
    R1 = rc.regional_matrix(alt, drop_flag_chars="u")
    key = ["x", "y", "wave", "income_control"]
    rows = []
    for ic in (False, True):
        f0, f1 = rs.eu_wide(R0, income_control=ic, n_perm=n_perm).assign(geography="baseline"), rs.eu_wide(R1, income_control=ic, n_perm=n_perm).assign(geography="with DE and BE at NUTS1")
        rows += [f0, f1]
    f = pd.concat(rows, ignore_index=True)
    cols = ["n", "rho", "q_value", "significant"]
    n1 = f[f.geography == "baseline"][[*key, *cols]].merge(f[f.geography != "baseline"][[*key, *cols, "loo_min_abs_rho", "loo_sign_flips"]], on=key, suffixes=("_baseline", "_with_de_be"), how="outer")
    p0, p1 = rs.italy_placebo(R0).assign(geography="baseline"), rs.italy_placebo(R1).assign(geography="with DE and BE at NUTS1")
    W1 = rs.within_country(rs.wave_table(R1, rs.PRIMARY_WAVE), income_control=True)
    from scipy import stats
    rows = []
    for c in de_be:
        sub = W1[W1.index.str[:2] == c]
        for x in ("eqi", "eqi_quality", "eqi_impartiality", "eqi_corruption"):
            for y in rs.OUTCOMES:
                d = sub[[x, y]].dropna()
                if len(d) >= 3:
                    rows.append({"country": c, "x": x, "y": y, "n": len(d), "rho": float(stats.spearmanr(d[x], d[y])[0])})
    results = {"n1_family_comparison": n1, "n2_italy_rank": pd.concat([p0, p1], ignore_index=True), "n3_de_be_within": pd.DataFrame(rows)}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results
