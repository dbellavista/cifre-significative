"""Regional "why the South" round: two designs side by side, descriptive associations only.

Track A (EU-wide, placebo-capable): indicators that exist for the NUTS2 regions of many countries (EQI governance index, structural business statistics) against the regional gaps
(income, output per hour, hours per inhabitant, employment, unemployment). Regions are demeaned by country, so only within-country differences count; the pre-specified family is the 2017 EQI wave with
2015-19 outcomes (BH false-discovery control over the family), the other waves are replications (sign agreement), and Italy is ranked among the other countries' slopes (`regional_corr.italy_slope_test`).
Track B (Italy only, exploratory): indicators that exist for the 21 Italian regions only (internal migration, public spending by region). With one country there are no placebo countries and the 21 regions
form one cluster (about 8 in the South and Islands), so the null is a permutation of the values across regions, once unrestricted and once within the three zones (North, Centre, South and Islands),
which asks whether an association survives the North-South split. Nothing here is a causal claim and Track B p-values are not corrected for the many indicators screened by eye.
"""
import numpy as np
import pandas as pd

from cifresignificative.analysis import corr
from cifresignificative.analysis import regional_corr as rc
from cifresignificative.analysis.data import italy_area

GEO = rc.GEO
# EQI wave -> window of the outcomes it is compared with (centred on the wave; EQI itself has one value, in the wave year)
WAVES = {2010: (2008, 2012), 2013: (2011, 2015), 2017: (2015, 2019), 2021: (2019, 2023)}
PRIMARY_WAVE = 2017  # matches the 2015-19 window of the earlier regional screens
OUTCOMES = ["gdp_pps_pc_eu27", "gdp_per_hour_eur", "hours_per_inhabitant", "employment_rate_20_64", "employment_rate_20_64_female", "unemployment_rate"]
INCOME_OUTCOMES = {"gdp_pps_pc_eu27", "gdp_per_hour_eur"}  # not purged of income in the "beyond income" variant
EU_WIDE = ["eqi", "eqi_quality", "eqi_impartiality", "eqi_corruption", "sbs_manufacturing_employment_share_pct", "sbs_local_units_per_1000_inhabitants", "sbs_wages_per_employee_keur"]
# pay per employee (SBS) against income or output per hour is near-definitional (both are labour income per unit), so those pairs are left out of every family
MECHANICAL = {frozenset((x, y)) for x in ("sbs_wages_per_employee_keur",) for y in ("gdp_pps_pc_eu27", "gdp_per_hour_eur")}
ZONES = {"North-West": "North", "North-East": "North", "Centre": "Centre", "South": "South and Islands", "Islands": "South and Islands"}
CPT_SECTORS = {"Amministrazione Generale": "general_administration", "Istruzione": "education", "Sanità": "health", "Ricerca e Sviluppo (R. & S.)": "research", "Giustizia": "justice",
               "Viabilità": "roads", "Altri trasporti": "other_transport", "Industria e Artigianato": "industry_crafts", "Lavoro": "labour_policy", "Previdenza e Integrazioni Salariali": "pensions",
               "Ambiente": "environment", "Interventi in campo sociale (assist. e benef.)": "social_assistance"}
CPT_LEVELS = {"Amministrazioni Centrali": "central", "Amministrazioni Regionali": "regional", "Amministrazioni Locali": "local"}  # public enterprises form the remainder
ITALY_ONLY = ["net_migration_per_1000", "inflow_per_1000", "outflow_per_1000", "moves_to_north_per_1000", "spend_total_pc_eur", "spend_capital_pc_eur", "capital_share_pct",
              *(f"spend_{v}_pc_eur" for v in CPT_SECTORS.values()), *(f"spend_{v}_level_pc_eur" for v in CPT_LEVELS.values())]


def zone_of(geo: str) -> str:
    """North / Centre / South and Islands for an Italian NUTS2 code."""
    return ZONES.get(italy_area(geo), "other")


def wave_table(R: pd.DataFrame, wave: int) -> pd.DataFrame:
    """Region x indicator window means for one EQI wave (`R` from `regional_corr.regional_matrix`, which carries the EQI and SBS columns)."""
    return corr.cross_section(R, "level", WAVES[wave], entity_level=GEO)


def within_country(X: pd.DataFrame, income_control: bool) -> pd.DataFrame:
    """Country-demeaned table; with `income_control` every non-income column is also purged of its fit on log income per head (what is left beyond income)."""
    W = rc.country_demean(X)
    if not income_control:
        return W
    inc = rc.country_demean(X[[rc.CONTROL]]) if rc.CONTROL in X else None
    if inc is None:
        return W
    keep = [c for c in W.columns if c not in INCOME_OUTCOMES and c != rc.CONTROL]
    resid = corr.residualise(pd.concat([W[keep], inc], axis=1), rc.CONTROL)
    return pd.concat([W[[c for c in W.columns if c in INCOME_OUTCOMES]], resid], axis=1)


def perm_p(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Two-sided permutation p of the pooled Spearman rho when x is shuffled across the rows of each group (country). The asymptotic p is too liberal for country-demeaned data:
    countries differ in dispersion, which widens the null of the pooled rho (a shuffled test rejected 9% of the time at the nominal 5%). Ranks are pooled, so the null permutes the pooled ranks within groups."""
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return np.nan
    null = np.empty((n_perm, len(rx)))
    for g in np.unique(groups):
        m = groups == g
        null[:, m] = rng.permuted(np.tile(rx[m], (n_perm, 1)), axis=1)
    rho_null = null @ ry / denom
    return float((1 + (np.abs(rho_null) >= abs(rx @ ry / denom) - 1e-12).sum()) / (1 + n_perm))


def screen_perm(X: pd.DataFrame, n_perm: int = 20000, seed: int = 0, n_min: int = 80, q: float = 0.05, exclusions=rc.MECHANICAL) -> pd.DataFrame:
    """The pair screen of `regional_corr.screen_regions` (size-scaled columns dropped, same pairs) with the within-country permutation p next to the asymptotic p, and BH over all pairs for both
    (`q_asymptotic`, `q_value`, `significant`, `significant_asymptotic`). The funnel's duplicate and robustness steps are not repeated: this checks the p-values behind the survivors."""
    X = X.drop(columns=corr.size_scaled(X.columns))
    t = corr.pair_screen(X, "spearman", n_min, exclusions=exclusions).rename(columns={"p": "p_asymptotic"})
    g, rng = rc.country_of(X.index), np.random.default_rng(seed)
    t["p"] = [perm_p(d[x].to_numpy(), d[y].to_numpy(), g.reindex(d.index).to_numpy(), n_perm, rng) for x, y in zip(t.x, t.y) for d in [X[[x, y]].dropna()]]
    a = corr.fdr(t, q, "p_asymptotic")
    out = corr.fdr(t, q).assign(q_asymptotic=a.q_value, significant_asymptotic=a.significant)
    return out.sort_values("p", ignore_index=True)


def pair_family(X: pd.DataFrame, xs: list[str], ys: list[str], groups: pd.Series | None = None, n_min: int = 20, n_perm: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Spearman rho of every (x, y) of the pre-specified family on the rows of `X`, with BH q over the family. With `groups` (e.g. countries) also leave-one-group-out robustness (smallest |rho|
    and the number of groups whose removal flips the sign); with `groups` and `n_perm` the p-value is the within-group permutation p (`perm_p`) instead of the asymptotic one."""
    rng = np.random.default_rng(seed)
    rows = []
    for x in xs:
        for y in ys:
            if x not in X or y not in X or x == y or frozenset((x, y)) in MECHANICAL:
                continue
            d = X[[x, y]].dropna()
            if len(d) < n_min:
                continue
            rho, p = corr._corr(d[x].to_numpy(), d[y].to_numpy(), "spearman")
            if np.isnan(rho):
                continue
            row = {"x": x, "y": y, "n": len(d), "rho": rho, "p": p}
            if groups is not None:
                g = groups.reindex(d.index).to_numpy()
                row["n_groups"] = int(pd.Series(g).nunique())
                loo = [corr._corr(d[x].to_numpy()[g != c], d[y].to_numpy()[g != c], "spearman")[0] for c in np.unique(g) if (g != c).sum() >= n_min]
                row["loo_min_abs_rho"] = float(np.nanmin(np.abs(loo))) if loo else np.nan
                row["loo_sign_flips"] = int(sum(np.sign(r) != np.sign(rho) for r in loo if np.isfinite(r)))
                if n_perm:
                    row["p_asymptotic"], row["p"] = p, perm_p(d[x].to_numpy(), d[y].to_numpy(), g, n_perm, rng)
            rows.append(row)
    out = pd.DataFrame(rows)
    return corr.fdr(out) if len(out) else out


def family_null(X: pd.DataFrame, xs: list[str], ys: list[str], groups: pd.Series, n_shuffle: int = 100, n_perm: int = 1000, seed: int = 0, n_min: int = 20) -> pd.DataFrame:
    """Calibration: shuffle each column's values across the rows of its group, rerun the family with the same permutation test, and return per shuffle the number of BH discoveries
    and the share of tests with p < 0.05 (should be about 0 and about 0.05)."""
    rng = np.random.default_rng(seed)
    cols = [c for c in dict.fromkeys([*xs, *ys]) if c in X]
    rows = []
    for k in range(n_shuffle):
        S = corr.shuffle_within(X[cols], groups, rng)
        t = pair_family(S, xs, ys, groups, n_min, n_perm=n_perm, seed=k)
        rows.append({"shuffle": k, "discoveries": int(t.significant.sum()), "share_p_below_005": float((t.p < 0.05).mean()), "n_tests": len(t)})
    return pd.DataFrame(rows)


def eu_wide(R: pd.DataFrame, waves=WAVES, xs=EU_WIDE, ys=OUTCOMES, income_control: bool = False, n_perm: int | None = 2000) -> pd.DataFrame:
    """Track A: the family for every EQI wave (rows: wave, x, y, rho, p, q within the wave, leave-one-country-out). Primary wave = `PRIMARY_WAVE`; the others replicate."""
    frames = []
    for w in waves:
        W = within_country(wave_table(R, w), income_control)
        t = pair_family(W, xs, ys, rc.country_of(W.index), n_perm=n_perm)
        if len(t):
            frames.append(t.assign(wave=w, primary=(w == PRIMARY_WAVE), income_control=income_control))
    out = pd.concat(frames, ignore_index=True)
    prim = out[out.wave.eq(PRIMARY_WAVE)].set_index(["x", "y"]).rho
    out["same_sign_as_primary"] = [np.sign(r) == np.sign(prim.get((x, y), np.nan)) for x, y, r in zip(out.x, out.y, out.rho)]
    return out


def italy_placebo(R: pd.DataFrame, wave: int = PRIMARY_WAVE, xs=EU_WIDE, ys=OUTCOMES) -> pd.DataFrame:
    """Italy's within-country slope against the other countries' (`regional_corr.italy_slope_test`: rank of Italy's slope difference among the countries with at least 5 regions)."""
    Xw = within_country(wave_table(R, wave), income_control=False)
    rows = [rc.italy_slope_test(Xw, x, y) for x in xs for y in ys if x in Xw and y in Xw and frozenset((x, y)) not in MECHANICAL]
    return pd.DataFrame([r for r in rows if r]).assign(wave=wave)


def italy_table(R: pd.DataFrame, flows: pd.DataFrame, spending: pd.DataFrame, window: tuple[int, int] = WAVES[PRIMARY_WAVE]) -> pd.DataFrame:
    """The 21 Italian regions x (outcomes, EU-wide indicators, Italy-only indicators), window means.
    Migration: moves in/out/to-the-North per 1000 inhabitants of the year (all citizenships and sexes; moves within the region excluded). Spending (CPT, extended public sector, nominal EUR):
    per inhabitant, by account, sector and level of government."""
    pop = R["population"].dropna()
    pop = pop[pop.index.get_level_values(GEO).str.startswith("IT")]
    X = corr.cross_section(R[R.index.get_level_values(GEO).str.startswith("IT")], "level", window, entity_level=GEO)
    f = flows[flows.sex.eq("T") & flows.citizenship.eq("total") & flows.origin.ne(flows.destination) & flows.year.between(*window)]
    north = f.destination.map(lambda g: zone_of(g) == "North")
    parts = {"inflow_per_1000": f.groupby(["destination", "year"]).value.sum(), "outflow_per_1000": f.groupby(["origin", "year"]).value.sum(), "moves_to_north_per_1000": f[north].groupby(["origin", "year"]).value.sum()}
    mig = {}
    for name, s in parts.items():
        s = s.rename_axis([GEO, "year"])
        mig[name] = (s / pop.reindex(s.index) * 1000).groupby(level=GEO).mean()
    mig = pd.DataFrame(mig)
    mig["net_migration_per_1000"] = mig.inflow_per_1000 - mig.outflow_per_1000
    s = spending[spending.year.between(*window)]
    popw = pop[pop.index.get_level_values("year").isin(range(window[0], window[1] + 1))]

    def per_capita(sub: pd.DataFrame) -> pd.Series:
        v = sub.groupby(["geo", "year"]).value.sum().rename_axis([GEO, "year"])
        return (v * 1e6 / popw.reindex(v.index)).groupby(level=GEO).mean()  # EUR million -> EUR per inhabitant of the year
    sp = {"spend_total_pc_eur": per_capita(s[s.account.eq("total")]), "spend_capital_pc_eur": per_capita(s[s.account.eq("capital")])}
    sp["capital_share_pct"] = sp["spend_capital_pc_eur"] / sp["spend_total_pc_eur"] * 100
    tot = s[s.account.eq("total")]
    for name, short in CPT_SECTORS.items():
        sp[f"spend_{short}_pc_eur"] = per_capita(tot[tot.sector.eq(name)])
    for name, short in CPT_LEVELS.items():
        sp[f"spend_{short}_level_pc_eur"] = per_capita(tot[tot.level.eq(name)])
    return X.join(mig).join(pd.DataFrame(sp))


def zone_gaps(T: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """South-and-Islands minus North mean of each column, in standard deviations across the 21 regions (descriptive: which indicators differ most between the two ends)."""
    z = pd.Series([zone_of(g) for g in T.index], index=T.index)
    rows = []
    for c in cols:
        if c in T and T[c].notna().sum() >= 10 and T[c].std() > 0:
            rows.append({"indicator": c, "north_mean": T.loc[z.eq("North"), c].mean(), "south_mean": T.loc[z.eq("South and Islands"), c].mean(), "gap_sd": (T.loc[z.eq("South and Islands"), c].mean() - T.loc[z.eq("North"), c].mean()) / T[c].std()})
    return pd.DataFrame(rows)


def italy_exploratory(T: pd.DataFrame, xs: list[str], ys: list[str] = OUTCOMES, n_perm: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Track B: Spearman rho across the Italian regions, raw and within the three zones (values demeaned by zone), with permutation p-values (values shuffled across all regions; within zones for the
    zone-demeaned version, which keeps the North-South split fixed), and the range of rho when one region is left out. `q_*` is BH over the family; treat it as exploratory (one cluster, few regions per zone)."""
    zone = pd.Series([zone_of(g) for g in T.index], index=T.index)
    Z = T - T.groupby(zone.to_numpy()).transform("mean")
    rng = np.random.default_rng(seed)
    rows = []
    for x in xs:
        for y in ys:
            if x not in T or y not in T or x == y or frozenset((x, y)) in MECHANICAL:
                continue
            d = T[[x, y]].dropna()
            if len(d) < 15:
                continue
            a, b = d[x].to_numpy(), d[y].to_numpy()
            rho, p_asym = corr._corr(a, b, "spearman")
            dz = Z.loc[d.index, [x, y]]
            rho_z = corr._corr(dz[x].to_numpy(), dz[y].to_numpy(), "spearman")[0]
            zg = zone.loc[d.index].to_numpy()
            null_raw = np.array([corr._corr(rng.permutation(a), b, "spearman")[0] for _ in range(n_perm)])
            null_z = np.empty(n_perm)
            for k in range(n_perm):
                xa = a.copy()
                for g in np.unique(zg):
                    m = zg == g
                    xa[m] = rng.permutation(a[m])
                xz = pd.Series(xa).groupby(zg).transform("mean")
                null_z[k] = corr._corr(xa - xz.to_numpy(), dz[y].to_numpy(), "spearman")[0]
            loo = [corr._corr(np.delete(a, i), np.delete(b, i), "spearman")[0] for i in range(len(d))]
            rows.append({"x": x, "y": y, "n": len(d), "rho": rho, "p_asymptotic": p_asym, "p_perm": float((np.abs(null_raw) >= abs(rho) - 1e-12).mean()), "rho_within_zone": rho_z,
                         "p_perm_within_zone": float((np.abs(null_z) >= abs(rho_z) - 1e-12).mean()), "loo_min_rho": float(np.nanmin(loo)), "loo_max_rho": float(np.nanmax(loo)),
                         "loo_sign_flips": int(sum(np.sign(r) != np.sign(rho) for r in loo))})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_perm"] = corr.fdr(out, p_col="p_perm").q_value.to_numpy()
        out["q_perm_within_zone"] = corr.fdr(out, p_col="p_perm_within_zone").q_value.to_numpy()
    return out


def compare_tracks(placebo: pd.DataFrame, italy: pd.DataFrame) -> pd.DataFrame:
    """One row per (x, y) of the EU-wide family with both answers: Track A (Italy's rho, other countries' pooled slope and Italy's rank among countries) and Track B
    (the same rho with its permutation p-values across Italian regions, raw and beyond the North-South split)."""
    a = placebo.rename(columns={"slope_italy": "slope_italy_A", "slope_others": "slope_others_A", "p_diff": "p_rank_among_countries", "n_italy": "n_italy_A"})
    keep = ["x", "y", "n_italy_A", "n_countries", "slope_italy_A", "slope_others_A", "slope_diff_italy", "p_rank_among_countries"]
    b = italy[["x", "y", "rho", "p_perm", "rho_within_zone", "p_perm_within_zone", "loo_sign_flips"]].rename(columns={c: f"{c}_B" for c in ("rho", "p_perm", "rho_within_zone", "p_perm_within_zone", "loo_sign_flips")})
    return a[keep].merge(b, on=["x", "y"], how="inner")
