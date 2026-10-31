"""Correlation discovery: screen indicator pairs, control the false-discovery rate, stamp each survivor with robustness checks.

Associations only, never causal. With ~176 indicators there are ~15k pairs and about 30 countries, so every screen is
followed by Benjamini-Hochberg correction and leave-one-out / without-Ireland / rank-vs-linear checks.
"""
from collections.abc import Set as AbstractSet
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from cifresignificative.studies.italy_productivity.analysis.data import (
    BENCHMARKS,
    EU27_ISO3,
    ITALY,
)

COUNTRIES = [*EU27_ISO3, *BENCHMARKS]
BAD_FLAGS = ("b", "p", "backcast", "linked")  # series breaks, provisional values, back-casts, spliced approximations
IRELAND = "IRL"  # GDP-type series distorted by multinationals


def panel_matrix(panel: pd.DataFrame, drop_flags=BAD_FLAGS, years: tuple[int, int] | None = None, countries=None) -> pd.DataFrame:
    """(iso3, year) x indicator table; observations carrying a flag in `drop_flags` are dropped, aggregates excluded."""
    d = panel[panel.iso3.isin(countries or COUNTRIES)]
    if drop_flags:
        d = d[~d.flag.isin(drop_flags)]
    if years:
        d = d[d.year.between(*years)]
    return d.pivot_table(index=["iso3", "year"], columns="indicator", values="value")


def cross_section(M: pd.DataFrame, kind: str = "level", window: tuple[int, int] = (2015, 2019), window0: tuple[int, int] | None = None,
                  entity_level: str = "iso3") -> pd.DataFrame:
    """entity x indicator table (entity = iso3, or geo for regions). 'level' = window mean; 'change' = window mean minus `window0` mean (needs window0)."""
    def mean(w):
        return M[M.index.get_level_values("year").isin(range(w[0], w[1] + 1))].groupby(level=entity_level).mean()
    if kind == "level":
        return mean(window)
    if kind == "change":
        if window0 is None:
            raise ValueError("kind='change' needs window0")
        return mean(window) - mean(window0)
    raise ValueError(kind)


def residualise(X: pd.DataFrame, control: str | list[str], log: bool = False) -> pd.DataFrame:
    """Each column minus its least-squares fit on `control` (fitted on the rows where both exist); control columns are dropped.

    Cross-country data share a 'development' factor (rich countries are better on almost everything), so a raw screen mostly
    rediscovers it; correlating the residuals asks what moves together beyond income.
    """
    ctrl = [control] if isinstance(control, str) else list(control)
    Z = X[ctrl].apply(np.log) if log else X[ctrl]
    out = {}
    for c in X.columns.difference(ctrl):
        d = pd.concat([X[c], Z], axis=1).dropna()
        if len(d) < len(ctrl) + 3:
            continue
        A = np.column_stack([np.ones(len(d)), d[ctrl].to_numpy()])
        beta, *_ = np.linalg.lstsq(A, d[c].to_numpy(), rcond=None)
        out[c] = pd.Series(d[c].to_numpy() - A @ beta, index=d.index)
    return pd.DataFrame(out).reindex(X.index)


def within_changes(M: pd.DataFrame, entity_level: str = "iso3") -> pd.DataFrame:
    """(iso3, year) x indicator year-on-year differences, demeaned by country and by year (two-way), so common shocks such as
    2009 or 2020 and country trends are removed and only country-specific co-movement is left."""
    d = M.groupby(level=entity_level).diff()
    d = d - d.groupby(level=entity_level).transform("mean")
    return d - d.groupby(level="year").transform("mean")


SIZE_UNITS = ("thousand", "million", "persons", "permits", "number")


def size_scaled(columns, unit_of: dict[str, str] | None = None) -> list[str]:
    """Indicators in absolute counts or amounts (unit from the glossary, or a _ths/_nr/_meur suffix, or population):
    they all scale with country size, so they correlate with each other for that reason alone."""
    if unit_of is None:
        from cifresignificative.core import glossary
        g = glossary.load()
        unit_of = g.unit.to_dict()
    return [c for c in columns if c in ("population", "nuts2_population") or c.endswith(("_ths", "_nr", "_meur", "_mio_eur"))
            or any(u in str(unit_of.get(c, "")).lower() for u in SIZE_UNITS)]


def _stem(name: str) -> str:
    for p in ("wb_", "imf_", "ameco_", "oecd_", "wgi_", "proc_", "piaac_", "nuts2_"):
        if name.startswith(p):
            return name[len(p):]
    return name


KNOWN_DUPLICATES = {frozenset(p) for p in [  # same or definitionally linked quantity under different names
    ("gdp_real_pc", "wb_gdp_pc_ppp_const"), ("gdp_real_pc", "gdp_pps_pc_eu27"), ("gdp_pps_pc_eu27", "wb_gdp_pc_ppp_const"),
    ("gov_debt_pct_gdp", "imf_gov_debt_pct_gdp"), ("gov_debt_pct_gdp", "wb_gov_debt_pct_gdp"), ("imf_gov_debt_pct_gdp", "wb_gov_debt_pct_gdp"),
    ("gov_balance_pct_gdp", "imf_gov_net_lending_pct_gdp"), ("gdp_growth", "wb_gdp_growth"), ("hicp_inflation", "wb_inflation_cpi"),
    ("unemployment_rate", "imf_unemployment_rate"), ("unemployment_rate", "wb_unemployment_ilo"), ("imf_unemployment_rate", "wb_unemployment_ilo"),
    ("current_account_pct_gdp", "imf_current_account_pct_gdp"), ("pension_spend_pct_gdp", "oecd_pension_exp_pct_gdp"),
    ("labour_productivity_pps_eu27", "ameco_tfp_index"), ("homicide_rate_police", "homicide_victims_per_100k"),
    ("homicide_rate_police", "wb_homicide_rate"), ("homicide_victims_per_100k", "wb_homicide_rate"), ("gdp_pps_pc_eu27", "labour_productivity_pps_eu27"),
    ("students_per_100_pop_20_24", "wb_tertiary_enrol_gross"), ("net_earnings_pps", "gross_earnings_pps"), ("gini", "wb_gini"),
]}


def is_trivial_pair(x: str, y: str, exclusions: AbstractSet[frozenset] = frozenset()) -> bool:
    """Same quantity from another source (`wb_x` vs `x`), a name that extends the other (`arope` / `arope_old`), the same two-token family (`employment_rate_55_64` / `employment_rate_60_64`), or curated."""
    if frozenset((x, y)) in exclusions or frozenset((x, y)) in KNOWN_DUPLICATES:
        return True
    sx, sy = _stem(x), _stem(y)
    same_family = sx.split("_")[:2] == sy.split("_")[:2] and len(sx.split("_")) > 1 and len(sy.split("_")) > 1  # employment_rate_55_64 / employment_rate_60_64
    return sx == sy or sx.startswith(sy + "_") or sy.startswith(sx + "_") or same_family


def _corr(a: np.ndarray, b: np.ndarray, method: str) -> tuple[float, float]:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan, np.nan
    r, p = (stats.spearmanr if method == "spearman" else stats.pearsonr)(a, b)
    return float(r), float(p)


def _fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return np.nan, np.nan
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def pair_screen(X: pd.DataFrame, method: str = "spearman", n_min: int = 20, skip_trivial: bool = True, exclusions: AbstractSet[frozenset] = frozenset()) -> pd.DataFrame:
    """All indicator pairs of X (rows = observations): tidy (x, y, n, rho, p, ci_lo, ci_hi), pairs with fewer than n_min complete rows dropped.

    For 'spearman' the ranks are taken on each pair's complete rows. CIs use the Fisher transform (approximate for Spearman).
    """
    arr = {c: X[c].to_numpy(dtype=float) for c in X.columns}
    rows = []
    for x, y in combinations(X.columns, 2):
        if skip_trivial and is_trivial_pair(x, y, exclusions):
            continue
        ok = ~(np.isnan(arr[x]) | np.isnan(arr[y]))
        n = int(ok.sum())
        if n < n_min:
            continue
        r, p = _corr(arr[x][ok], arr[y][ok], method)
        if np.isnan(r):
            continue
        lo, hi = _fisher_ci(r, n)
        rows.append((x, y, n, r, p, lo, hi))
    return pd.DataFrame(rows, columns=["x", "y", "n", "rho", "p", "ci_lo", "ci_hi"])


def fdr(df: pd.DataFrame, q: float = 0.05, p_col: str = "p") -> pd.DataFrame:
    """Benjamini-Hochberg: adds `q_value` (monotone adjusted p) and `significant` (q_value <= q)."""
    out = df.copy()
    m = len(out)
    if m == 0:
        out["q_value"], out["significant"] = [], []
        return out
    order = np.argsort(out[p_col].to_numpy())
    p_sorted = out[p_col].to_numpy()[order]
    adj = np.minimum.accumulate((p_sorted * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1)
    qv = np.empty(m)
    qv[order] = adj
    out["q_value"] = qv
    out["significant"] = out.q_value <= q
    return out


def robustness(pairs: pd.DataFrame, X: pd.DataFrame, method: str = "spearman", n_min: int = 20, strength: float = 0.5,
               groups: pd.Series | None = None, irl_label: str = IRELAND) -> pd.DataFrame:
    """Adds checks per pair (X is iso3 x indicator for cross-sections; for regions pass `groups`, a country label per row, so
    leave-one-out drops whole countries and `irl_label` is the Irish label, e.g. "IE"):
    `loo_min_abs` / `loo_sign_stable` (drop each country in turn), `rho_ex_irl`, `rho_other` (the other of Spearman/Pearson),
    and `robust` = every leave-one-out rho keeps the sign and at least `strength` of |rho|, |rho| without Ireland is
    at least `strength` of |rho|, and the other method agrees in sign.
    """
    other = "pearson" if method == "spearman" else "spearman"
    res = []
    for x, y in zip(pairs.x, pairs.y):
        d = X[[x, y]].dropna()
        r0, _ = _corr(d[x].to_numpy(), d[y].to_numpy(), method)
        g = d.index.to_series() if groups is None else groups.reindex(d.index)
        loo = []
        for c in g.unique():
            s = d[(g != c).to_numpy()]
            loo.append(_corr(s[x].to_numpy(), s[y].to_numpy(), method)[0] if len(s) >= n_min - 1 else np.nan)
        loo = np.array(loo)
        s = d[(g != irl_label).to_numpy()]
        r_irl = _corr(s[x].to_numpy(), s[y].to_numpy(), method)[0] if len(s) >= n_min - 1 else np.nan
        r_oth = _corr(d[x].to_numpy(), d[y].to_numpy(), other)[0]
        stable = bool(np.all(np.sign(loo[~np.isnan(loo)]) == np.sign(r0)))
        loo_min = np.nanmin(np.abs(loo)) if np.isfinite(loo).any() else np.nan
        res.append({"loo_min_abs": loo_min, "loo_sign_stable": stable,
                    "rho_ex_irl": r_irl, "rho_other": r_oth,
                    "robust": stable and loo_min >= strength * abs(r0) and abs(r_irl) >= strength * abs(r0) and np.sign(r_oth) == np.sign(r0)})
    return pd.concat([pairs.reset_index(drop=True), pd.DataFrame(res)], axis=1)


DUPLICATE_RHO = 0.95  # |rho| above this between differently named series is almost surely the same quantity (other source, other base year)


def screen(X: pd.DataFrame, method: str = "spearman", q: float = 0.05, n_min: int = 20, check_robust: bool = True,
           drop_size_scaled: bool = True, dup_rho: float | None = DUPLICATE_RHO, groups: pd.Series | None = None, irl_label: str = IRELAND,
           exclusions: AbstractSet[frozenset] = frozenset()) -> pd.DataFrame:
    """Funnel in one call: [drop size-scaled columns] -> pair screen -> BH-FDR (over the tested pairs) -> [drop near-duplicates]
    -> robustness stamps on the survivors, ranked by |rho|."""
    if drop_size_scaled:
        X = X.drop(columns=size_scaled(X.columns))
    pairs = fdr(pair_screen(X, method, n_min, exclusions=exclusions), q)
    sig = pairs[pairs.significant]
    if dup_rho is not None:
        sig = sig[sig.rho.abs() < dup_rho]
    if check_robust and len(sig):
        sig = robustness(sig, X, method, n_min, groups=groups, irl_label=irl_label)
    return sig.assign(abs_rho=sig.rho.abs()).sort_values("abs_rho", ascending=False).drop(columns="abs_rho").reset_index(drop=True)


SPLIT_WINDOWS = ((2005, 2014), (2015, 2024))


def split_sample(M: pd.DataFrame, windows=SPLIT_WINDOWS, control: str | None = None, method: str = "spearman", q: float = 0.05, n_min: int = 20,
                 exclusions: AbstractSet[frozenset] = frozenset()) -> pd.DataFrame:
    """Split-sample replication of the cross-section level screen: the same funnel (size-scaled columns dropped, optional logged `control`
    removed, pair screen, BH over the tested pairs) is run separately on the window means of each half of the years. One row per pair
    significant in at least one half: rho / q_value / n per half (suffix _a, _b), `replicated` = significant in both with the same sign.
    This discriminates far more than the leave-one-out stamps, since a pair carried by one period (a crisis) fails the other half."""
    halves = []
    for w in windows:
        X = cross_section(M, "level", w)
        X = X.drop(columns=size_scaled(X.columns))
        if control is not None:
            X = residualise(X, control, log=True)
        halves.append(fdr(pair_screen(X, method, n_min, exclusions=exclusions), q).set_index(["x", "y"]))
    a, b = (h[["n", "rho", "q_value", "significant"]] for h in halves)
    out = a.join(b, how="outer", lsuffix="_a", rsuffix="_b").reset_index()
    sig = out[["significant_a", "significant_b"]].fillna(False).astype(bool)
    out["replicated"] = sig.all(axis=1) & (np.sign(out.rho_a) == np.sign(out.rho_b))
    out = out[sig.any(axis=1) & ~(out[["rho_a", "rho_b"]].abs() >= DUPLICATE_RHO).any(axis=1)].drop(columns=["significant_a", "significant_b"])  # near-duplicates dropped as in `screen`
    return out.assign(min_abs_rho=out[["rho_a", "rho_b"]].abs().min(axis=1)).sort_values(["replicated", "min_abs_rho"], ascending=False).reset_index(drop=True)


def split_sample_null(M: pd.DataFrame, n_perm: int = 5, seed: int = 0, **kw) -> list[int]:
    """Number of 'replicated' pairs after independently shuffling each indicator within each half's cross-section; should be ~0."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        S = M.copy()
        for w in kw.get("windows", SPLIT_WINDOWS):
            rows = np.flatnonzero(M.index.get_level_values("year").isin(range(w[0], w[1] + 1)))
            for j in range(M.shape[1]):
                S.iloc[rows, j] = rng.permutation(M.iloc[rows, j].to_numpy())
        out.append(int(split_sample(S, **kw).replicated.sum()))
    return out


def shuffle_within(X: pd.DataFrame, groups: pd.Series | None, rng: np.random.Generator) -> pd.DataFrame:
    """Independently permute each column's values across all rows, or across the rows of each group when `groups` is given."""
    if groups is None:
        return X.apply(lambda col: pd.Series(rng.permutation(col.to_numpy()), index=col.index))
    gi = groups.reindex(X.index).to_numpy()
    S = X.copy()
    for g in np.unique(gi):
        rows = np.flatnonzero(gi == g)
        for j in range(X.shape[1]):
            S.iloc[rows, j] = rng.permutation(X.iloc[rows, j].to_numpy())
    return S


def permutation_null(X: pd.DataFrame, method: str = "spearman", q: float = 0.05, n_min: int = 20, n_perm: int = 5, seed: int = 0,
                     groups: pd.Series | None = None) -> list[int]:
    """Number of FDR 'discoveries' after independently shuffling each indicator's values across countries (or, with `groups`,
    across the regions of each country, which keeps the country structure); should be ~0."""
    rng = np.random.default_rng(seed)
    return [int(fdr(pair_screen(shuffle_within(X, groups, rng), method, n_min), q).significant.sum()) for _ in range(n_perm)]


# --- Italy-specific experiments (E5, E6) and panel models (E4, E7) -------------------------------------------------


def italy_gap_series(panel: pd.DataFrame, indicator: str) -> pd.Series:
    """Year series: Italy minus the median of the other EU27 members (see `data.eu_median_ex_italy` for the coverage rule)."""
    from cifresignificative.studies.italy_productivity.analysis.data import (
        eu_median_ex_italy,
        wide,
    )
    w = wide(panel, indicator, EU27_ISO3)
    if ITALY not in w:
        return pd.Series(dtype=float)
    return (w[ITALY] - eu_median_ex_italy(w)).dropna()


def italy_gap_corr(panel: pd.DataFrame, target: str, indicators=None, n_min: int = 12, q: float = 0.05) -> pd.DataFrame:
    """E5: which indicators' Italy-vs-EU gaps move with the gap in `target`? Spearman on year-on-year changes of the gaps
    (gap levels trend, so correlating them would be spurious), BH-FDR over the indicators tested."""
    gaps = {i: italy_gap_series(panel, i) for i in (indicators or sorted(panel.indicator.unique()))}
    t = gaps[target].diff() if target in gaps else italy_gap_series(panel, target).diff()
    rows = []
    for i, g in gaps.items():
        if i == target or is_trivial_pair(i, target):
            continue
        d = pd.concat([t, g.diff()], axis=1, keys=["t", "g"]).dropna()
        if len(d) < n_min:
            continue
        r, p = _corr(d.t.to_numpy(), d.g.to_numpy(), "spearman")
        if not np.isnan(r):
            rows.append((target, i, len(d), r, p))
    out = fdr(pd.DataFrame(rows, columns=["target", "indicator", "n", "rho", "p"]), q)
    return out.reindex(out.rho.abs().sort_values(ascending=False).index).reset_index(drop=True)


def italy_fit(X: pd.DataFrame, x: str, y: str) -> dict:
    """E6: fit y = a + b*x on every country except Italy (iso3 x indicator table); Italy's observed vs predicted y and the residual
    in units of the other countries' residual SD (|z| > 2 = Italy sits off the EU line)."""
    d = X[[x, y]].dropna()
    if ITALY not in d.index:
        return {}
    rest = d.drop(ITALY)
    b, a = np.polyfit(rest[x], rest[y], 1)
    resid = rest[y] - (a + b * rest[x])
    pred = a + b * d.at[ITALY, x]
    return {"x": x, "y": y, "n": len(rest), "slope": b, "r_ex_italy": float(rest[x].corr(rest[y], method="spearman")),
            "italy_y": d.at[ITALY, y], "italy_pred": pred, "resid_z": float((d.at[ITALY, y] - pred) / resid.std(ddof=2))}


def panel_fe(M: pd.DataFrame, y: str, xs: list[str], lag: int = 0, exclude: tuple[str, ...] = (), entity_level: str = "iso3",
             cluster_country: bool = False) -> dict:
    """Entity + year fixed effects, entity-clustered SEs (linearmodels). `lag` shifts the regressors back `lag` years within entity.
    For regional data pass entity_level="geo" and cluster_country=True (SEs clustered by the country prefix of the region code)."""
    from linearmodels.panel import PanelOLS
    D = M[[y, *xs]].drop(index=list(exclude), level=entity_level, errors="ignore")
    if lag:
        D = pd.concat([D[y], D[xs].groupby(level=entity_level).shift(lag)], axis=1)
    D = D.dropna()
    if len(D) < 50 or D.index.get_level_values(entity_level).nunique() < 10:
        return {}
    model = PanelOLS(D[y], D[xs], entity_effects=True, time_effects=True, drop_absorbed=True)
    if cluster_country:
        clusters = pd.Series(pd.factorize(D.index.get_level_values(entity_level).str[:2])[0], index=D.index)
        res = model.fit(cov_type="clustered", clusters=clusters.to_frame())
    else:
        res = model.fit(cov_type="clustered", cluster_entity=True)
    return {"n_obs": int(res.nobs), "n_countries": int(res.entity_info["total"]), "coef": res.params.to_dict(), "p": res.pvalues.to_dict(),
            "r2_within": float(res.rsquared_within)}


def lead_lag(M: pd.DataFrame, x: str, y: str, lags=(1, 2, 3), **fe_kwargs) -> pd.DataFrame:
    """E4: FE coefficient of x(t-k) on y(t) against y(t-k) on x(t). Predictive precedence only, never causation."""
    rows = []
    for k in lags:
        for direction, (dep, ind) in {"x_leads_y": (y, x), "y_leads_x": (x, y)}.items():
            r = panel_fe(M, dep, [ind], lag=k, **fe_kwargs)
            if r:
                rows.append({"x": x, "y": y, "lag": k, "direction": direction, "coef": r["coef"][ind], "p": r["p"][ind], "n_obs": r["n_obs"]})
    return pd.DataFrame(rows)
