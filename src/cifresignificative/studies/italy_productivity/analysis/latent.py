"""Latent structure of the national indicators: PCA on the country x indicator cross-section, with a parallel-analysis null, bootstrap stability and Italy's position over time.

Replaces ranked pair lists by a few dimensions. Components are descriptive summaries of co-variation (a 'development' axis is expected first), not causes.
Italy's position is read as a rank among countries (a placebo ranking), never tested as a cluster. Axes are fitted once on the base window and later windows are projected onto them, so scores are comparable over time.
"""
import numpy as np
import pandas as pd

from cifresignificative.studies.italy_productivity.analysis import corr
from cifresignificative.studies.italy_productivity.analysis.data import EU27_ISO3, ITALY

BASE = (2015, 2019)
WINDOWS = {"2005-09": (2005, 2009), "2010-14": (2010, 2014), "2015-19": (2015, 2019), "2020-24": (2020, 2024)}


def prepare(M: pd.DataFrame, window: tuple[int, int] = BASE, min_cov: float = 0.8, min_country_cov: float = 0.7, countries=EU27_ISO3,
            dup_rho: float = corr.DUPLICATE_RHO) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Country x indicator window means for `countries`: size-scaled and sparse columns dropped (coverage < `min_cov` of countries), then countries with
    fewer than `min_country_cov` of the kept indicators dropped, then near-duplicates removed greedily (|Spearman| >= `dup_rho` or a trivial same-family pair with an already kept column).
    Returns (raw table, column means, column SDs); `standardise` applies them, winsorising at +-3 SD and setting missing values to 0 (the mean)."""
    X = corr.cross_section(M, "level", window)
    X = X[X.index.isin(countries)].drop(columns=corr.size_scaled(X.columns))
    X = X.loc[:, X.notna().mean() >= min_cov]
    X = X[X.notna().mean(axis=1) >= min_country_cov].dropna(axis=1, how="all")
    R = X.rank().corr().abs()
    keep: list[str] = []
    for c in X.columns:
        if not any(R.loc[c, k] >= dup_rho or corr.is_trivial_pair(c, k) for k in keep):
            keep.append(c)
    X = X[keep]
    return X, X.mean(), X.std(ddof=0).replace(0, np.nan)


def standardise(X: pd.DataFrame, mu: pd.Series, sd: pd.Series, winsor: float = 3.0) -> pd.DataFrame:
    return ((X[mu.index] - mu) / sd).clip(-winsor, winsor).fillna(0.0)


def pca(Z: pd.DataFrame, k: int | None = None) -> dict:
    """SVD of the standardised table: eigenvalues (of the correlation matrix, sum = number of columns), explained share, loadings (columns x components), scores (rows x components)."""
    _, s, Vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    ev = s**2 / len(Z)
    k = k or len(ev)
    comp = [f"PC{i + 1}" for i in range(k)]
    V = pd.DataFrame(Vt[:k].T, index=Z.columns, columns=comp)
    for c in comp:  # sign convention: the largest absolute loading is positive
        if V[c].iloc[V[c].abs().argmax()] < 0:
            V[c] = -V[c]
    return {"eigenvalues": ev, "explained": ev / ev.sum(), "loadings": V, "scores": pd.DataFrame(Z.to_numpy() @ V.to_numpy(), index=Z.index, columns=comp)}


def parallel_analysis(Z: pd.DataFrame, n_perm: int = 200, pct: float = 95.0, seed: int = 0) -> dict:
    """Horn's parallel analysis: eigenvalues of the real table against the `pct` percentile of eigenvalues after independently shuffling every column; retain components above the null."""
    rng = np.random.default_rng(seed)
    A = Z.to_numpy()
    real = np.linalg.svd(A, compute_uv=False) ** 2 / len(A)
    null = np.array([np.linalg.svd(np.column_stack([rng.permutation(A[:, j]) for j in range(A.shape[1])]), compute_uv=False) ** 2 / len(A) for _ in range(n_perm)])
    thr = np.percentile(null, pct, axis=0)
    above = real > thr
    return {"real": real, "threshold": thr, "n_retain": int(np.argmin(above)) if not above.all() else len(above), "null_first_share": float((null[:, 0] / null.sum(axis=1)).mean())}


def bootstrap_stability(Z: pd.DataFrame, k: int, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Resample countries with replacement; for each of the first `k` full-sample components report the median and 5th percentile of the best absolute cosine
    between its loading vector and the bootstrap loadings (any of the first k bootstrap components, so order swaps do not count as instability)."""
    rng = np.random.default_rng(seed)
    V = pca(Z, k)["loadings"].to_numpy()
    best = []
    for _ in range(n_boot):
        B = Z.iloc[rng.integers(0, len(Z), len(Z))]
        B = B.loc[:, B.std() > 0]
        Vb = pca(B - B.mean(), k)["loadings"].reindex(Z.columns).fillna(0).to_numpy()
        best.append(np.abs(V.T @ Vb / (np.linalg.norm(V, axis=0)[:, None] * np.linalg.norm(Vb, axis=0)[None, :])).max(axis=1))
    best = np.array(best)
    return pd.DataFrame({"component": [f"PC{i + 1}" for i in range(k)], "median_cosine": np.median(best, axis=0), "p05_cosine": np.percentile(best, 5, axis=0)})


def split_half_congruence(M: pd.DataFrame, k: int, halves=((2005, 2014), (2015, 2024))) -> pd.DataFrame:
    """Loadings fitted separately on the two halves' cross-sections (same columns); best absolute cosine of each first-half component with the second-half components."""
    (Xa, mua, sda), (Xb, mub, sdb) = (prepare(M, w) for w in halves)
    cols = Xa.columns.intersection(Xb.columns)
    Va = pca(standardise(Xa, mua[cols], sda[cols]), k)["loadings"].to_numpy()
    Vb = pca(standardise(Xb, mub[cols], sdb[cols]), k)["loadings"].to_numpy()
    C = np.abs(Va.T @ Vb) / (np.linalg.norm(Va, axis=0)[:, None] * np.linalg.norm(Vb, axis=0)[None, :])
    return pd.DataFrame({"component": [f"PC{i + 1}" for i in range(k)], "best_cosine": C.max(axis=1), "same_order": C.argmax(axis=1) == np.arange(k), "n_indicators": len(cols)})


def italy_over_time(M: pd.DataFrame, V: pd.DataFrame, mu: pd.Series, sd: pd.Series, windows: dict[str, tuple[int, int]] = WINDOWS) -> pd.DataFrame:
    """Project each window's country table onto the base-window axes `V` (same means and SDs); per window and component: Italy's score, the median of the other countries, Italy's rank (1 = highest) and the number of countries."""
    rows = []
    for name, w in windows.items():
        X = corr.cross_section(M, "level", w)
        X = X[X.index.isin(EU27_ISO3)].reindex(columns=mu.index)
        X = X[X.notna().mean(axis=1) >= 0.5]
        if ITALY not in X.index:
            continue
        S = standardise(X, mu, sd) @ V
        for c in V.columns:
            s = S[c]
            rows.append({"window": name, "component": c, "italy": float(s[ITALY]), "median_others": float(s.drop(ITALY).median()), "rank_from_top": int((s > s[ITALY]).sum() + 1), "n_countries": len(s),
                         "coverage_italy": float(X.loc[ITALY].notna().mean())})
    return pd.DataFrame(rows)
