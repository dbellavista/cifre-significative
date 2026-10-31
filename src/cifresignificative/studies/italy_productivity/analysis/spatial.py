"""Spatial dependence of the regional screens: neighbour structure from the NUTS 2021 boundaries, Moran's I and eigenvector spatial filtering (Griffith).

The regional tests (`regional_corr`, `regional_south`) demean by country and permute within countries, which absorbs country clustering but not the similarity of neighbouring regions: two spatially smooth
variables can correlate by chance more often than the permutation null allows. Here the neighbours are queen contiguous regions of the same country (islands are linked to their nearest same-country region), and each
column is purged of the spatial patterns (eigenvectors of the centred weights matrix) that carry its autocorrelation, chosen greedily until Moran's I of the remainder is no longer clearly positive.
`spatial_null` asks the calibration question directly: with simulated spatially smooth x, how often does the plain within-country permutation test reject, and how often the filtered one. Descriptive, never causal.
"""
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

from cifresignificative.core.clean import nuts
from cifresignificative.studies.italy_productivity.analysis import regional_corr as rc
from cifresignificative.studies.italy_productivity.analysis import regional_south as rs

TOL_M = 2000  # generalised (20M) boundaries leave small gaps between neighbours: regions closer than this count as touching
MIN_EIG_RATIO = 0.25  # candidate eigenvectors: eigenvalue at least this share of the largest (positive spatial autocorrelation)
K_MAX = 20
Z_STOP = 1.645  # stop adding eigenvectors when the standardised Moran's I of the remainder falls below this


def contiguity(geos: list[str], geometries: dict | None = None, tol: float = TOL_M) -> np.ndarray:
    """Symmetric boolean matrix: queen contiguity (within `tol` metres) between regions of the same country; a region with no such neighbour (an island) is linked to the nearest same-country region by centroid distance."""
    geometries = nuts.regions(nuts.TARGET) if geometries is None else geometries
    g = [geometries[c] for c in geos]
    country = np.array([c[:2] for c in geos])
    tree = STRtree(g)
    A = np.zeros((len(geos), len(geos)), bool)
    for i, gi in enumerate(g):
        for j in tree.query(gi.buffer(tol)):
            if j != i and country[i] == country[j] and gi.distance(g[j]) <= tol:
                A[i, j] = A[j, i] = True
    cen = np.array([[x.centroid.x, x.centroid.y] for x in g])
    for i in np.flatnonzero(~A.any(1)):
        same = np.flatnonzero((country == country[i]) & (np.arange(len(geos)) != i))
        if len(same):
            j = same[np.argmin(np.hypot(*(cen[same] - cen[i]).T))]
            A[i, j] = A[j, i] = True
    return A


def row_standardise(A: np.ndarray) -> np.ndarray:
    s = A.sum(1, keepdims=True)
    return np.divide(A, s, out=np.zeros(A.shape), where=s > 0)


def _moments(W: np.ndarray) -> tuple[float, float, float]:
    S0 = W.sum()
    return S0, 0.5 * ((W + W.T) ** 2).sum(), ((W.sum(0) + W.sum(1)) ** 2).sum()


def morans_i(x: np.ndarray, W: np.ndarray) -> tuple[float, float]:
    """Moran's I of `x` for the row-standardised matrix `W` and its z-score under normality (E = -1/(n-1))."""
    n = len(x)
    z = x - x.mean()
    S0, S1, S2 = _moments(W)
    I = n / S0 * (z @ W @ z) / (z @ z)
    e = -1 / (n - 1)
    var = (n * n * S1 - n * S2 + 3 * S0 * S0) / (S0 * S0 * (n * n - 1)) - e * e
    return float(I), float((I - e) / np.sqrt(var))


def morans_perm(x: np.ndarray, W: np.ndarray, groups: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    """Moran's I and its permutation p (one-sided, positive autocorrelation) with `x` shuffled within the groups (countries), the null the regional tests use."""
    I = morans_i(x, W)[0]
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    null = np.empty(n_perm)
    for k in range(n_perm):
        y = x.copy()
        for ix in idx:
            y[ix] = x[rng.permutation(ix)]
        null[k] = morans_i(y, W)[0]
    return I, float((1 + (null >= I).sum()) / (n_perm + 1))


def candidate_vectors(W: np.ndarray, groups: np.ndarray, min_ratio: float = MIN_EIG_RATIO) -> np.ndarray:
    """Eigenvectors of M A M (A the symmetrised weights, M removing country means) with eigenvalue >= min_ratio x the largest: the smooth spatial patterns inside countries, one per column, orthogonal to the country dummies."""
    D = (groups[:, None] == np.unique(groups)[None, :]).astype(float)
    M = np.eye(len(groups)) - D @ np.linalg.pinv(D)
    val, vec = np.linalg.eigh(M @ ((W + W.T) / 2) @ M)
    keep = val >= min_ratio * val.max()
    return vec[:, keep][:, np.argsort(-val[keep])]


def filter_column(x: np.ndarray, W: np.ndarray, E: np.ndarray, z_stop: float = Z_STOP, k_max: int = K_MAX) -> tuple[np.ndarray, list[int]]:
    """Greedy eigenvector filter: add the candidate that lowers Moran's I of the residual most, until its z-score is below `z_stop` or `k_max` vectors are used. Returns the residual and the chosen columns of `E`."""
    chosen: list[int] = []
    e = x - x.mean()
    while len(chosen) < min(k_max, E.shape[1]) and morans_i(e, W)[1] > z_stop:
        best, best_i, best_r = -1, np.inf, e
        for j in sorted(set(range(E.shape[1])) - set(chosen)):
            Z = E[:, [*chosen, j]]
            r = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
            i = morans_i(r, W)[0]
            if i < best_i:
                best, best_i, best_r = j, i, r
        chosen.append(best)
        e = best_r
    return e, chosen


class Neighbours:
    """Weights for a set of regions (index of 2021 codes); `filter_table` handles missing values per column by dropping those regions from the weights."""

    def __init__(self, geos: list[str], geometries: dict | None = None, tol: float = TOL_M):
        self.geos = list(geos)
        self.pos = {g: i for i, g in enumerate(self.geos)}
        self.A = contiguity(self.geos, geometries, tol)
        self._cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def sub(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row-standardised weights and candidate eigenvectors of the regions in `idx` (cached by region set)."""
        key = idx.tobytes()
        if key not in self._cache:
            W = row_standardise(self.A[np.ix_(idx, idx)])
            self._cache[key] = (W, candidate_vectors(W, np.array([self.geos[i][:2] for i in idx])))
        W, E = self._cache[key]
        return W, E, np.array([self.geos[i][:2] for i in idx])

    def filter_table(self, X: pd.DataFrame, min_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Spatially filtered copy of a country-demeaned region x indicator table, and per column: n, Moran's I and z before and after, number of eigenvectors used."""
        out, rows = X.copy() * np.nan, []
        for c in X:
            s = X[c].dropna()
            s = s[s.index.isin(self.pos)]
            if len(s) < min_n:
                continue
            idx = np.array([self.pos[g] for g in s.index])
            W, E, _ = self.sub(idx)
            if E.shape[1] == 0 or not W.any():
                out.loc[s.index, c] = s
                continue
            r, ch = filter_column(s.to_numpy(), W, E)
            (i0, z0), (i1, z1) = morans_i(s.to_numpy(), W), morans_i(r, W)
            out.loc[s.index, c] = r
            rows.append({"indicator": c, "n": len(s), "moran_i": i0, "moran_z": z0, "moran_i_filtered": i1, "moran_z_filtered": z1, "k_vectors": len(ch)})
        return out, pd.DataFrame(rows)


def smooth_noise(n_like: np.ndarray, W: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Spatially autocorrelated noise (I - rho W)^-1 e with e standard normal: x independent of everything else but with neighbours alike."""
    return np.linalg.solve(np.eye(len(n_like)) - rho * W, rng.standard_normal(len(n_like)))


def spatial_null(X: pd.DataFrame, nb: Neighbours, xs: list[str], ys: list[str], n_sim: int = 100, n_perm: int = 500, rho: float = 0.8, seed: int = 0, n_min: int = 20) -> pd.DataFrame:
    """Calibration: replace every x by independent spatially smooth noise (autocorrelation `rho`, then demeaned by country), keep the real y, and run the family with the within-country permutation p, plain and after
    spatial filtering of both sides. Returns per simulation the mean Moran's I of the simulated x, the share of p < 0.05 and the number of BH discoveries for each pipeline (should be about 0.05 and 0 if the pipeline is calibrated)."""
    rng = np.random.default_rng(seed)
    Y = X[[c for c in ys if c in X]]
    Yf, _ = nb.filter_table(Y)
    rows = []
    for k in range(n_sim):
        S = {}
        for x in [c for c in xs if c in X]:
            s = X[x].dropna()
            s = s[s.index.isin(nb.pos)]
            W = nb.sub(np.array([nb.pos[g] for g in s.index]))[0]
            S[x] = pd.Series(rc.country_demean(pd.DataFrame({x: smooth_noise(s.to_numpy(), W, rho, rng)}, index=s.index))[x], index=s.index)
        Xs = pd.DataFrame(S)
        Xsf, dsim = nb.filter_table(rc.country_demean(Xs))
        for name, (A, B) in {"plain": (Xs, Y), "filtered": (Xsf, Yf)}.items():
            t = rs.pair_family(pd.concat([A, B], axis=1), list(A), list(B), rc.country_of(pd.concat([A, B], axis=1).index), n_min, n_perm=n_perm, seed=k)
            rows.append({"sim": k, "pipeline": name, "rho": rho, "moran_i_x": float(dsim.moran_i.mean()), "share_p_below_005": float((t.p < 0.05).mean()), "discoveries": int(t.significant.sum()), "n_tests": len(t)})
    return pd.DataFrame(rows)


def compare_family(X: pd.DataFrame, nb: Neighbours, xs: list[str], ys: list[str], n_perm: int = 2000, seed: int = 0, n_min: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The pre-specified family on a country-demeaned table, plain and after spatial filtering of every column (same pairs, within-country permutation p, BH over the family each way).
    Returns one row per pair with both answers (`rho`, `p`, `q_value`, `significant` and the `_filtered` versions, `same_sign`) and the per-column Moran table."""
    XF, diag = nb.filter_table(X)
    cols = [c for c in dict.fromkeys([*xs, *ys]) if c in X]
    plain = rs.pair_family(X[cols], xs, ys, rc.country_of(X.index), n_min, n_perm=n_perm, seed=seed)
    filt = rs.pair_family(XF[cols], xs, ys, rc.country_of(XF.index), n_min, n_perm=n_perm, seed=seed)
    keep = ["x", "y", "n", "rho", "p", "q_value", "significant"]
    out = plain[keep].merge(filt[keep], on=["x", "y"], suffixes=("", "_filtered"), how="outer")
    out["same_sign"] = np.sign(out.rho) == np.sign(out.rho_filtered)
    return out, diag


def italy_placebo_filtered(X: pd.DataFrame, nb: Neighbours, xs: list[str], ys: list[str]) -> pd.DataFrame:
    """Italy's within-country slope against the other countries' (`regional_corr.italy_slope_test`), plain and on spatially filtered columns, side by side."""
    XF, _ = nb.filter_table(X)
    rows = []
    for x in xs:
        for y in ys:
            if x in X and y in X and frozenset((x, y)) not in rs.MECHANICAL:
                a, b = rc.italy_slope_test(X, x, y), rc.italy_slope_test(XF, x, y)
                if a and b:
                    rows.append({"x": x, "y": y, "slope_diff": a["slope_diff_italy"], "p_rank": a["p_diff"], "slope_diff_filtered": b["slope_diff_italy"], "p_rank_filtered": b["p_diff"], "n_countries": a["n_countries"]})
    return pd.DataFrame(rows)
