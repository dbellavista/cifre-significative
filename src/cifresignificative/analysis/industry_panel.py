"""Industry panel (country x ~55 market industries), Rajan-Zingales style: do industries that rely more on ICT or intangible capital grow faster in some countries than in others?

Outcome: log growth per year of real GVA per hour in `sectors64` (2-digit industries). Industry intensity: the contribution of ICT (or intangible) capital to labour
productivity growth in the EU KLEMS section the industry belongs to, taken as the median over the other countries in 1996-2004 (Italy excluded, so the
intensity is a benchmark that does not use Italy's own outcome and predates the outcome window). Per country: slope of growth on intensity (country intercept, SEs clustered by section);
Italy is judged by its rank among the countries (placebo ranking), never by a clustered test: Italy is one cluster. Descriptive associations, not causal.
"""
import numpy as np
import pandas as pd

from cifresignificative.analysis.data import EU27_ISO3, ITALY, load_sectors64

SECTION = {"A": "A", "B": "B", "C": "C", "D": "D-E", "E": "D-E", "F": "F", "G": "G", "H": "H", "I": "I", "J": "J", "K": "K", "M": "M-N", "N": "M-N", "R": "R-S", "S": "R-S"}
INTENSITIES = {"ict": "LP1ConTangICT", "intangible": "LP1ConIntang"}
WINDOWS = {"full": ((2005, 2007), (2017, 2019)), "early": ((2005, 2007), (2010, 2012)), "late": ((2012, 2014), (2017, 2019))}  # (start window, end window) of annual means


def section_of(nace: str) -> str:
    return SECTION[nace[0]]


def intensity(ek: pd.DataFrame, var: str, years: tuple[int, int] = (1996, 2004), exclude: str = ITALY) -> pd.Series:
    """KLEMS section -> median over countries (EU27 members and GBR, `exclude` left out) of the mean contribution `var` in `years`."""
    d = ek[(ek.variable == var) & ek.year.between(*years) & ~ek.iso3.isin([exclude, "EU27", "USA", "JPN"])]
    return d.groupby(["nace_r2_code", "iso3"]).value.mean().groupby("nace_r2_code").median()


def industry_growth(s: pd.DataFrame, start: tuple[int, int], end: tuple[int, int]) -> pd.DataFrame:
    """iso3, nace_r2 -> `g`: log change of mean real productivity between the two windows, per year; EU27 aggregate dropped, outliers winsorised at the 1st/99th percentile."""
    s = s[s.iso3.isin([*EU27_ISO3, "GBR"]) & (s["prod"] > 0)]
    mean = lambda w: s[s.year.between(*w)].groupby(["iso3", "nace_r2"])["prod"].mean()
    mid = lambda w: (w[0] + w[1]) / 2
    g = (np.log(mean(end)) - np.log(mean(start))).dropna() / (mid(end) - mid(start))
    return g.clip(*g.quantile([0.01, 0.99])).rename("g").reset_index()


def _slope(g: np.ndarray, x: np.ndarray, cl: np.ndarray) -> tuple[float, float]:
    """OLS slope of g on x with intercept and its cluster-robust (CR1) standard error; clusters = KLEMS sections, since the intensity only varies between sections."""
    X = np.column_stack([np.ones(len(x)), x])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ g
    e = g - X @ b
    meat = sum(np.outer(sc, sc) for sc in (X[cl == k].T @ e[cl == k] for k in np.unique(cl)))
    G, n = len(np.unique(cl)), len(g)
    V = XtXi @ meat @ XtXi * (G / (G - 1)) * ((n - 1) / (n - 2))
    return float(b[1]), float(np.sqrt(V[1, 1]))


def _perm_p(g: np.ndarray, secs: np.ndarray, xs: pd.Series, b: float, n_perm: int, rng: np.random.Generator) -> float:
    """Two-sided randomisation p of the slope: the intensity is permuted across the sections (the unit at which it varies), which keeps the within-section dependence."""
    keys = np.array(sorted(xs.index)); vals = xs.loc[keys].to_numpy()
    idx = np.searchsorted(keys, secs)
    gc = g - g.mean()
    def slope(v):
        x = v[idx]; xc = x - x.mean()
        return (xc @ gc) / (xc @ xc) if xc @ xc > 0 else 0.0
    sims = np.array([slope(rng.permutation(vals)) for _ in range(n_perm)])
    return float(((np.abs(sims) >= abs(b) - 1e-12).sum() + 1) / (n_perm + 1))


def country_slopes(G: pd.DataFrame, x: pd.Series, min_n: int = 20, n_perm: int = 0, seed: int = 0) -> pd.DataFrame:
    """Per country: slope of industry growth `g` on intensity `x` (indexed by KLEMS section; sections without a value are dropped), cluster-robust `se`,
    and with `n_perm` > 0 the section-permutation p `p_perm` (the valid test here: with ~14 clusters the clustered t over-rejects)."""
    rng = np.random.default_rng(seed)
    d = G.assign(x=G.nace_r2.map(lambda n: x.get(section_of(n), np.nan))).dropna(subset=["g", "x"])
    rows = []
    for c, P in d.groupby("iso3"):
        if len(P) >= min_n and P.x.nunique() > 2:
            b, se = _slope(P.g.to_numpy(), P.x.to_numpy(), P.nace_r2.map(section_of).to_numpy())
            secs = P.nace_r2.map(section_of).to_numpy()
            rows.append({"iso3": c, "n_industries": len(P), "slope": b, "se": se, **({"p_perm": _perm_p(P.g.to_numpy(), secs, x[x.index.isin(secs)], b, n_perm, rng)} if n_perm else {})})
    return pd.DataFrame(rows)


def italy_rank(S: pd.DataFrame, country: str = ITALY) -> dict:
    """Placebo ranking: Italy's slope among the countries' slopes (1 = largest), the median of the others and the share of other countries with a slope at most as large as Italy's (one-sided empirical p of 'Italy is low')."""
    oth = S[S.iso3 != country].slope
    it = S.loc[S.iso3 == country, "slope"]
    if it.empty:
        return {}
    v = float(it.iloc[0])
    return {"italy_slope": v, "median_others": float(oth.median()), "rank_from_top": int((S.slope > v).sum() + 1), "n_countries": len(S), "p_low": float((((oth <= v).sum()) + 1) / (len(oth) + 1))}


def run(ek: pd.DataFrame, s: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Slopes per country for each intensity and window, plus Italy's placebo rank and a leave-one-section-out check on Italy's slope."""
    s = load_sectors64() if s is None else s
    slopes, ranks, loo = [], [], []
    for wname, (w0, w1) in WINDOWS.items():
        G = industry_growth(s, w0, w1)
        for iname, var in INTENSITIES.items():
            x = intensity(ek, var)
            S = country_slopes(G, x, n_perm=499).assign(window=wname, intensity=iname)
            slopes.append(S)
            ranks.append({"window": wname, "intensity": iname, **italy_rank(S)})
            if wname == "full":
                for sec in sorted(x.index.intersection(list({section_of(n) for n in G.nace_r2}))):
                    Gs = G[G.nace_r2.map(section_of) != sec]
                    Ss = country_slopes(Gs, x)
                    loo.append({"intensity": iname, "dropped_section": sec, **italy_rank(Ss)})
    return {"slopes": pd.concat(slopes, ignore_index=True), "italy_rank": pd.DataFrame(ranks), "italy_loo_section": pd.DataFrame(loo)}


def placebo_null(ek: pd.DataFrame, s: pd.DataFrame | None = None, n_rep: int = 40, n_perm: int = 99, seed: int = 0, intensity_name: str = "ict") -> dict:
    """Calibration: replace the intensity by a random permutation across sections, then recompute every country's `p_perm`; the share below 0.05 should be about 5%."""
    s = load_sectors64() if s is None else s
    G = industry_growth(s, *WINDOWS["full"])
    x = intensity(ek, INTENSITIES[intensity_name])
    rng = np.random.default_rng(seed)
    share = []
    for r in range(n_rep):
        S = country_slopes(G, pd.Series(rng.permutation(x.to_numpy()), index=x.index), n_perm=n_perm, seed=r)
        share.append(float((S.p_perm < 0.05).mean()))
    return {"mean_share_p_below_005": float(np.mean(share)), "n_rep": n_rep}
