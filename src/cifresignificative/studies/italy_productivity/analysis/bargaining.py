"""Wage-setting institutions and wage pass-through: is the country's pass-through of inflation to wages (notebook 26, `hf.passthrough_by_country`, 2005Q1-2021Q2) related to collective bargaining
coverage or trade union density (OECD/AIAS, `sources/oecd_bargaining.py`; country means over 2005-2019)? About 20 countries: one cross-section, two pre-specified institution variables, Spearman with a
permutation p (the institution shuffled across countries), Benjamini-Hochberg over the two, robustness without the two countries with negative estimates and with 2010-2019 means, leave-one-country-out, a shuffled null,
and Italy by placebo ranking (Italy's coverage is recorded as 100% every year, a top-coded point). Descriptive only: institutions and pass-through can both follow a third factor."""
import numpy as np
import pandas as pd

from cifresignificative.studies.italy_productivity.analysis import corr
from cifresignificative.studies.italy_productivity.analysis.data import ITALY

VARIABLES = ("cbc", "tud")


def institution_means(cbc: pd.DataFrame, tud: pd.DataFrame, years=(2005, 2019), min_years: int = 3) -> pd.DataFrame:
    """iso3 -> mean coverage (`cbc`) and union density (`tud`) over the years of `years` with a value, for countries with at least `min_years` observations of that variable (coverage is observed in few years for some countries: 2-4 values)."""
    out = {}
    for name, d in zip(VARIABLES, (cbc, tud), strict=True):
        g = d[d.year.between(*years)].groupby("iso3").value
        out[name] = g.mean()[g.count() >= min_years]
    return pd.DataFrame(out)


def _perm_p(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    null = rng.permuted(np.tile(rx, (n_perm, 1)), axis=1) @ ry / denom
    return float((1 + (np.abs(null) >= abs(rx @ ry / denom) - 1e-12).sum()) / (1 + n_perm))


def association(pt: pd.DataFrame, inst: pd.DataFrame, n_perm: int = 20000, seed: int = 0, q: float = 0.05, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    """One row per institution variable: n, Spearman rho, asymptotic and permutation p, leave-one-country-out minimum |rho| and sign flips; BH q over the rows. `pt` has `iso3`, `pass_through`."""
    rng = np.random.default_rng(seed)
    d = pt.set_index("iso3").pass_through.rename("pt").to_frame().join(inst, how="inner").drop(index=list(exclude), errors="ignore")
    rows = []
    for v in VARIABLES:
        x = d[[v, "pt"]].dropna()
        rho, p = corr._corr(x[v].to_numpy(), x.pt.to_numpy(), "spearman")
        loo = [corr._corr(x[v].drop(c).to_numpy(), x.pt.drop(c).to_numpy(), "spearman")[0] for c in x.index]
        rows.append({"variable": v, "n": len(x), "rho": rho, "p_asymptotic": p, "p": _perm_p(x[v].to_numpy(), x.pt.to_numpy(), n_perm, rng), "loo_min_abs_rho": float(np.nanmin(np.abs(loo))),
                     "loo_sign_flips": int(sum(np.sign(r) != np.sign(rho) for r in loo))})
    return corr.fdr(pd.DataFrame(rows), q)


def italy_position(pt: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """Italy's value, the median of the other countries and Italy's rank (1 = highest) for pass-through and each institution, among the countries with both pass-through and institutions."""
    d = pt.set_index("iso3").pass_through.rename("pass_through").to_frame().join(inst, how="inner")
    rows = []
    for c in ("pass_through", *VARIABLES):
        s = d[c].dropna()
        if ITALY in s.index:
            rows.append({"variable": c, "n": len(s), "italy": s[ITALY], "median_others": s.drop(ITALY).median(), "rank_1_highest": int((s > s[ITALY]).sum() + 1)})
    return pd.DataFrame(rows)


def shuffled_null(pt: pd.DataFrame, inst: pd.DataFrame, n_rep: int = 500, n_perm: int = 500, seed: int = 0) -> dict:
    """Calibration: shuffle the pass-through across countries and rerun both tests: discoveries per repetition about 0, share of p < 0.05 about 5%."""
    rng = np.random.default_rng(seed)
    d = pt.set_index("iso3").pass_through
    disc, small = [], []
    for k in range(n_rep):
        t = pd.DataFrame({"iso3": d.index, "pass_through": rng.permutation(d.to_numpy())})
        r = association(t, inst, n_perm=n_perm, seed=k)
        disc.append(int(r.significant.sum())); small.append(float((r.p < 0.05).mean()))
    return {"n_rep": n_rep, "mean_discoveries": float(np.mean(disc)), "share_reps_with_discovery": float(np.mean(np.array(disc) > 0)), "mean_share_p_below_005": float(np.mean(small))}
