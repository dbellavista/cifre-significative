"""High-frequency (monthly / quarterly) dynamics: stationarity checks, prewhitened cross-correlation, Granger and panel lead-lag,
a spread model, placebo event studies and wage pass-through.

Levels of yields, indices and prices trend, so every comparison is made on a transformed (approximately stationary) series;
autocorrelation is handled by prewhitening, HAC or country-clustered errors, and events are judged against a placebo
distribution built from the same series. Associations and predictive order only, never causal.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cifresignificative.analysis import corr
from cifresignificative.analysis.data import ITALY, PROCESSED

HF_PATH = PROCESSED / "highfreq.parquet"
FREQ_ALIAS = {"M": "ME", "Q": "QE"}
TRANSFORMS = {  # indicator -> transformation applied before any correlation
    "m_unemployment_rate": "diff", "m_youth_unemployment_rate": "diff",
    "m_hicp_inflation": "diff", "m_hicp_energy_inflation": "diff", "m_hicp_core_inflation": "diff",
    "m_industrial_production": "dlog", "m_retail_volume": "dlog", "m_long_term_yield": "diff",
    "m_esi": "level", "m_consumer_confidence": "diff", "m_industry_confidence": "level", "m_services_confidence": "level",
    "q_gdp_growth": "level", "q_employment_ths": "dlog", "q_hours_ths": "dlog", "q_labour_cost_yoy": "diff",
}
REGIMES = {"full sample": ("2005-01-01", "2026-12-31"), "2005-2011": ("2005-01-01", "2011-12-31"),
           "2012-2019": ("2012-01-01", "2019-12-31"), "2020 onwards": ("2020-01-01", "2026-12-31")}
MIN_OBS = 60  # a monthly series needs five years to enter a time-series test


def load_highfreq() -> pd.DataFrame:
    return pd.read_parquet(HF_PATH)


def wide_hf(hf: pd.DataFrame, indicator: str, drop_flag_chars: str = "u") -> pd.DataFrame:
    """date x iso3 table on the complete period grid (gaps are NaN so lags and differences are exact)."""
    d = hf[hf.indicator == indicator]
    if drop_flag_chars:
        d = d[~d.flag.fillna("").map(lambda f: any(c in f for c in drop_flag_chars))]
    w = d.pivot(index="date", columns="iso3", values="value").sort_index()
    return w.reindex(pd.date_range(w.index.min(), w.index.max(), freq=FREQ_ALIAS[d.freq.iloc[0]]))


def transform(w: pd.DataFrame | pd.Series, how: str):
    if how == "level":
        return w
    if how == "diff":
        return w.diff()
    if how == "dlog":
        return 100 * w.transform(np.log).diff()
    raise ValueError(how)


def transformed(hf: pd.DataFrame, indicator: str, how: str | None = None) -> pd.DataFrame:
    return transform(wide_hf(hf, indicator), how or TRANSFORMS[indicator])


def hf_panel(hf: pd.DataFrame, indicators: list[str], how: dict[str, str] | None = None, countries: list[str] | None = None) -> pd.DataFrame:
    """(iso3, date) x indicator panel of transformed series (all indicators must share one frequency); aggregates excluded."""
    cols = {}
    for i in indicators:
        w = transformed(hf, i, (how or {}).get(i))
        w = w.drop(columns=[c for c in w.columns if c == "EU27"], errors="ignore")
        if countries:
            w = w[[c for c in w.columns if c in countries]]
        cols[i] = w.rename_axis(index="date", columns="iso3").unstack()  # keeps NaN rows, so row-based lags never jump over a gap
    return pd.DataFrame(cols).sort_index()


# --- stationarity ----------------------------------------------------------------------------------------------------


def stationarity_report(hf: pd.DataFrame, items: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) per (indicator, transformation, country); `stationary` = ADF p < 0.05 and KPSS p > 0.05."""
    from statsmodels.tsa.stattools import adfuller, kpss
    items = items or list(TRANSFORMS.items())
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ind, how in items:
            w = transformed(hf, ind, how)
            for c in w.columns:
                s = w[c].dropna()
                if len(s) < MIN_OBS or s.std() == 0:
                    continue
                rows.append({"indicator": ind, "transform": how, "iso3": c, "n": len(s), "adf_p": adfuller(s, autolag="AIC")[1], "kpss_p": kpss(s, regression="c", nlags="auto")[1]})
    out = pd.DataFrame(rows)
    out["stationary"] = (out.adf_p < 0.05) & (out.kpss_p > 0.05)
    return out


def stationarity_summary(rep: pd.DataFrame) -> pd.DataFrame:
    return rep.groupby(["indicator", "transform"]).agg(countries=("iso3", "nunique"), share_stationary=("stationary", "mean"), median_adf_p=("adf_p", "median"),
                                                       median_kpss_p=("kpss_p", "median")).reset_index()


# --- time-series lead-lag --------------------------------------------------------------------------------------------


def _prewhiten(x: np.ndarray, y: np.ndarray, max_ar: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Box-Jenkins prewhitening: fit an AR(p) (p by BIC) to x and apply the same filter to both series."""
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order
    sel = ar_select_order(x, maxlag=max_ar, ic="bic", trend="c")
    p = max(len(sel.ar_lags or []), 0)
    if p == 0:
        return x, y
    phi = AutoReg(x, lags=p, trend="c").fit().params[1:]
    filt = lambda v: v[p:] - sum(phi[k] * v[p - 1 - k:len(v) - 1 - k] for k in range(p))
    return filt(x), filt(y)


def cross_corr(x: pd.Series, y: pd.Series, max_lag: int = 12, prewhiten: bool = True) -> pd.DataFrame:
    """Correlation of x(t-k) with y(t) for k = -max_lag..max_lag (k > 0: x leads y), on prewhitened series, with 95% white-noise bands."""
    d = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    if len(d) < MIN_OBS:
        return pd.DataFrame(columns=["lag", "r", "n", "band"])
    xv, yv = (_prewhiten(d.x.to_numpy(), d.y.to_numpy()) if prewhiten else (d.x.to_numpy(), d.y.to_numpy()))
    a, b = pd.Series(xv), pd.Series(yv)
    rows = []
    for k in range(-max_lag, max_lag + 1):
        z = pd.concat([a.shift(k), b], axis=1).dropna()
        if len(z) >= 30:
            rows.append({"lag": k, "r": float(z.iloc[:, 0].corr(z.iloc[:, 1])), "n": len(z), "band": 1.96 / np.sqrt(len(z))})
    return pd.DataFrame(rows)


def best_lead(cc: pd.DataFrame, min_lag: int = 0) -> dict:
    """Lag (>= min_lag, x leading) with the largest |r| and whether it clears the 95% band."""
    c = cc[cc.lag >= min_lag]
    if c.empty:
        return {}
    r = c.loc[c.r.abs().idxmax()]
    return {"lag": int(r.lag), "r": float(r.r), "n": int(r.n), "significant": bool(abs(r.r) > r.band)}


def granger(x: pd.Series, y: pd.Series, max_lag: int = 6) -> dict:
    """Granger tests in both directions at the VAR lag chosen by BIC; p-values from the F test of the restricted vs full regression."""
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import grangercausalitytests
    d = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    if len(d) < MIN_OBS:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lag = max(1, VAR(d).select_order(maxlags=max_lag).selected_orders["bic"])
        px = grangercausalitytests(d[["y", "x"]], [lag])[lag][0]["ssr_ftest"][1]  # does x help predict y?
        py = grangercausalitytests(d[["x", "y"]], [lag])[lag][0]["ssr_ftest"][1]  # does y help predict x?
    return {"lag": lag, "p_x_leads_y": float(px), "p_y_leads_x": float(py), "n": len(d)}


def verdict(p_xy: float, p_yx: float, alpha: float = 0.05) -> str:
    return {(True, False): "x leads", (False, True): "y leads", (True, True): "both"}.get((p_xy < alpha, p_yx < alpha), "neither")


def _granger_panel(M: pd.DataFrame, dep: str, ind: str, k: int, own_lags: int) -> dict:
    """FE regression of dep(t) on ind(t-k) and dep(t-1..t-own_lags) (country + date effects, country-clustered SEs): a panel Granger test.
    Without the own lags a persistent dependent series would 'be predicted' by anything that is itself persistent."""
    D = M[[dep, ind]].copy()
    g = D.groupby(level="iso3")
    D["_x"] = g[ind].shift(k)
    for j in range(1, own_lags + 1):
        D[f"_own{j}"] = g[dep].shift(j)
    xs = ["_x", *[f"_own{j}" for j in range(1, own_lags + 1)]]
    r = corr.panel_fe(D.drop(columns=[ind]), dep, xs)
    return {"coef": r["coef"]["_x"], "p": r["p"]["_x"], "n_obs": r["n_obs"], "n_countries": r["n_countries"]} if r else {}


def lead_lag_table(M: pd.DataFrame, pairs: list[tuple[str, str]], lags=(1, 3, 6, 12), own_lags: int = 3) -> pd.DataFrame:
    """Panel Granger tests in both directions for each (x, y) pair and lag k: does x(t-k) help predict y(t) beyond y's own recent past,
    and does y(t-k) help predict x(t) beyond x's own past?"""
    rows = []
    for x, y in pairs:
        for k in lags:
            r1, r2 = _granger_panel(M, y, x, k, own_lags), _granger_panel(M, x, y, k, own_lags)
            if r1 and r2:
                rows.append({"x": x, "y": y, "lag": k, "coef_x_leads_y": r1["coef"], "p_x_leads_y": r1["p"], "coef_y_leads_x": r2["coef"],
                             "p_y_leads_x": r2["p"], "n_obs": r1["n_obs"], "n_countries": r1["n_countries"], "verdict": verdict(r1["p"], r2["p"])})
    return pd.DataFrame(rows)


def country_lead(hf: pd.DataFrame, x: str, y: str, max_lag: int = 12) -> pd.DataFrame:
    """Per country: lag (0..max_lag, x leading) with the largest prewhitened |cross-correlation| between transformed x and y."""
    X, Y = transformed(hf, x), transformed(hf, y)
    rows = []
    for c in sorted(set(X.columns) & set(Y.columns) - {"EU27"}):
        b = best_lead(cross_corr(X[c], Y[c], max_lag))
        if b:
            rows.append({"x": x, "y": y, "iso3": c, **b})
    return pd.DataFrame(rows)


def italy_rank(df: pd.DataFrame, col: str = "r") -> dict:
    """Italy's rank among countries by |col| (1 = strongest), the placebo-ranking device used for single-country comparisons."""
    d = df.set_index("iso3")[col].abs().sort_values(ascending=False)
    if ITALY not in d.index:
        return {}
    return {"italy_value": float(df.set_index("iso3").loc[ITALY, col]), "rank": int(d.index.get_loc(ITALY) + 1), "n_countries": len(d)}


# --- spread model ----------------------------------------------------------------------------------------------------


def spread_series(hf: pd.DataFrame, country: str = ITALY, ref: str = "DEU") -> pd.Series:
    w = wide_hf(hf, "m_long_term_yield")
    return (w[country] - w[ref]).rename(f"spread_{country}_{ref}")


def spread_model(hf: pd.DataFrame, country: str = ITALY, ref: str = "DEU", lag: int = 1, hac_lags: int = 6) -> pd.DataFrame:
    """Monthly change in the yield spread over `ref` on the previous month's (standardised) differentials with `ref`: economic sentiment
    (level), change in unemployment, change in core inflation and industrial-production growth. OLS with Newey-West SEs, by regime."""
    def diff(ind: str, how: str) -> pd.Series:
        w = transform(wide_hf(hf, ind), how)
        return w[country] - w[ref]
    spread = spread_series(hf, country, ref)
    X = pd.DataFrame({"sentiment_gap": diff("m_esi", "level"), "unemployment_change_gap": diff("m_unemployment_rate", "diff"),
                      "core_inflation_change_gap": diff("m_hicp_core_inflation", "diff"), "industrial_growth_gap": diff("m_industrial_production", "dlog")})
    X = X.shift(lag)
    d = pd.concat([spread.diff().rename("d_spread"), X], axis=1).dropna()
    rows = []
    for regime, (a, b) in REGIMES.items():
        s = d.loc[a:b]
        if len(s) < 40:
            continue
        Z = (s.drop(columns="d_spread") - s.drop(columns="d_spread").mean()) / s.drop(columns="d_spread").std()
        m = sm.OLS(s.d_spread, sm.add_constant(Z)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        for v in Z.columns:
            rows.append({"country": country, "ref": ref, "regime": regime, "regressor": v, "coef_pp_per_sd": float(m.params[v]), "se": float(m.bse[v]),
                         "p": float(m.pvalues[v]), "n": len(s), "r2": float(m.rsquared)})
    return pd.DataFrame(rows)


# --- events ----------------------------------------------------------------------------------------------------------


def event_study(x: pd.Series, dates, window: int = 6, min_null: int = 30) -> pd.DataFrame:
    """Change in `x` around each event date: mean of the `window` months after minus mean of the `window` months before (the event
    month sits in neither), against a placebo distribution from every other month at least `window` months away from the event.
    p is two-sided and exact over the placebo months (+1 smoothing); z is the standardised change."""
    x = x.copy()
    x.index = pd.DatetimeIndex(x.index)
    r = x.rolling(window, min_periods=window).mean()
    stat = r.shift(-window) - r.shift(1)  # post window [t+1, t+w] minus pre window [t-w, t-1]
    pos = {d: i for i, d in enumerate(stat.index)}
    rows = []
    for name, when in dates:
        t = pd.Timestamp(when) + pd.offsets.MonthEnd(0)
        if t not in pos or not np.isfinite(stat.get(t, np.nan)):
            continue
        i = pos[t]
        mask = np.abs(np.arange(len(stat)) - i) > window
        null = stat.to_numpy()[mask]
        null = null[np.isfinite(null)]
        if len(null) < min_null:
            continue
        obs = stat.iloc[i]
        rows.append({"event": name, "date": pd.Timestamp(when), "change": float(obs), "z": float((obs - null.mean()) / null.std()),
                     "p": float((1 + (np.abs(null) >= abs(obs)).sum()) / (1 + len(null))), "n_placebo": len(null)})
    return pd.DataFrame(rows)


# --- pass-through ----------------------------------------------------------------------------------------------------


def quarterly(hf: pd.DataFrame, indicator: str, how: str = "level") -> pd.DataFrame:
    """Monthly series averaged to quarters (quarter-end dates), then transformed."""
    w = wide_hf(hf, indicator).resample("QE").mean()
    return transform(w, how)


def passthrough_panel(hf: pd.DataFrame, window: tuple[str, str], n_lags: int = 5, exclude: tuple[str, ...] = ()) -> dict:
    """Labour-cost growth (year on year) on the average inflation of the current and previous n_lags - 1 quarters, controlling for
    unemployment; country + quarter fixed effects, country-clustered SEs. The coefficient on the inflation average is the cumulative
    pass-through (percentage points of wage growth per point of inflation)."""
    wage = wide_hf(hf, "q_labour_cost_yoy")
    infl = quarterly(hf, "m_hicp_inflation").rolling(n_lags, min_periods=n_lags).mean()
    unemp = quarterly(hf, "m_unemployment_rate")
    cols = {"wage_growth": wage, "inflation_avg": infl, "unemployment": unemp}
    M = pd.DataFrame({k: v.loc[window[0]:window[1]].drop(columns=["EU27"], errors="ignore").stack() for k, v in cols.items()}).dropna()
    M.index = M.index.set_names(["date", "iso3"])
    M = M.swaplevel().sort_index()
    r = corr.panel_fe(M, "wage_growth", ["inflation_avg", "unemployment"], exclude=exclude)
    if not r:
        return {}
    return {"pass_through": r["coef"]["inflation_avg"], "p": r["p"]["inflation_avg"], "unemployment_coef": r["coef"]["unemployment"], "n_obs": r["n_obs"], "n_countries": r["n_countries"]}


def passthrough_by_country(hf: pd.DataFrame, window: tuple[str, str] = ("2005-01-01", "2026-12-31"), n_lags: int = 5) -> pd.DataFrame:
    """Same regression country by country (OLS, Newey-West SEs, no fixed effects) so Italy can be ranked among the others."""
    wage = wide_hf(hf, "q_labour_cost_yoy")
    infl = quarterly(hf, "m_hicp_inflation").rolling(n_lags, min_periods=n_lags).mean()
    unemp = quarterly(hf, "m_unemployment_rate")
    rows = []
    for c in sorted(set(wage.columns) & set(infl.columns) & set(unemp.columns) - {"EU27"}):
        d = pd.concat([wage[c], infl[c], unemp[c]], axis=1, keys=["w", "pi", "u"]).loc[window[0]:window[1]].dropna()
        if len(d) < 40:
            continue
        m = sm.OLS(d.w, sm.add_constant(d[["pi", "u"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        rows.append({"iso3": c, "pass_through": float(m.params["pi"]), "se": float(m.bse["pi"]), "p": float(m.pvalues["pi"]), "n": len(d)})
    return pd.DataFrame(rows)
