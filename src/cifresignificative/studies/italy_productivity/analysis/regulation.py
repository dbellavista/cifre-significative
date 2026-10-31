"""Regulation and productivity: where Italy stands in the OECD product market regulation (PMR) and employment protection (EPL) indices, and whether sectors that are more
regulated in a country (PMR 2018 sector indicator) show slower productivity growth in that sector (EU KLEMS-style `sectors64`, 2005-07 to 2017-19).

Design: country x sector pairs for the sectors with a matching PMR indicator (`SECTORS`); within each sector both variables are turned into ranks (so sector level differences drop out), the statistic is
the pooled correlation of those within-sector ranks, and the p-value shuffles the regulation ranks across the countries of each sector (`perm_pooled`). Italy is judged by a placebo ranking among the countries,
not by a test (one country). PMR is measured in 2018, at the end of the growth window and with no earlier wave in this flow, so the association cannot say which came first. Descriptive only."""
import numpy as np
import pandas as pd

from cifresignificative.studies.italy_productivity.analysis import corr
from cifresignificative.studies.italy_productivity.analysis import industry_panel as IP
from cifresignificative.studies.italy_productivity.analysis.data import (
    EU27_ISO3,
    ITALY,
    load_sectors64,
)

SECTORS = {"G47": ("retail", ["RETAIL_TRADE"]), "M69_M70": ("legal and accounting", ["LAWYERS", "ACCOUNTANTS"]), "M71": ("architecture and engineering", ["ARCHITECTS", "ENGINEERS"]),
           "H49": ("land transport", ["RAIL", "ROAD"]), "J61": ("telecommunications", ["ECOMM"]), "D": ("electricity and gas", ["ENERGY"])}  # sectors64 code -> (label, PMR sector indicators averaged)
OVERALL = ["PMR", "STATE_INVOL", "INVOL_BUSINESS", "BARRIER_ENTRY", "NETWORK_SECTORS", "SERVBARRIER", "PUBLIC_PROCUREMENT", "ADBURDEN"]
POSITION = [*OVERALL, "LAWYERS", "ACCOUNTANTS", "ARCHITECTS", "ENGINEERS", "NOTARIES", "RETAIL_TRADE", "ENERGY", "TRANSPORT", "ECOMM"]
MIN_COUNTRIES = 15  # a sector enters the test with at least this many countries


def position(pmr: pd.DataFrame, measures=POSITION, years=(2018, 2023), eu=EU27_ISO3) -> pd.DataFrame:
    """Italy's PMR value, the median of the other EU27 members and Italy's rank (1 = most restrictive) among the EU27 members with a value, per indicator and wave."""
    rows = []
    for m in measures:
        for y in years:
            s = pmr[(pmr.measure == m) & (pmr.year == y) & pmr.iso3.isin(eu)].set_index("iso3").value
            if ITALY in s.index:
                rows.append({"measure": m, "year": y, "n_eu": len(s), "italy": s[ITALY], "median_others": s.drop(ITALY).median(), "rank": int((s > s[ITALY]).sum() + 1)})
    return pd.DataFrame(rows)


def epl_gap(epl: pd.DataFrame, measure: str = "EPL_OV", last: int = 2019, eu=EU27_ISO3) -> pd.DataFrame:
    """Year table: Italy, the median of the other EU27 members present, the number of members and Italy's rank (1 = strictest). The VERSION1 series is not updated after `last` (values repeat)."""
    w = epl[(epl.measure == measure) & epl.iso3.isin(eu) & (epl.year <= last)].pivot(index="year", columns="iso3", values="value")
    oth = w.drop(columns=ITALY)
    return pd.DataFrame({"italy": w[ITALY], "median_others": oth.median(axis=1), "n_others": oth.notna().sum(axis=1), "rank": w.rank(axis=1, ascending=False, method="min")[ITALY]})


def sector_regulation(pmr: pd.DataFrame, year: int = 2018) -> pd.DataFrame:
    """iso3, nace_r2 -> `reg`: mean of the PMR sector indicators matched to the sector (`SECTORS`), wave `year`."""
    p = pmr[pmr.year == year].pivot(index="iso3", columns="measure", values="value")
    rows = [(c, code, p.loc[c, ms].mean()) for code, (_, ms) in SECTORS.items() for c in p.index if set(ms) <= set(p.columns) and p.loc[c, ms].notna().all()]
    return pd.DataFrame(rows, columns=["iso3", "nace_r2", "reg"])


def _within_rank(d: pd.DataFrame, col: str) -> pd.Series:
    r = d.groupby("nace_r2")[col].rank()
    n = d.groupby("nace_r2")[col].transform("count")
    return (r - (n + 1) / 2) / np.sqrt((n ** 2 - 1) / 12)  # centred and scaled ranks: each sector has mean 0 and sd 1


def perm_pooled(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    """Pooled correlation of the within-group standardised ranks and its two-sided permutation p (x shuffled within each group)."""
    obs = float(np.mean(x * y))
    parts = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    sh = np.empty((n_perm, len(x)))
    for idx in parts:
        sh[:, idx] = rng.permuted(np.tile(x[idx], (n_perm, 1)), axis=1)
    null = sh @ y / len(x)
    return obs, float((1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / (1 + n_perm))


def sector_data(pmr: pd.DataFrame, s: pd.DataFrame | None = None, window=IP.WINDOWS["full"], year: int = 2018) -> pd.DataFrame:
    """Country x sector table: `g` (log growth per year of real GVA per hour between the two windows), `reg`, and their within-sector standardised ranks `rg`, `rr`; sectors with fewer than MIN_COUNTRIES dropped."""
    s = load_sectors64() if s is None else s
    G = IP.industry_growth(s, *window)
    d = G[G.nace_r2.isin(SECTORS)].merge(sector_regulation(pmr, year), on=["iso3", "nace_r2"]).dropna()
    d = d[d.groupby("nace_r2").iso3.transform("nunique") >= MIN_COUNTRIES].reset_index(drop=True)
    return d.assign(rg=lambda t: _within_rank(t, "g"), rr=lambda t: _within_rank(t, "reg"))


def test(d: pd.DataFrame, n_perm: int = 20000, seed: int = 0, q: float = 0.05) -> pd.DataFrame:
    """Pooled row (`sector` = "pooled") and one row per sector: Spearman rho of growth on regulation, n, permutation p, BH q over all rows. Negative rho: more regulated sectors grow more slowly."""
    rng = np.random.default_rng(seed)
    rho, p = perm_pooled(d.rr.to_numpy(), d.rg.to_numpy(), d.nace_r2.to_numpy(), n_perm, rng)
    rows = [{"sector": "pooled", "n": len(d), "n_sectors": d.nace_r2.nunique(), "rho": rho, "p": p}]
    for code, g in d.groupby("nace_r2"):
        r = corr._corr(g.reg.to_numpy(), g.g.to_numpy(), "spearman")[0]
        rows.append({"sector": code, "n": len(g), "n_sectors": 1, "rho": r, "p": perm_pooled(g.rr.to_numpy(), g.rg.to_numpy(), g.nace_r2.to_numpy(), n_perm, rng)[1]})
    return corr.fdr(pd.DataFrame(rows), q)


def italy_rows(d: pd.DataFrame) -> pd.DataFrame:
    """Italy's regulation rank (1 = most restrictive) and growth rank (1 = fastest) in each sector, with the country count."""
    out = []
    for code, g in d.groupby("nace_r2"):
        code = str(code)
        if ITALY in set(g.iso3):
            it = g[g.iso3 == ITALY].iloc[0]
            out.append({"nace_r2": code, "label": SECTORS[code][0], "n": len(g), "reg_italy": it.reg, "reg_median_others": g[g.iso3 != ITALY].reg.median(), "reg_rank": int((g.reg > it.reg).sum() + 1),
                        "growth_italy_pct": it.g * 100, "growth_median_others_pct": g[g.iso3 != ITALY].g.median() * 100, "growth_rank": int((g.g > it.g).sum() + 1)})
    return pd.DataFrame(out)


def italy_placebo(d: pd.DataFrame, min_sectors: int = 4) -> pd.DataFrame:
    """Italy against the pooled relation: for each country with at least `min_sectors` sectors, mean over its sectors of (growth rank - slope x regulation rank), the slope being the pooled one.
    A negative gap = slower growth than the pooled relation predicts from the country's regulation. `rank` = Italy's place among the countries (1 = most negative)."""
    b = float(np.mean(d.rr * d.rg))
    gap = (d.rg - b * d.rr).groupby(d.iso3).agg(["mean", "size"])
    gap = gap[gap["size"] >= min_sectors]["mean"].sort_values()
    return pd.DataFrame({"iso3": gap.index, "gap": gap.to_numpy(), "rank": np.arange(1, len(gap) + 1)}).assign(n_countries=len(gap), pooled_slope=b)


def shuffled_null(d: pd.DataFrame, n_rep: int = 200, n_perm: int = 500, seed: int = 0, q: float = 0.05) -> dict:
    """Calibration: shuffle the regulation across the countries of each sector, rerun the family (pooled + sectors) with the same permutation test; discoveries per repetition must be about 0 and p < 0.05 in about 5%."""
    rng = np.random.default_rng(seed)
    disc, small = [], []
    for k in range(n_rep):
        t = d.copy()
        t["reg"] = t.groupby("nace_r2").reg.transform(lambda v: rng.permutation(v.to_numpy()))
        t["rr"] = _within_rank(t, "reg")
        r = test(t, n_perm=n_perm, seed=k, q=q)
        disc.append(int(r.significant.sum())); small.append(float((r.p < 0.05).mean()))
    return {"n_rep": n_rep, "mean_discoveries": float(np.mean(disc)), "share_reps_with_discovery": float(np.mean(np.array(disc) > 0)), "mean_share_p_below_005": float(np.mean(small))}
