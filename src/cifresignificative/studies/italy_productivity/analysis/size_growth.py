"""Firm size against the *level* and the *growth* of productivity: does the size structure that lowers Italy's level (`size_class`) also line up with the growth Italy lost?

Three descriptive checks. (1) Over time: Italy's micro share and the gain of its own mix change (`mix_change`), within stretches without a series break, against real value added per hour. (2) Across countries: the micro share against the
level and against the growth of hourly productivity (`growth_link`: rank correlations with a permutation p, subsets, and a slope with the initial level held fixed, because poorer members catch up). (3) Across sections: Italy's micro excess against its
growth gap, the same statistic for every member, and a pooled country and section demeaned correlation with a within-country permutation (`section_link`). `coverage_sensitivity` moves value added missing from the size-class statistics into Italy's
micro class, an illustrative bound for under-reporting. Productivity comes from the national accounts of the business economy (`BUSINESS`: no agriculture, finance, real estate, public services), which is closest to the size-class scope. Nothing here says which causes which."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from cifresignificative.studies.italy_productivity.analysis import size_class as SC
from cifresignificative.studies.italy_productivity.analysis.data import EU27_ISO3, ITALY

BUSINESS = ["B-E", "F", "G-I", "J", "M_N"]
OLD_MEMBERS = ["AUT", "BEL", "DEU", "DNK", "ESP", "FIN", "FRA", "GRC", "IRL", "ITA", "LUX", "NLD", "PRT", "SWE"]  # EU15 without the UK: no catch-up from a low base after 2004
SECTIONS = {"C": "Manifattura", "F": "Costruzioni", "G": "Commercio", "H": "Trasporti", "I": "Alloggio e ristoraz.", "J": "Informazione", "M": "Serv. professionali", "N": "Serv. amministrativi"}  # real estate left out: imputed rents in the national accounts
MIN_SECTIONS = 7


def micro_shares(size: pd.DataFrame, vintage: str, nace: str | None = None) -> pd.DataFrame:
    """Micro share of persons employed (%), iso3 x year, from the 0-9 class over the published total, so a year counts even when the 20-49 or 250+ cells are confidential."""
    d = size[(size.vintage == vintage) & (size.nace_r2 == (nace or SC.ECONOMY[vintage])) & size.iso3.isin(EU27_ISO3) & size.size_emp.isin(["0-9", "TOTAL"])]
    p = d.pivot_table(index=["iso3", "year"], columns="size_emp", values="persons")
    return (p["0-9"] / p["TOTAL"]).unstack("year") * 100


def business_accounts(sectors: pd.DataFrame, size: pd.DataFrame) -> pd.DataFrame:
    """One row per (iso3, year) of the business economy: `real` (chain-linked value added per hour), `level` (current value added per hour, euro at the EU27 price level; GDP-wide price index, so approximate); a year needs all the sections."""
    d = sectors[sectors.nace_r2.isin(BUSINESS) & sectors.iso3.isin(EU27_ISO3)]
    g = d.groupby(["iso3", "year"]).agg(cp=("gva_cp_meur", lambda x: x.sum(min_count=len(BUSINESS))), clv=("gva_clv20_meur", lambda x: x.sum(min_count=len(BUSINESS))), hours=("hours_ths", lambda x: x.sum(min_count=len(BUSINESS)))).reset_index()
    pli = size.dropna(subset=["pli"]).drop_duplicates(["iso3", "year"]).set_index(["iso3", "year"]).pli
    g["pli"] = pli.reindex(pd.MultiIndex.from_frame(g[["iso3", "year"]])).to_numpy()
    return g.assign(real=g.clv / g.hours, level=g.cp * 1e3 / g.hours / (g.pli / 100))


def mix_change(size: pd.DataFrame, vintage: str, start: int, end: int, iso3: str = ITALY) -> float:
    """Log points (x100) that a country's aggregate productivity gains from the change of its size mix between two years, at the end-year class productivities: log(sum s1 x1 / sum s0 x1) x 100. Only meaningful without a break between the years."""
    t = SC.class_table(size, vintage)
    t = t[t.iso3 == iso3]
    P, V = t.pivot(index="year", columns="size_emp", values="persons")[SC.CLASSES], t.pivot(index="year", columns="size_emp", values="va_meur")[SC.CLASSES]
    s0, s1, x1 = P.loc[start] / P.loc[start].sum(), P.loc[end] / P.loc[end].sum(), V.loc[end] / P.loc[end]
    return float(np.log((s1 * x1).sum() / (s0 * x1).sum()) * 100)


def rank_link(x: pd.Series, y: pd.Series, n_perm: int = 19999, seed: int = 3) -> dict[str, float]:
    """Spearman correlation of two series on their common non-missing index, with a two-sided permutation p."""
    d = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    a, b = rankdata(d.x), rankdata(d.y)
    obs = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    hits = sum(abs(np.corrcoef(rng.permutation(a), b)[0, 1]) >= abs(obs) for _ in range(n_perm))
    return {"n": len(d), "rho": obs, "p": (hits + 1) / (n_perm + 1)}


def partial_slope(micro: pd.Series, growth: pd.Series, initial: pd.Series, n_perm: int = 9999, seed: int = 1) -> dict[str, float]:
    """OLS of growth on the micro share and the log initial level: the slope in points a year per 10 points of micro share, a permutation p that shuffles the micro share, and each country's residual (`res_<iso3>` for Italy only)."""
    d = pd.concat([micro, growth, initial.map(np.log)], axis=1, keys=["m", "g", "l"]).dropna()
    X = np.column_stack([np.ones(len(d)), d.m, d.l])
    beta = np.linalg.lstsq(X, d.g, rcond=None)[0]
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        Xp = X.copy()
        Xp[:, 1] = rng.permutation(d.m.to_numpy())
        hits += abs(np.linalg.lstsq(Xp, d.g, rcond=None)[0][1]) >= abs(beta[1])
    res = d.g - X @ beta
    return {"n": len(d), "slope_per_10": float(beta[1] * 10), "p": (hits + 1) / (n_perm + 1), "residual_italy": float(res.get(ITALY, np.nan))}


def growth_link(size: pd.DataFrame, sectors: pd.DataFrame, start: int = 2005, end: int = 2024, micro_year: int = 2023, initial_year: int = 2008) -> dict[str, pd.DataFrame]:
    """`table`: micro share, level and growth per member. `links`: rank correlations of the micro share with the level and the growth, all members and subsets. `partial`: the slope with the initial level held fixed, all members and without Ireland or Greece."""
    acc = business_accounts(sectors, size)
    R, L = acc.pivot(index="year", columns="iso3", values="real"), acc.pivot(index="year", columns="iso3", values="level")
    m = micro_shares(size, "new")[micro_year]
    g = (np.log(R.loc[end]) - np.log(R.loc[start])) / (end - start) * 100
    table = pd.DataFrame({"micro": m, "level": L.loc[micro_year], "growth_log_year": g, "growth_pct": (R.loc[end] / R.loc[start] - 1) * 100})
    old = [c for c in OLD_MEMBERS if c in table.index]
    links = {"level, all": rank_link(m, np.log(L.loc[micro_year])), "level, old members": rank_link(m.reindex(old), np.log(L.loc[micro_year])), "level, no Ireland": rank_link(m.drop("IRL"), np.log(L.loc[micro_year])),
             "growth, all": rank_link(m, g), "growth, old members": rank_link(m.reindex(old), g), "growth, old members no Greece": rank_link(m.reindex([c for c in old if c != "GRC"]), g),
             "growth, new members": rank_link(m.drop(old), g), "growth, micro share 2012": rank_link(micro_shares(size, "old")[2012], g)}
    init = L.loc[initial_year]
    partial = {"all": partial_slope(m, g, init), "no Ireland": partial_slope(m.drop("IRL"), g, init), "no Greece": partial_slope(m.drop("GRC"), g, init), "no Ireland, Greece": partial_slope(m.drop(["IRL", "GRC"]), g, init)}
    return {"table": table, "links": pd.DataFrame(links).T, "partial": pd.DataFrame(partial).T}


def section_growth(s64: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Log growth of real value added per hour (points a year), iso3 x section, summing the A*64 industries that a member has in both years."""
    d = s64[s64.iso3.isin(EU27_ISO3) & s64.year.isin([start, end])].assign(section=lambda x: x.nace_r2.str[0]).dropna(subset=["gva_clv20_meur", "hours_ths"])
    d = d[d.section.isin(SECTIONS) & (d.groupby(["iso3", "nace_r2"]).year.transform("nunique") == 2)]
    w = d.groupby(["iso3", "section", "year"])[["gva_clv20_meur", "hours_ths"]].sum().eval("gva_clv20_meur / hours_ths").unstack("year")
    return (np.log(w[end] / w[start]) / (end - start) * 100).unstack("section")


def section_micro(size: pd.DataFrame, vintage: str = "new", year: int = 2023) -> pd.DataFrame:
    """Micro share of persons employed (%), iso3 x section, one year."""
    d = size[(size.vintage == vintage) & (size.year == year) & size.nace_r2.isin(SECTIONS) & size.iso3.isin(EU27_ISO3) & size.size_emp.isin(["0-9", "TOTAL"])]
    p = d.pivot_table(index=["iso3", "nace_r2"], columns="size_emp", values="persons")
    return (p["0-9"] / p["TOTAL"]).unstack("nace_r2") * 100


def _two_way(d: pd.DataFrame) -> pd.DataFrame:
    z = d - d.groupby(level=0).transform("mean")
    return z - z.groupby(level=1).transform("mean")


def section_link(size: pd.DataFrame, s64: pd.DataFrame, start: int = 2005, end: int = 2024, vintage: str = "new", year: int = 2023, n_perm: int = 4999, seed: int = 7) -> dict[str, pd.DataFrame]:
    """`italy`: per section, Italy's growth gap to the median of the others and its micro excess. `countries`: per member with at least MIN_SECTIONS sections, the Spearman correlation of those two. `pooled`: the correlation after demeaning by country and
    section, with a permutation of the micro share inside each country."""
    G, M = section_growth(s64, start, end), section_micro(size, vintage, year)
    rows = {}
    for c in G.index.intersection(M.index):
        d = pd.concat([G.loc[c] - G.drop(index=c).median(), M.loc[c] - M.drop(index=c).median()], axis=1, keys=["gap", "excess"]).dropna()
        if len(d) >= MIN_SECTIONS:
            rows[c] = d.corr(method="spearman").iloc[0, 1]
    italy = pd.DataFrame({"gap": G.loc[ITALY] - G.drop(index=ITALY).median(), "italy_growth": G.loc[ITALY], "median_growth": G.drop(index=ITALY).median(), "n_growth": G.drop(index=ITALY).notna().sum(), "micro": M.loc[ITALY], "median_micro": M.drop(index=ITALY).median()})
    italy["excess"] = italy.micro - italy.median_micro
    L = pd.concat([G.stack().rename("g"), M.stack().rename("m")], axis=1).dropna()
    z = _two_way(L)
    obs = float(z.corr(method="spearman").iloc[0, 1])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        p = L.copy()
        p["m"] = p.groupby(level=0).m.transform(lambda q: rng.permutation(q.to_numpy()))
        hits += abs(_two_way(p).corr(method="spearman").iloc[0, 1]) >= abs(obs)
    pooled = pd.DataFrame([{"n": len(L), "countries": L.index.get_level_values(0).nunique(), "rho": obs, "p": (hits + 1) / (n_perm + 1)}])
    return {"italy": italy, "countries": pd.Series(rows, name="rho").sort_values().to_frame(), "pooled": pooled}


def coverage_sensitivity(size: pd.DataFrame, sectors: pd.DataFrame, panel: pd.DataFrame, year: int = 2019) -> dict[str, pd.DataFrame]:
    """`ratio`: value added of the size-class statistics (old vintage, without real estate) over the national accounts of the business economy, per member. `scenarios`: Italy's mix effect, gap per hour to Germany with its size-mix and within terms, and gradient, when the value
    added that would bring its ratio to the median of the others or to Germany's is added to its micro class. An illustrative bound: the difference also holds definitions (factor cost, scope), and it is put in the micro class in full, with Germany left as published."""
    o = size[(size.vintage == "old") & (size.year == year) & (size.size_emp == "TOTAL") & size.iso3.isin(EU27_ISO3)]
    total = o[o.nace_r2 == SC.ECONOMY["old"]].set_index("iso3").va_meur - o[o.nace_r2 == "L"].set_index("iso3").va_meur
    na = sectors[sectors.nace_r2.isin(BUSINESS) & (sectors.year == year) & sectors.iso3.isin(EU27_ISO3)].groupby("iso3").gva_cp_meur.sum(min_count=len(BUSINESS))
    ratio = (total / na).dropna().rename("ratio")
    t = SC.class_table(size, "old")
    t = t[t.year == year]
    P, V = t.pivot(index="iso3", columns="size_emp", values="persons")[SC.CLASSES], t.pivot(index="iso3", columns="size_emp", values="va_meur")[SC.CLASSES]
    S = P.div(P.sum(axis=1), axis=0)
    base = float(SC.gap_decomposition(size, panel, "old", year, basis="hour").set_index("reference").loc["DEU", "total"]) * 100  # the added value added moves only Italy's numerator, so hours and prices stay out of the change

    def stats(values: pd.DataFrame, added: float, name: str) -> dict[str, float | str]:
        x = (values / P).where((P > 0) & (values > 0))
        it = x.loc[ITALY]
        mix, total = float(np.log((S.loc[ITALY] * it).sum() / (S.loc["DEU"] * it).sum())) * 100, base + float(np.log(values.loc[ITALY].sum() / V.loc[ITALY].sum())) * 100
        return {"scenario": name, "added_meur": added, "micro_va_up_pct": added / V.loc[ITALY, "0-9"] * 100, "mix_effect_typical": float(SC.mix_effect(S, x).loc[ITALY]) * 100, "total_vs_germany": total, "mix_vs_germany": mix, "within_vs_germany": total - mix, "mix_share": mix / total * 100, "gradient": float(np.log(it["GE250"] / it["0-9"]))}

    rows = [stats(V, 0.0, "as published")]
    for name, target in (("to the median ratio", ratio.drop(ITALY).median()), ("to Germany's ratio", ratio["DEU"])):
        add = float((target - ratio[ITALY]) * na[ITALY])
        v2 = V.copy()
        v2.loc[ITALY, "0-9"] += add
        rows.append(stats(v2, add, name))
    return {"ratio": ratio.sort_values().to_frame(), "scenarios": pd.DataFrame(rows)}
