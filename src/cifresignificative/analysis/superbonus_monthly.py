"""When and where the Superbonus was used: the ENEA monthly reports by region (`sources/enea_monthly.py`, 62 month ends from August 2021 to September 2026, SuperEcobonus only, cumulative since July 2020, by transmission date) against population and regional income.

Descriptive. Flows are differences of the cumulative series over windows (the first window is the cumulative to August 2021, so it covers July 2020 to August 2021); late transmissions move works between windows and ENEA's own revisions can make a regional flow slightly negative (May 2026).
One pre-specified family, the question "when did the income gradient of the new works arise": the Spearman correlation across the 20 regions between the admitted investment of each window per inhabitant and GDP per inhabitant (Eurostat NUTS2, workplace based, `superbonus_regions.region_table`), permutation p over regions, Benjamini-Hochberg over the windows;
a placebo that shuffles the regional incomes and must give about zero discoveries. The completed-works version, the cumulative gradient by month and the zone shares are shown as robustness and description, not as a family. Never causal: the credit followed the works, which followed the housing stock, ownership and the ability to advance the cost.
"""
import numpy as np
import pandas as pd
from scipy import stats

from cifresignificative.analysis import superbonus_regions as SBR

ZONES = ["North", "Centre", "South and Islands"]
# (label, previous month end that is subtracted (None = cumulative), month end): the windows of the flow tables
WINDOWS = [("to Aug 2021", None, "2021-08-31"), ("Sep-Dec 2021", "2021-08-31", "2021-12-31"), ("H1 2022", "2021-12-31", "2022-06-30"), ("H2 2022", "2022-06-30", "2022-12-31"),
           ("H1 2023", "2022-12-31", "2023-06-30"), ("H2 2023", "2023-06-30", "2023-12-31"), ("2024", "2023-12-31", "2024-12-31"), ("2025 to Sep 2026", "2024-12-31", "2026-09-30")]


def regions_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.region != "Italia"]


def cumulative(df: pd.DataFrame, field: str = "inv_completed") -> pd.DataFrame:
    """Month end x region matrix of a cumulative field (EUR), the 20 regions."""
    return regions_only(df).pivot(index="date", columns="region", values=field).sort_index()


def attributes(df: pd.DataFrame, regional: pd.DataFrame, year: int = 2021) -> pd.DataFrame:
    """Region -> zone, population and GDP per inhabitant (Eurostat NUTS2, `year`), through `superbonus_regions.region_table` on the last month end."""
    e = df.rename(columns={"date": "snapshot"}).assign(total=lambda x: x.inv_completed, condominiums=lambda x: x.condo_completed)
    t = SBR.region_table(e, regional, str(df.date.max().date()), year)
    return t.set_index("region")[["zone", "population", "gdp_per_inhabitant_eur"]]


def window_flows(df: pd.DataFrame, regional: pd.DataFrame, field: str = "inv_admitted") -> pd.DataFrame:
    """Long table window x region: the flow of `field` (EUR), per inhabitant, and the zone."""
    cum, att = cumulative(df, field), attributes(df, regional)
    rows = []
    for label, a, b in WINDOWS:
        flow = cum.loc[pd.Timestamp(b)] - (cum.loc[pd.Timestamp(a)] if a else 0.0)
        rows.append(pd.DataFrame({"window": label, "region": flow.index, "flow_eur": flow.to_numpy()}))
    out = pd.concat(rows, ignore_index=True).join(att, on="region")
    out["per_inhabitant_eur"] = out.flow_eur / out.population
    return out


def zone_flow_shares(flows: pd.DataFrame) -> pd.DataFrame:
    """Window x zone: flow (EUR million), share of the window's flow, population share and the ratio of the two (1 = in proportion to population)."""
    g = flows.groupby(["window", "zone"]).agg(flow_meur=("flow_eur", lambda s: s.sum() / 1e6), population=("population", "sum")).reset_index()
    g["share_of_flow"] = g.flow_meur / g.groupby("window").flow_meur.transform("sum")
    g["share_of_population"] = g.population / g.groupby("window").population.transform("sum")
    g["ratio_of_shares"] = g.share_of_flow / g.share_of_population
    order = {w[0]: i for i, w in enumerate(WINDOWS)}
    return g.assign(_o=g.window.map(order), _z=g.zone.map({z: i for i, z in enumerate(ZONES)})).sort_values(["_o", "_z"]).drop(columns=["_o", "_z", "population"]).reset_index(drop=True)


def zone_completion(df: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    """Share of the admitted investment already completed, by zone, at the month ends `dates`."""
    z = regions_only(df).assign(zone=lambda x: x.nuts2.str[:3].map(SBR.ZONE))
    g = z[z.date.isin(pd.to_datetime(dates))].groupby(["date", "zone"])[["inv_completed", "inv_admitted"]].sum()
    return (g.inv_completed / g.inv_admitted).unstack().reindex(columns=ZONES).rename_axis(None, axis=1)


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def _perm_p(x: np.ndarray, y: np.ndarray, perms: np.ndarray) -> tuple[float, float]:
    """Spearman rho of x and y and a two-sided permutation p with the fixed permutations `perms` (rows of indices) of the ranks of x."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rxc, ryc = rx - rx.mean(), ry - ry.mean()
    rho = float(rxc @ ryc / np.sqrt((rxc**2).sum() * (ryc**2).sum()))
    null = rxc[perms] @ ryc / np.sqrt((rxc**2).sum() * (ryc**2).sum())
    return rho, float((1 + (np.abs(null) >= abs(rho) - 1e-12).sum()) / (1 + len(perms)))


def window_gradient(flows: pd.DataFrame, x: str = "gdp_per_inhabitant_eur", y: str = "per_inhabitant_eur", n_perm: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Per window: Spearman rho between GDP per inhabitant and the flow per inhabitant over the regions, permutation p and BH q over the windows (the pre-specified family)."""
    rng = np.random.default_rng(seed)
    n = flows.region.nunique()
    perms = np.argsort(rng.random((n_perm, n)), axis=1)
    rows = []
    for w in [w[0] for w in WINDOWS]:
        d = flows[flows.window == w].dropna(subset=[x, y])
        rho, p = _perm_p(d[x].to_numpy(), d[y].to_numpy(), perms)
        rows.append({"window": w, "n": len(d), "rho": rho, "p": p})
    out = pd.DataFrame(rows)
    out["q"] = _bh(out.p.to_numpy())
    return out


def gradient_placebo(flows: pd.DataFrame, n_shuffle: int = 200, n_perm: int = 2000, seed: int = 1, alpha: float = 0.05) -> dict:
    """Placebo for the window family: shuffle the regional incomes (one fixed shuffle for all windows), rerun the family; the share of shuffles with any q < alpha and the mean number of windows below it should be near zero."""
    rng = np.random.default_rng(seed)
    wide = flows.pivot(index="region", columns="window", values="per_inhabitant_eur")[[w[0] for w in WINDOWS]]
    inc = flows.drop_duplicates("region").set_index("region").gdp_per_inhabitant_eur.reindex(wide.index).to_numpy()
    perms = np.argsort(rng.random((n_perm, len(wide))), axis=1)
    any_q, n_q = 0, []
    for _ in range(n_shuffle):
        x = rng.permutation(inc)
        p = np.array([_perm_p(x, wide[w].to_numpy(), perms)[1] for w in wide.columns])
        k = int((_bh(p) < alpha).sum())
        any_q += k > 0
        n_q.append(k)
    return {"n_shuffle": n_shuffle, "share_with_discovery": any_q / n_shuffle, "mean_discoveries": float(np.mean(n_q))}


def gradient_path(df: pd.DataFrame, regional: pd.DataFrame, field: str = "inv_completed") -> pd.DataFrame:
    """Month end -> Spearman rho (no p) between the cumulative investment per inhabitant and GDP per inhabitant over the 20 regions: how the cumulative gradient built up."""
    att = attributes(df, regional)
    cum = cumulative(df, field)
    per = cum.div(att.population.reindex(cum.columns), axis=1)
    inc = att.gdp_per_inhabitant_eur.reindex(cum.columns)
    return pd.DataFrame({"date": cum.index, "rho": [float(stats.spearmanr(inc, per.loc[d])[0]) for d in cum.index]})


def monthly_flows(df: pd.DataFrame, field: str = "inv_admitted") -> pd.DataFrame:
    """Month x zone flow of `field` (EUR million; difference of the cumulative; the first month is missing)."""
    z = regions_only(df).assign(zone=lambda x: x.nuts2.str[:3].map(SBR.ZONE))
    g = z.groupby(["date", "zone"])[field].sum().unstack().reindex(columns=ZONES).diff() / 1e6
    return g.dropna(how="all").rename_axis(None, axis=1)


def basis_check(monthly: pd.DataFrame, annual_regions: pd.DataFrame, national: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(a) Italy: the monthly report against the transcribed national snapshots (`enea_superbonus.csv`) at the same month end, ratio of completed investment; (b) regions: the annual report's regional table (works ended by 31 December, later extraction) against the monthly report of the same month end, ratio and range over the regions."""
    it = monthly[monthly.region == "Italia"].set_index("date")
    a = national.set_index("snapshot").investments_completed_eur
    a = a[a.index.isin(it.index)]
    nat = pd.DataFrame({"annual_or_bulletin_meur": a / 1e6, "monthly_meur": it.inv_completed.reindex(a.index) / 1e6}).assign(ratio=lambda x: x.monthly_meur / x.annual_or_bulletin_meur).rename_axis("snapshot").reset_index()
    rows = []
    for snap, g in annual_regions[annual_regions.region != "Italia"].groupby("snapshot"):
        m = monthly[(monthly.date == snap) & (monthly.region != "Italia")].set_index("region").inv_completed
        r = (m / g.set_index("region").total).dropna()
        rows.append({"snapshot": snap, "n": len(r), "ratio_min": r.min(), "ratio_median": r.median(), "ratio_max": r.max(), "rho_between_bases": float(stats.spearmanr(m.reindex(r.index), g.set_index("region").total.reindex(r.index))[0])})
    return nat, pd.DataFrame(rows)
