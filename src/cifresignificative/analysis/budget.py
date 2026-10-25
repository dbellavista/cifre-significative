"""How the Italian budget laws were financed: coverage ratios from the resources-and-uses tables of the Servizi Bilancio dossiers (`sources/budget_laws.py`, `data/external/budget_laws.csv`).

*Impieghi* (uses) are the interventions that worsen the balance (higher spending, lower revenue); *risorse* (resources) are what pays for them (higher revenue, lower spending); the manovra's effect on the balance is resources minus uses.
Per budget year and horizon year (h = 0 is the year the law is meant to act on):
- **coverage ratio** = resources / uses (1 = fully covered inside the manovra, 0 = entirely financed by deficit);
- **deficit-financed share** = 1 - coverage = -balance / uses, and the deficit effect in % of GDP;
- how the resources split between higher revenue and lower spending, and how the uses split between higher spending and lower revenue.
Four scopes: `bill` (the budget bill alone), `budget` (the bill plus the decree-law that goes with it, which is how the dossiers define the manovra in 2020 and 2024), `pandemic` (the 2020-21 emergency decrees, kept apart) and `all` (budget plus pandemic).
These are ex-ante estimates from the government's technical reports, not realised effects. 2022 has no published gross split (the Servizio del Bilancio issued no dossier): its uses and resources are rebuilt from the measure lines of the Nota di lettura 274 (`evidence` = `reconstructed_from_measure_lines`; signs by measure, net totals reproduced), so they are comparable but not the government's own table. Descriptive, never causal.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cifresignificative.analysis import corr
from cifresignificative.analysis.data import debt_decomposition, load_spending, wide
from cifresignificative.sources.budget_laws import CSV
from cifresignificative.sources.deficit_targets import CSV as TARGETS_CSV

GOVERNMENTS = Path("data/external/italy_governments.csv")
ITEMS = ["totale_impieghi", "totale_risorse", "manovra", "totale_maggiori_spese", "totale_minori_entrate", "totale_maggiori_entrate", "totale_minori_spese", "spesa_per_interessi", "rimodulazione_pnrr"]
SCOPES = {"bill": lambda r: r.role.isin(["bill", "total"]) & r.group.eq("budget"), "budget": lambda r: r.role.isin(["bill", "total", "decree"]) & r.group.eq("budget"), "pandemic": lambda r: r.group.eq("pandemic"), "all": lambda r: r.group.isin(["budget", "pandemic"])}


def load(path: Path = CSV) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["ref_date"])


def parts(d: pd.DataFrame, aggregate: str = "indebitamento") -> pd.DataFrame:
    """One row per (budget year, table, horizon year) with the items as columns (EUR million). The 2018 first-reading changes (`role` changes) are not a manovra and are dropped."""
    d = d[(d["aggregate"] == aggregate) & d["item"].isin(ITEMS) & (d.role != "changes")]
    w = d.pivot_table(index=["budget_year", "role", "group", "ref_date", "doc", "table", "horizon_year"], columns="item", values="value_meur", aggfunc="first")
    return w.reindex(columns=ITEMS).reset_index()


def _sum(g: pd.DataFrame) -> pd.Series:
    return g[ITEMS].sum(min_count=1)  # a missing item stays missing


def measures(d: pd.DataFrame, gdp: pd.Series | None = None, aggregate: str = "indebitamento") -> pd.DataFrame:
    """Coverage measures per (scope, budget year, horizon year). `gdp` = nominal GDP (EUR million) by calendar year, for the shares of GDP (missing years give NaN)."""
    w = parts(d, aggregate)
    rows = []
    for scope, pick in SCOPES.items():
        sel = w[pick(w)]
        for (by, hy), g in sel.groupby(["budget_year", "horizon_year"]):
            s = _sum(g)
            rows.append({"scope": scope, "budget_year": by, "horizon_year": hy, "h": hy - by, "ref_date": g.ref_date.min(), "n_parts": len(g), **s})
    m = pd.DataFrame(rows)
    m["uses"], m["resources"] = m.totale_impieghi, m.totale_risorse
    m["balance"] = m.manovra.where(m.manovra.notna(), m.resources - m.uses)
    m["coverage_ratio"] = m.resources / m.uses
    m["deficit_financed_share"] = 1 - m.coverage_ratio
    m["resources_from_revenue"] = m.totale_maggiori_entrate / m.uses  # coverage contributed by higher revenue
    m["resources_from_spending"] = m.totale_minori_spese / m.uses  # coverage contributed by lower spending
    m["uses_spending_share"] = m.totale_maggiori_spese / m.uses  # share of the uses that is higher spending (the rest is lower revenue, plus interest in 2026)
    g = m.horizon_year.map(gdp) if gdp is not None else np.nan
    m["gdp_meur"] = g
    for c in ("uses", "resources", "balance"):
        m[f"{c}_pct_gdp"] = m[c] / m.gdp_meur * 100
    m["deficit_pct_gdp"] = -m.balance_pct_gdp
    return m.drop(columns=ITEMS).sort_values(["scope", "budget_year", "horizon_year"], ignore_index=True)


def government_at(date: pd.Timestamp, path: Path = GOVERNMENTS) -> str:
    """Name of the government in office on `date` (italy_governments.csv)."""
    g = pd.read_csv(path, parse_dates=["start", "end"])
    row = g[(g.start <= date) & (g.end.isna() | (g.end > date))]
    return row.government.iloc[-1] if len(row) else ""


def first_year(m: pd.DataFrame) -> pd.DataFrame:
    """The year each law is meant to act on (h = 0), with the government that proposed it."""
    f = m[m.h == 0].copy()
    f["government"] = f.ref_date.map(government_at)
    return f.reset_index(drop=True)


def three_year(m: pd.DataFrame) -> pd.DataFrame:
    """Sums over the three years of the horizon (only budget years with all three), coverage and deficit-financed share of the sums."""
    full = m.groupby(["scope", "budget_year"]).filter(lambda g: set(g.h) >= {0, 1, 2} and g.uses.notna().all())
    s = full[full.h <= 2].groupby(["scope", "budget_year"])[["uses", "resources", "balance"]].sum().reset_index()
    s["coverage_ratio"] = s.resources / s.uses
    s["deficit_financed_share"] = 1 - s.coverage_ratio
    return s


# --- link to the realised deficit and debt ---------------------------------------------------------------------------------------------------------------------------------------------------------
# Pre-specified family. A manovra can only act on the primary balance, so the debt is the weakest link by construction; the signs are expected to be positive (a larger planned deficit effect goes with a larger realised increase).
XS = ["deficit_pct_gdp", "deficit_financed_share"]
YS = ["delta_deficit", "delta_primary_deficit", "delta_debt"]
YS_SURPRISE = ["deficit_surprise"]  # second pre-specified family: the manovra's planned effect and deficit-financed share against the miss of the deficit target


def outcomes(panel: pd.DataFrame, iso3: str = "ITA") -> pd.DataFrame:
    """Realised outcomes by calendar year t (Eurostat, latest vintage): net borrowing and primary deficit in % of GDP and their change from t-1 (positive = looser), the change in debt/GDP split into primary balance, interest, nominal growth and
    stock-flow (`data.debt_decomposition`, one year), and real GDP growth for context. The change in net borrowing from one year to the next is not the manovra's effect: the cycle, interest and one-offs move it too."""
    bal, intr, debt = (wide(panel, k)[iso3] for k in ("gov_balance_pct_gdp", "gov_interest_pct_gdp", "gov_debt_pct_gdp"))
    o = pd.DataFrame({"deficit": -bal, "primary_deficit": -(bal + intr), "debt": debt, "real_growth": wide(panel, "gdp_growth")[iso3]})
    o["delta_deficit"], o["delta_primary_deficit"], o["delta_debt"] = o.deficit.diff(), o.primary_deficit.diff(), o.debt.diff()
    dec = pd.DataFrame({t: debt_decomposition(panel, iso3, t - 1, t) for t in o.index[1:]}).T
    return o.join(dec.rename(columns=lambda c: f"debt_{c.split()[0]}"))


def targets(path: Path = TARGETS_CSV) -> pd.DataFrame:
    """Programmatic deficit targets by budget year (`sources/deficit_targets.py`), as positive deficits in % of GDP: `target` as proposed, `target_final` after any revision before the law passed."""
    t = pd.read_csv(path).set_index("budget_year")
    return pd.DataFrame({"target": -t.target_pct_gdp, "target_final": -t.target_final_pct_gdp})


def link_table(m: pd.DataFrame, out: pd.DataFrame, scope: str = "bill", tg: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per budget year: the manovra's first-year measures (`scope`) next to the outcomes of the same calendar year (t) and of the year before (t-1, for the lead placebo). With the targets, also the deficit target of the
    year, the **deficit surprise** (outturn deficit minus target, pp of GDP, positive = a larger deficit than planned; against the target as proposed and as revised) and the change the plan implied (target minus last year's outturn)."""
    f = first_year(m[m.scope == scope]).set_index("budget_year")
    t = f[["deficit_pct_gdp", "deficit_financed_share", "coverage_ratio", "uses_pct_gdp", "government"]].join(out[YS + ["deficit", "real_growth"]])
    lag = out[YS].shift(1).rename(columns=lambda c: f"{c}_prev")
    t = t.join(lag)
    if tg is not None:
        t = t.join(tg)
        t["deficit_surprise"], t["deficit_surprise_final"] = t.deficit - t.target, t.deficit - t.target_final
        t["planned_delta_deficit"] = t.target - out.deficit.shift(1).reindex(t.index)
    return t


def spearman_perm(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    """Spearman rho and the two-sided permutation p (years shuffled; the +1 correction keeps p above zero). With about ten years the asymptotic p is unreliable."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    perm = np.array([rng.permutation(len(ry)) for _ in range(n_perm)])
    null = _rows_corr(rx, ry[perm])
    return rho, float((1 + (np.abs(null) >= abs(rho) - 1e-12).sum()) / (n_perm + 1))


def _rows_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - a.mean()
    b0 = b - b.mean(axis=1, keepdims=True)
    return (b0 @ a0) / np.sqrt((a0 @ a0) * (b0 * b0).sum(axis=1))


def link_family(T: pd.DataFrame, xs=XS, ys=YS, n_perm: int = 20000, seed: int = 0, drop_years=(), n_min: int = 6) -> pd.DataFrame:
    """Spearman rho, permutation p and BH q over the pre-specified family (every x with every y), on the years left after `drop_years`, with the leave-one-year-out range of rho."""
    rng = np.random.default_rng(seed)
    rows = []
    for x in xs:
        for y in ys:
            d = T.drop(index=[t for t in drop_years if t in T.index])[[x, y]].dropna()
            if len(d) < n_min:
                continue
            rho, p = spearman_perm(d[x].to_numpy(), d[y].to_numpy(), n_perm, rng)
            loo = [stats.spearmanr(d[x].drop(i), d[y].drop(i))[0] for i in d.index]
            rows.append({"x": x, "y": y, "n": len(d), "rho": rho, "p": p, "loo_min": float(np.nanmin(loo)), "loo_max": float(np.nanmax(loo)), "loo_sign_flips": int(sum(np.sign(r) != np.sign(rho) for r in loo))})
    return corr.fdr(pd.DataFrame(rows))


def year_shuffle_null(T: pd.DataFrame, xs=XS, ys=YS, n_shuffle: int = 300, n_perm: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Calibration: shuffle every outcome across the years, rerun the family and count the BH discoveries and the share of p < 0.05 (should be about 0 and 0.05)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_shuffle):
        S = T.copy()
        for y in ys:
            S[y] = rng.permutation(T[y].to_numpy())
        t = link_family(S, xs, ys, n_perm=n_perm, seed=k)
        rows.append({"shuffle": k, "discoveries": int(t.significant.sum()), "share_p_below_005": float((t.p < 0.05).mean()), "n_tests": len(t)})
    return pd.DataFrame(rows)


def lead_placebo(T: pd.DataFrame, xs=XS, ys=YS, n_perm: int = 20000, seed: int = 0) -> pd.DataFrame:
    """The same family against the outcomes of the previous year (t-1), which the manovra cannot have caused: a clear association here would point to persistence or common shocks, not to the manovra."""
    P = T.drop(columns=ys).join(T[[f"{y}_prev" for y in ys]].rename(columns=lambda c: c.removesuffix("_prev")))
    return link_family(P, xs, ys, n_perm=n_perm, seed=seed)


def plan_vs_outturn(T: pd.DataFrame, n_perm: int = 20000, seed: int = 0) -> pd.DataFrame:
    """How well the programmatic target anticipated the direction and size of the change in the deficit: Spearman between the change the plan implied (target minus last year's outturn) and the realised change, all years and without 2020-21.
    A check on the targets, not on the manovra: the change implied by the plan is mostly the government's forecast of the cycle and of the baseline."""
    rows = []
    for name, drop in (("all years", ()), ("without 2020-21", (2020, 2021))):
        d = T.drop(index=[t for t in drop if t in T.index])[["planned_delta_deficit", "delta_deficit"]].dropna()
        rho, p = spearman_perm(d.planned_delta_deficit.to_numpy(), d.delta_deficit.to_numpy(), n_perm, np.random.default_rng(seed))
        rows.append({"years": name, "n": len(d), "rho": rho, "p": p, "same_sign": int((np.sign(d.planned_delta_deficit) == np.sign(d.delta_deficit)).sum())})
    return pd.DataFrame(rows)


def outturn_components(panel: pd.DataFrame, spending: pd.DataFrame | None = None, iso3: str = "ITA") -> pd.DataFrame:
    """Outturn by calendar year (Eurostat, latest vintage), % of GDP: revenue (TR), expenditure (TE), interest (D41PAY), primary expenditure (TE - interest); nominal GDP (EUR million) and the growth of real and nominal GDP (%)."""
    sp = load_spending() if spending is None else spending
    x = sp[(sp.iso3 == iso3) & (sp.kind == "nature") & sp["item"].isin(["TR", "TE", "D41PAY"])].pivot_table(index="year", columns="item", values="value")
    c = pd.DataFrame({"revenue": x.TR, "expenditure": x.TE, "interest": x.D41PAY})
    c["primary_expenditure"] = c.expenditure - c.interest
    c["nominal_gdp"] = wide(panel, "gdp_nominal_meur")[iso3]
    c["real_growth"] = wide(panel, "gdp_growth")[iso3]
    c["nominal_growth"] = c.nominal_gdp.pct_change(fill_method=None) * 100
    return c


def target_components(path: Path = TARGETS_CSV) -> pd.DataFrame:
    """Revenue, expenditure and interest targets (% of GDP), the forecast growth of real and nominal GDP and the forecast nominal GDP level (EUR million) of the budget year, from the draft budgetary plans (2016-2025).
    The forecast level is the plan's level of year t-2 grown by its forecast growth of t-1 and t."""
    t = pd.read_csv(path).set_index("budget_year")
    d = t[["revenue_target", "expenditure_target", "interest_target", "real_growth_forecast", "nominal_growth_forecast", *[c for c in t if c.startswith("target_D") or c in ("target_P2", "target_P51G", "target_other")]]].copy()
    d["primary_expenditure_target"] = d.expenditure_target - d.interest_target
    d["nominal_gdp_forecast"] = t.nominal_gdp_level_tm2 * (1 + t.nominal_growth_tm1 / 100) * (1 + t.nominal_growth_forecast / 100)
    return d


def decompose_surprise(T: pd.DataFrame, comp: pd.DataFrame, tc: pd.DataFrame) -> pd.DataFrame:
    """Split the deficit surprise (outturn minus target, pp of GDP, positive = larger deficit) into the misses of revenue, primary expenditure and interest measured as *levels* (EUR, expressed in % of the outturn GDP) and a denominator effect.
    With r = forecast nominal GDP / outturn nominal GDP and a ratio k: k_out - k_target = (k_out - k_target * r) + k_target * (r - 1). The first term is the level miss, the second moves with the GDP forecast error alone (a GDP above the forecast
    shrinks every ratio). The deficit target in the identity is revenue-target minus expenditure-target, so what is left over is the rounding of the published figures. Also the forecast errors of real and nominal GDP growth (outturn minus forecast)."""
    d = T[["deficit", "target", "deficit_surprise"]].join(tc).join(comp, how="left")
    d["gdp_ratio"] = d.nominal_gdp_forecast / d.nominal_gdp  # > 1: GDP came in below the forecast
    for k, tgt in (("revenue", "revenue_target"), ("primary_expenditure", "primary_expenditure_target"), ("interest", "interest_target")):
        d[f"{k}_level_miss"] = d[k] - d[tgt] * d.gdp_ratio
    d["revenue_shortfall"] = -d.revenue_level_miss
    d["denominator_effect"] = (d.primary_expenditure_target + d.interest_target - d.revenue_target) * (d.gdp_ratio - 1)
    d["components_sum"] = d.revenue_shortfall + d.primary_expenditure_level_miss + d.interest_level_miss + d.denominator_effect
    d["rounding"] = d.deficit_surprise - d.components_sum
    d["real_growth_error"] = d.real_growth - d.real_growth_forecast
    d["nominal_growth_error"] = d.nominal_growth - d.nominal_growth_forecast
    return d.rename(columns={"primary_expenditure_level_miss": "primary_expenditure_overrun", "interest_level_miss": "interest_overrun"})[["deficit", "target", "deficit_surprise", "revenue_shortfall", "primary_expenditure_overrun", "interest_overrun", "denominator_effect", "rounding", "gdp_ratio", "real_growth_forecast", "real_growth", "real_growth_error", "nominal_growth_forecast", "nominal_growth", "nominal_growth_error"]]


def surprise_drivers(D: pd.DataFrame, n_perm: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Spearman (permutation p) of the deficit surprise and its parts against the growth errors, all years and without 2020-21. Descriptive: ten years, and the parts are not independent of each other (the denominator effect is
    mechanically tied to the nominal GDP error)."""
    rng = np.random.default_rng(seed)
    pairs = [("deficit_surprise", "real_growth_error"), ("deficit_surprise", "nominal_growth_error"), ("revenue_shortfall", "nominal_growth_error"), ("primary_expenditure_overrun", "nominal_growth_error"),
             ("interest_overrun", "nominal_growth_error"), ("denominator_effect", "nominal_growth_error")]
    rows = []
    for name, drop in (("all years", ()), ("without 2020-21", (2020, 2021))):
        E = D.drop(index=[t for t in drop if t in D.index])
        for x, y in pairs:
            d = E[[x, y]].dropna()
            rho, p = spearman_perm(d[x].to_numpy(), d[y].to_numpy(), n_perm, rng)
            rows.append({"years": name, "x": x, "y": y, "n": len(d), "rho": rho, "p": p})
    return pd.DataFrame(rows)


# expenditure categories of the draft budgetary plans (rows 2.1-2.8 of the target table) and their Eurostat counterparts (`gov_10a_main`, % of GDP)
EXP_CATEGORIES = {"compensation of employees": ("D1PAY", "target_D1"), "intermediate consumption": ("P2", "target_P2"), "social payments": ("D62_D632PAY", "target_D62_D632"), "interest": ("D41PAY", "interest_target"),
                  "subsidies": ("D3PAY", "target_D3"), "gross fixed capital formation": ("P51G", "target_P51G"), "capital transfers": ("D9PAY", "target_D9"), "other": (None, "target_other")}


def expenditure_misses(T: pd.DataFrame, tc: pd.DataFrame, comp: pd.DataFrame, spending: pd.DataFrame | None = None, iso3: str = "ITA") -> pd.DataFrame:
    """Where the expenditure miss came from: for each category of the draft budgetary plans, outturn minus target as a euro-level miss in % of the outturn GDP (outturn ratio minus target ratio times forecast / outturn nominal GDP, as in
    `decompose_surprise`; positive = more spending than planned). `other` is the outturn total expenditure minus the other categories, like the plan's residual row 2.8. Index = budget year; the last column is the sum, which equals the
    miss of total expenditure (primary expenditure plus interest) up to the rounding of the published targets."""
    sp = load_spending() if spending is None else spending
    x = sp[(sp.iso3 == iso3) & (sp.kind == "nature")].pivot_table(index="year", columns="item", values="value")
    r = (tc.nominal_gdp_forecast / comp.nominal_gdp.reindex(tc.index)).reindex(T.index)
    out = pd.DataFrame(index=T.index)
    named = [c for c, _ in (v for v in EXP_CATEGORIES.values()) if c]
    for name, (item, tgt) in EXP_CATEGORIES.items():
        ratio_out = x[item] if item else x.TE - x[named].sum(axis=1)
        out[name] = ratio_out.reindex(T.index) - tc[tgt].reindex(T.index) * r
    out["total"] = out.sum(axis=1)
    out["total_expenditure_miss"] = x.TE.reindex(T.index) - tc.expenditure_target.reindex(T.index) * r
    return out
