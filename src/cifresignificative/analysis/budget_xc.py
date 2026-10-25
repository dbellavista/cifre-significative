"""Italy's deficit surprise against the Commission's autumn forecast, ranked among the EU27 (placebo ranking for notebook 38).

For each country and budget year B (2017-2025) the **surprise** is the outturn (Eurostat, latest vintage) minus the one-year-ahead forecast of the Commission's autumn forecast of B-1 (`sources/ec_forecasts.py`), for net borrowing and its parts, in % of GDP,
positive = a larger deficit, a revenue shortfall or more spending than forecast. The autumn forecast comes after the draft budgetary plans and normally includes their measures, so it is the outside view of the plan, not the government's own target; for Italy the two are put side by side.
The ratios are in % of the outturn GDP on the outturn side and of the forecast GDP on the forecast side, so a GDP error moves every part (the Italian decomposition in `budget.py` separates this term, here the forecast GDP levels are not parsed).
Italy is one country: the claim is a **rank** (share of the 27 members at least as extreme as Italy, p = rank / 27, one-sided, with Italy among them), not a clustered test. Two measures: the raw surprise and the **idiosyncratic** one, minus the median of the other 26 members in the same
year (the common shock of 2020-23 moves everyone). Per-country sign-flip tests of the mean idiosyncratic surprise with BH over the 27 countries, and a sign-randomised null (should give about 0 discoveries). Descriptive, never causal.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from cifresignificative.analysis import corr
from cifresignificative.analysis.data import EU27_ISO3, load_spending, wide
from cifresignificative.sources.ec_forecasts import CSV as FORECAST_CSV

YEARS = list(range(2016, 2026))  # 2016 comes from the autumn 2015 annex: a sensitivity window only, the primary windows start in 2017
# windows pre-specified before the ranks were computed: the whole period without the pandemic year, and the years of the building credits in Italy
WINDOWS = {"2017-25 without 2020": [y for y in YEARS if y >= 2017 and y != 2020], "2021-23": [2021, 2022, 2023], "2017-19": [2017, 2018, 2019], "2024-25": [2024, 2025], "2016-25 without 2020": [y for y in YEARS if y != 2020]}
PRIMARY = ["2017-25 without 2020", "2021-23"]
COMPONENTS = ["deficit", "revenue_shortfall", "primary_expenditure_overrun", "interest_overrun"]
EURO_AREA_2025 = [c for c in EU27_ISO3 if c not in ("BGR", "CZE", "DNK", "HUN", "POL", "ROU", "SWE")]  # 20 euro-area members at the end of 2025 (Bulgaria joined in 2026)


def load_forecasts(path: Path = FORECAST_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def forecast_wide(F: pd.DataFrame, variable: str, lead: int = 1) -> pd.DataFrame:
    """Budget year x country: the value for year B in the autumn forecast of B - lead (lead 1 = the forecast made when the budget is prepared)."""
    d = F[(F.variable == variable) & (F.vintage + lead == F.year)]
    return d.pivot(index="year", columns="iso3", values="value").sort_index()


def outturn_wide(panel: pd.DataFrame, spending: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Eurostat outturns by year and country, % of GDP: net lending (B9), revenue, expenditure, interest."""
    sp = load_spending() if spending is None else spending
    x = sp[(sp.kind == "nature") & sp["item"].isin(["TR", "TE", "D41PAY"])]
    g = {k: x[x["item"] == k].pivot(index="year", columns="iso3", values="value").sort_index() for k in ("TR", "TE", "D41PAY")}
    return {"net_lending": wide(panel, "gov_balance_pct_gdp"), "revenue": g["TR"], "expenditure": g["TE"], "interest": g["D41PAY"]}


def surprises(F: pd.DataFrame, O: dict[str, pd.DataFrame], lead: int = 1, countries: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Year x country surprises (outturn minus forecast, positive = worse): `deficit`, `revenue_shortfall`, `primary_expenditure_overrun`, `interest_overrun`. The first equals the sum of the other three (up to rounding of the published figures)."""
    cs = EU27_ISO3 if countries is None else countries
    f = {k: forecast_wide(F, k, lead) for k in ("net_lending", "revenue", "expenditure", "interest")}
    def at(d: pd.DataFrame) -> pd.DataFrame:
        return d.reindex(index=YEARS, columns=cs)
    fo = {k: at(v) for k, v in f.items()}
    oo = {k: at(v) for k, v in O.items()}
    return {"deficit": fo["net_lending"] - oo["net_lending"], "revenue_shortfall": fo["revenue"] - oo["revenue"],
            "primary_expenditure_overrun": (oo["expenditure"] - oo["interest"]) - (fo["expenditure"] - fo["interest"]), "interest_overrun": oo["interest"] - fo["interest"]}


def idiosyncratic(S: pd.DataFrame) -> pd.DataFrame:
    """Surprise minus the median of the *other* countries in the same year (leave-one-out, so Italy does not move its own benchmark)."""
    out = S.copy()
    for c in S.columns:
        out[c] = S[c] - S.drop(columns=c).median(axis=1)
    return out


def window_mean(S: pd.DataFrame, years: list[int]) -> pd.Series:
    """Mean over the window; countries missing any year of it are dropped (no partial means)."""
    w = S.loc[years]
    return w.mean().where(w.notna().all())


def italy_rank(S: pd.DataFrame, years: list[int], iso3: str = "ITA", absolute: bool = False) -> dict:
    """Rank of Italy's window mean among all countries (1 = largest overrun, or largest absolute miss), the number of countries and the one-sided placebo p = rank / N (ties count against Italy)."""
    m = window_mean(S, years).dropna()
    m = m.abs() if absolute else m
    rank = int((m >= m[iso3] - 1e-12).sum())
    return {"value": float(m[iso3]), "rank": rank, "n": len(m), "p": rank / len(m), "median_others": float(m.drop(iso3).median())}


def yearly_ranks(S: dict[str, pd.DataFrame], iso3: str = "ITA") -> pd.DataFrame:
    """Italy's rank in each budget year for each component (1 = the largest overrun among the countries with data), raw and idiosyncratic: shows which years carry a window result."""
    rows = []
    for comp, s in S.items():
        for measure, X in (("raw", s), ("idiosyncratic", idiosyncratic(s))):
            for y in X.index:
                r = X.loc[y].dropna()
                if iso3 in r:
                    rows.append({"component": comp, "measure": measure, "year": y, "value": float(r[iso3]), "rank": int((r >= r[iso3] - 1e-12).sum()), "n": len(r)})
    return pd.DataFrame(rows)


def lead_sensitivity(F: pd.DataFrame, O: dict[str, pd.DataFrame], leads=(0, 1, 2), years: list[int] | None = None, iso3: str = "ITA") -> pd.DataFrame:
    """The deficit result against the autumn forecast of the same year (lead 0, made after the budget is known), of the year before (lead 1, the primary one) and of two years before (lead 2), on the years every lead has (2018-25) so that only
    the horizon changes: Italy's rank for the window 2021-23 and for the period without 2020."""
    ys = list(range(2018, 2026)) if years is None else years
    rows = []
    for lead in leads:
        S = surprises(F, O, lead)["deficit"]
        for name, w in (("2021-23", [2021, 2022, 2023]), (f"{ys[0]}-{ys[-1]} without 2020", [y for y in ys if y != 2020])):
            for measure, X in (("raw", S), ("idiosyncratic", idiosyncratic(S))):
                rows.append({"lead": lead, "window": name, "measure": measure, **italy_rank(X, w, iso3)})
    return pd.DataFrame(rows)


def rank_table(S: dict[str, pd.DataFrame], windows: dict[str, list[int]] = WINDOWS, iso3: str = "ITA", countries: list[str] | None = None) -> pd.DataFrame:
    """Italy's rank for every component, window and measure (raw, idiosyncratic) with BH over the primary family (the two primary windows, all components, both measures); the other windows are reported without a q."""
    rows = []
    for comp, s in S.items():
        s = s if countries is None else s[countries]
        for measure, X in (("raw", s), ("idiosyncratic", idiosyncratic(s))):
            for wname, yrs in windows.items():
                rows.append({"component": comp, "measure": measure, "window": wname, "primary": wname in PRIMARY, **italy_rank(X, yrs, iso3)})
    d = pd.DataFrame(rows)
    prim = corr.fdr(d[d.primary].reset_index(drop=True))
    return d.merge(prim[["component", "measure", "window", "q_value"]], how="left", on=["component", "measure", "window"])


def _screen(X: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    """Mean per column and the two-sided sign-flip p (each year's value flipped at random, exact under a symmetric null around 0), BH over columns."""
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(X)))
    rows = []
    for c in X.columns:
        v = X[c].to_numpy()
        mean = float(v.mean())
        rows.append({"iso3": c, "mean": mean, "p": float((1 + (np.abs((signs @ v) / len(v)) >= abs(mean) - 1e-12).sum()) / (n_perm + 1))})
    return corr.fdr(pd.DataFrame(rows))


def country_sign_flip(S: pd.DataFrame, years: list[int], n_perm: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Per country: mean idiosyncratic surprise over `years` and the sign-flip p, BH over countries. Years are not independent (a revision of the outturn repeats in several years), so the p is an approximation: it ranks countries,
    it is not a calibrated claim for one of them."""
    return _screen(idiosyncratic(S).loc[years].dropna(axis=1), n_perm, seed).sort_values("mean", ascending=False, ignore_index=True)


def sign_null(S: pd.DataFrame, years: list[int], n_shuffle: int = 200, n_perm: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Calibration of the country screen: multiply every idiosyncratic value by a random sign (a symmetric null by construction), rerun the screen and count the BH discoveries (should be about 0) and the share of p < 0.05 (about 0.05)."""
    rng = np.random.default_rng(seed)
    X = idiosyncratic(S).loc[years].dropna(axis=1)
    rows = []
    for k in range(n_shuffle):
        t = _screen(X * rng.choice([-1.0, 1.0], size=X.shape), n_perm, k)
        rows.append({"shuffle": k, "discoveries": int(t.significant.sum()), "share_p_below_005": float((t.p < 0.05).mean())})
    return pd.DataFrame(rows)


def first_estimate_sensitivity(F: pd.DataFrame, O: dict[str, pd.DataFrame], iso3: str = "ITA") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The deficit result with an early outturn instead of the latest Eurostat vintage: the estimate for the year just ended in the autumn forecast of B+1 (the Commission's own figure after the April notification, before later revisions of the accounts; budget years 2017-2024,
    the 2026 annex is not in the table). Returns (ranks, revisions): Italy's rank for 2021-23 and for 2017-24 without 2020, raw and idiosyncratic, with each outturn; and Italy's revision (latest minus early estimate of the deficit, % of GDP, positive = a larger deficit after revision)
    next to the median revision of the others, by year. A revision that is large for Italy in 2021 is the accounting treatment of the building credits, not a forecast error."""
    ys = [y for y in YEARS if 2017 <= y <= 2024]
    f, early = forecast_wide(F, "net_lending", 1).reindex(index=ys, columns=EU27_ISO3), forecast_wide(F, "net_lending", -1).reindex(index=ys, columns=EU27_ISO3)
    late = O["net_lending"].reindex(index=ys, columns=EU27_ISO3)
    rows = []
    for outturn, name in ((early, "early estimate"), (late, "latest vintage")):
        S = f - outturn
        for wname, w in (("2021-23", [2021, 2022, 2023]), ("2017-24 without 2020", [y for y in ys if y != 2020])):
            for measure, X in (("raw", S), ("idiosyncratic", idiosyncratic(S))):
                rows.append({"outturn": name, "window": wname, "measure": measure, **italy_rank(X, w, iso3)})
    rev = early - late  # deficit revision = (-late) - (-early) = early - late
    rev = rev.rename_axis("year")
    return pd.DataFrame(rows), pd.DataFrame({"italy": rev[iso3], "median_others": rev.drop(columns=iso3).median(axis=1), "max_others": rev.drop(columns=iso3).max(axis=1)}).reset_index()


def gdp_ratio(F: pd.DataFrame, gdp_nominal: pd.DataFrame) -> pd.DataFrame:
    """Forecast over outturn nominal GDP for budget year B, year x country. The forecast level is built from the latest outturn of B-2 and the nominal growth of the autumn forecast of B-1 for B-1 and B (real growth and GDP deflator compounded);
    the outturn is the latest Eurostat vintage (current prices). Above 1 = the forecast expected a larger economy than came out, which makes the forecast ratios smaller than they would be at the outturn GDP."""
    def nominal(lead: int) -> pd.DataFrame:  # nominal growth of year B-lead+... : lead 1 = year B, lead 0 = year B-1, both read in the autumn forecast of B-1
        g = {k: forecast_wide(F, k, lead) for k in ("real_growth", "deflator")}
        return ((1 + g["real_growth"] / 100) * (1 + g["deflator"] / 100)).reindex(index=[y - 1 + lead for y in YEARS], columns=EU27_ISO3).set_axis(YEARS, axis=0)
    n_prev, n_b = nominal(0), nominal(1)
    base = gdp_nominal.reindex(index=[y - 2 for y in YEARS], columns=EU27_ISO3).set_axis(YEARS, axis=0)
    return base * n_prev * n_b / gdp_nominal.reindex(index=YEARS, columns=EU27_ISO3)


def denominator_split(S_deficit: pd.DataFrame, F: pd.DataFrame, gdp_ratio_: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the deficit surprise (pp of GDP) into a denominator term and the rest: surprise = (D_out - D_fc) / G_out + D_fc/G_fc * (G_fc/G_out - 1), D the deficit in euro. The second term, `denominator`, is the part due to the GDP forecast error alone
    (positive when GDP came out below the forecast while the forecast deficit was positive, or the reverse for a surplus); `level` is the rest: the miss of the deficit in euro, in % of the outturn GDP."""
    d_fc = -forecast_wide(F, "net_lending", 1).reindex(index=YEARS, columns=S_deficit.columns)
    den = d_fc * (gdp_ratio_.reindex(columns=S_deficit.columns) - 1)
    return {"denominator": den, "level": S_deficit - den}


def italy_two_views(S_deficit: pd.DataFrame, net_lending: pd.DataFrame, targets: pd.DataFrame, iso3: str = "ITA") -> pd.DataFrame:
    """Italy's miss against the Commission's autumn forecast next to the miss against the government's own target (`budget.targets`, % of GDP, positive = a larger deficit than planned), by budget year, with the two references
    (`commission_deficit` = outturn deficit minus the surprise; `government_deficit` = the target as proposed) and their gap (positive = the Commission expected a larger deficit than the government planned)."""
    out = pd.DataFrame({"outturn_deficit": -net_lending[iso3].reindex(YEARS), "commission_surprise": S_deficit[iso3]})
    out["commission_deficit"] = out.outturn_deficit - out.commission_surprise
    out["government_deficit"] = targets["target"].reindex(YEARS)
    out["government_surprise"] = out.outturn_deficit - out.government_deficit
    out["reference_gap"] = out.commission_deficit - out.government_deficit
    return out


DBP_CSV = "data/external/dbp_targets.csv"
DBP_WINDOWS = {"2018-25 without 2020": [y for y in range(2018, 2026) if y != 2020], "2021-23": [2021, 2022, 2023]}
DBP_MIN_YEARS = {"2018-25 without 2020": 4, "2021-23": 2}  # the plans parsed cover 4-9 countries a year: a window mean needs at least this many years of a country


def dbp_plans(F: pd.DataFrame, italy_targets: pd.DataFrame, csv: str = DBP_CSV, max_gap: float = 2.0) -> pd.DataFrame:
    """Year x country net lending planned in the October draft budgetary plan (% of GDP, negative = deficit), `sources/dbp_euro.py`, arithmetic-verified rows only (`identity_*`; label-only rows are left out). A plausibility filter drops an identity-only row that differs from the Commission's autumn forecast
    of the same autumn by more than `max_gap` points (a column misread: Malta, Lithuania and Greece 2021 differ by 2.4 to 6.3 points; the rows also found in an S.13 row are kept). Italy's own plans come from the parsed plans where available and from `deficit_targets.csv` otherwise (2022)."""
    d = pd.read_csv(csv)
    d = d[d.evidence.str.startswith("identity")]
    fc = forecast_wide(F, "net_lending", 1)
    d = d.assign(forecast=[fc.loc[y, c] if y in fc.index and c in fc.columns else np.nan for y, c in zip(d.budget_year, d.iso3, strict=True)])
    d = d[(d.evidence == "identity_and_subsector_row") | ((d.target_pct_gdp - d.forecast).abs() <= max_gap)]
    P = d.pivot(index="budget_year", columns="iso3", values="target_pct_gdp").reindex(index=YEARS)
    for y, v in (-italy_targets.target).items():
        if y in P.index and pd.isna(P.loc[y, "ITA"]):
            P.loc[y, ["ITA"]] = v
    return P


def dbp_surprises(P: pd.DataFrame, O: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Planned minus outturn net lending (positive = a larger deficit than the government's own plan)."""
    return P - O["net_lending"].reindex(index=P.index, columns=P.columns)


def window_mean_min(S: pd.DataFrame, years: list[int], min_years: int) -> pd.Series:
    w = S.loc[years]
    return w.mean().where(w.notna().sum() >= min_years)


def dbp_rank_table(S: pd.DataFrame, comm: pd.DataFrame, windows: dict[str, list[int]] = DBP_WINDOWS, iso3: str = "ITA") -> pd.DataFrame:
    """Italy's rank (1 = the largest overrun) among the countries with a window mean from the plans (at least `DBP_MIN_YEARS`), raw and idiosyncratic (minus the median of the others in the same year, within the sample), next to the same ranks of the Commission-based surprise `comm` for exactly
    the same country-years (the plans' sample is thin: this separates the effect of the sample from that of the reference). One-sided p = rank / N, ties against Italy."""
    rows = []
    mask = S.notna()
    for wn, yrs in windows.items():
        for ref, X in (("own plan", S), ("Commission forecast, same country-years", comm.reindex(index=S.index, columns=S.columns).where(mask))):
            for measure, Z in (("raw", X), ("idiosyncratic", idiosyncratic(X))):
                m = window_mean_min(Z, yrs, DBP_MIN_YEARS[wn]).dropna()
                if iso3 not in m:
                    continue
                rank = int((m >= m[iso3] - 1e-12).sum())
                rows.append({"window": wn, "reference": ref, "measure": measure, "italy": float(m[iso3]), "rank": rank, "n": len(m), "p": rank / len(m), "median_others": float(m.drop(iso3).median()), "countries": ",".join(m.index)})
    return pd.DataFrame(rows)
