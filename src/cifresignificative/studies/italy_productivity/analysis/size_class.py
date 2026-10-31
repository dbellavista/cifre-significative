"""Firm size and productivity: Italy's employment by enterprise size class and apparent labour productivity (value added per person employed) by class, against the other EU27 members.

Source: Eurostat structural business statistics by employment size class (`build.build_size`, `data/processed/size_class.parquet`). Two vintages that are never combined: `old` (sbs_sc_sca_r2, 2005-2020, value added at
factor cost, business economy B-N without K) and `new` (sbs_sc_ovw, 2021 onwards, basic prices, wider scope). A (country, year) enters only when all five disjoint classes (0-9, 10-19, 20-49, 50-249, 250+) have persons
and value added. Euro levels are not compared across countries (no price-level index in the panel): every statistic is a share, a within-country ratio, or a shift-share of the country's own productivity.

Statistics: micro and large employment shares; productivity of a class relative to the country's total; `gradient` = log(productivity of 250+ over 0-9); `mix_effect` = log of the country's productivity recomputed with the
median size mix of the other members over its actual one (positive: it would be more productive with the others' typical mix). Italy is judged by its rank among the countries (placebo ranking, one cluster), never by a test.
The composition identity splits the micro share into the sector mix and the micro share within each sector. Descriptive only: size and productivity are jointly determined, nothing here says which causes which."""
import numpy as np
import pandas as pd

from cifresignificative.studies.italy_productivity.analysis import corr
from cifresignificative.studies.italy_productivity.analysis.data import (
    EU27_ISO3,
    ITALY,
    PROCESSED,
    load_panel,
    wide,
)

CLASSES = ["0-9", "10-19", "20-49", "50-249", "GE250"]
ECONOMY = {"old": "B-N_S95_X_K", "new": "B-S_X_O_S94"}
SECTIONS = {"C": "manufacturing", "F": "construction", "G": "trade", "H": "transport and storage", "I": "accommodation and food", "J": "information and communication", "L": "real estate", "M": "professional services", "N": "administrative services"}
STATS = ["micro_share", "large_share", "micro_lp_rel", "large_lp_rel", "gradient", "mix_effect", "mean_firm_size"]
PRIMARY = ("old", 2019)  # the main snapshot: last year before the pandemic with the most countries (23 members)
MIN_COUNTRIES = 12
PEERS = ["DEU", "FRA", "ESP", "NLD"]  # larger members with size-class data in the main year (Portugal and Austria are missing)


def load_size() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "size_class.parquet")


def class_table(size: pd.DataFrame, vintage: str, nace: str | None = None) -> pd.DataFrame:
    """Long table (iso3, year, size_emp, persons, va_meur, enterprises) of the EU27 members for one vintage and one NACE code (default: the economy aggregate of the vintage); only (iso3, year) with all five classes."""
    d = size[(size.vintage == vintage) & (size.nace_r2 == (nace or ECONOMY[vintage])) & size.iso3.isin(EU27_ISO3) & size.size_emp.isin(CLASSES)].dropna(subset=["persons", "va_meur"])
    n = d.groupby(["iso3", "year"]).size()
    keep = n[n == len(CLASSES)].index
    return d.set_index(["iso3", "year"]).loc[keep].reset_index()


def others_mix(S: pd.DataFrame, how: str = "median") -> pd.DataFrame:
    """For each row of S (iso3 x class shares of one year), the median (or mean) share by class of the OTHER rows, renormalised to sum to 1."""
    rows = {}
    for c in S.index:
        o = S.drop(index=c)
        m = o.median() if how == "median" else o.mean()
        rows[c] = m / m.sum()
    return pd.DataFrame(rows).T


def mix_effect(S: pd.DataFrame, LP: pd.DataFrame, how: str = "median") -> pd.Series:
    """log(sum_k typical_share_k x LP_k / sum_k share_k x LP_k) per country of one year: the productivity change of the country's own class productivities under the other members' typical size mix."""
    typ = others_mix(S, how)
    return np.log((typ * LP).sum(axis=1, skipna=False) / (S * LP).sum(axis=1, skipna=False))  # NaN when a class is empty (no productivity to apply)


def structure(t: pd.DataFrame, how: str = "median") -> pd.DataFrame:
    """One row per (iso3, year) with the `STATS`, `lp_total_keur` (euro level, for reading only: not compared across countries) and the five class shares `s_<class>`."""
    key = ["iso3", "year"]
    P = t.pivot_table(index=key, columns="size_emp", values="persons")[CLASSES]
    V = t.pivot_table(index=key, columns="size_emp", values="va_meur")[CLASSES]
    E = t.pivot_table(index=key, columns="size_emp", values="enterprises")[CLASSES]
    S = P.div(P.sum(axis=1), axis=0)
    LP = (V.div(P) * 1000).where((P > 0) & (V > 0))  # an empty class (0 persons, 0 value added) has no productivity
    tot = V.sum(axis=1) / P.sum(axis=1) * 1000
    out = pd.DataFrame({"micro_share": S["0-9"], "large_share": S["GE250"], "micro_lp_rel": LP["0-9"] / tot, "large_lp_rel": LP["GE250"] / tot, "gradient": np.log(LP["GE250"] / LP["0-9"]),
                        "mean_firm_size": P.sum(axis=1) / E.sum(axis=1).where(E.notna().all(axis=1)), "lp_total_keur": tot,
                        "broken": t.assign(b=t.flag.fillna("").str.contains("b")).groupby(key).b.any()})
    years = S.index.get_level_values("year").unique()
    mix = pd.concat({y: mix_effect(S.xs(y, level="year"), LP.xs(y, level="year"), how) for y in years}, names=["year", "iso3"])
    out["mix_effect"] = mix.reorder_levels(key).reindex(out.index)
    for c in CLASSES:
        out[f"s_{c}"] = S[c]
    return out.reset_index()


def position(st: pd.DataFrame, year: int, stats: list[str] = STATS, iso3: str = ITALY) -> pd.DataFrame:
    """Italy's value, the median of the other members, Italy's rank (1 = highest) among the countries with a value, and the two-sided placebo p = 2 min(rank from the top, rank from the bottom) / n (capped at 1).
    `italy_break` / `n_break`: Italy's / the number of countries' values flagged as a break in series (Eurostat flag `b`); a rank compares countries within one year, so it is not a trend."""
    rows = []
    d = st[st.year == year].set_index("iso3")
    for s in stats:
        v = d[s].dropna()
        if iso3 not in v.index or len(v) < MIN_COUNTRIES:
            continue
        top = int((v >= v[iso3] - 1e-12).sum())
        bottom = int((v <= v[iso3] + 1e-12).sum())
        rows.append({"statistic": s, "year": year, "n": len(v), "italy": float(v[iso3]), "median_others": float(v.drop(iso3).median()), "rank": top, "p": min(1.0, 2 * min(top, bottom) / len(v)),
                     "italy_break": bool(d.broken[iso3]), "n_break": int(d.broken.reindex(v.index).sum())})
    return pd.DataFrame(rows)


def position_all(size: pd.DataFrame, primary: tuple[str, int] = PRIMARY) -> pd.DataFrame:
    """Italy's position for every vintage, year and statistic. BH q over the primary family (the main snapshot, all statistics); other years carry no q."""
    out = []
    for vintage in ("old", "new"):
        st = structure(class_table(size, vintage))
        for y in sorted(st[st.iso3 == ITALY].year.unique()):
            out.append(position(st, int(y)).assign(vintage=vintage, primary=(vintage, int(y)) == primary))
    d = pd.concat(out, ignore_index=True)
    prim = corr.fdr(d[d.primary].reset_index(drop=True))[["statistic", "q_value"]]
    return d.merge(prim, how="left", on="statistic").assign(q_value=lambda x: x.q_value.where(x.primary))


def sector_position(size: pd.DataFrame, vintage: str = "old", year: int = 2019, sections: dict[str, str] = SECTIONS) -> pd.DataFrame:
    """Position of Italy (`position`) in each NACE section with at least MIN_COUNTRIES members that have all five classes in that section."""
    out = []
    for code, label in sections.items():
        st = structure(class_table(size, vintage, code))
        p = position(st, year, [s for s in STATS if s != "mean_firm_size"])
        if not p.empty:
            out.append(p.assign(nace_r2=code, label=label, vintage=vintage))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def composition(size: pd.DataFrame, vintage: str = "old", year: int = 2019, sections: list[str] | None = None, how: str = "median") -> pd.DataFrame:
    """Per country: micro share of employment over `sections`, split exactly into `composition` (actual minus the same within-sector micro shares under the other members' typical sector mix: sector weights)
    and `within` (those within-sector micro shares under the typical mix minus the typical micro shares of the other members' sectors under the typical mix): micro - reference = composition + within.
    Only countries with persons for the total and the 0-9 class in every section."""
    sections = sections or list(SECTIONS)
    d = size[(size.vintage == vintage) & (size.year == year) & size.nace_r2.isin(sections) & size.iso3.isin(EU27_ISO3) & size.size_emp.isin(["TOTAL", "0-9"])]
    P = d.pivot_table(index=["iso3", "nace_r2"], columns="size_emp", values="persons")
    P = P.dropna()
    ok = P.groupby("iso3").size()
    P = P.loc[ok[ok == len(sections)].index]
    W = P["TOTAL"].unstack("nace_r2")[sections]
    W = W.div(W.sum(axis=1), axis=0)
    M = (P["0-9"] / P["TOTAL"]).unstack("nace_r2")[sections]
    micro = (W * M).sum(axis=1)
    rows = []
    for c in W.index:
        wbar = (W.drop(index=c).median() if how == "median" else W.drop(index=c).mean())
        wbar = wbar / wbar.sum()
        mbar = M.drop(index=c).median() if how == "median" else M.drop(index=c).mean()
        cf = (wbar * M.loc[c]).sum()
        ref = (wbar * mbar).sum()
        rows.append({"iso3": c, "micro_share": micro[c], "counterfactual": cf, "reference": ref, "composition": micro[c] - cf, "within": cf - ref, "total_gap": micro[c] - ref})
    return pd.DataFrame(rows)


def composition_rank(c: pd.DataFrame, iso3: str = ITALY) -> pd.DataFrame:
    """Italy's value, median of the others and rank (1 = highest) of the micro share, the composition and the within-sector terms."""
    rows = []
    d = c.set_index("iso3")
    for s in ("micro_share", "total_gap", "composition", "within"):
        v = d[s]
        top = int((v >= v[iso3] - 1e-12).sum())
        bottom = int((v <= v[iso3] + 1e-12).sum())
        rows.append({"statistic": s, "n": len(v), "italy": float(v[iso3]), "median_others": float(v.drop(iso3).median()), "rank": top, "p": min(1.0, 2 * min(top, bottom) / len(v))})
    return pd.DataFrame(rows)


def level_gap(size: pd.DataFrame, panel: pd.DataFrame, vintage: str = "old", year: int = 2019, nace: str | None = None, peers: list[str] = PEERS, iso3: str = ITALY) -> pd.DataFrame:
    """Value added per person (`head`) and per hour (`hour`) by size class, put on a common scale with the GDP price level index of the country-year (EUR at the EU27 price level) and, per hour, with the economy-wide annual hours per
    worker (AMECO `ameco_hours_per_worker`, assumed equal in every class): Italy's value, the median of the others, the ratio, Italy's rank (1 = highest) and the ratio to each peer. Approximate: the price index is GDP-wide, hours are
    not by class, and the persons are heads, so a country with much part-time work looks less productive per head. Read the ratios as orders of magnitude, not as estimates; Ireland's multinational-driven values stay in."""
    t = class_table(size, vintage, nace)
    t = t[t.year == year]
    pli = size[(size.year == year)].drop_duplicates("iso3").set_index("iso3").pli / 100
    hours = wide(panel, "ameco_hours_per_worker", EU27_ISO3).loc[year]
    P = t.pivot(index="iso3", columns="size_emp", values="persons")[CLASSES]
    V = t.pivot(index="iso3", columns="size_emp", values="va_meur")[CLASSES]
    head = (V / P * 1e6).where((P > 0) & (V > 0))
    head["total"] = V.sum(axis=1) / P.sum(axis=1) * 1e6
    rows = []
    for basis, X in (("head", head.div(pli.reindex(head.index), axis=0)), ("hour", head.div(pli.reindex(head.index), axis=0).div(hours.reindex(head.index), axis=0))):
        for cl in [*CLASSES, "total"]:
            v = X[cl].dropna()
            if iso3 not in v.index or len(v) < MIN_COUNTRIES:
                continue
            row = {"basis": basis, "size_emp": cl, "year": year, "n": len(v), "italy": float(v[iso3]), "median_others": float(v.drop(iso3).median()), "ratio_to_median": float(v[iso3] / v.drop(iso3).median()), "rank": int((v >= v[iso3] - 1e-12).sum())}
            row |= {f"ratio_{p}": float(v[iso3] / v[p]) if p in v.index else np.nan for p in peers}
            rows.append(row)
    return pd.DataFrame(rows)


def _adjusted(size: pd.DataFrame, panel: pd.DataFrame, vintage: str, year: int, nace: str | None, basis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shares S and productivity X (EUR at the EU27 price level, per head or per hour) by class of one year; classes with no persons or no value added give NaN in X."""
    t = class_table(size, vintage, nace)
    t = t[t.year == year]
    pli = size[size.year == year].drop_duplicates("iso3").set_index("iso3").pli / 100
    P = t.pivot(index="iso3", columns="size_emp", values="persons")[CLASSES]
    V = t.pivot(index="iso3", columns="size_emp", values="va_meur")[CLASSES]
    X = (V / P * 1e6).where((P > 0) & (V > 0)).div(pli.reindex(P.index), axis=0)
    if basis == "hour":
        X = X.div(wide(panel, "ameco_hours_per_worker", EU27_ISO3).loc[year].reindex(P.index), axis=0)
    return P.div(P.sum(axis=1), axis=0), X


def gap_decomposition(size: pd.DataFrame, panel: pd.DataFrame, vintage: str = "old", year: int = 2019, nace: str | None = None, basis: str = "hour", peers: list[str] = PEERS, iso3: str = ITALY) -> pd.DataFrame:
    """Exact split of the log productivity gap of Italy to a reference into `size_mix` and `within`: with s the class shares and X the class productivity (same scale), total = log(sum s_i X_i / sum s_r X_r) =
    size_mix + within, size_mix = log(sum s_i X_i / sum s_r X_i) (Italy's mix against the reference mix, at Italy's own productivities: negative = the mix costs Italy) and within = log(sum s_r X_i / sum s_r X_r) (Italy's class productivities
    against the reference's, at the reference mix). References: each of `peers` and `typical` (the other members' median shares and median class productivity: a synthetic benchmark, not a country). `basis` hour or head; same approximations as `level_gap`."""
    S, X = _adjusted(size, panel, vintage, year, nace, basis)
    ok = X.notna().all(axis=1)
    S, X = S[ok], X[ok]
    refs = {p: (S.loc[p], X.loc[p]) for p in peers if p in S.index}
    o = S.index != iso3
    typ = S[o].median()
    refs["typical"] = (typ / typ.sum(), X[o].median())
    rows = []
    for name, (sr, xr) in refs.items():
        own = (S.loc[iso3] * X.loc[iso3]).sum()
        at_ref_mix = (sr * X.loc[iso3]).sum()
        rows.append({"reference": name, "basis": basis, "year": year, "n": int(ok.sum()), "total": np.log(own / (sr * xr).sum()), "size_mix": np.log(own / at_ref_mix), "within": np.log(at_ref_mix / (sr * xr).sum())})
    return pd.DataFrame(rows)


def mix_leave_one_out(size: pd.DataFrame, vintage: str = "old", year: int = 2019, iso3: str = ITALY, how: str = "median") -> pd.DataFrame:
    """Italy's mix effect with each other member removed from the reference group in turn (and from nothing else): `dropped`, `mix_effect`. Shows whether one country (Greece, Slovakia) carries the reference."""
    t = class_table(size, vintage)
    t = t[t.year == year]
    P = t.pivot(index="iso3", columns="size_emp", values="persons")[CLASSES]
    V = t.pivot(index="iso3", columns="size_emp", values="va_meur")[CLASSES]
    S, LP = P.div(P.sum(axis=1), axis=0), (V.div(P) * 1000).where((P > 0) & (V > 0))
    rows = [{"dropped": c, "mix_effect": float(mix_effect(S.drop(index=c), LP.drop(index=c), how)[iso3])} for c in S.index if c != iso3]
    return pd.DataFrame(rows)


def run_size_all(out=None) -> dict[str, pd.DataFrame]:
    from .experiments import OUT
    out = OUT if out is None else out
    out.mkdir(parents=True, exist_ok=True)
    size, panel = load_size(), load_panel()
    old = structure(class_table(size, "old"))
    new = structure(class_table(size, "new"))
    comp = composition(size)
    res = {"s1_structure": pd.concat([old.assign(vintage="old"), new.assign(vintage="new")], ignore_index=True), "s2_position": position_all(size), "s3_sector_position": sector_position(size),
           "s4_composition": comp, "s5_composition_rank": composition_rank(comp),
           "s6_mean_mix": position(structure(class_table(size, "old"), how="mean"), PRIMARY[1], ["mix_effect"]).assign(variant="mean of the others"),
           "s9_mix_leave_one_out": mix_leave_one_out(size), "s7_level_gap": pd.concat([level_gap(size, panel, nace=n).assign(section=n or "economy") for n in (None, "C", "G")], ignore_index=True),
           "s8_gap_decomposition": pd.concat([gap_decomposition(size, panel, nace=n, basis=b).assign(section=n or "economy") for n in (None, "C", "G") for b in ("hour", "head")], ignore_index=True)}
    for k, v in res.items():
        v.to_parquet(out / f"{k}.parquet")
    return res
