"""Contract timing and the real wage loss after the 2021-23 inflation shock, across Italy's collective-agreement groups (Istat contractual wages, `sources/istat_contracts.py`).

Contractual wages move when an increment of an agreement takes effect, so each group's hourly index is a staircase. For every leaf agreement group (no group below it: 67 private, 13 public) the notebook measures
- **staleness** at the start of a window: months since the group's last visible increment (a step of at least 0.25% in one month) up to that month, and the number of months without a step in the 24 before it;
- the **real change** of the contractual wage over the window: the index change corrected for the cumulative December-on-December HICP inflation of Italy over the same years.
Question (exploratory, two-sided): do groups whose contract was already old when the shock started end up with a different real loss than groups whose increments were fresh? Spearman rank correlation with a permutation p (the x values shuffled
within the private and the public branch, so the different regime of public-sector bargaining does not drive it; also within the parent group), BH over the pre-specified family, a calm-period placebo (December 2016 to December 2019) and a null that
shuffles the outcome across groups. The groups are not employees: no weights, the number of workers per agreement is not in the data. Contractual wages exclude wage drift and bonuses outside the agreements. Descriptive, never causal.
"""
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from cifresignificative.analysis import corr
from cifresignificative.analysis import regional_south as rs

STEP = 0.0025  # a month-on-month rise of the index of at least 0.25% counts as an increment taking effect
XS = ["staleness_months", "standstill_24m"]
XS_CNEL = ["months_since_renewal", "months_expired"]  # the same two questions with the renewal dates of the CNEL archive: age of the last renewal and months of expiry at the start of the window
WINDOWS = {"shock 2021-24": (2021, 2024), "shock 2021-25": (2021, 2025)}
PLACEBO = (2016, 2019)


def wide_index(wages: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Month x group table of the hourly contractual wage index for the given groups."""
    return wages[wages.group.isin(groups)].pivot(index="month", columns="group", values="index")


def december_inflation(hf: pd.DataFrame, iso3: str = "ITA", indicator: str = "m_hicp_inflation") -> pd.Series:
    """December-on-December HICP inflation (%) of a country by year, from the monthly annual rates of the high-frequency table (`m_hicp_inflation`, or another HICP component: `m_hicp_services_inflation`, `m_hicp_goods_inflation`)."""
    s = hf[(hf.iso3 == iso3) & hf.indicator.eq(indicator)].set_index("date").value
    return s[s.index.month == 12].rename(lambda d: d.year)


def cumulative_inflation(dec: pd.Series, start: int, end: int) -> float:
    """Price change (%) between December `start` and December `end`."""
    return float((np.prod([1 + dec[y] / 100 for y in range(start + 1, end + 1)]) - 1) * 100)


def timing(idx: pd.DataFrame, start: int, step: float = STEP) -> pd.DataFrame:
    """Per group, at December `start`: months since the last step and months without a step among the 24 up to it (months before the series starts count as steps unknown: the series begins in 2005, long before every window)."""
    rise = idx.pct_change(fill_method=None) >= step
    cut = pd.Timestamp(year=start, month=12, day=31)
    r = rise.loc[:cut]
    out = {}
    for g in r:
        s = r[g]
        steps = s[s].index
        out[g] = {"staleness_months": (cut.year - steps[-1].year) * 12 + (cut.month - steps[-1].month) if len(steps) else np.nan, "standstill_24m": int(24 - s.iloc[-24:].sum())}
    return pd.DataFrame(out).T


def window_table(wages: pd.DataFrame, groups: pd.DataFrame, dec: pd.Series, start: int, end: int, leaves_only: bool = True, step: float = STEP) -> pd.DataFrame:
    """One row per agreement group: staleness at December `start`, nominal and real change of the contractual wage between December `start` and December `end`, the number of months with a step inside the window (`steps_after`), the group's branch and parent."""
    g = groups[groups.leaf] if leaves_only else groups
    idx = wide_index(wages, list(g.code))
    a, b = idx.loc[pd.Timestamp(year=start, month=12, day=31)], idx.loc[pd.Timestamp(year=end, month=12, day=31)]
    t = timing(idx, start, step)
    rise = idx.pct_change(fill_method=None) >= step
    after = rise.loc[pd.Timestamp(year=start, month=12, day=31):pd.Timestamp(year=end, month=12, day=31)].iloc[1:]
    infl = cumulative_inflation(dec, start, end)
    d = t.join(pd.DataFrame({"nominal_change": (b / a - 1) * 100, "steps_after": after.sum()})).join(g.set_index("code")[["name_it", "name_en", "branch", "parent"]])
    d["real_change"] = ((1 + d.nominal_change / 100) / (1 + infl / 100) - 1) * 100
    d["inflation"] = infl
    return d.dropna(subset=["staleness_months", "real_change"]).rename_axis("group")


def family(d: pd.DataFrame, ys=("real_change",), xs=XS, by: str = "branch", n_perm: int = 20000, seed: int = 0, n_min: int = 20) -> pd.DataFrame:
    """Spearman rho and the permutation p (x shuffled within `by`: branch or parent), BH over the family, plus the leave-one-group-out range of rho."""
    rng = np.random.default_rng(seed)
    rows = []
    for x in xs:
        for y in ys:
            s = d[[x, y, by]].dropna()
            if len(s) < n_min:
                continue
            rho = float(stats.spearmanr(s[x], s[y])[0])
            p = rs.perm_p(s[x].to_numpy(), s[y].to_numpy(), s[by].to_numpy(), n_perm, rng)
            loo = [stats.spearmanr(s[x].drop(i), s[y].drop(i))[0] for i in s.index]
            rows.append({"x": x, "y": y, "n": len(s), "rho": rho, "p": p, "loo_min": float(np.nanmin(loo)), "loo_max": float(np.nanmax(loo))})
    return corr.fdr(pd.DataFrame(rows))


def shuffle_null(d: pd.DataFrame, ys=("real_change",), xs=XS, by: str = "branch", n_shuffle: int = 200, n_perm: int = 500, seed: int = 0) -> pd.DataFrame:
    """Calibration: shuffle the outcome across groups (within `by`), rerun the family and count the BH discoveries (should be about 0) and the share of p < 0.05 (about 0.05)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_shuffle):
        S = d.copy()
        for y in ys:
            S[y] = S.groupby(by)[y].transform(lambda v: rng.permutation(v.to_numpy()))
        t = family(S, ys, xs, by, n_perm=n_perm, seed=k)
        rows.append({"shuffle": k, "discoveries": int(t.significant.sum()), "share_p_below_005": float((t.p < 0.05).mean()), "n_tests": len(t)})
    return pd.DataFrame(rows)


def tension_table(t: pd.DataFrame, years=range(2015, 2026), month: int = 12, base_switch: int = 2021) -> pd.DataFrame:
    """Waiting-time indicators at December of each year by sector, from the base-2015 series up to 2020 and the base-2021 series from 2021 on (the two bases are not spliced into one series: the column `base` says which one each value is)."""
    d = t[t.month.dt.month.eq(month) & t.month.dt.year.isin(list(years))].copy()
    d["year"] = d.month.dt.year
    d = d[((d.base == "2015") & (d.year < base_switch)) | ((d.base == "2021") & (d.year >= base_switch))]
    return d.pivot_table(index=["year", "base"], columns=["indicator", "sector"], values="value").reset_index()


def cluster_table(d: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    """One row per parent group (the unit of the primary analysis): the sub-sectors under one parent share the same national agreements, so their staircases are not independent observations. Means of the leaves' staleness, standstill,
    steps after the start and real change (and of the `extra` columns); `n_leaves` is the number of sub-sectors."""
    c = d.groupby("parent").agg(n_leaves=("real_change", "size"), staleness_months=("staleness_months", "mean"), standstill_24m=("standstill_24m", "mean"), steps_after=("steps_after", "mean"), nominal_change=("nominal_change", "mean"),
                                real_change=("real_change", "mean"), branch=("branch", "first"), inflation=("inflation", "first"), **{x: (x, "mean") for x in extra})
    return c.rename_axis("group")


def aggregates_table(wages: pd.DataFrame, groups: pd.DataFrame, dec: pd.Series, start: int, end: int, codes=("Z0010", "Z0060", "Z1240", "Z2520", "Z3620")) -> pd.DataFrame:
    """Nominal and real change of the contractual wage of the aggregates (agriculture, industry, private services, private sector, total economy) between two Decembers."""
    d = window_table(wages, groups, dec, start, end, leaves_only=False)
    return d[d.index.isin(codes)][["name_en", "staleness_months", "standstill_24m", "nominal_change", "real_change", "inflation"]]


def persistence(d: pd.DataFrame) -> pd.DataFrame:
    """Does a group that had few increments before the shock also have few after it? Spearman between `standstill_24m` and `steps_after`, and between `steps_after` and the real change (mechanism check for the main association)."""
    rows = []
    for x, y in (("standstill_24m", "steps_after"), ("steps_after", "real_change"), ("staleness_months", "steps_after")):
        s = d[[x, y]].dropna()
        rows.append({"x": x, "y": y, "n": len(s), "rho": float(stats.spearmanr(s[x], s[y])[0])})
    return pd.DataFrame(rows)


def _months(a, b) -> int:  # Timestamps (ty types a Series element as Timestamp | NaT, so no annotation)
    return (b.year - a.year) * 12 + (b.month - a.month)


def renewal_timing(cycles: pd.DataFrame, mapping: pd.DataFrame, start: int, asof) -> pd.DataFrame:
    """Per mapped Istat leaf group, from the renewal cycles of its CNEL contract (`sources.cnel_contracts.primary_cycles`), at December `start`:
    `months_since_renewal` (from the signing of the last renewal signed up to then), `months_expired` (months the contract in force had been expired, 0 if still valid; empty when the archive has no expiry for that cycle),
    `months_to_renewal` (from December `start` to the next signing, or to `asof` when none: censored, `renewed_after` False), plus `cnel_code` and `evidence`. A leaf with no renewal signed before December `start` has no row values."""
    cut = pd.Timestamp(year=start, month=12, day=31)
    rows = {}
    for r in mapping[mapping.cnel_code != ""].itertuples():
        c = cycles[cycles.code == r.cnel_code].sort_values("signed")
        before, after = c[c.signed <= cut], c[c.signed > cut]
        row: dict = {"cnel_code": r.cnel_code, "evidence": r.evidence, "renewed_after": bool(len(after))}
        if len(before):
            last = before.iloc[-1]
            row["months_since_renewal"] = _months(last.signed, cut)
            row["months_expired"] = max(0, _months(last.expiry, cut)) if pd.notna(last.expiry) else np.nan
        row["months_to_renewal"] = _months(cut, after.signed.iloc[0] if len(after) else asof)
        rows[r.istat_group] = row
    out = pd.DataFrame(rows).T.rename_axis("group")
    out["renewed_after"] = out.renewed_after.astype(bool)
    for c in ("months_since_renewal", "months_expired", "months_to_renewal"):
        out[c] = pd.to_numeric(out[c])
    return out


def with_renewals(d: pd.DataFrame, timing: pd.DataFrame, evidence: tuple[str, ...] = ("verified",)) -> pd.DataFrame:
    """The leaf table `d` (`window_table`) restricted to the leaves whose CNEL mapping has one of the given evidence labels, with the renewal timing joined on."""
    t = timing[timing.evidence.isin(evidence)].drop(columns="evidence")
    return d.join(t, how="inner")


def proxy_agreement(d: pd.DataFrame) -> pd.DataFrame:
    """How far the staircase proxy agrees with the renewal dates on the same leaves: Spearman between staleness and the months since the last signed renewal, between the 24-month standstill and the months of expiry, and between the months to the next renewal and the steps seen after the start."""
    rows = []
    for x, y in (("staleness_months", "months_since_renewal"), ("standstill_24m", "months_expired"), ("months_to_renewal", "steps_after")):
        s = d[[x, y]].dropna()
        rows.append({"x": x, "y": y, "n": len(s), "rho": float(stats.spearmanr(s[x], s[y])[0])})
    return pd.DataFrame(rows)


DEFLATOR_OF = {"Z0010": "goods", "Z0060": "goods", "Z1240": "services"}  # top-level sector -> HICP component; the public sector keeps the all-items index
DEFLATOR_INDICATORS = {"all-items": "m_hicp_inflation", "goods": "m_hicp_goods_inflation", "services": "m_hicp_services_inflation"}


def deflator_of(groups: pd.DataFrame) -> pd.Series:
    """Per agreement group, the HICP component used by the sector deflator: goods for agriculture and industry (including construction), services for private services, all-items for the public sector. Consumer prices are not the prices
    of a branch's output, so this is a sensitivity check of the national deflator, not a better measure of the loss."""
    parent = dict(zip(groups.code, groups.parent, strict=True))
    out = {}
    for code in groups.code:
        seen, c = [], code
        while isinstance(c, str) and c not in seen:
            seen.append(c)
            c = parent.get(c)
        out[code] = next((DEFLATOR_OF[a] for a in seen if a in DEFLATOR_OF), "all-items")
    return pd.Series(out, name="deflator")


def sector_real_change(d: pd.DataFrame, deflator: pd.Series, decs: dict[str, pd.Series], start: int, end: int, col: str = "nominal_change") -> pd.Series:
    """Real change (%) between two Decembers of the contractual wage in column `col` of `d`, each group deflated with its HICP component (`decs`: December inflation by component name)."""
    infl = deflator.reindex(d.index).map({k: cumulative_inflation(v, start, end) for k, v in decs.items()})
    return ((1 + d[col] / 100) / (1 + infl / 100) - 1) * 100


def leaf_weights(mapping: pd.DataFrame, workers: pd.DataFrame) -> pd.Series:
    """Employees per Istat leaf group: the 2025 CNEL-INPS count of the mapped agreement code, split equally among the leaves that share the code (S005 covers three groups, Metalmeccanici several); NaN when the code has no count (public sector, no code)."""
    m = mapping[mapping.cnel_code != ""].set_index("istat_group")
    emp = m.cnel_code.map(workers.drop_duplicates("code").set_index("code").employees)
    return (emp / m.groupby("cnel_code").cnel_code.transform("size")).rename("employees")


def public_leaf_weights(mapping: pd.DataFrame, headcount: pd.DataFrame, extended: bool = False) -> pd.Series:
    """Heads per public Istat leaf group from the Conto annuale (`headcount` of one year): the sum of the heads of the contracts listed in the map (`+`-separated codes); `extended` adds the `extra_contracts` column (special-statute regions
    and autonomous provinces, the Bolzano and Trento schools); a code missing from the headcount table raises."""
    hc = headcount.set_index("contract").heads
    out = {}
    for group, contracts, extra in zip(mapping.istat_group, mapping.ca_contract, mapping.get("extra_contracts", pd.Series("", index=mapping.index))):
        codes = [c for c in (contracts + ("+" + extra if extended and extra else "")).split("+") if c]
        if not set(codes) <= set(hc.index):
            raise ValueError(f"{group}: contract codes not in the headcount table: {sorted(set(codes) - set(hc.index))}")
        out[group] = float(hc[codes].sum())
    return pd.Series(out, name="employees")


def all_leaf_weights(private: pd.Series, public: pd.Series) -> pd.Series:
    """Private CNEL-INPS counts (2025) and public Conto annuale heads (2021) in one series; a group with both keeps the private count (there is none: the public groups have no CNEL-INPS count)."""
    both = private.dropna().index.intersection(public.index)
    if len(both):
        raise ValueError(f"groups with both a private and a public count: {list(both)}")
    return pd.concat([private.dropna(), public]).rename("employees")


def parent_weights(leaves: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Employees per parent group: the sum over its leaves that have a count (NaN when none has)."""
    x = w.reindex(leaves.index)
    return x.groupby(leaves.parent).sum(min_count=1).rename("employees")


def _wcorr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    a, b = a - (w * a).sum() / w.sum(), b - (w * b).sum() / w.sum()
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def weighted_family(d: pd.DataFrame, w: pd.Series, ys=("real_change",), xs=XS, by: str = "branch", n_perm: int = 20000, seed: int = 0, n_min: int = 15) -> pd.DataFrame:
    """The family of `family` with each group weighted by its employees: weighted Spearman (the weighted Pearson correlation of the ranks, ranks taken over the rows used), two-sided permutation p with x shuffled within `by` (weights stay with the outcome row),
    BH over the family; `rho` is the unweighted Spearman of the same rows for comparison and `n_eff` the Kish effective number of groups (sum w)^2 / sum w^2 (a few large agreements make it much smaller than n)."""
    rng = np.random.default_rng(seed)
    rows = []
    for x in xs:
        for y in ys:
            s = d[[x, y, by]].join(w.rename("w")).dropna()
            s = s[s.w > 0]
            if len(s) < n_min:
                continue
            rx, ry, ww, g = s[x].rank().to_numpy(), s[y].rank().to_numpy(), s.w.to_numpy(), s[by].to_numpy()
            obs = _wcorr(rx, ry, ww)
            null = np.empty(n_perm)
            for k in range(n_perm):
                px = rx.copy()
                for gg in np.unique(g):
                    m = g == gg
                    px[m] = rng.permutation(rx[m])
                null[k] = _wcorr(px, ry, ww)
            rows.append({"x": x, "y": y, "n": len(s), "n_eff": float(ww.sum() ** 2 / (ww ** 2).sum()), "rho": float(stats.spearmanr(s[x], s[y])[0]), "rho_weighted": obs,
                         "p": float((1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / (1 + n_perm))})
    return corr.fdr(pd.DataFrame(rows))


def weighted_summary(d: pd.DataFrame, w: pd.Series, col: str = "real_change") -> pd.DataFrame:
    """Employee-weighted against unweighted mean of the outcome, and the share of employees (and of groups) with a real loss, by branch and in total; groups without a count are left out and their number reported."""
    s = d[[col, "branch"]].join(w.rename("w"))
    rows = []
    for name, t in [("all", s), *s.groupby("branch")]:
        u = t.dropna(subset=["w"])
        if u.empty:
            rows.append({"group": name, "n": len(t), "n_with_count": 0, "employees": 0.0})
            continue
        rows.append({"group": name, "n": len(t), "n_with_count": len(u), "employees": float(u.w.sum()), "mean": float(u[col].mean()), "weighted_mean": float((u.w * u[col]).sum() / u.w.sum()),
                     "share_groups_losing": float((u[col] < 0).mean()), "share_employees_losing": float(u.w[u[col] < 0].sum() / u.w.sum())})
    return pd.DataFrame(rows)


def weighted_null(d: pd.DataFrame, w: pd.Series, ys=("real_change",), xs=XS, by: str = "branch", n_shuffle: int = 100, n_perm: int = 300, seed: int = 0) -> pd.DataFrame:
    """Calibration of `weighted_family`: shuffle the outcome across groups within `by` (weights stay with the group), rerun the family and count the BH discoveries (should be about 0) and the share of p < 0.05 (about 0.05)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_shuffle):
        S = d.copy()
        for y in ys:
            S[y] = S.groupby(by)[y].transform(lambda v: rng.permutation(v.to_numpy()))
        t = weighted_family(S, w, ys, xs, by, n_perm=n_perm, seed=k)
        rows.append({"shuffle": k, "discoveries": int(t.significant.sum()), "share_p_below_005": float((t.p < 0.05).mean())})
    return pd.DataFrame(rows)


def weighted_cluster_table(d: pd.DataFrame, w: pd.Series, extra: Sequence[str] = ()) -> pd.DataFrame:
    """The parent table of `cluster_table` with the leaves averaged by their employee weights instead of equally (leaves without a weight are left out, a parent without any is dropped): staleness, standstill, real change (both horizons) and the `extra`
    columns are weighted means over the parent's leaves, so a parent blending large and small leaves follows the large ones. Also returns `employees` (the parent's weight, the sum of its leaves')."""
    cols = ["staleness_months", "standstill_24m", "real_change", "real_change_25", *extra]
    x = d.join(w.rename("w"))
    x = x[x.w.notna() & (x.w > 0)]
    num = x[cols].mul(x.w, axis=0).where(x[cols].notna())
    den = x[cols].notna().mul(x.w, axis=0)
    out = num.groupby(x.parent).sum(min_count=1) / den.groupby(x.parent).sum()
    out["n_leaves"] = x.groupby("parent").size()
    out["branch"] = x.groupby("parent").branch.first()
    out["employees"] = x.groupby("parent").w.sum()
    return out.rename_axis("group")
