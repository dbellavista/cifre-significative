"""Self-employment, Italy vs the EU27 (EU labour force survey, EU-SILC via Eurostat): level and mix, sector composition, age profile, poverty by activity status.

Descriptive. Italy is ranked among the members with the placebo ranking of `income_dist.placebo_gap` (Italy's gap to the median of the others against every country's own
gap), never tested as a cluster. Breaks (`b`) and provisional values (`p`) are dropped from window means. `NSAL` in the poverty table is "employed persons except employees":
self-employed plus contributing family workers, so it is a close but not exact stand-in for the self-employed.
"""
import pandas as pd

from cifresignificative.analysis import income_dist as ID
from cifresignificative.analysis.data import EU27_ISO3, ITALY, PROCESSED

DROP_FLAGS = ("b", "p")
SECTIONS = list("ABCDEFGHIJKLMNOPQRS")  # NACE Rev. 2 sections without households (T), extraterritorial bodies (U) and no response


def load_selfemp(path=PROCESSED / "selfemp.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def _clean(se: pd.DataFrame, table: str, countries=EU27_ISO3, drop_flags=DROP_FLAGS) -> pd.DataFrame:
    return se[(se.table == table) & se.iso3.isin(countries) & ~se.flag.isin(drop_flags)]


def share_wide(se: pd.DataFrame, sex: str = "T") -> pd.DataFrame:
    """(iso3, year) x [SELF, SELF_S, SELF_NS]: % of employment 15-64 (employers = SELF_S, own-account workers without employees = SELF_NS)."""
    d = _clean(se, "selfemp_share_status")
    return d[d.sex == sex].pivot_table(index=["iso3", "year"], columns="wstatus", values="value")


def age_wide(se: pd.DataFrame) -> pd.DataFrame:
    """(iso3, year) x age band: % of employment in the band that is self-employed."""
    return _clean(se, "selfemp_share_age").pivot_table(index=["iso3", "year"], columns="age", values="value")


def poverty_wide(se: pd.DataFrame) -> pd.DataFrame:
    """(iso3, year) x [EMP, SAL, NSAL, UNE]: at-risk-of-poverty rate by activity status, plus `gap_nsal_sal` (percentage points) and `ratio_nsal_sal`."""
    W = _clean(se, "arop_by_status").pivot_table(index=["iso3", "year"], columns="wstatus", values="value")
    return W.assign(gap_nsal_sal=W.NSAL - W.SAL, ratio_nsal_sal=W.NSAL / W.SAL)


def sector_frame(se: pd.DataFrame, year: int, min_cover: float = 0.97) -> pd.DataFrame:
    """Country x NACE section: self-employed and employed (thousands, 15-64) in `year`. A cross-section, so break flags are kept (only provisional values dropped).
    Eurostat leaves out small cells: a missing self-employed count in a section with employment is set to 0, and a country is kept only when employment is present in all 19 sections and the sections add up to
    at least `min_cover` of the published self-employed TOTAL. Households as employers (T) and extraterritorial bodies (U) are left out of both counts."""
    def get(table):
        d = _clean(se, table, drop_flags=("p",))
        d = d[d.year == year]
        return d[d.nace_r2.isin(SECTIONS)].pivot_table(index="iso3", columns="nace_r2", values="value"), d[d.nace_r2 == "TOTAL"].set_index("iso3").value
    (s, ts), (e, _) = get("selfemp_nace_ths"), get("employed_nace_ths")
    ok = e.notna().sum(axis=1).eq(len(SECTIONS)) & ((s.reindex(e.index).sum(axis=1) / ts.reindex(e.index)) >= min_cover)
    keep = ok[ok].index
    return pd.concat({"self": s.reindex(index=keep, columns=SECTIONS).fillna(0.0), "emp": e.loc[keep, SECTIONS]}, axis=1)


def decompose(F: pd.DataFrame, country: str = ITALY) -> dict:
    """Self-employment share of `country` minus the share of the pooled others, split exactly into composition (sector weights differ) and within-sector rates:
    share = sum_s w_s r_s; gap = sum (w - wbar) rbar + sum w (r - rbar), with wbar, rbar the others' pooled sector weights and self-employment rates (pp)."""
    if country not in F.index:
        return {}
    oth = F.drop(country)
    wbar = oth["emp"].sum() / oth["emp"].to_numpy().sum()
    rbar = oth["self"].sum() / oth["emp"].sum()
    e, s = F.loc[country, "emp"], F.loc[country, "self"]
    w, r = e / e.sum(), s / e
    comp, within = float(((w - wbar) * rbar).sum() * 100), float((w * (r - rbar)).sum() * 100)
    return {"share": float(s.sum() / e.sum() * 100), "pooled_others": float((wbar * rbar).sum() * 100), "composition": comp, "within_sector": within}


def decompose_all(F: pd.DataFrame) -> pd.DataFrame:
    """`decompose` for every country in the frame, against the pooled rest (the placebo distribution: Italy's composition and within-sector parts ranked among them)."""
    return pd.DataFrame({c: decompose(F, c) for c in F.index}).T.rename_axis("iso3")


def rank_in(table: pd.Series, country: str = ITALY) -> dict:
    """Rank (1 = largest), n and share of countries with an absolute value at least as large as `country`'s (placebo ranking over a per-country gap)."""
    x = table.dropna()
    return {"value": float(x[country]), "rank_from_top": int((x > x[country]).sum() + 1), "n_countries": len(x), "p_abs": float((x.abs() >= abs(x[country]) - 1e-12).mean())}


def change_between(W: pd.DataFrame, col: str, start: tuple[int, int], end: tuple[int, int]) -> pd.Series:
    """Country change in the mean of `col` between two year windows."""
    yr = W.index.get_level_values("year")
    m = lambda w: W[col][(yr >= w[0]) & (yr <= w[1])].groupby(level="iso3").mean()
    return (m(end) - m(start)).dropna()


def window_gap(W: pd.DataFrame, col: str, windows: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Italy's window mean, the median of the others, the gap and the placebo ranking, per window."""
    rows = []
    for name, w in windows.items():
        T = ID.window_table(W, w)
        rows.append({"window": name, **ID.placebo_gap(T[col])})
    return pd.DataFrame(rows)
