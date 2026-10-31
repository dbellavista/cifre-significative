"""Loading and comparison helpers shared by the notebooks."""
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from cifresignificative.core.countries import EU27

PROCESSED = Path("data/processed")
EU27_ISO3 = sorted(EU27.values())
PEERS = ["DEU", "FRA", "ESP", "PRT", "GRC", "NLD", "POL"]  # large economies + southern/eastern comparators
BENCHMARKS = ["GBR", "USA", "JPN"]
ITALY = "ITA"
BASE_YEAR = 2005


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "panel.parquet")


def load_regional() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "regional.parquet")


def load_regional_extra() -> pd.DataFrame:
    """EQI and structural-business-statistics indicators of the "why the South" round (separate from `load_regional` so earlier screens keep their test counts)."""
    return pd.read_parquet(PROCESSED / "regional_extra.parquet")


ITALY_AREAS = {"ITC": "North-West", "ITH": "North-East", "ITI": "Centre", "ITF": "South", "ITG": "Islands"}


def italy_area(geo: str) -> str:
    """Macro-area of an Italian NUTS2 code (ITC4 -> North-West); 'other' for non-Italian regions."""
    return ITALY_AREAS.get(geo[:3], "other")


def wide(panel: pd.DataFrame, indicator: str, countries=None) -> pd.DataFrame:
    """year x iso3 table for one indicator (EU27 members + benchmarks; aggregates excluded)."""
    d = panel[panel.indicator == indicator]
    d = d[d.iso3.isin(countries or [*EU27_ISO3, *BENCHMARKS])]
    return d.pivot(index="year", columns="iso3", values="value").sort_index()


def eu_median_ex_italy(w: pd.DataFrame, min_share: float = 0.7) -> pd.Series:
    """Median of the other EU27 members (Italy excluded so it is not compared with itself).

    A year is left empty when fewer than `min_share` of the countries that ever report this series have a value, so a
    handful of early reporters (e.g. provisional last-year data) cannot masquerade as the EU median.
    """
    others = w[[c for c in w.columns if c in EU27_ISO3 and c != ITALY]]
    needed = max(3, int(np.ceil(min_share * others.notna().any().sum())))
    return others.median(axis=1).where(others.notna().sum(axis=1) >= needed)


def rebase(w: pd.DataFrame, base: int = BASE_YEAR) -> pd.DataFrame:
    return w / w.loc[base] * 100


def italy_rank(w: pd.DataFrame, year: int, ascending: bool = False) -> tuple[int, int]:
    """(rank of Italy among EU27 members with data, number of members) in a year; rank 1 = highest unless ascending."""
    row = w.loc[year, [c for c in w.columns if c in EU27_ISO3]].dropna()
    return int(row.rank(ascending=ascending)[ITALY]), len(row)


def zscore_gap(panel: pd.DataFrame, indicators: list[str], year_window: tuple[int, int]) -> pd.Series:
    """Italy's mean over the window minus the EU27-ex-Italy median, in cross-country standard deviations."""
    out = {}
    for ind in indicators:
        w = wide(panel, ind, EU27_ISO3).loc[year_window[0]:year_window[1]]
        means = w.mean()
        if ITALY not in means.dropna().index:
            continue
        others = means.drop(ITALY).dropna()
        out[ind] = (means[ITALY] - others.median()) / others.std()
    return pd.Series(out)


def cagr(start: float, end: float, years: int) -> float:
    return (np.power(end / start, 1 / years) - 1) * 100


def implicit_interest_rate(panel: pd.DataFrame) -> pd.DataFrame:
    """Interest paid (% GDP) / previous-year debt (% GDP) x 100, year x iso3."""
    interest = wide(panel, "gov_interest_pct_gdp")
    debt = wide(panel, "gov_debt_pct_gdp")
    return interest / debt.shift(1) * 100


def nominal_growth(panel: pd.DataFrame) -> pd.DataFrame:
    return wide(panel, "gdp_nominal_meur").pct_change(fill_method=None) * 100


def debt_decomposition(panel: pd.DataFrame, iso3: str, start: int, end: int) -> pd.Series:
    """Change in debt/GDP between `start` and `end` (pp), split into contributions (pp of GDP).

    d_t - d_{t-1} = -primary_balance_t + (i_t - g_t)/(1+g_t) * d_{t-1} (+ stock-flow residual), with
    primary balance = balance + interest paid, i = implicit rate. We split (i-g) into interest and growth effects:
    interest = i/(1+g) * d_{t-1},  growth = -g/(1+g) * d_{t-1}.
    """
    d = wide(panel, "gov_debt_pct_gdp")[iso3]
    bal = wide(panel, "gov_balance_pct_gdp")[iso3]
    intr = wide(panel, "gov_interest_pct_gdp")[iso3]
    g = nominal_growth(panel)[iso3] / 100
    rows = []
    for t in range(start + 1, end + 1):
        d0 = d[t - 1]
        interest = intr[t] / (1 + g[t])  # = i*d0/(1+g)
        growth = -g[t] / (1 + g[t]) * d0
        primary = -(bal[t] + intr[t]) / (1 + g[t])
        total = d[t] - d0
        rows.append({"primary balance": primary, "interest": interest, "nominal growth": growth, "other (stock-flow)": total - primary - interest - growth})
    return pd.DataFrame(rows).sum()


def gdp_pc_growth_decomposition(panel: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Log change in real GDP per head (x100), split into employment rate, hours per worker, and a productivity/demography residual.

    Proxies: employment = employment rate 20-64, hours = AMECO annual hours per worker, residual = real output per
    hour worked *plus* shifts in the 20-64 share of population (not separable with this panel).
    """
    y = wide(panel, "gdp_real_pc")
    e = wide(panel, "employment_rate_20_64")
    h = wide(panel, "ameco_hours_per_worker")
    cols = [c for c in y.columns if c in e.columns and c in h.columns]
    f = lambda w: np.log(w.loc[end, cols] / w.loc[start, cols]) * 100
    out = pd.DataFrame({"employment rate": f(e), "hours per worker": f(h)})
    out["productivity & demography (residual)"] = f(y) - out["employment rate"] - out["hours per worker"]
    out["real GDP per head"] = f(y)
    return out.dropna()


NINE_SECTIONS = ["A", "B-E", "F", "G-I", "J", "K", "M_N", "O-Q", "R-U"]  # NACE sections; L (real estate, imputed rents) excluded; O-Q (public admin, education, health) is non-market, output largely measured from inputs
MARKET_SECTIONS = [n for n in NINE_SECTIONS if n != "O-Q"]
SECTOR_LABELS = {"A": "Agriculture", "B-E": "Industry (mining, manufacturing, utilities)", "F": "Construction", "G-I": "Trade, transport, hospitality",
                 "J": "Information & communication", "K": "Finance & insurance", "M_N": "Professional & admin services",
                 "O-Q": "Public admin, education, health", "R-U": "Arts, other services"}


def load_sectors() -> pd.DataFrame:
    s = pd.read_parquet(PROCESSED / "sectors.parquet")
    s = s[s.nace_r2.isin(NINE_SECTIONS)].copy()
    s["prod"] = s.gva_clv20_meur / s.hours_ths * 1000  # real EUR (2020 prices) per hour worked
    return s


def aggregate_productivity(s: pd.DataFrame) -> pd.DataFrame:
    """year x iso3 real value added per hour summed over the sections in `s` (NINE_SECTIONS by default; chain-linked volumes are only approximately additive)."""
    g = s.dropna(subset=["gva_clv20_meur", "hours_ths"]).groupby(["iso3", "year"])
    return (g.gva_clv20_meur.sum() / g.hours_ths.sum() * 1000).unstack(0)


def shift_share(s: pd.DataFrame, iso3: str, start: int, end: int) -> pd.Series:
    """Change in aggregate real output per hour (% of the start level), split into
    within-sector productivity, reallocation of hours between sectors, and their interaction."""
    d = s[(s.iso3 == iso3) & s.year.isin([start, end])].pivot(index="nace_r2", columns="year", values=["prod", "hours_ths"]).dropna()  # only the two years: a missing year elsewhere must not drop the sector
    p0, p1 = d["prod"][start], d["prod"][end]
    h0, h1 = d["hours_ths"][start], d["hours_ths"][end]
    s0, s1 = h0 / h0.sum(), h1 / h1.sum()
    base = (s0 * p0).sum()
    return pd.Series({
        "within-sector productivity": (s0 * (p1 - p0)).sum() / base * 100,
        "reallocation of hours": ((s1 - s0) * p0).sum() / base * 100,
        "interaction": ((s1 - s0) * (p1 - p0)).sum() / base * 100,
    })


COFOG_GROUPS = {  # label -> (COFOG items added, items subtracted); the 13 groups add up to total expenditure
    "Debt interest": (["GF0107"], []),
    "General services (excl. interest)": (["GF01"], ["GF0107"]),
    "Defence": (["GF02"], []),
    "Public order & safety": (["GF03"], []),
    "Economic affairs": (["GF04"], []),
    "Environment": (["GF05"], []),
    "Housing & community": (["GF06"], []),
    "Health": (["GF07"], []),
    "Recreation & culture": (["GF08"], []),
    "Education": (["GF09"], []),
    "Old-age pensions": (["GF1002"], []),
    "Survivors' pensions": (["GF1003"], []),
    "Other social protection": (["GF10"], ["GF1002", "GF1003"]),
}
NATURE_LABELS = {"D1PAY": "Public sector wages", "P2": "Intermediate consumption (goods & services)", "D62_D632PAY": "Social benefits",
                 "D41PAY": "Interest", "D3PAY": "Subsidies", "P51G": "Public investment (GFCF)", "D9PAY": "Capital transfers", "D7PAY": "Other current transfers"}


def load_spending() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "spending.parquet")


def spend_wide(spending: pd.DataFrame, item: str, kind: str = "cofog", unit: str = "PC_GDP") -> pd.DataFrame:
    d = spending[(spending.kind == kind) & (spending.item == item) & (spending.unit == unit) & spending.iso3.isin([*EU27_ISO3, *BENCHMARKS])]
    return d.pivot(index="year", columns="iso3", values="value").sort_index()


def cofog_groups(spending: pd.DataFrame, year: int, unit: str = "PC_GDP") -> pd.DataFrame:
    """iso3 x group table (% of GDP or % of total spending) for one year."""
    out = {}
    for label, (plus, minus) in COFOG_GROUPS.items():
        w = sum(spend_wide(spending, i, unit=unit).loc[year] for i in plus) - sum((spend_wide(spending, i, unit=unit).loc[year] for i in minus), 0)
        out[label] = w
    return pd.DataFrame(out)


def italy_cofog_euro(spending: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Italy's COFOG groups in million euro (current prices) for the given years, plus the share of total spending; euro levels are Italy-only (not comparable across countries)."""
    lv = pd.DataFrame({g: cofog_group_series(spending, g, "MIO_EUR")[ITALY] for g in COFOG_GROUPS}).loc[years].T
    out = lv.copy(); out.columns = [f"{y} MEUR" for y in years]
    for y in years: out[f"{y} % of total"] = 100 * lv[y] / lv[y].sum()
    return out


GAP_GROUPS = {  # theme group -> [(indicator, label)]: the indicators used for the "Italy vs the rest" overview
    "Output & investment": [("gdp_real_pc", "Real GDP per head"), ("gdp_growth", "GDP growth"), ("gfcf_pct_gdp", "Investment % GDP"),
                            ("gov_pubinv_pct_gdp", "Public investment % GDP"), ("rd_pct_gdp", "R&D % GDP")],
    "Debt & prices": [("gov_debt_pct_gdp", "Government debt % GDP"), ("gov_interest_pct_gdp", "Interest % GDP"), ("long_term_yield", "10y yield"),
                      ("hicp_inflation", "Inflation")],
    "Labour": [("employment_rate_20_64", "Employment 20-64"), ("employment_rate_20_64_female", "Female employment"), ("employment_rate_55_64", "Employment 55-64"),
               ("youth_unemployment_rate", "Youth unemployment"), ("neet_rate_15_29", "NEET 15-29"), ("nominal_ulc", "Unit labour cost index")],
    "Demography & pensions": [("old_age_dependency", "Old-age dependency"), ("median_age", "Median age"), ("fertility_rate", "Fertility"),
                              ("pension_spend_pct_gdp", "Old-age spending % GDP"), ("arop_65plus", "Poverty 65+")],
    "Education": [("tertiary_attainment_25_34", "Tertiary 25-34"), ("early_leavers_18_24", "Early leavers"), ("edu_spend_pct_gdp", "Education spending % GDP"),
                  ("stem_share_grads_pct", "STEM share of graduates"), ("women_share_stem_grads_pct", "Women among STEM graduates"),
                  ("phd_grads_per_1000_pop_25_34", "PhD graduates per 1000"), ("mean_age_grads_bachelor", "Age of bachelor graduates"),
                  ("recent_grads_employment_rate", "Recent graduates employed")],
    "Gender & wages": [("part_time_female_20_64", "Women part-time"), ("involuntary_part_time_female_15_64", "Involuntary part-time (women)"),
                       ("women_managers_share_pct", "Women among managers"), ("gender_pension_gap_65", "Gender pension gap"), ("inwork_poverty_18_64", "In-work poverty")],
    "Governance": [("wgi_government_effectiveness", "Government effectiveness"), ("wgi_control_of_corruption", "Control of corruption"),
                   ("vat_compliance_gap_pct", "VAT compliance gap")],
}


def gap_table(panel: pd.DataFrame, window: tuple[int, int]) -> pd.DataFrame:
    """One row per GAP_GROUPS indicator: Italy and median-of-others (window mean), and the gap in cross-country SDs."""
    rows = []
    for group, items in GAP_GROUPS.items():
        for ind, label in items:
            w = wide(panel, ind, EU27_ISO3).loc[window[0]:window[1]]
            means = w.mean()
            if ITALY not in means.dropna().index:
                continue
            others = means.drop(ITALY).dropna()
            rows.append({"group": group, "indicator": ind, "label": label, "italy": means[ITALY], "median_others": others.median(),
                         "z_gap": (means[ITALY] - others.median()) / others.std()})
    return pd.DataFrame(rows)


def cofog_group_series(spending: pd.DataFrame, label: str, unit: str = "PC_GDP") -> pd.DataFrame:
    """year x iso3 table for one COFOG_GROUPS entry."""
    plus, minus = COFOG_GROUPS[label]
    out = cast(pd.DataFrame, sum(spend_wide(spending, i, unit=unit) for i in plus))
    for i in minus:
        out = out - spend_wide(spending, i, unit=unit)
    return out


FIELD_LABELS = {"F00": "Generic programmes", "F01": "Education", "F02": "Arts & humanities", "F03": "Social sciences & journalism",
                "F04": "Business, administration & law", "F05": "Natural sciences, maths & statistics", "F06": "ICT",
                "F07": "Engineering, manufacturing & construction", "F08": "Agriculture & veterinary", "F09": "Health & welfare", "F10": "Services"}
STEM_FIELDS = ["F05", "F06", "F07"]


def load_highered() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "highered.parquet")


def broad_year(he: pd.DataFrame, table: str, min_countries: int = 24) -> int:
    """Latest year in which at least `min_countries` EU27 members report (so medians are not built on a few early reporters)."""
    d = he[(he.table == table) & he.iso3.isin(EU27_ISO3)]
    n = d.groupby("year").iso3.nunique()
    return int(n[n >= min_countries].index.max())


def field_shares(he: pd.DataFrame, table: str, level: str, sex: str, year: int) -> pd.DataFrame:
    """iso3 x ISCED-F broad field table: % of all graduates (or students) of that level/sex."""
    d = he[(he.table == table) & (he.isced11 == level) & (he.sex == sex) & (he.year == year)]
    w = d.pivot_table(index="iso3", columns="iscedf13", values="value")
    return w[list(FIELD_LABELS)].div(w["TOTAL"], axis=0) * 100


def women_share_by_field(he: pd.DataFrame, table: str, level: str, year: int) -> pd.DataFrame:
    d = he[(he.table == table) & (he.isced11 == level) & (he.year == year) & he.sex.isin(["T", "F"])]
    w = d.pivot_table(index=["iso3", "sex"], columns="iscedf13", values="value")
    return (w.xs("F", level="sex") / w.xs("T", level="sex") * 100)[[*FIELD_LABELS, "TOTAL"]]


def age_profile(he: pd.DataFrame, level: str, year: int, ages: list[str]) -> pd.DataFrame:
    """iso3 x single-year age: % of that level's graduates (age TOTAL as denominator)."""
    d = he[(he.table == "he_grads_age") & (he.isced11 == level) & (he.sex == "T") & (he.year == year)]
    w = d.pivot_table(index="iso3", columns="age", values="value")
    return w[ages].div(w["TOTAL"], axis=0) * 100


def load_sectors64() -> pd.DataFrame:
    s = pd.read_parquet(PROCESSED / "sectors64.parquet")
    s = s[~s.nace_r2.isin(["L68", "O84", "P85", "Q86", "Q87_Q88", "T", "U"])].copy()  # market sectors: no real estate (imputed rents) or non-market services
    s["prod"] = s.gva_clv20_meur / s.hours_ths * 1000
    return s


def load_ageing() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "ageing_report.parquet")


def ageing_series(ar: pd.DataFrame, section: str, variable: str) -> pd.DataFrame:
    """year x iso3 table for one Ageing Report variable."""
    d = ar[(ar.section == section) & (ar.variable == variable)]
    return d.pivot(index="year", columns="iso3", values="value").sort_index()


def load_long() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "long_history.parquet")


def long_wide(lh: pd.DataFrame, variable: str) -> pd.DataFrame:
    return lh[lh.variable == variable].pivot(index="year", columns="iso3", values="value").sort_index()


A64_LABELS = {
    "A01": "Crop & animal production", "A02": "Forestry", "A03": "Fishing", "B": "Mining", "C10-C12": "Food, beverages, tobacco", "C13-C15": "Textiles, clothing, leather",
    "C16": "Wood products", "C17": "Paper", "C18": "Printing", "C19": "Coke & refined petroleum", "C20": "Chemicals", "C21": "Pharmaceuticals", "C22": "Rubber & plastics",
    "C23": "Non-metallic minerals", "C24": "Basic metals", "C25": "Fabricated metal products", "C26": "Computers & electronics", "C27": "Electrical equipment", "C28": "Machinery",
    "C29": "Motor vehicles", "C30": "Other transport equipment", "C31_C32": "Furniture & other manufacturing", "C33": "Repair of machinery", "D": "Electricity & gas", "E36": "Water supply",
    "E37-E39": "Sewerage & waste", "F": "Construction", "G45": "Motor vehicle trade & repair", "G46": "Wholesale trade", "G47": "Retail trade", "H49": "Land transport", "H50": "Water transport",
    "H51": "Air transport", "H52": "Warehousing & transport support", "H53": "Postal & courier", "I": "Accommodation & food service", "J58": "Publishing", "J59_J60": "Film, TV & broadcasting",
    "J61": "Telecommunications", "J62_J63": "IT services", "K64": "Financial services", "K65": "Insurance & pensions", "K66": "Auxiliary financial services", "M69_M70": "Legal, accounting, head offices",
    "M71": "Architecture & engineering", "M72": "Scientific R&D", "M73": "Advertising & market research", "M74_M75": "Other professional & veterinary", "N77": "Rental & leasing",
    "N78": "Employment agencies", "N79": "Travel agencies", "N80-N82": "Security, buildings & office support", "R90-R92": "Creative arts, libraries, gambling", "R93": "Sports & recreation",
    "S94": "Membership organisations", "S95": "Repair of computers & household goods", "S96": "Other personal services",
}


def weighted_median_ex_italy(w: pd.DataFrame, weights: pd.DataFrame, min_share: float = 0.7) -> pd.Series:
    """Population- (or any year x iso3 weight-) weighted median of the other EU27 members, same coverage rule as `eu_median_ex_italy`."""
    others = [c for c in w.columns if c in EU27_ISO3 and c != ITALY]
    needed = max(3, int(np.ceil(min_share * w[others].notna().any().sum())))
    out = {}
    for y in w.index:
        v = w.loc[y, others].dropna()
        wt = weights.loc[y, v.index] if y in weights.index else pd.Series(dtype=float)
        v = v[v.index.isin(wt.dropna().index)]
        if len(v) < needed:
            out[y] = np.nan
            continue
        order = v.sort_values(); cum = wt.loc[order.index].cumsum() / wt.loc[order.index].sum()
        out[y] = float(order[cum >= 0.5].iloc[0])
    return pd.Series(out)
