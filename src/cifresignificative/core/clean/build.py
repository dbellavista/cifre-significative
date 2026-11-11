"""Build processed tables from raw files: panel.parquet, regional.parquet, pensions_oecd_2024.parquet."""
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from cifresignificative.core.catalog import (
    EUROSTAT,
    EUROSTAT_EARNINGS,
    EUROSTAT_HIGHERED,
    EUROSTAT_HIGHFREQ,
    EUROSTAT_INCOME,
    EUROSTAT_MIGRATION,
    EUROSTAT_PROJECTIONS,
    EUROSTAT_REGIONAL,
    EUROSTAT_REGIONAL_SBS,
    EUROSTAT_SECTORS,
    EUROSTAT_SECTORS64,
    EUROSTAT_SELFEMP,
    EUROSTAT_SPENDING,
    EXCLUDED_POINTS,
    SBS_SECTIONS,
)
from cifresignificative.core.clean import jsonstat, nuts, vintages
from cifresignificative.core.sources import (
    ageing,
    ameco,
    bready,
    cohesion,
    cpt,
    daily_yields,
    eqi,
    euklems,
    gisco_nuts,
    imf,
    istat,
    istat_migration,
    istat_unobserved,
    manual,
    mef_gap,
    mef_irpef,
    oecd,
    procurement,
    vatgap,
    worldbank,
)

RAW = Path("data/raw/eurostat")
RAW_SBS = Path("data/raw/eurostat_nuts2_sbs")
RAW_NUTS2 = Path("data/raw/eurostat_nuts2")
RAW_HIGHFREQ = Path("data/raw/eurostat_highfreq")
RAW_SECTORS = Path("data/raw/eurostat_sectors")
RAW_SPENDING = Path("data/raw/eurostat_spending")
RAW_HIGHERED = Path("data/raw/eurostat_highered")
RAW_PROJECTIONS = Path("data/raw/eurostat_projections")
RAW_SECTORS64 = Path("data/raw/eurostat_sectors64")
RAW_EARNINGS = Path("data/raw/eurostat_earnings")
RAW_MIGRATION = Path("data/raw/eurostat_migration")
RAW_INCOME = Path("data/raw/eurostat_income")
RAW_SELFEMP = Path("data/raw/eurostat_selfemp")
RAW_SIZE = Path("data/raw/eurostat_size")
OUT_DIR = Path("data/processed")
COLS = ["iso3", "geo", "year", "indicator", "value", "flag", "theme", "source"]


def _collect(loaders, errors):
    frames = []
    for label, load in loaders:
        try:
            frames.append(load())
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{label}: {exc}")
    return frames


def build_panel() -> pd.DataFrame:
    errors: list[str] = []
    loaders: list[tuple[str, Callable[[], pd.DataFrame]]] = [(i.name, lambda i=i: jsonstat.to_panel(i.name, i.theme, i.dataset, jsonstat.parse(RAW / f"{i.name}.json"))) for i in EUROSTAT]
    loaders += [(n, lambda n=n: worldbank.load(n)) for n in worldbank.WDI]
    loaders += [(n, lambda n=n: imf.load(n)) for n in imf.WEO]
    loaders += [("ameco", ameco.load), ("oecd_series", oecd.load_series), ("manual", manual.load), ("vat_gap", vatgap.load), ("procurement", procurement.load)]
    frames = _collect(loaders, errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
    panel = pd.concat(frames, ignore_index=True)
    if panel.duplicated(["indicator", "geo", "year"]).any():
        raise ValueError("duplicate (indicator, geo, year) keys")
    panel = pd.concat([panel[COLS], _women_managers_share(panel), _highered_derived(he := build_highered())], ignore_index=True)
    panel = pd.concat([panel, _migration_crime_derived(panel, build_migration(), he)], ignore_index=True)
    panel = pd.concat([panel, _arope_linked(panel)], ignore_index=True)
    drop = pd.Series(False, index=panel.index)
    for (indicator, year) in EXCLUDED_POINTS:
        drop |= (panel.indicator == indicator) & (panel.year == year)
    return panel[~drop].reset_index(drop=True)


def _women_managers_share(panel: pd.DataFrame) -> pd.DataFrame:
    """Women as % of employed managers (ISCO major group 1), derived from the two helper headcounts."""
    w = panel[panel.indicator.isin(["managers_total_ths", "managers_female_ths"])].pivot_table(index=["iso3", "geo", "year"], columns="indicator", values="value")
    out = (w["managers_female_ths"] / w["managers_total_ths"] * 100).dropna().rename("value").reset_index()
    return out.assign(indicator="women_managers_share_pct", flag=None, theme="gender", source="eurostat:lfsa_egais (derived)")[COLS]


def _arope_linked(panel: pd.DataFrame) -> pd.DataFrame:
    """AROPE 2005-2025 on the current definition: `arope_old` (2005-2020) rescaled by each country's mean new/old ratio in the overlap years
    2015-2020, spliced with `arope` (2015+). Pre-2015 values are therefore approximations; rows carry the flag `linked`."""
    w = panel[panel.indicator.isin(["arope", "arope_old"])].pivot_table(index=["iso3", "geo", "year"], columns="indicator", values="value")
    ov = w.loc[(slice(None), slice(None), slice(2015, 2020)), :].dropna()
    ratio = (ov["arope"] / ov["arope_old"]).groupby(level=["iso3", "geo"]).mean()
    out = w["arope"].copy()
    for (iso, geo), r in ratio.items():
        for y in range(2005, 2015):
            if (iso, geo, y) in w.index and pd.notna(w.loc[(iso, geo, y), "arope_old"]):
                out.loc[(iso, geo, y)] = w.loc[(iso, geo, y), "arope_old"] * r
    d = out.dropna().rename("value").reset_index()
    return d.assign(indicator="arope_linked", flag=d.year.map(lambda y: "linked" if y < 2015 else None), theme="inequality", source="eurostat:ilc_peps01+ilc_peps01n (linked)")[COLS]


def build_highered() -> dict[str, pd.DataFrame]:
    """Higher-education detail tables keyed by catalog name (graduates/students by field, graduates by age, population groups)."""
    out = {}
    for ind in EUROSTAT_HIGHERED:
        df = jsonstat.parse(RAW_HIGHERED / f"{ind.name}.json", keep=ind.keep)
        out[ind.name] = df.assign(iso3=df["geo"].map(jsonstat.ISO3))
    return out


STEM_FIELDS = ["F05", "F06", "F07"]  # natural sciences/maths/statistics, ICT, engineering/manufacturing/construction (ISCED-F 2013)
_AGE_VALUE = {**{f"Y{a}": float(a) for a in range(20, 50)}, "Y50-54": 52.0, "Y55-59": 57.0, "Y60-64": 62.0, "Y_GE65": 67.0}  # ranges at midpoints


def _highered_derived(he: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Derived rates: STEM shares, women's shares, doctoral and graduate rates per population, mean graduation age."""
    rows = []

    def emit(series: pd.Series, name: str, source: str) -> None:
        d = series.dropna().rename("value").reset_index()
        d = d[d.iso3.notna()]
        rows.append(d.assign(indicator=name, flag=None, theme="education", source=f"eurostat:{source} (derived)")[COLS])

    def by_field(table: str, level: str) -> pd.DataFrame:
        d = he[table]
        d = d[d.isced11 == level]
        return d.pivot_table(index=["iso3", "geo", "year", "sex"], columns="iscedf13", values="value")

    def stem(w: pd.DataFrame) -> pd.Series:
        return w[STEM_FIELDS].sum(axis=1, min_count=len(STEM_FIELDS))

    g = by_field("he_grads_field", "ED5-8")
    st = stem(g)
    emit((st / g["TOTAL"] * 100).xs("T", level="sex"), "stem_share_grads_pct", "educ_uoe_grad02")
    emit((st / g["TOTAL"] * 100).xs("F", level="sex"), "stem_share_grads_female_pct", "educ_uoe_grad02")
    emit((st / g["TOTAL"] * 100).xs("M", level="sex"), "stem_share_grads_male_pct", "educ_uoe_grad02")
    emit(st.xs("F", level="sex") / st.xs("T", level="sex") * 100, "women_share_stem_grads_pct", "educ_uoe_grad02")
    emit(g["F06"].xs("F", level="sex") / g["F06"].xs("T", level="sex") * 100, "women_share_ict_grads_pct", "educ_uoe_grad02")
    emit((g["F06"] / g["TOTAL"] * 100).xs("T", level="sex"), "ict_share_grads_pct", "educ_uoe_grad02")
    emit(g["TOTAL"].xs("F", level="sex") / g["TOTAL"].xs("T", level="sex") * 100, "women_share_grads_pct", "educ_uoe_grad02")
    phd = by_field("he_grads_field", "ED8")
    emit(phd["TOTAL"].xs("F", level="sex") / phd["TOTAL"].xs("T", level="sex") * 100, "phd_women_share_pct", "educ_uoe_grad02")
    emit((stem(phd) / phd["TOTAL"] * 100).xs("T", level="sex"), "phd_stem_share_pct", "educ_uoe_grad02")
    s = by_field("he_students_field", "ED5-8")
    emit((stem(s) / s["TOTAL"] * 100).xs("T", level="sex"), "stem_share_students_pct", "educ_uoe_enrt03")
    emit(s["TOTAL"].xs("F", level="sex") / s["TOTAL"].xs("T", level="sex") * 100, "women_share_students_pct", "educ_uoe_enrt03")
    pop = he["he_pop_age"].pivot_table(index=["iso3", "geo", "year"], columns="age", values="value")
    p_20_29, p_25_34, p_20_24 = pop["Y20-24"] + pop["Y25-29"], pop["Y25-29"] + pop["Y30-34"], pop["Y20-24"]
    emit(g["TOTAL"].xs("T", level="sex") / p_20_29 * 1000, "grads_per_1000_pop_20_29", "educ_uoe_grad02")
    emit(stem(g).xs("T", level="sex") / p_20_29 * 1000, "stem_grads_per_1000_pop_20_29", "educ_uoe_grad02")
    emit(phd["TOTAL"].xs("T", level="sex") / p_25_34 * 1000, "phd_grads_per_1000_pop_25_34", "educ_uoe_grad02")
    emit(s["TOTAL"].xs("T", level="sex") / p_20_24 * 100, "students_per_100_pop_20_24", "educ_uoe_enrt03")
    a = he["he_grads_age"]
    a = a[(a.sex == "T") & a.age.isin(_AGE_VALUE)].assign(age_v=lambda d: d.age.map(_AGE_VALUE))
    for level, name in [("ED6", "bachelor"), ("ED7", "master"), ("ED8", "phd")]:
        x = a[a.isced11 == level]
        grp = x.assign(w=x.value * x.age_v).groupby(["iso3", "geo", "year"])
        emit(grp.w.sum() / grp.value.sum(), f"mean_age_grads_{name}", "educ_uoe_grad01")
    b = a[a.isced11 == "ED6"]
    old = b[b.age_v >= 30].groupby(["iso3", "geo", "year"]).value.sum() / b.groupby(["iso3", "geo", "year"]).value.sum() * 100
    emit(old, "bachelor_grads_30plus_pct", "educ_uoe_grad01")
    return pd.concat(rows, ignore_index=True)


def build_sectors64() -> pd.DataFrame:
    """A*64 industry table: (iso3, geo, year, nace_r2) with real/nominal value added and hours."""
    cols = {}
    for ind in EUROSTAT_SECTORS64:
        df = jsonstat.parse(RAW_SECTORS64 / f"{ind.name}.json", keep=ind.keep)
        cols[ind.name.replace("sector64_", "")] = df.set_index(["geo", "year", "nace_r2"])["value"]
    out = pd.DataFrame(cols).reset_index()
    out["iso3"] = out["geo"].map(jsonstat.ISO3)
    return out.rename(columns={"gva_clv20_meur": "gva_clv20_meur", "hours_ths": "hours_ths"})


def build_earnings_cases() -> pd.DataFrame:
    ind = EUROSTAT_EARNINGS[0]
    df = jsonstat.parse(RAW_EARNINGS / f"{ind.name}.json", keep=ind.keep)
    return df.assign(iso3=df["geo"].map(jsonstat.ISO3))[["iso3", "geo", "year", "estruct", "ecase", "value", "flag"]]


def build_income() -> pd.DataFrame:
    """Income distribution (EU-SILC): one long table, `table` = catalog name, `quant_inc` set only for `income_share`; Italy and every country with data, all years requested."""
    frames = []
    for ind in EUROSTAT_INCOME:
        df = jsonstat.parse(RAW_INCOME / f"{ind.name}.json", keep=ind.keep)
        frames.append(df.assign(iso3=df["geo"].map(jsonstat.ISO3), table=ind.name))
    return pd.concat(frames, ignore_index=True)


def build_selfemp() -> pd.DataFrame:
    """Self-employment tables in one long table: `table` = catalog name; the kept dimensions (`wstatus`, `sex`, `age`, `nace_r2`) are NaN where a table does not have them."""
    frames = []
    for ind in EUROSTAT_SELFEMP:
        df = jsonstat.parse(RAW_SELFEMP / f"{ind.name}.json", keep=ind.keep)
        frames.append(df.assign(iso3=df["geo"].map(jsonstat.ISO3), table=ind.name))
    return pd.concat(frames, ignore_index=True)


def build_migration() -> dict[str, pd.DataFrame]:
    """Migration detail tables keyed by catalog name (flows and stock by citizenship and age)."""
    out = {}
    for ind in EUROSTAT_MIGRATION:
        df = jsonstat.parse(RAW_MIGRATION / f"{ind.name}.json", keep=ind.keep)
        out[ind.name] = df.assign(iso3=df["geo"].map(jsonstat.ISO3))
    return out


def _migration_crime_derived(panel: pd.DataFrame, mig: dict[str, pd.DataFrame], he: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-capita migration, asylum and permit rates, net migration of nationals (incl. ages 25-34), foreign share, prison occupancy."""
    rows = []
    keys = ["iso3", "geo", "year"]

    def emit(series: pd.Series, name: str, theme: str, source: str) -> None:
        d = series.replace([float("inf"), -float("inf")], float("nan")).dropna().rename("value").reset_index()
        rows.append(d.assign(indicator=name, flag=None, theme=theme, source=f"{source} (derived)")[COLS])

    def ind(name: str) -> pd.Series:
        d = panel[panel.indicator == name]
        return d.set_index(keys).value

    pop = ind("population")
    emit(ind("immigration_total_nr") / pop * 1000, "immigration_per_1000", "migration", "eurostat:migr_imm8")
    emit(ind("emigration_total_nr") / pop * 1000, "emigration_per_1000", "migration", "eurostat:migr_emi2")
    emit((ind("immigration_total_nr") - ind("emigration_total_nr")) / pop * 1000, "net_migration_registered_per_1000", "migration", "eurostat:migr_imm8/migr_emi2")
    emit(ind("asylum_first_applicants_nr") / pop * 1e5, "asylum_first_applicants_per_100k", "migration", "eurostat:migr_asyappctza")
    emit(ind("first_permits_nr") / pop * 1000, "first_permits_per_1000", "migration", "eurostat:migr_resfirst")
    emit(ind("prison_held_nr") / ind("prison_capacity_nr") * 100, "prison_occupancy_pct", "crime", "eurostat:crim_pris_cap")

    def mt(name: str, citizen: str, ages: list[str]) -> pd.Series:
        d = mig[name]
        d = d[(d.citizen == citizen) & d.age.isin(ages)]
        return d.groupby(keys).value.sum(min_count=len(ages))

    emit(mt("mig_emi_citizen", "NAT", ["TOTAL"]) / pop * 1000, "emigration_nationals_per_1000", "migration", "eurostat:migr_emi1ctz")
    emit((mt("mig_imm_citizen", "NAT", ["TOTAL"]) - mt("mig_emi_citizen", "NAT", ["TOTAL"])) / pop * 1000, "net_migration_nationals_per_1000", "migration", "eurostat:migr_imm1ctz/migr_emi1ctz")
    yp = he["he_pop_age"].pivot_table(index=keys, columns="age", values="value")
    pop_25_34 = yp["Y25-29"] + yp["Y30-34"]
    young = ["Y25-29", "Y30-34"]
    emit(mt("mig_emi_citizen", "NAT", young) / pop_25_34 * 1000, "emigration_nationals_25_34_per_1000", "migration", "eurostat:migr_emi1ctz")
    emit((mt("mig_imm_citizen", "NAT", young) - mt("mig_emi_citizen", "NAT", young)) / pop_25_34 * 1000, "net_migration_nationals_25_34_per_1000", "migration", "eurostat:migr_imm1ctz/migr_emi1ctz")
    st = mig["mig_pop_citizen"].pivot_table(index=keys, columns="citizen", values="value")
    emit((st["TOTAL"] - st["NAT"]) / st["TOTAL"] * 100, "foreign_citizens_share_pct", "migration", "eurostat:migr_pop1ctz")
    return pd.concat(rows, ignore_index=True)


def build_projections() -> pd.DataFrame:
    """Eurostat EUROPOP2023 projections (long: iso3, geo, year, indic_de, projection, value)."""
    ind = EUROSTAT_PROJECTIONS[0]
    df = jsonstat.parse(RAW_PROJECTIONS / f"{ind.name}.json", keep=ind.keep)
    return df.assign(iso3=df["geo"].map(jsonstat.ISO3))[["iso3", "geo", "year", "indic_de", "projection", "value", "flag"]]


def build_spending() -> pd.DataFrame:
    """Government spending by function and by economic nature (long format, one row per geo/year/item/unit)."""
    frames = []
    for ind, keep, kind, item in [(EUROSTAT_SPENDING[0], ("cofog99", "unit"), "cofog", "cofog99"), (EUROSTAT_SPENDING[1], ("na_item",), "nature", "na_item")]:
        df = jsonstat.parse(RAW_SPENDING / f"{ind.name}.json", keep=keep)
        df = df.rename(columns={item: "item"}).assign(kind=kind)
        if "unit" not in df:
            df["unit"] = "PC_GDP"
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["iso3"] = out["geo"].map(jsonstat.ISO3)
    return out[["iso3", "geo", "year", "kind", "item", "unit", "value", "flag"]]


def build_regional() -> pd.DataFrame:
    """NUTS2 regions (4-character codes) of EU27 countries, long format."""
    frames = []
    for ind in EUROSTAT_REGIONAL:
        path = RAW_NUTS2 / f"{ind.name}.json"
        if not path.exists():
            print(f"{ind.name}: raw file missing", file=sys.stderr)
            continue
        df = jsonstat.parse(path)
        df = df[df.geo.str.len().eq(4) & df.geo.str[:2].isin(jsonstat.NUTS_COUNTRIES) & ~df.geo.str.endswith("ZZ")]  # ZZ = extra-regio, not a region
        frames.append(df.assign(indicator=ind.name, theme=ind.theme, source=f"eurostat:{ind.dataset}", country=df.geo.str[:2]))
    return pd.concat(frames, ignore_index=True)


def build_regional_extra(regional: pd.DataFrame) -> pd.DataFrame:
    """Indicators of the "why the South" round (EQI waves and structural business statistics), in the regional long format but in their own table:
    merging them into `regional.parquet` would change the number of tests, and so the false-discovery cut-offs, of the earlier regional screens (notebook 25)."""
    frames = [build_regional_sbs(regional[regional.indicator.eq("nuts2_population")])]
    if eqi.RAW.exists():
        frames.append(eqi.load())  # one value per region and survey year
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


def build_regional_sbs(population: pd.DataFrame) -> pd.DataFrame:
    """Per-region structure indicators from the NUTS2 structural business statistics (sections B-N without K; 2008-2020), in the regional long format.
    A (region, year) is kept only when all 12 sections are present for the indicators it needs, so confidential cells do not bias the totals; the flag is the union of the components' flag characters.
    `population` = rows of `nuts2_population` (geo, year, value) for the per-inhabitant ratio."""
    raw = {}
    for ind in EUROSTAT_REGIONAL_SBS:
        path = RAW_SBS / f"{ind.name}.json"
        if not path.exists():
            print(f"{ind.name}: raw file missing", file=sys.stderr)
            return pd.DataFrame()
        df = jsonstat.parse(path, keep=("nace_r2",))
        df = df[df.geo.str.len().eq(4) & df.geo.str[:2].isin(jsonstat.NUTS_COUNTRIES) & ~df.geo.str.endswith("ZZ")]
        g = df.groupby(["geo", "year"])
        full = g.nace_r2.nunique().eq(len(SBS_SECTIONS))
        tot = g.value.sum().rename("total")[full]
        flag = g.flag.agg(lambda f: "".join(sorted(set("".join(f.dropna()))))).rename("flag")
        manuf = df[df.nace_r2.eq("C")].set_index(["geo", "year"]).value.rename("manufacturing")
        raw[ind.name] = pd.concat([tot, manuf, flag], axis=1).dropna(subset=["total"])
    units, persons, wages = (raw[i.name] for i in EUROSTAT_REGIONAL_SBS)
    pop = population.set_index(["geo", "year"]).value
    flags = pd.concat([units.flag, persons.flag, wages.flag], axis=1).fillna("").agg("".join, axis=1).map(lambda f: "".join(sorted(set(f))))
    out = {
        "nuts2_sbs_manufacturing_employment_share_pct": persons.manufacturing / persons.total * 100,
        "nuts2_sbs_local_units_per_1000_inhabitants": units.total / pop * 1000,
        "nuts2_sbs_wages_per_employee_keur": wages.total / persons.total * 1000,
    }
    frames = [s.dropna().rename("value").reset_index().assign(indicator=n) for n, s in out.items()]
    d = pd.concat(frames, ignore_index=True)
    d["flag"] = [flags.get((g, y)) or None for g, y in zip(d.geo, d.year)]
    return d.assign(theme="structure", source="eurostat:sbs_r_nuts06_r2", country=d.geo.str[:2])


def build_highfreq() -> pd.DataFrame:
    """Monthly and quarterly series for the EU27, benchmarks and aggregates, long format (period label plus period-end date)."""
    frames = []
    for ind in EUROSTAT_HIGHFREQ:
        path = RAW_HIGHFREQ / f"{ind.name}.json"
        if not path.exists():
            print(f"{ind.name}: raw file missing", file=sys.stderr)
            continue
        df = jsonstat.parse_periodic(path)
        df = df[df.geo.isin(jsonstat.ISO3)]
        frames.append(df.assign(indicator=ind.name, theme=ind.theme, source=f"eurostat:{ind.dataset}", iso3=df.geo.map(jsonstat.ISO3)))
    return pd.concat(frames, ignore_index=True)[["iso3", "geo", "period", "date", "freq", "indicator", "value", "flag", "theme", "source"]]


def build_size() -> pd.DataFrame:
    """Enterprise statistics by employment size class: one row per (iso3, year, vintage, nace_r2, size_emp) with enterprises, persons employed and value added (EUR million) and `flag`, the union of the Eurostat flag characters of the three.
    `pli` = GDP price level index of the country-year (EU27 = 100, `prc_ppp_ind`; an approximation for value added). `vintage` = `old` (sbs_sc_sca_r2, 2005-2020, value added at factor cost, economy = `B-N_S95_X_K`) or `new` (sbs_sc_ovw, 2021 onwards, basic prices, economy = `B-S_X_O_S94`, wider scope): never combine the two.
    Sums of sections are not built: they keep only 8-12 countries a year because small sections have confidential large-firm cells."""
    out = []
    for vintage in ("old", "new"):
        cols, flags = {}, []
        for kind in ("enterprises", "persons", "va_meur"):
            df = jsonstat.parse(RAW_SIZE / f"size_{vintage}_{kind}.json", keep=("nace_r2", "size_emp"))
            df = df[df.geo.isin(jsonstat.ISO3)].set_index(["geo", "year", "nace_r2", "size_emp"])
            cols[kind] = df["value"]
            flags.append(df["flag"].fillna(""))
        w = pd.DataFrame(cols)
        w["flag"] = pd.concat(flags, axis=1).fillna("").agg("".join, axis=1).map(lambda f: "".join(sorted(set(f))) or None)
        out.append(w.reset_index().assign(vintage=vintage))
    d = pd.concat(out, ignore_index=True)
    d["iso3"] = d.geo.map(jsonstat.ISO3)
    pli = jsonstat.parse(RAW_SIZE / "size_price_level.json")[["geo", "year", "value"]].rename(columns={"value": "pli"})
    d = d.merge(pli, how="left", on=["geo", "year"])
    return d[["iso3", "geo", "year", "vintage", "nace_r2", "size_emp", "enterprises", "persons", "va_meur", "pli", "flag"]]


def build_sectors() -> pd.DataFrame:
    """Industry table: one row per (iso3, year, nace_r2) with real/nominal gross value added and hours worked."""
    cols = {}
    for ind in EUROSTAT_SECTORS:
        df = jsonstat.parse(RAW_SECTORS / f"{ind.name}.json", keep=("nace_r2",))
        cols[ind.name] = df.set_index(["geo", "year", "nace_r2"])["value"]
    out = pd.DataFrame(cols).reset_index()
    out["iso3"] = out["geo"].map(jsonstat.ISO3)
    return out.rename(columns={"sector_gva_clv20_meur": "gva_clv20_meur", "sector_gva_cp_meur": "gva_cp_meur", "sector_hours_ths": "hours_ths"})


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    panel.to_parquet(OUT_DIR / "panel.parquet", index=False)
    snap = vintages.snapshot(panel)
    print(f"vintage snapshot: {snap.name if snap else 'unchanged since last snapshot'}")
    print(f"panel: {len(panel)} rows, {panel.indicator.nunique()} indicators")
    regional = build_regional()
    regional.to_parquet(OUT_DIR / "regional.parquet", index=False)
    print(f"vintage snapshot (regional): {getattr(vintages.snapshot(regional, 'regional'), 'name', 'unchanged')}")
    print(f"regional: {len(regional)} rows, {regional.indicator.nunique()} indicators, {regional.geo.nunique()} regions")
    extra = build_regional_extra(regional)
    extra.to_parquet(OUT_DIR / "regional_extra.parquet", index=False)
    print(f"vintage snapshot (regional extra): {getattr(vintages.snapshot(extra, 'regional_extra'), 'name', 'unchanged')}")
    print(f"regional extra: {len(extra)} rows, {extra.indicator.nunique()} indicators, {extra.geo.nunique()} regions")
    if all(gisco_nuts.path(y).exists() for y in gisco_nuts.YEARS):
        cw = nuts.crosswalk()
        reg21, extra21 = nuts.harmonise(regional, cw), nuts.harmonise(extra, cw)
        reg21.to_parquet(OUT_DIR / "regional_nuts2021.parquet", index=False)
        extra21.to_parquet(OUT_DIR / "regional_extra_nuts2021.parquet", index=False)
        print(f"vintage snapshot (regional nuts2021): {getattr(vintages.snapshot(reg21, 'regional_nuts2021'), 'name', 'unchanged')}")
        print(f"NUTS 2021 geography: regional {reg21.geo.nunique()} regions ({int(reg21.geo_method.eq('renamed').sum())} renamed rows, {regional.geo.nunique() - reg21.geo.nunique()} codes dropped or merged), extra {extra21.geo.nunique()} regions")
    highfreq = build_highfreq()
    highfreq.to_parquet(OUT_DIR / "highfreq.parquet", index=False)
    print(f"vintage snapshot (highfreq): {getattr(vintages.snapshot(highfreq, 'highfreq'), 'name', 'unchanged')}")
    print(f"highfreq: {len(highfreq)} rows, {highfreq.indicator.nunique()} indicators")
    highered = pd.concat([d.assign(table=k) for k, d in build_highered().items()], ignore_index=True)
    highered.to_parquet(OUT_DIR / "highered.parquet", index=False)
    print(f"highered: {len(highered)} rows")
    if any(daily_yields.SRC_DIR.glob("*_10y_*.csv")):  # manual Investing.com exports, not redistributed (see the README of the folder)
        dy = daily_yields.load()
        dy.to_parquet(OUT_DIR / "daily_yields.parquet", index=False)
        print(f"daily yields: {len(dy)} rows, {dy.date.min().date()} to {dy.date.max().date()}")
    else:
        print("daily yields: manual exports not present, skipped (see data/external/source/daily_yields/README.md)", file=sys.stderr)
    istat.load().to_parquet(OUT_DIR / "istat_irregular_work.parquet", index=False)
    if mef_gap.PDF.exists():
        mef_gap.load().to_parquet(OUT_DIR / "italy_tax_gap_mef.parquet", index=False)
    else:
        print(f"MEF tax gap: {mef_gap.PDF.name} not present, skipped (see docs/reproducibility.md)", file=sys.stderr)
    if mef_irpef.SRC.exists():
        irpef = mef_irpef.load()
        irpef.to_parquet(OUT_DIR / "mef_irpef.parquet", index=False)
        print(f"MEF IRPEF returns: {len(irpef)} rows, tax years {irpef.tax_year.min()}-{irpef.tax_year.max()}, {irpef.code.nunique()} tables, {int(irpef.jump.sum())} jump-flagged cells")
        dims = mef_irpef.load_dims()
        dims.to_parquet(OUT_DIR / "mef_irpef_dims.parquet", index=False)
        print(f"MEF IRPEF returns by region, sector, age, turnover and income twentile: {len(dims)} rows, {dims.groupby('dimension').code.nunique().to_dict()} tables")
    if istat_migration.RAW.exists():
        flows = istat_migration.load_flows()
        flows.to_parquet(OUT_DIR / "italy_internal_migration.parquet", index=False)
        print(f"Italian internal migration: {len(flows)} origin-destination rows, {flows.year.min()}-{flows.year.max()}")
    if cpt.RAW_DIR.exists():
        spend = cpt.load()
        spend.to_parquet(OUT_DIR / "italy_cpt_spending.parquet", index=False)
        print(f"CPT regional spending: {len(spend)} rows, {spend.year.min()}-{spend.year.max()}, {spend.sector.nunique()} sectors")
    if istat_unobserved.XLSX.exists():
        istat_unobserved.load_components().to_parquet(OUT_DIR / "italy_unobserved_economy.parquet", index=False)
        istat_unobserved.load_by_industry().to_parquet(OUT_DIR / "italy_underground_by_industry.parquet", index=False)
    else:
        print(f"Istat non-observed economy: {istat_unobserved.XLSX.name} not present, skipped (see docs/manual_data.md section 4)", file=sys.stderr)
    cohesion.load().to_parquet(OUT_DIR / "esif_absorption.parquet", index=False)
    if bready.XLSX.exists():
        bready.load().to_parquet(OUT_DIR / "italy_bready_cities.parquet", index=False)
    if euklems.RAW.exists():  # large optional download (not in git)
        euklems.load().to_parquet(OUT_DIR / "euklems_growth.parquet", index=False)
    else:
        print("euklems: raw CSV not present, skipped (see docs/data_dictionary.md)", file=sys.stderr)
    lh = ameco.load_long()
    lh.to_parquet(OUT_DIR / "long_history.parquet", index=False)
    print(f"long history: {len(lh)} rows, {lh.variable.nunique()} variables, {lh.year.min()}-{lh.year.max()}")
    se = build_selfemp()
    se.to_parquet(OUT_DIR / "selfemp.parquet", index=False)
    size = build_size()
    size.to_parquet(OUT_DIR / "size_class.parquet", index=False)
    print(f"size class: {len(size)} rows, {size.iso3.nunique()} countries, vintages {size.groupby('vintage').year.agg(['min', 'max']).to_dict('index')}")
    print(f"self-employment: {len(se)} rows")
    inc = build_income()
    inc.to_parquet(OUT_DIR / "income_dist.parquet", index=False)
    print(f"income distribution: {len(inc)} rows")
    mg = pd.concat([d.assign(table=k) for k, d in build_migration().items()], ignore_index=True)
    mg.to_parquet(OUT_DIR / "migration.parquet", index=False)
    print(f"migration: {len(mg)} rows")
    build_sectors64().to_parquet(OUT_DIR / "sectors64.parquet", index=False)
    ec = build_earnings_cases()  # break in 2024 (flag b, new basis of the average wage: catalog.SERIES_BREAKS): rates within a year are fine, levels across it are not
    ec.to_parquet(OUT_DIR / "earnings_cases.parquet", index=False)
    proj = build_projections()
    proj.to_parquet(OUT_DIR / "projections_eurostat.parquet", index=False)
    ar = ageing.load()
    ar.to_parquet(OUT_DIR / "ageing_report.parquet", index=False)
    print(f"projections: eurostat {len(proj)} rows, ageing report {len(ar)} rows ({ar.variable.nunique()} variables)")
    spending = build_spending()
    spending.to_parquet(OUT_DIR / "spending.parquet", index=False)
    print(f"spending: {len(spending)} rows, {spending.item.nunique()} items")
    sectors = build_sectors()
    sectors.to_parquet(OUT_DIR / "sectors.parquet", index=False)
    print(f"sectors: {len(sectors)} rows, {sectors.geo.nunique()} geos, {sectors.nace_r2.nunique()} industries")
    try:
        snap = oecd.load_snapshot()
        snap.to_parquet(OUT_DIR / "pensions_oecd_2024.parquet", index=False)
        print(f"pensions snapshot: {len(snap)} rows")
    except FileNotFoundError as exc:
        print(f"pensions snapshot skipped: {exc}", file=sys.stderr)
