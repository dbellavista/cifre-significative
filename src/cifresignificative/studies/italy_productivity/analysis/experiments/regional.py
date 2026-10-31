"""NUTS2 experiments: regional screens, permutation checks, convergence and the 'why the South' round."""
from pathlib import Path

import pandas as pd

from cifresignificative.studies.italy_productivity.analysis import corr
from cifresignificative.studies.italy_productivity.analysis import regional_corr as rc
from cifresignificative.studies.italy_productivity.analysis import regional_south as rs
from cifresignificative.studies.italy_productivity.analysis.data import (
    PROCESSED,
    load_regional,
    load_regional_extra,
)

from .common import OUT, WINDOWS


def run_regional_screen(regional: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """R1 / R1b / R2: region screens. Within-country (country-demeaned, income removed), the same between countries, and changes."""
    regional = load_regional() if regional is None else regional
    R = rc.regional_matrix(regional, drop_flag_chars="u")
    Rb = rc.regional_matrix(regional, drop_flag_chars="ubp")  # changes across windows: drop breaks and provisional values too
    out = {
        "r1_levels_within": rc.screen_regions(rc.within_country_table(R, "level", WINDOWS["recent"])).assign(experiment="r1_levels_within"),
        "r1_levels_between": corr.screen(rc.between_country_table(R, WINDOWS["recent"]), exclusions=rc.MECHANICAL).assign(experiment="r1_levels_between"),
        "r2_changes_within": rc.screen_regions(rc.within_country_table(Rb, "change", WINDOWS["recent"], WINDOWS["early"])).assign(experiment="r2_changes_within"),
    }
    return out


def run_regional_perm_check(regional: pd.DataFrame | None = None, n_perm: int = 20000) -> dict[str, pd.DataFrame]:
    """RP1 / RP2: the R1 and R2 pair screens again with the within-country permutation p (`regional_south.screen_perm`); compare `significant` with `significant_asymptotic`."""
    regional = load_regional() if regional is None else regional
    R = rc.regional_matrix(regional, drop_flag_chars="u")
    Rb = rc.regional_matrix(regional, drop_flag_chars="ubp")
    return {
        "rp1_levels_perm": rs.screen_perm(rc.within_country_table(R, "level", WINDOWS["recent"]), n_perm).assign(experiment="rp1_levels_perm"),
        "rp2_changes_perm": rs.screen_perm(rc.within_country_table(Rb, "change", WINDOWS["recent"], WINDOWS["early"]), n_perm).assign(experiment="rp2_changes_perm"),
    }


def run_regional_hypotheses(regional: pd.DataFrame | None = None, screens: dict[str, pd.DataFrame] | None = None, n_extra: int = 8) -> dict[str, pd.DataFrame]:
    """R3 (region + year FE, SEs clustered by country, with and without Ireland, lead-lag verdicts) and R4 (does Italy's within-country
    slope differ?) for the regional analogues of the literature hypotheses plus the top `n_extra` robust R1 survivors."""
    regional = load_regional() if regional is None else regional
    R = rc.regional_matrix(regional, drop_flag_chars="ubp")
    Xw = rc.within_country_table(rc.regional_matrix(regional, drop_flag_chars="u"), "level", WINDOWS["recent"], income_control=False)
    pairs = [(h[0], h[1], h[2], h[3], h[4], h[5]) for h in rc.HYPOTHESES]
    if screens and "r1_levels_within" in screens:
        top = screens["r1_levels_within"]
        theme_of = regional.assign(indicator=regional.indicator.str.removeprefix("nuts2_")).groupby("indicator").theme.first()
        known = {frozenset((h[2], h[3])) for h in rc.HYPOTHESES}
        top = top[top.robust & (top.x.map(theme_of) != top.y.map(theme_of)) & ~top.apply(lambda r: frozenset((r.x, r.y)) in known, axis=1)].head(n_extra)  # same-theme pairs (labour vs labour) are near-definitional
        pairs += [(f"S{i + 1}", "R1 survivor", r.y, r.x, 0, f"{r.x} ~ {r.y} (R1)") for i, r in enumerate(top.itertuples())]
    irl = tuple(g for g in R.index.get_level_values("geo").unique() if g.startswith("IE"))
    rows, lag_rows, slope_rows = [], [], []
    for hid, theme, y, x, sign, label in pairs:
        if x not in R or y not in R:
            continue
        r, r_ex = corr.panel_fe(R, y, [x], entity_level="geo", cluster_country=True), corr.panel_fe(R, y, [x], exclude=irl, entity_level="geo", cluster_country=True)
        sl = rc.italy_slope_test(Xw, x, y)
        if sl:
            slope_rows.append({"id": hid, **sl})
        if not r:
            continue
        coef = r["coef"][x]
        rows.append({"id": hid, "theme": theme, "y": y, "x": x, "label": label, "expected_sign": sign, "selected_from_screen": hid.startswith("S"), "coef": coef, "p": r["p"][x], "n_obs": r["n_obs"],
                     "n_regions": r["n_countries"], "coef_ex_irl": r_ex.get("coef", {}).get(x), "p_ex_irl": r_ex.get("p", {}).get(x),
                     "sign_agrees": bool(coef * sign > 0) if sign else None})
        ll = corr.lead_lag(R, x, y, entity_level="geo", cluster_country=True)
        for k, g in ll.groupby("lag"):
            g = g.set_index("direction")
            if {"x_leads_y", "y_leads_x"} - set(g.index):
                continue
            sx, sy = g.p["x_leads_y"] < 0.05, g.p["y_leads_x"] < 0.05
            lag_rows.append({"id": hid, "x": x, "y": y, "lag": k, "p_x_leads_y": g.p["x_leads_y"], "p_y_leads_x": g.p["y_leads_x"], "coef_x_leads_y": g.coef["x_leads_y"],
                             "coef_y_leads_x": g.coef["y_leads_x"], "verdict": {(True, False): "x leads", (False, True): "y leads", (True, True): "both"}.get((bool(sx), bool(sy)), "neither")})
    panel_fe = pd.DataFrame(rows)
    if len(panel_fe):  # FDR over the pre-specified hypotheses only: pairs picked from the screen were selected on the same kind of data
        pre = ~panel_fe.selected_from_screen
        panel_fe.loc[pre, "q_value"] = corr.fdr(panel_fe[pre], 0.05).q_value
    slopes = pd.DataFrame(slope_rows)
    return {"r3_region_panel": panel_fe, "r3_lead_lag": pd.DataFrame(lag_rows), "r4_italy_slope": slopes}


def run_convergence(regional: pd.DataFrame | None = None) -> pd.DataFrame:
    """R5: beta-convergence of regional GDP per head (PPS), with and without the richest region of each country."""
    regional = load_regional() if regional is None else regional
    R = rc.regional_matrix(regional, drop_flag_chars="ubp")
    return pd.concat([rc.convergence(R, drop_top_region=False), rc.convergence(R, drop_top_region=True)], ignore_index=True)


def run_regional_all(regional: pd.DataFrame | None = None, out: Path = OUT) -> dict[str, pd.DataFrame]:
    regional = load_regional() if regional is None else regional
    out.mkdir(parents=True, exist_ok=True)
    results = run_regional_screen(regional)
    results.update(run_regional_perm_check(regional))
    results.update(run_regional_hypotheses(regional, results))
    results["r5_convergence"] = run_convergence(regional)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_south_all(regional: pd.DataFrame | None = None, extra: pd.DataFrame | None = None, flows: pd.DataFrame | None = None, spending: pd.DataFrame | None = None,
                  out: Path = OUT, n_perm: int = 2000) -> dict[str, pd.DataFrame]:
    """The "why the South" round (`regional_south`): S1 EU-wide family by EQI wave (plain and beyond income) with its permutation null, S2 Italy's rank among countries, S3 Italy-only exploratory family and null,
    S4 the two tracks side by side, S5 North-South gaps in standard deviations."""
    regional = load_regional() if regional is None else regional
    extra = load_regional_extra() if extra is None else extra
    flows = pd.read_parquet(PROCESSED / "italy_internal_migration.parquet") if flows is None else flows
    spending = pd.read_parquet(PROCESSED / "italy_cpt_spending.parquet") if spending is None else spending
    R = rc.regional_matrix(pd.concat([regional, extra], ignore_index=True), drop_flag_chars="u")
    s1 = pd.concat([rs.eu_wide(R, income_control=False), rs.eu_wide(R, income_control=True)], ignore_index=True)
    W = rs.within_country(rs.wave_table(R, rs.PRIMARY_WAVE), income_control=False)
    s1_null = rs.family_null(W, rs.EU_WIDE, rs.OUTCOMES, rc.country_of(W.index))
    s2 = rs.italy_placebo(R)
    T = rs.italy_table(R, flows, spending)
    s3 = rs.italy_exploratory(T, [*rs.EU_WIDE, *rs.ITALY_ONLY], n_perm=n_perm)
    s4 = rs.compare_tracks(s2, s3)
    s5 = rs.zone_gaps(T, [*rs.OUTCOMES, *rs.EU_WIDE, *rs.ITALY_ONLY])
    results = {"s1_eu_wide": s1, "s1_null": s1_null, "s2_italy_rank": s2, "s3_italy_only": s3, "s4_compare": s4, "s5_zone_gaps": s5, "s_italy_table": T.reset_index()}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results
