"""Regulation (PMR/EPL) and collective bargaining experiments."""
from pathlib import Path

import pandas as pd

from cifresignificative.core.sources import oecd_bargaining as OB
from cifresignificative.core.sources import oecd_regulation as OR
from cifresignificative.studies.italy_productivity.analysis import bargaining as BG
from cifresignificative.studies.italy_productivity.analysis import hf as H
from cifresignificative.studies.italy_productivity.analysis import industry_panel as IP
from cifresignificative.studies.italy_productivity.analysis import regulation as RG_
from cifresignificative.studies.italy_productivity.analysis.data import (
    load_sectors64,
)

from .common import OUT


def run_regulation_all(out: Path = OUT, n_perm: int = 20000, n_rep: int = 200) -> dict[str, pd.DataFrame]:
    """G1-G6: Italy's position in the OECD PMR and EPL, and the sector test (regulation of the sector in 2018 against productivity growth of the sector, 2005-07 to 2017-19 as the primary specification,
    robustness: early and late windows, PMR 2023). BH over the primary family (pooled + the sectors with MIN_COUNTRIES countries) only; the robustness rows are descriptive."""
    pmr, epl = OR.load_pmr(), OR.load_epl()
    s = load_sectors64()
    rows, primary = [], None
    for wname, w in IP.WINDOWS.items():
        for year in (2018, 2023):
            d = RG_.sector_data(pmr, s, w, year)
            if (wname, year) == ("full", 2018):
                primary = d
            rows.append(RG_.test(d, n_perm=n_perm).assign(window=wname, pmr_year=year, primary=(wname, year) == ("full", 2018)))
    assert primary is not None
    results = {"g1_pmr_position": RG_.position(pmr), "g2_epl_gap": RG_.epl_gap(epl).reset_index(), "g3_sector_test": pd.concat(rows, ignore_index=True), "g4_italy_sectors": RG_.italy_rows(primary),
               "g5_italy_placebo": RG_.italy_placebo(primary), "g6_null": pd.DataFrame([RG_.shuffled_null(primary, n_rep=n_rep)])}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results


def run_bargaining_all(out: Path = OUT, n_perm: int = 20000, n_rep: int = 500) -> dict[str, pd.DataFrame]:
    """W1-W4: bargaining coverage and union density against the country pass-through of notebook 26 (`hf3_passthrough_countries`, rebuilt from the monthly data). BH over the two primary tests; the robustness
    rows (without Greece and Portugal, 2010-19 means, at least 6 observations) are descriptive."""
    cbc, tud = OB.load("cbc"), OB.load("tud")
    pt = H.passthrough_by_country(H.load_highfreq(), ("2005-01-01", "2021-06-30"))
    specs = {"primary": (BG.institution_means(cbc, tud), ()), "without GRC and PRT": (BG.institution_means(cbc, tud), ("GRC", "PRT")),
             "2010-19 means": (BG.institution_means(cbc, tud, (2010, 2019)), ()), "at least 6 observations": (BG.institution_means(cbc, tud, min_years=6), ())}
    assoc = pd.concat([BG.association(pt, inst, n_perm, exclude=ex).assign(spec=k, primary=k == "primary") for k, (inst, ex) in specs.items()], ignore_index=True)
    inst = specs["primary"][0]
    results = {"w1_institutions": inst.rename_axis("iso3").reset_index(), "w2_association": assoc, "w3_italy": BG.italy_position(pt, inst), "w4_null": pd.DataFrame([BG.shuffled_null(pt, inst, n_rep=n_rep)])}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results
