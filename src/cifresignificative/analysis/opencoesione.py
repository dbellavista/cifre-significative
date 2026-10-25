"""OpenCoesione 2014-2020 by region against the notebook 34 outcomes (Italy-only track, exploratory).

Predictors per region (21 areas, Trentino-Alto Adige as its two provinces; `sources/opencoesione.py`: project financing divided equally among a project's regions, net of economies), in euro per inhabitant (mean population 2014-2019):
the EU part of all funds (`oc_eu_pc_eur`), the national co-financing of the EU programmes (national public part of the ERDF and ESF projects, `oc_cofin_pc_eur`), the national Development and Cohesion Fund (`oc_fsc_pc_eur`, 80% to the South by law) and the share of the financing paid out by mid-2026
(`oc_payment_rate_pct`, all funds: execution). Outcomes: the notebook 34 ones, window means 2015-19. The design is `regional_south.italy_exploratory` (Spearman across the 21 regions, permutation p across regions and within the three zones, leave-one-region-out, BH over the family): the pre-specified family is 4 predictors x 6 outcomes;
a shuffled-predictor null must give about zero discoveries. There are no placebo countries for Italy-only data and the funds are allocated by rule to the poorer regions, so the reading is on the within-zone column. Descriptive, never causal.
"""
import numpy as np
import pandas as pd

from cifresignificative.analysis import regional_south as rs

XS = ["oc_eu_pc_eur", "oc_cofin_pc_eur", "oc_fsc_pc_eur", "oc_payment_rate_pct"]
WINDOW = (2014, 2019)


def region_predictors(oc: pd.DataFrame, regional: pd.DataFrame, window: tuple[int, int] = WINDOW) -> pd.DataFrame:
    """Region (NUTS2) x predictors, euro per inhabitant (population: mean of the window)."""
    pop = regional[regional.indicator.eq("nuts2_population") & regional.year.between(*window)].groupby("geo").value.mean()
    by = oc.groupby(["area", "fund"])[["eu_meur", "national_public_meur", "total_public_meur", "payments_meur"]].sum()
    eu_programmes = oc[oc.fund.isin(["FESR", "FSE"])].groupby("area").national_public_meur.sum()
    fsc = oc[oc.fund.eq("FSC")].groupby("area").total_public_meur.sum()
    tot = by.groupby("area").sum()
    out = pd.DataFrame({"oc_eu_pc_eur": tot.eu_meur, "oc_cofin_pc_eur": eu_programmes, "oc_fsc_pc_eur": fsc}).mul(1e6).div(pop.reindex(tot.index), axis=0)
    out["oc_payment_rate_pct"] = tot.payments_meur / tot.total_public_meur * 100
    out.index.name = "geo"
    return out.loc[[g for g in out.index if g in pop.index]]


def link(T: pd.DataFrame, pred: pd.DataFrame, n_perm: int = 2000, seed: int = 0) -> pd.DataFrame:
    """The pre-specified family on the notebook 34 table `T` (outcomes of the 21 regions) joined with the predictors."""
    X = T[rs.OUTCOMES].join(pred[XS], how="inner")
    return rs.italy_exploratory(X, XS, rs.OUTCOMES, n_perm=n_perm, seed=seed)


def null(T: pd.DataFrame, pred: pd.DataFrame, n_shuffle: int = 100, n_perm: int = 300, seed: int = 0) -> dict:
    """Calibration: the predictors shuffled across regions (jointly, so their mutual correlation stays), the family rerun; share of shuffles with any BH discovery (q_perm <= 0.05) and the mean number of discoveries, in both the unrestricted and the within-zone column."""
    rng = np.random.default_rng(seed)
    base = T[rs.OUTCOMES].join(pred[XS], how="inner")
    hits, hits_z = [], []
    for _ in range(n_shuffle):
        sh = base.copy()
        sh[XS] = base[XS].iloc[rng.permutation(len(base))].to_numpy()
        f = rs.italy_exploratory(sh, XS, rs.OUTCOMES, n_perm=n_perm, seed=int(rng.integers(1 << 30)))
        hits.append(int((f.q_perm <= 0.05).sum()))
        hits_z.append(int((f.q_perm_within_zone <= 0.05).sum()))
    return {"n_shuffle": n_shuffle, "share_any_discovery": float(np.mean(np.array(hits) > 0)), "mean_discoveries": float(np.mean(hits)), "share_any_discovery_within_zone": float(np.mean(np.array(hits_z) > 0)), "mean_discoveries_within_zone": float(np.mean(hits_z))}


def zone_table(pred: pd.DataFrame) -> pd.DataFrame:
    """Mean of each predictor by zone (North, Centre, South and Islands) and the South-to-North ratio."""
    z = pd.Series([rs.zone_of(g) for g in pred.index], index=pred.index)
    g = pred[XS].groupby(z).mean().reindex(["North", "Centre", "South and Islands"])
    g.loc["South / North"] = g.loc["South and Islands"] / g.loc["North"]
    return g
