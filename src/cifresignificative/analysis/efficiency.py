"""Spending-against-outcome pairs (health, education, old-age) used in notebook 15 and the dashboard."""
import numpy as np
import pandas as pd

from cifresignificative.analysis.data import EU27_ISO3, ITALY, spend_wide, wide

# title, x label, y label, lower_is_better
PAIRS = {
    "health_le": ("Health spending vs life expectancy", "Public health spending, % GDP", "Life expectancy at birth, years", False),
    "health_tm": ("Health spending vs treatable mortality", "Public health spending, % GDP", "Treatable deaths per 100,000", True),
    "edu_pisa": ("Education spending vs PISA", "Public education spending, % GDP", "PISA mean score (2015, 2018)", False),
    "pension_arop": ("Old-age spending vs poverty of the old", "Public old-age spending, % GDP", "At-risk-of-poverty rate, 65+ (%)", True),
}


def pair_data(panel: pd.DataFrame, spending: pd.DataFrame, years: tuple[int, int] = (2015, 2019)) -> dict[str, tuple[pd.Series, pd.Series]]:
    """key -> (x, y) EU27 series averaged over `years`."""
    def avg(w: pd.DataFrame) -> pd.Series:
        return w.loc[years[0]:years[1], [c for c in EU27_ISO3 if c in w]].mean()
    pisa = pd.concat([wide(panel, f"wb_pisa_{k}").loc[[2015, 2018]].mean() for k in ("math", "reading", "science")], axis=1).mean(axis=1).reindex(EU27_ISO3)
    return {"health_le": (avg(spend_wide(spending, "GF07")), avg(wide(panel, "life_expectancy_birth"))),
            "health_tm": (avg(spend_wide(spending, "GF07")), avg(wide(panel, "treatable_mortality_rate"))),
            "edu_pisa": (avg(spend_wide(spending, "GF09")), pisa),
            "pension_arop": (avg(spend_wide(spending, "GF1002")), avg(wide(panel, "arop_65plus")))}


def residual_summary(x: pd.Series, y: pd.Series, lower_better: bool) -> dict:
    """Italy's distance from the line fitted on the other members, in standard deviations of the others' residuals."""
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    o = d.drop(ITALY)
    b = np.polyfit(o.x, o.y, 1)
    resid = d.y - np.polyval(b, d.x)
    z = resid / resid.drop(ITALY).std()
    better = -z if lower_better else z
    return {"Italy x": d.loc[ITALY, "x"], "Italy y": d.loc[ITALY, "y"], "fitted y": float(np.polyval(b, d.loc[ITALY, "x"])), "residual (SD)": float(z[ITALY]),
            "rank (1 = best)": int(better.rank(ascending=False)[ITALY]), "n": len(d), "corr (others)": float(o.x.corr(o.y))}
