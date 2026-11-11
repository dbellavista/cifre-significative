"""Cells of notebook 24_correlation_discovery."""
from . import NOTEBOOKS, C, M

NOTEBOOKS["24_correlation_discovery"] = [
    (M, """# Correlation discovery
Which indicators move together across the EU27 + GBR/USA/JPN, and where does Italy sit relative to those relationships?
**Associations only, never causal.** About 30 countries and ~15,000 indicator pairs means a naive screen is mostly noise,
so every screen below is followed by Benjamini-Hochberg false-discovery control and robustness stamps
(leave-one-country-out, without Ireland, Spearman vs Pearson). Size-scaled series (counts, EUR millions) and near-duplicates
(|rho| >= 0.95 between differently named series) are dropped. Code: `src/cifresignificative/studies/italy_productivity/analysis/corr.py`, `experiments.py`."""),
    (C, '''from cifresignificative.studies.italy_productivity.analysis import corr, experiments as ex
M_ = corr.panel_matrix(panel)
res = ex.run_all(panel)
funnel = pd.DataFrame({"survivors (FDR 5%, not duplicate)": {k: len(v) for k, v in res.items() if k.startswith(("e1", "e2", "e3"))},
                       "robust": {k: int(v["robust"].sum()) if "robust" in v else np.nan for k, v in res.items() if k.startswith(("e1", "e2", "e3"))}})
display(funnel)'''),
    (M, "## Why we control for income\nRich countries are better on almost everything, so a raw cross-section mostly rediscovers one 'development' factor. `e1_levels` correlates what is left after removing the log-GDP-per-head fit; `e1_levels_raw` does not."),
    (C, '''top = res["e1_levels"].head(15)[["x", "y", "n", "rho", "q_value", "robust"]]
display(top.round(3))
# null check: shuffling each indicator across countries should leave ~0 discoveries
X1 = corr.cross_section(M_, "level", (2015, 2019))
print("discoveries on shuffled data:", corr.permutation_null(X1, n_perm=3))'''),
    (M, "## E2 - do improvements co-occur? (change 2005-09 to 2015-19)"),
    (C, 'display(res["e2_changes"].head(15)[["x", "y", "n", "rho", "q_value", "robust"]].round(3))'),
    (M, "## E3 - within-country co-movement (year-on-year changes, demeaned by country)"),
    (C, 'display(res["e3_within"].head(15)[["x", "y", "n", "rho", "q_value"]].round(3))'),
    (M, "## E5 - what moves with Italy's gap?\nSpearman correlation of year-on-year changes in the gap (Italy minus median of the other EU27) for each indicator against the gap in the target."),
    (C, '''for t in ("e5_italy_gap_gdp", "e5_italy_gap_employment"):
    print(t); display(res[t].head(8).round(3))'''),
    (M, "## E7 - literature hypotheses with panel fixed effects\nEntity + year FE, entity-clustered SEs, also without Ireland. `italy_resid_z` = Italy's distance from the cross-country line fitted on the other countries (2015-19), in residual SDs."),
    (C, '''h = res["e7_hypotheses"]
display(h[["id", "label", "expected_sign", "coef", "p", "q_value", "sign_agrees", "p_ex_irl", "italy_resid_z", "n_obs"]].round(3))'''),
    (M, "## E4 - lead-lag\nFor each hypothesis: does x(t-k) predict y(t) (entity + year FE), and does y(t-k) predict x(t)? Verdict 'x leads' / 'y leads' when only one direction is significant at 5%. Predictive order only."),
    (C, '''ll = res["e4_lead_lag"]
display(ll.pivot_table(index=["id", "x", "y"], columns="lag", values="verdict", aggfunc="first"))'''),
    (C, '''fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (x, y) in zip(axes, [("pension_spend_pct_gdp", "neet_rate_15_29"), ("early_leavers_18_24", "neet_rate_15_29"), ("gdp_growth", "gov_debt_pct_gdp")]):
    d = X1[[x, y]].dropna()
    ax.scatter(d[x], d[y], color="0.6")
    ax.scatter(d.loc[["ITA"], x], d.loc["ITA", y], color="C0", zorder=3)
    b, a = np.polyfit(d.drop("ITA")[x], d.drop("ITA")[y], 1)
    xs = np.linspace(d[x].min(), d[x].max(), 20); ax.plot(xs, a + b * xs, "--", color="C1")
    ax.set_xlabel(x); ax.set_ylabel(y)
plt.tight_layout(); plt.show()'''),
]

NOTEBOOKS["24_correlation_discovery"].append((M, """## Takeaways (Eurostat, World Bank, IMF, AMECO, OECD; descriptive associations only)
- **Income drives most of the raw signal.** The 2015-19 cross-section yields about 1,000 FDR-significant pairs; after removing the log-GDP-per-head fit about 260 remain (206 before Eurostat's release of 5 November 2026, which revised GDP per head for five members: the count is sensitive to the income control). Shuffling indicators across countries gives 0 discoveries, so the screen is not producing noise on its own.
- **The leaders are mostly definitional or same-family pairs** (life expectancy vs preventable mortality, PISA maths vs science, unemployment vs youth unemployment). Name-based filters catch the obvious ones; the list still needs human curation before any pair is read as a finding. Robustness checks barely discriminate among FDR survivors (nearly all pass), so they flag fragile pairs rather than rank good ones.
- **Italy's gap in GDP per head moves with its gaps in debt and pensions.** Year-on-year changes in the gap correlate at -0.71 with government debt/GDP, -0.71 with the IMF debt series (the same quantity, so not independent) and -0.68 with pension spending (q = 0.03 to 0.05, n = 19-20 years). No indicator passes FDR for the employment-rate gap (best: current account, rho = -0.62, q = 0.19).
- **Literature hypotheses (panel FE, entity + year effects):** 9 of 13 are significant at q < 0.05, and 7 of those go the expected way (TFP-GDP, yields-investment, growth-debt, pensions-55-64 employment, pensions-NEET, female employment-GDP, early leaving-NEET). Two significant results go the *opposite* way to the stated expectation (H4 public investment vs balance, H5 pensions vs education spending) and should be treated as refuted or mis-specified, not as support. Four show no significant association (graduates-emigration, childcare-female employment, TFP-net earnings, government effectiveness-GDP; q > 0.2).
- **Italy against the cross-country line (2015-19):** NEET is 2.6 residual SDs above the line implied by pension spending and 2.6 above the one implied by early school leaving; female employment is 2.3 SDs below the line implied by full-time childcare coverage.
- **Lead-lag (predictive order, 1-3 years):** female employment predicts GDP per head at every lag but not the reverse (H9); long-term yields predict investment (H2); early school leaving predicts NEET at 2-3 years (H13); pension spending predicts 55-64 employment at 2-3 years (H7). The reverse direction dominates for childcare vs female employment (H10), budget balance vs public investment (H4) and pension vs education spending (H5), so the stated direction of those hypotheses is not supported. TFP-GDP (H1) is significant both ways, and TFP-earnings, government effectiveness-GDP and graduates-emigration show nothing either way.
- **Limits:** ~30 countries and 15-20 years; provisional and broken observations are dropped; same-quantity pairs under different names can survive; no causal identification, and lead-lag tests only show predictive order."""))
