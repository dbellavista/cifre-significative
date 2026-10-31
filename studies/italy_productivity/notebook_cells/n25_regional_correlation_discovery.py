"""Cells of notebook 25_regional_correlation_discovery."""
from . import NOTEBOOKS, C, M

NOTEBOOKS["25_regional_correlation_discovery"] = [
    (M, """# Regional correlation discovery (NUTS2)
The national screen (notebook 24) has about 30 countries. Here the unit is the **region** (about 230 NUTS2 regions with data in the 2015-19 window, 26 indicators plus derived ratios), which gives far more observations but needs care:
regions of one country share institutions, policy and the national cycle, and neighbours are not independent. So:
- the default screen works **within countries** (each indicator minus its country mean; countries with fewer than 3 regions drop out) and removes the log-GDP-per-head fit;
- robustness drops **whole countries** (leave-one-country-out); the permutation null **shuffles regions inside each country**;
- derived ratios that share a numerator or denominator (output per hour, pay per hour, labour share, hours per inhabitant, investment rate) are not paired with each other;
- Italy is tested with a **placebo ranking** (the same slope difference for every country with at least 5 regions) because Italy is a single cluster.
**Associations only, never causal.** Code: `src/cifresignificative/studies/italy_productivity/analysis/regional_corr.py`, `experiments.py`."""),
    (C, '''from cifresignificative.studies.italy_productivity.analysis import corr, regional_corr as rc, experiments as ex
reg = load_regional()
res = ex.run_regional_all(reg)
funnel = pd.DataFrame({"survivors": {k: len(v) for k, v in res.items() if k.startswith(("r1", "r2"))},
                       "robust": {k: int(v["robust"].sum()) for k, v in res.items() if k.startswith(("r1", "r2"))}})
display(funnel)
R = rc.regional_matrix(reg, drop_flag_chars="u")
X1 = rc.within_country_table(R, "level", (2015, 2019))
print("regions with data per indicator (2015-19):"); display(X1.notna().sum().sort_values().to_frame("regions").T)
print("discoveries after shuffling regions within countries:", rc.region_permutation_null(X1, n_min=80, n_perm=3))'''),
    (M, "## R1 - within-country associations, 2015-19 (income removed)\nSame-theme pairs (labour vs labour, demography vs demography) are near-definitional; the table keeps robust pairs from different themes."),
    (C, '''theme_of = reg.assign(indicator=reg.indicator.str.removeprefix("nuts2_")).groupby("indicator").theme.first()
r1 = res["r1_levels_within"]
cross = r1[r1.robust & (r1.x.map(theme_of) != r1.y.map(theme_of))].assign(x_theme=lambda d: d.x.map(theme_of), y_theme=lambda d: d.y.map(theme_of))
display(cross.head(20)[["x", "x_theme", "y", "y_theme", "n", "rho", "q_value"]].round(3))
print("between countries (n = countries):"); display(res["r1_levels_between"].head(8)[["x", "y", "n", "rho", "q_value", "robust"]].round(3))'''),
    (M, "## R2 - within-country associations in changes (2005-09 to 2015-19; break-flagged and provisional values dropped)"),
    (C, 'display(res["r2_changes_within"].head(15)[["x", "y", "n", "rho", "q_value", "robust"]].round(3))'),
    (M, "## R2b - permutation check of the p-values behind R1 and R2\nThe pooled Spearman p assumes countries share one dispersion; they do not, and a shuffled test rejects more than the nominal 5%. Here every pair of R1 and R2 is retested with the within-country permutation p (`regional_south.perm_p`, 20,000 shuffles, ranks pooled), BH over all pairs. The funnel's duplicate and robustness steps are not repeated."),
    (C, '''from cifresignificative.studies.italy_productivity.analysis import regional_south as rs
chk = pd.DataFrame({k: {"pairs tested": len(res[k]), "BH survivors, asymptotic p": int(res[k].significant_asymptotic.sum()), "BH survivors, permutation p": int(res[k].significant.sum()),
                        "both": int((res[k].significant & res[k].significant_asymptotic).sum()), "share p < 0.05, asymptotic": round(float((res[k].p_asymptotic < 0.05).mean()), 3), "share p < 0.05, permutation": round(float((res[k].p < 0.05).mean()), 3)}
                    for k in ["rp1_levels_perm", "rp2_changes_perm"]})
display(chk)
lost = pd.concat([res[k][res[k].significant_asymptotic & ~res[k].significant].assign(screen=k[:3]) for k in ["rp1_levels_perm", "rp2_changes_perm"]])
print("asymptotic survivors that lose significance under the permutation p:"); display(lost[["screen", "x", "y", "n", "rho", "p_asymptotic", "p", "q_value"]].round(4))
rng = np.random.default_rng(0); g = rc.country_of(X1.index); Xn = X1.drop(columns=corr.size_scaled(X1.columns)); null = []
for k in range(50):
    t = rs.screen_perm(corr.shuffle_within(Xn, g, rng), n_perm=1000, seed=k)
    null.append({"discoveries, asymptotic": int(t.significant_asymptotic.sum()), "discoveries, permutation": int(t.significant.sum()), "share p < 0.05, asymptotic": (t.p_asymptotic < 0.05).mean(), "share p < 0.05, permutation": (t.p < 0.05).mean()})
print("shuffled regions within countries, 50 runs (R1 table), mean:"); display(pd.DataFrame(null).mean().round(3).to_frame("mean"))'''),
    (M, "## R3 - regional hypotheses with region + year fixed effects, SEs clustered by country\nThe `RH` rows were specified in advance (regional analogues of the literature hypotheses); FDR is applied to them only. `S` rows were picked from the R1 survivors, so their p-values are not independent evidence."),
    (C, '''h = res["r3_region_panel"]
display(h[["id", "label", "expected_sign", "coef", "p", "q_value", "sign_agrees", "p_ex_irl", "n_obs", "n_regions"]].round(3))
ll = res["r3_lead_lag"]
display(ll.pivot_table(index=["id", "x", "y"], columns="lag", values="verdict", aggfunc="first"))'''),
    (M, "## R4 - is Italy's within-country slope different?\nSlope of y on x across regions after removing country means, Italy against all other countries, and the same difference for every other country as a placebo (p = share of countries with an equal or larger |difference|; the smallest possible p is 1 / number of countries). The last columns split Italy into North and South + Islands (exploratory, about 10 regions each)."),
    (C, '''s = res["r4_italy_slope"]
display(s[["id", "x", "y", "slope_italy", "slope_others", "slope_diff_italy", "p_diff", "n_countries", "slope_north", "slope_south"]].round(3))'''),
    (M, "## R5 - convergence of regional income\nAnnualised log growth of GDP per head (PPS) between 2005-07 and 2017-19 on the initial log level, country fixed effects, SEs clustered by country. beta < 0 means poorer regions catch up. Second block drops the richest region of each country."),
    (C, '''display(res["r5_convergence"].round(3))
lv0 = corr.cross_section(R[["gdp_pps_pc_eu27"]], "level", (2005, 2007), entity_level="geo")["gdp_pps_pc_eu27"]
lv1 = corr.cross_section(R[["gdp_pps_pc_eu27"]], "level", (2017, 2019), entity_level="geo")["gdp_pps_pc_eu27"]
g = pd.DataFrame({"ln0": np.log(lv0), "growth": (np.log(lv1) - np.log(lv0)) / 12 * 100}).dropna()
g["country"] = g.index.str[:2]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, cs, title in [(axes[0], ["IT"], "Italy"), (axes[1], ["DE", "ES", "FR", "PL"], "Germany, Spain, France, Poland")]:
    sub = g[g.country.isin(cs)]
    for c, grp in sub.groupby("country"):
        ax.scatter(grp.ln0, grp.growth, label=c, s=28)
        if len(grp) > 3: ax.plot(np.sort(grp.ln0), np.polyval(np.polyfit(grp.ln0, grp.growth, 1), np.sort(grp.ln0)), lw=1)
    ax.set_title(title); ax.set_xlabel("log GDP per head, 2005-07 (PPS)"); ax.legend(fontsize=8)
axes[0].set_ylabel("growth, % a year, 2005-07 to 2017-19")
plt.tight_layout(); plt.show()'''),
]

NOTEBOOKS["25_regional_correlation_discovery"].append((M, """## Takeaways (Eurostat NUTS2, 2015-19 unless stated; descriptive associations only)
- **The funnel (within countries, income removed):** 165 pairs survive false-discovery control in the 2015-19 levels (161 pass the leave-one-country-out and rank-vs-linear checks), 59 in the 2005-09 to 2015-19 changes, and 88 between countries (n about 27). Shuffling regions inside countries gives 0 discoveries in 3 runs. Most of the strongest pairs are labour-market pairs (long-term vs total unemployment, employment vs unemployment), so the cross-theme rows are more informative: AROPE with unemployment (rho 0.73, 137 regions), early school leaving with NEET (0.71), R&D personnel and physician density with tertiary attainment (about 0.6).
- **Permutation check (R2b):** retesting every pair with the within-country permutation p keeps 164 of the 165 levels survivors and 53 of the 59 changes survivors; the seven that drop out are weak (|rho| 0.14-0.21, permutation q 0.050-0.073). The check is needed: on shuffled regions the asymptotic p falls below 0.05 in 6.1% of pairs (0.12 discoveries per run, 50 runs), the permutation p in 5.0% (0 discoveries).
- **Pre-specified regional hypotheses (region + year fixed effects, country-clustered SEs):** 5 of 10 are significant at q < 0.05 and 4 go the expected way: early leaving and NEET, net migration and income per head, output and pay per hour (shares the hours denominator), unemployment and the investment rate. Broadband coverage and output per hour move in the *opposite* direction (coefficient -0.14), most likely because coverage saturates while productivity keeps changing. Graduate share and output per hour is borderline (q = 0.059); R&D intensity, graduates and female employment, and physicians and life expectancy show nothing.
- **Cross-sectional links do not survive region fixed effects.** All six AROPE pairs picked from the within-country screen are unrelated once region and year effects are added (p between 0.54 and 0.92; AROPE only starts in 2015), so differences between regions are persistent, not something that moves with time.
- **Lead-lag (1-3 years):** unemployment leads the investment rate (all lags); early leaving leads NEET at 2-3 years; old-age dependency leads old-age employment; output per hour leads the graduate share at 2-3 years (not the reverse); broadband-output, income-migration and output-pay hold in both directions, and R&D, graduates-female employment and physicians-life expectancy show no order.
- **Italy's slopes are the steepest of 14-18 countries in three relations** (graduate share and output per hour 1.18 vs 0.55; graduate share and female employment 3.15 vs 0.40; early leaving and NEET 1.62 vs 0.81), which is the smallest p the placebo ranking can give (1/17, 1/18, 1/14, so p = 0.06 to 0.07): suggestive, not significant, because Italy is one country. They are concentrated in the South and Islands (graduates and female employment: slope 3.03 there, -0.84 in the North; rho 0.79 vs -0.35), and about 10 regions each, so exploratory.
- **Convergence of regional income (2005-07 to 2017-19, country effects):** pooled beta -0.11 (p = 0.60, no convergence), Germany -1.13 (p < 0.001), Italy +0.47 (p = 0.21; +0.17 without the richest region), Spain +0.66, France +0.23, Poland +0.26 (none significant). Italy's South did not catch up with the North in this window.
- **Limits:** about 230 regions but only about 27 countries as clusters; NUTS codes mix vintages (coverage differs by indicator: 137 regions for AROPE, 235 for GDP-based ratios); LFS series carry many break flags (dropped for changes and panels, kept for single-window levels); output per hour uses current prices, not PPPs, so only within-country comparisons are reliable; no causal identification."""))
