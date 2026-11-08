"""Cells of notebook 05_policy_evaluation (Phase 3: Administrative Cycles, Lagged Two-Way FE & Placebo Ranking)."""
from . import NOTEBOOKS, C, M

NOTEBOOKS["05_policy_evaluation"] = [
    (M, """# Phase 3 — Comparative Evaluation: Administrative Cycles, Lagged Two-Way FE & Placebo Ranking

This notebook synthesizes the four pillars across **Cesena** (`040007`) and the 12 Emilia-Romagna peer municipalities (`N = 13` municipalities, `2008–2025`) to answer two empirical questions:
1. **How did Cesena perform across its administrative cycles (`Lucchi I 2009–2014`, `Lucchi II 2014–2019`, `Lattuca I 2019–2024`, `Lattuca II 2024+`) relative to Core Peers and All ER Peers?**
2. **Across the 13 Emilia-Romagna municipalities, do municipal fiscal and investment levers at year $t$ predict real-economy, income, or demographic changes at $t+1 \\dots t+4$ after absorbing municipality and year Fixed Effects?**

**Statistical Rigor & Identification Limits:**
- **Stationarity transforms:** Outcome series are transformed to annual log growth rates (`dlog_*`, $\\times 100$) or first differences (`diff_*`) before entering the panel regressions.
- **Two-Way Fixed Effects (`twoway_fe_regression`):** Every specification absorbs municipality fixed effects ($\\alpha_c$, time-invariant city traits) and year fixed effects ($\\tau_t$, common regional macro shocks), with standard errors clustered by municipality (`CR1` small-sample correction) and a within-year permutation $p$-value (`p_perm`, 5,000 permutations).
- **Benjamini-Hochberg FDR (`bh_fdr`) & Shuffled Null Check (`shuffled_null_check`):** We control the false discovery rate across all 140 tested predictor-outcome-lag pairs (`q_bh_perm`, `q_bh_cluster`) and run repeated shuffled panels to check calibration.
- **Placebo Ranking for Cesena-Specific Claims (`cesena_placebo_ranking`):** Because Cesena is a single cluster (`N = 1`), any claim about Cesena's long-run trajectory is evaluated via its rank among the 13 ER municipalities (`rank_1_to_13`). With 13 municipalities, the smallest attainable two-sided placebo rank proportion is $2/13 = 0.154$, so single-city rankings are reported as descriptive orderings. Descriptive only, never causal."""),
    (C, '''from cifresignificative.studies.cesena.analysis.data import (
    load_panel_comuni, load_administrations, load_scorecard, load_lagged_fe,
    load_placebo_ranking, load_extended_placebo_ranking, load_synthetic_cesena, load_synthetic_summary,
    load_synthetic_in_space, load_synthetic_in_time, load_flood_shock,
)
from cifresignificative.studies.cesena.analysis.experiments import (
    shuffled_null_check,
)
from cifresignificative.studies.cesena.analysis.flood import (
    compare_flood_tiers,
)

panel = load_panel_comuni()
admins = load_administrations()
scorecard = load_scorecard()
lagged_fe = load_lagged_fe()
ranking = load_placebo_ranking()
ext_ranking = load_extended_placebo_ranking()
synth_series = load_synthetic_cesena()
synth_summary = load_synthetic_summary()
synth_in_space = load_synthetic_in_space()
synth_in_time = load_synthetic_in_time()
flood_df = load_flood_shock()
flood_tiers = compare_flood_tiers(flood_df)

print("Cesena municipal administrative cycles (data/external/cesena_administrations.csv):")
display(admins)'''),
    (M, """## 1. Administrative Cycle Scorecard: Cesena vs. Core Peers & All ER Peers
We compare Cesena's period averages against the 6 Core Peers and all 12 ER Peers across `Lucchi I (2009–2014)`, `Lucchi II (2014–2019)`, `Lattuca I (2019–2024)`, and `Lattuca II (2024+)`. Recall that `Lucchi I` falls under the `D.P.R. 194/1996` accounting regime while `Lattuca I` falls under harmonized `D.Lgs. 118/2011` accounting (`Lucchi II` spans the 2016 transition), so fiscal levels are always compared *against peers in the same cycle*, never as an unadjusted cross-regime time trend."""),
    (C, '''piv_sc = scorecard.pivot(
    index=["metric", "metric_label"],
    columns="administration",
    values=["cesena", "core_peer_median", "gap_vs_core", "rank_1_to_13"],
)
order_cols = ["Lucchi I (2009-2014)", "Lucchi II (2014-2019)", "Lattuca I (2019-2024)", "Lattuca II (2024+)"]
avail_cols = [c for c in order_cols if c in scorecard["administration"].unique()]
for cyc in avail_cols:
    print(f"\\n=== {cyc} ===")
    sub = scorecard[scorecard["administration"] == cyc][[
        "metric_label", "cesena", "core_peer_median", "er_peer_median", "gap_vs_core", "gap_vs_er", "rank_1_to_13", "n_comuni"
    ]]
    display(sub.round(2))'''),
    (M, """## 2. Cesena's Long-Run Placebo Ranking Among the 13 Emilia-Romagna Municipalities
Where does Cesena (`040007`) rank among all 13 Emilia-Romagna municipalities on cumulative employment growth, local unit dynamics, firm scale, real household income growth, inequality, demographic aging, working-age attractiveness, and harmonized fiscal execution?"""),
    (C, '''print("Cesena Long-Run Placebo Ranking among the 13 Emilia-Romagna Municipalities (rank 1 = highest value):")
display(ranking[[
    "label", "period", "cesena_value", "peer_median_ex_cesena",
    "peer_min", "peer_max", "gap_vs_peer_median", "rank_1_to_13", "n_comuni", "placebo_two_sided_prop",
]].round(2))

fig, ax = plt.subplots(figsize=(9.5, 4.2))
r_plot = ranking.iloc[::-1]
y = np.arange(len(r_plot))
ax.scatter(r_plot["rank_1_to_13"], y, s=80, color=ITALY_C, zorder=3)
for i, (_, r) in enumerate(r_plot.iterrows()):
    ax.plot([1, r["n_comuni"]], [i, i], color=PEER_C, lw=1.2, zorder=1)
    ax.text(r["rank_1_to_13"] + 0.25, i, f"#{int(r['rank_1_to_13'])} ({r['cesena_value']:+.1f} vs med {r['peer_median_ex_cesena']:+.1f})", fontsize=8, va="center")
ax.axvline(7, color=EU_C, ls="--", lw=1.2, label="Median rank (#7 of 13)")
ax.set_yticks(y, r_plot["label"], fontsize=8.5)
ax.set_xlim(0.5, 16.5)
ax.set_xticks(range(1, 14))
ax.set_xlabel("Rank among 13 Emilia-Romagna Municipalities (1 = highest)")
ax.set_title("Cesena's Rank Across Key Multi-Year Trajectories (Placebo Ordering)", fontsize=9.5)
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.show()'''),
    (M, """## 3. Two-Way Fixed Effects Lagged Screen ($t \\to t+1 \\dots t+4$), BH-FDR Control & Shuffled Null Check
We screen 5 policy/fiscal predictors against 7 stationary economic, income, and demographic outcomes across lags $k \\in \\{1, 2, 3, 4\\}$ years (`140` tests total), controlling for municipality FE and year FE. We also run repeated shuffled panels; they show the test over-rejects, so it could not support a positive result."""),
    (C, '''null_res = shuffled_null_check(panel, lags=(1, 2, 3, 4), seed=12345)
print("Shuffled-municipality placebo null check:", null_res)

print(f"\\nTotal lagged Two-Way FE tests: {len(lagged_fe)}")
print(f"Tests with unadjusted p_perm < 0.05: {int((lagged_fe['p_perm'] < 0.05).sum())} | p_cluster < 0.05: {int((lagged_fe['p_cluster'] < 0.05).sum())}")
print(f"Tests surviving BH-FDR q_bh_perm < 0.05: {int((lagged_fe['q_bh_perm'] < 0.05).sum())} | q_bh_cluster < 0.05: {int((lagged_fe['q_bh_cluster'] < 0.05).sum())}")
print(f"Tests surviving q_bh_perm < 0.10: {int((lagged_fe['q_bh_perm'] < 0.10).sum())} | q_bh_cluster < 0.10: {int((lagged_fe['q_bh_cluster'] < 0.10).sum())}")

print("\\nTop 15 Lagged Two-Way FE Associations (sorted by permutation p-value):")
display(lagged_fe.head(15)[[
    "predictor_label", "outcome_label", "lag_years", "regime_scope",
    "n_obs", "n_comuni", "n_years", "beta_std", "t_stat", "p_cluster", "p_perm", "q_bh_cluster", "q_bh_perm", "r2_within",
]].round(3))'''),
    (M, """## 4. Expanding Cross-Sectional Power ($N = 36, 45, 73, 83$ Municipal Pools) & Synthetic Control for Cesena (`2019–2024`)
Because the 13-city regional benchmark (`N = 13`) caps the smallest attainable two-sided placebo rank proportion at $2/13 = 0.154$, we perform two complementary robustness exercises:
1. **Extended MEF Municipal Placebo Ranking (`extended_mef_placebo_ranking`):** Using the full national MEF IRPEF open-data archives (`2008` vs `2024`), we rank Cesena (`C573`, `77,117` taxpayers in 2024) across four progressively broader municipal donor pools:
   - **All 36 Emilia-Romagna municipalities with $\\ge 15,000$ taxpayers** (`ER_15k_payers`, $N = 36$);
   - **All 73 Emilia-Romagna municipalities with $\\ge 10,000$ taxpayers** (`ER_10k_payers`, $N = 73$);
   - **All 45 Northern & Central Italian mid-sized cities with `40,000–120,000` taxpayers** (`NC_midsize_40k_120k`, $N = 45$);
   - **All 83 Italian mid-sized cities with `40,000–120,000` taxpayers** (`IT_midsize_40k_120k`, $N = 83$).
2. **Convex Synthetic Control for Cesena (`synthetic_cesena`):** Following the synthetic-control design (`w_j >= 0`, `sum(w_j) = 1`), we match Cesena's pre-2019 cumulative trajectory (`t_0 .. 2018`; 2019 is a split chosen after seeing the data, so ranks are descriptive) from the donor pool of ER peers and evaluate the `2019–2024` post-2018 gap.
3. **In-Space and In-Time Placebos (`synthetic_control_placebos`):**
   - **In-space placebos (donor permutations):** Iteratively assign each of the 12 peer municipalities as a pseudo-treated unit, evaluating the distribution of post-treatment gaps and post-to-pre RMSPE ratios ($r_j = \\text{RMSPE}_{\\text{post}, j} / \\text{RMSPE}_{\\text{pre}, j}$) to compute exact permutation $p$-values.
   - **In-time placebos (pseudo-treatment year 2014):** Re-fit Synthetic Cesena on pre-2015 trajectories ($\\le 2014$) to verify that no artificial gap emerges in `2015–2018` before the true 2019 divergence."""),
    (C, '''print("1. Extended MEF Municipal Placebo Ranking (N=36, 73 ER municipalities; N=45 Northern/Central & N=83 Italian mid-sized cities):")
display(ext_ranking[[
    "pool_label", "metric_label", "cesena_value", "peer_median_ex_cesena",
    "gap_vs_peer_median", "rank", "n_comuni", "placebo_two_sided_p", "below_05",
]].round(3))

print("\\n2. Synthetic Cesena (Convex Donor Weights Matched on Pre-2019 Trajectory, Evaluated in 2024):")
display(synth_summary[[
    "label", "pre_period", "eval_year", "pre_rmse_pp",
    "cesena_end_pct", "synthetic_end_pct", "peer_median_end_pct",
    "gap_vs_synthetic_pp", "gap_vs_peer_median_pp", "donor_weights",
]].round(2))

print("\\n3. Synthetic Control In-Space Placebos (Cesena Rank and Exact Permutation p among All 13 Municipalities):")
ces_space = synth_in_space[synth_in_space["is_cesena"]].copy()
display(ces_space[[
    "label", "gap_end_pp", "rank_gap_desc", "perm_p_gap",
    "ratio_post_pre_rmspe", "rank_ratio_desc", "perm_p_ratio", "n_units",
]].round(3))

print("\\n4. Synthetic Control In-Time Placebo (Pseudo-Treatment Year 2014, Pre-Match <= 2014):")
display(synth_in_time[[
    "label", "pseudo_year", "pre_rmse_pp", "pseudo_post_mean_gap_pp", "true_post_mean_gap_pp", "end_gap_pp",
]].round(2))

fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.8))
for ax, (m_col, title) in zip(axes.flat, [
    ("employees", "Private Employment Cumulative Growth (%)"),
    ("avg_firm_size", "Average Unit Size Cumulative Growth (%)"),
    ("mean_income_real_eur", "Real Mean IRPEF Income Cumulative Growth (%)"),
    ("median_income_real_eur", "Real Median IRPEF Income Cumulative Growth (%)"),
]):
    sub = synth_series[synth_series["metric"] == m_col]
    ax.plot(sub["year"], sub["peer_median_cum_pct"], color=PEER_C, ls=":", lw=1.6, label="ER peer median (ex Cesena)")
    ax.plot(sub["year"], sub["synthetic_cesena_cum_pct"], color=EU_C, ls="--", lw=2.0, label="Synthetic Cesena (pre-2019 match)")
    ax.plot(sub["year"], sub["cesena_cum_pct"], color=ITALY_C, lw=2.3, marker="o", ms=3.5, label="Cesena")
    ax.axvline(2018.5, color="gray", lw=0.9, ls="--")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(title, fontsize=9.5)
    ax.set_ylabel("Cumulative change from base year (%)")
axes[0, 0].legend(fontsize=8)
plt.tight_layout()
plt.show()'''),
    (M, """## 5. The May 2023 Flood: Descriptive Comparison (2022–2025)

Municipalities are grouped by how Allegato 1 of D.L. 61/2023 lists them: whole territory (Cesena, Forlì, Faenza, Ravenna), part of the territory (Imola, Bologna) and not listed (Rimini, Reggio nell'Emilia, Parma, Modena, Piacenza, Carpi, Ferrara). A listing is a legal perimeter, not a measure of damage, and this is not a causal design.
- **Income (MEF IRPEF 2022 $\\to$ 2023).**
- **Capital cash payments, all purposes (SIOPE):** from 2022 and from a 2019–21 base. PNRR works ran in the same years, and nothing here identifies reconstruction money.
- **Employment and units (ASIA 2022 $\\to$ 2024):** with and without food manufacturing, because of Cesena's 2022 break.
- **Net migration (ISTAT):** internal plus international."""),
    (C, '''print("1. Multi-Tier Flood Impact Comparison (Median Values across Tiers):")
display(flood_tiers[[
    "tier_label", "n_municipalities",
    "income_growth_22_23_pct_median", "taxpayer_growth_22_23_pct_median",
    "capitale_cassa_pc_2022_median", "capitale_cassa_pc_2024_median", "capitale_cassa_growth_22_24_pct_median",
    "emp_growth_22_24_pct_median", "units_growth_23_24_pct_median", "real_va_growth_22_23_pct_median",
    "net_migr_rate_2023_median", "net_migr_rate_2024_median", "pop_growth_22_25_pct_median",
]].round(2))

print("\\n2. City-Level Shock & Resilience Profile across Severely Impacted Poles vs Controls:")
display(flood_df[[
    "comune_name", "flood_tier",
    "income_growth_22_23_pct", "taxpayer_growth_22_23_pct",
    "capitale_cassa_pc_2022", "capitale_cassa_pc_2024", "capitale_cassa_growth_22_24_pct",
    "emp_growth_22_24_pct", "units_growth_23_24_pct", "real_va_growth_22_23_pct",
    "net_migr_rate_2023", "pop_growth_22_25_pct",
]].round(2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

# Left panel: Capital Reconstruction Cash Surge (2022 vs 2024)
sub_sev = flood_df[flood_df["flood_tier"] == "whole_territory"].sort_values("capitale_cassa_pc_2024", ascending=False)
x = np.arange(len(sub_sev))
w = 0.35
ax1.bar(x - w/2, sub_sev["capitale_cassa_pc_2022"], width=w, color=PEER_C, label="2022 (Pre-flood baseline)")
ax1.bar(x + w/2, sub_sev["capitale_cassa_pc_2024"], width=w, color=ITALY_C, label="2024 (Reconstruction outlays)")
ax1.set_xticks(x)
ax1.set_xticklabels(sub_sev["comune_name"], rotation=15, ha="right", fontsize=9)
ax1.set_ylabel("Capital Cash Outlays (EUR / resident)")
ax1.set_title("Reconstruction Cash Outlay Wave (MEF RGS SIOPE 2022 vs 2024)", fontsize=10)
ax1.legend(fontsize=8.5)

# Right panel: Income shock vs Employment Resilience
for _, r in flood_df.iterrows():
    c = ITALY_C if r["is_cesena"] else (EU_C if r["flood_tier"] == "whole_territory" else PEER_C)
    ax2.scatter(r["income_growth_22_23_pct"], r["emp_growth_22_24_pct"], color=c, s=70, zorder=3)
    ax2.text(r["income_growth_22_23_pct"] + 0.15, r["emp_growth_22_24_pct"], r["comune_name"], fontsize=8, va="center")
ax2.axvline(0, color="gray", lw=0.8, ls="--")
ax2.axhline(0, color="gray", lw=0.8, ls="--")
ax2.set_xlabel("Real Mean IRPEF Income Growth 2022->2023 (%)")
ax2.set_ylabel("Private Employment Growth 2022->2024 (%)")
ax2.set_title("Income Shock vs. Employment Resilience (2022-2024)", fontsize=10)

plt.tight_layout()
plt.show()'''),
    (M, """## 6. Sensitivity Under the Rules Fixed by the 2026-10-30 Review

All pool results with corrected q-values, how the rank moves across size windows and regions, the synthetic control without each donor, the fixed-effects screen with regime effects, and the repeated shuffled null."""),
    (C, '''P = "data/processed/cesena/"
display(pd.read_parquet(P + "extended_placebo_ranking.parquet")[["pool_key", "metric", "rank", "n_comuni", "placebo_two_sided_p", "q_bh", "is_main_pool"]].round(3))
s_pool = pd.read_parquet(P + "extended_pool_sensitivity.parquet")
display(s_pool[s_pool["metric"].str.startswith("mean")].pivot(index="taxpayers_lo", columns="region", values="rank"))
space = pd.read_parquet(P + "synthetic_in_space.parquet")
display(space[space["is_cesena"]][["metric", "gap_end_pp", "rank_gap_desc", "ratio_post_pre_rmspe", "rank_ratio_desc"]].round(2))
display(pd.read_parquet(P + "synthetic_in_time.parquet").drop(columns="label").round(2))
display(pd.read_parquet(P + "synthetic_leave_one_out.parquet").drop(columns="label").round(2))
regime = pd.read_parquet(P + "lagged_fe_regime_results.parquet")
print("Regime-effects screen: valid tests", int(regime["p_perm"].notna().sum()), "smallest q", round(float(regime["q_bh_perm"].min()), 3))
display(pd.read_parquet(P + "shuffled_null.parquet").round(3))'''),
    (M, """## 6b. Every Emilia-Romagna Municipality by Taxpayer Class

The wider pools put Cesena lower as smaller municipalities enter. Here every municipality of the region (any size, joined 2008-2024 by cadastral code; those created by a merger after 2008 drop out) is grouped by its 2024 taxpayers. A mean over few taxpayers moves with a handful of incomes, so small classes spread more and fill both tails: a lower rank in a pool with many small municipalities is expected even without any difference in the typical municipality."""),
    (C, '''from cifresignificative.studies.cesena.analysis.experiments import mef_region_universe, size_class_growth

er_all = mef_region_universe()
print(f"Emilia-Romagna municipalities in both years: {len(er_all)}; dropped by mergers: {len(er_all.attrs['dropped'])} ({', '.join(er_all.attrs['dropped'])})")
classes = size_class_growth(er_all, top=5)
print(f"Cesena: {classes['focal_value'].iloc[0]:+.2f}%; other municipalities growing more: {int(classes['n_above'].sum())} of {int(classes['n'].sum())}")
display(classes.drop(columns="focal_value").round(2))
print("Municipalities above 40k taxpayers:", sorted(er_all.loc[er_all["taxpayers_total"] >= 40000, "comune"]))'''),
    (M, """## Takeaways (Phase 3 — Evaluation; Descriptive)

- **Wider pools (`sensitive`).** The income rank is high among mid-sized cities and falls as smaller municipalities enter. No pool result stands out after correction, and the pools were examined after Cesena's rank was known.
- **Size classes in Emilia-Romagna (descriptive).** Of the 316 other municipalities (13 merged ones drop out), 110 grew more than Cesena, all with fewer than 40k taxpayers; none of the 12 above 40k did (they are the 12 peers). The class median falls from +8.34% under 2k taxpayers to -0.90% above 40k, and the standard deviation from 8.39 to 1.63 points.
- **Synthetic control (`sensitive`).** The mean-income gap is the largest of the 13 at the end, but it is already present before 2019 and rests on one donor. On the RMSPE ratio it is mid-table. The median-income, employment and unit-size results are weak or inherit the food-sector break.
- **Lagged fixed effects.** No discoveries. With 13 cities this has little power, so it does not show that budgets have no effect. The test also over-rejects on shuffled data, so it could not support a positive result.
- **Flood (`sensitive`; the income shock is `retracted`).** No visible income shock in any group: the earlier fall in Faenza and Imola came from a blank bracket in the source. Capital payments rose in most cities, with listed cities higher, and nothing identifies reconstruction money.
- **Limits.** All statements are descriptive associations for one municipality, never causal claims. The figures are in the table below and in `reports/findings.md`, both computed by `analysis/claims.py`."""),
    (C, '''from cifresignificative.studies.cesena.analysis.claims import claims_for_notebook

print("Figures behind this notebook, computed from the processed tables (the same ones quoted in reports/findings.md):")
with pd.option_context("display.max_rows", None, "display.max_colwidth", 60):
    display(claims_for_notebook("05"))'''),
]
