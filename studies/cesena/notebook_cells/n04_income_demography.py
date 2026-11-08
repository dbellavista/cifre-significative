"""Cells of notebook 04_income_demography (Pillar 4: Socio-Economic & Demographic Outcomes)."""
from . import NOTEBOOKS, C, M

NOTEBOOKS["04_income_demography"] = [
    (M, """# Pillar 4 — Socio-Economic & Demographic Outcomes: Income Distribution, Inequality & Working-Age Attractiveness (`2008–2026`)

This notebook evaluates whether Cesena's economic trajectory translated into broad-based household prosperity and demographic vitality compared with its Emilia-Romagna peers, using two administrative panels:
1. **MEF Municipal IRPEF Declarations (`Finanze.gov.it`, tax years `2008–2024`, 17 annual waves):** Taxpayer counts and taxable/total income across the 8 statutory income brackets (`<=0`, `0–10k`, `10–15k`, `15–26k`, `26–55k`, `55–75k`, `75–120k`, `>120k` EUR) plus income composition (dependent employment, pensions, self-employment/entrepreneurship).
2. **ISTAT Demo Municipal Population by Single Year of Age (`POSAS`, `2008–2026`) & Demographic Balance (`BILANCIO`, `2008–2025`):** Resident population, old-age index (`aging_index = pop_65_plus / pop_0_14 * 100`), dependency ratio, net migration rate (internal plus international), and our **exact working-age (`25–44`) cohort net attractiveness rate** ($\\text{NetCohort}_{25-44, t} = P_{25-44, t} - P_{24-43, t-1}$ per 1,000 cohort members aged `25–44`).

**Methodological notes:**
- **Bracket-creep caveat & dual Gini index:** MEF's statutory brackets are fixed in nominal euros (`0–10k` through `>120k`). Inflation pushes taxpayers into higher nominal brackets over time, so we report both the standard trapezoidal Lorentz approximation (`gini_trap`) and a **Pareto-tail-adjusted Gini (`gini_pareto`)** that corrects for top-bracket concentration, alongside real mean and interpolated median income in constant 2020 EUR.
- **Descriptive only:** All comparisons and peer rankings are descriptive."""),
    (C, '''from cifresignificative.studies.cesena.analysis.data import (
    load_irpef_summary, load_irpef_brackets, load_demography, load_panel_comuni,
    load_belt_comuni, cesena_vs_peers_series,
)
from cifresignificative.studies.cesena.analysis.outcomes import summarize_comprensorio_spillovers

irpef = load_irpef_summary()
brackets = load_irpef_brackets()
demo = load_demography()
panel = load_panel_comuni()
belt_df = load_belt_comuni()

print(f"MEF IRPEF summary rows: {len(irpef)} (years {irpef['year'].min()}-{irpef['year'].max()}, {irpef['comune_id'].nunique()} municipalities)")
print(f"ISTAT demography rows: {len(demo)} (years {demo['year'].min()}-{demo['year'].max()}, {demo['comune_id'].nunique()} municipalities)")
print(f"Comprensorio Cesenate (Cesena + 16 belt comuni) rows: {len(belt_df)} ({belt_df['comune_id'].nunique()} municipalities)")'''),
    (M, """## 1. Real Mean & Interpolated Median IRPEF Income (`2008–2024`, Constant 2020 EUR)
We compare Cesena's real mean income per positive taxpayer (`mean_income_real_eur`), real interpolated median income (`median_income_real_eur`), and real IRPEF tax base per resident (`irpef_taxbase_real_pc`) against the 6 Core Peers and all 12 Emilia-Romagna peers."""),
    (C, '''for col, lbl in [
    ("mean_income_real_eur", "Real mean IRPEF income per taxpayer (2020 EUR)"),
    ("median_income_real_eur", "Real interpolated median IRPEF income per taxpayer (2020 EUR)"),
    ("irpef_taxbase_real_pc", "Real total IRPEF income per resident (2020 EUR)"),
]:
    s = cesena_vs_peers_series(panel[panel["year"] <= 2024], col, core_only=True)
    print(f"\\n--- {lbl} (2008-2024) ---")
    display(s[["year", "cesena", "core_peer_median", "er_peer_median", "gap_vs_benchmark", "n_core_peers", "missing_core_peers"]].round(1))

# Rank of all 13 ER municipalities in 2024 and cumulative real growth 2008-2024
inc_08 = panel[panel["year"] == 2008].set_index("comune")[["mean_income_real_eur", "median_income_real_eur", "gini_pareto"]]
inc_24 = panel[panel["year"] == 2024].set_index("comune")[["mean_income_real_eur", "median_income_real_eur", "gini_pareto", "below_15k_payer_share_pct", "top_55k_income_share_pct"]]
inc_cmp = pd.DataFrame({
    "mean_2008": inc_08["mean_income_real_eur"],
    "mean_2024": inc_24["mean_income_real_eur"],
    "mean_growth_08_24_pct": (inc_24["mean_income_real_eur"] / inc_08["mean_income_real_eur"] - 1) * 100,
    "median_2008": inc_08["median_income_real_eur"],
    "median_2024": inc_24["median_income_real_eur"],
    "median_growth_08_24_pct": (inc_24["median_income_real_eur"] / inc_08["median_income_real_eur"] - 1) * 100,
    "gini_pareto_2008": inc_08["gini_pareto"],
    "gini_pareto_2024": inc_24["gini_pareto"],
    "gini_change_pts": inc_24["gini_pareto"] - inc_08["gini_pareto"],
    "below_15k_share_2024_pct": inc_24["below_15k_payer_share_pct"],
    "top_55k_inc_share_2024_pct": inc_24["top_55k_income_share_pct"],
}).sort_values("mean_growth_08_24_pct", ascending=False)
print("\\nAll 13 ER Municipalities — Real IRPEF Income Growth and Inequality (2008 vs 2024, sorted by real mean income growth):")
display(inc_cmp.round(2))'''),
    (M, """## 2. Income Inequality Across the 8 Statutory Brackets (`gini_trap` vs `gini_pareto`) & Income Sources
We examine the distribution of taxpayers and total income across the 8 MEF brackets in Cesena (`2008`, `2016`, `2024`) and compare the trajectory of the Pareto-adjusted Gini index (`gini_pareto`) and low-income taxpayer share (`<15,000 EUR`) against peers."""),
    (C, '''ces_b = brackets[(brackets["comune_id"] == "040007") & (brackets["tax_year"].isin([2008, 2016, 2024])) & (brackets["class_order"] >= 1)].copy()
ces_b["payer_share_pct"] = ces_b.groupby("tax_year")["taxpayers"].transform(lambda s: s / s.sum() * 100)
ces_b["income_share_pct"] = ces_b.groupby("tax_year")["income_eur"].transform(lambda s: s / s.sum() * 100)
piv_b = ces_b.pivot(index=["class_order", "class_label"], columns="tax_year", values=["payer_share_pct", "income_share_pct"])
print("Cesena Positive Taxpayer Share (%) and Total Income Share (%) across the 7 Positive MEF Brackets (2008, 2016, 2024):")
display(piv_b.round(2))

for col, lbl in [
    ("gini_pareto", "Pareto-adjusted Gini index across 8 IRPEF brackets"),
    ("gini_trap", "Trapezoidal lower-bound Gini index across 8 IRPEF brackets"),
    ("below_15k_payer_share_pct", "Taxpayers declaring <15,000 EUR (% of positive taxpayers)"),
    ("top_55k_income_share_pct", "Share of total IRPEF income held by >55,000 EUR brackets (%)"),
]:
    s = cesena_vs_peers_series(panel[panel["year"] <= 2024], col, core_only=True)
    print(f"\\n--- {lbl} ---")
    display(s[s["year"].isin([2008, 2012, 2016, 2019, 2021, 2024])][["year", "cesena", "core_peer_median", "er_peer_median", "gap_vs_benchmark"]].round(3))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
for ax, (col, title, yl) in zip(axes, [
    ("mean_income_real_eur", "Real Mean IRPEF Income", "2020 EUR per taxpayer"),
    ("median_income_real_eur", "Real Interpolated Median Income", "2020 EUR per taxpayer"),
    ("gini_pareto", "Local Income Inequality (Pareto Gini)", "Gini index (0-1)"),
]):
    s = cesena_vs_peers_series(panel[panel["year"] <= 2024], col, core_only=True)
    ax.plot(s["year"], s["core_peer_median"], color=EU_C, ls="--", lw=1.8, label="Core peer median")
    ax.fill_between(s["year"], s["core_peer_p25"], s["core_peer_p75"], color=PEER_C, alpha=0.3)
    ax.plot(s["year"], s["cesena"], color=ITALY_C, lw=2.3, marker="o", ms=3.5, label="Cesena")
    ax.set_title(title, fontsize=9.5)
    ax.set_ylabel(yl)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()'''),
    (M, """## 3. Demographic Vitality: Population, Aging Index, and Working-Age (`25–44`) Cohort Net Attractiveness (`2008–2026`)
Because municipal migration registers do not publish age-specific inflows/outflows across the entire 2008–2025 span, we measure young-adult demographic attractiveness directly from the exact single-year-of-age population register (`POSAS`):
$$\\text{NetCohort}_{25-44, t} = P_{25-44, t} - P_{24-43, t-1}$$
Since mortality between ages 24 and 44 is below 0.6 per 1,000, this cohort survival difference captures net migration of 25–44 year-olds per 1,000 cohort members aged `25–44` (`net_cohort_25_44_per_1k`). It also contains census and registry revisions, which are not migration (for example 2012–2013 and 2020–2021; `net_cohort_25_44_3y_ma` smooths single years), so read it by sub-period (Section 8)."""),
    (C, '''for col, lbl in [
    ("pop_total", "Total resident population (Jan 1, 2008-2026)"),
    ("aging_index", "Old-age / aging index (65+ per 100 aged 0-14)"),
    ("share_25_44_pct", "Working-age 25-44 cohort share of population (%)"),
    ("net_migr_rate_per_1k", "Total net migration rate per 1,000 residents"),
    ("net_cohort_25_44_per_1k", "25-44 working-age cohort net attractiveness per 1,000 cohort members"),
]:
    s = cesena_vs_peers_series(demo, col, core_only=True)
    print(f"\\n--- {lbl} ---")
    display(s[s["year"].isin([2008, 2011, 2014, 2017, 2019, 2021, 2023, 2025, 2026])][
        ["year", "cesena", "core_peer_median", "er_peer_median", "gap_vs_benchmark", "n_core_peers"]
    ].round(2))

dem_08 = demo[demo["year"] == 2008].set_index("comune")
dem_25 = demo[demo["year"] == 2025].set_index("comune")
cohort_mean = demo[demo["year"].between(2014, 2025)].groupby("comune")["net_cohort_25_44_per_1k"].mean()
migr_mean = demo[demo["year"].between(2014, 2025)].groupby("comune")["net_migr_rate_per_1k"].mean()
dem_cmp = pd.DataFrame({
    "pop_2008": dem_08["pop_total"],
    "pop_2025": dem_25["pop_total"],
    "pop_change_08_25_pct": (dem_25["pop_total"] / dem_08["pop_total"] - 1) * 100,
    "aging_2008": dem_08["aging_index"],
    "aging_2025": dem_25["aging_index"],
    "aging_change_pts": dem_25["aging_index"] - dem_08["aging_index"],
    "share_25_44_2025_pct": dem_25["share_25_44_pct"],
    "mean_net_migr_14_25_per_1k": migr_mean,
    "mean_25_44_attract_14_25_per_1k": cohort_mean,
}).sort_values("mean_25_44_attract_14_25_per_1k", ascending=False)
print("\\nAll 13 ER Municipalities — Demographic Trajectory & 25-44 Cohort Attractiveness (2008-2025):")
display(dem_cmp.round(2))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
for ax, (col, title, yl) in zip(axes, [
    ("aging_index", "Aging Index (65+ / 0-14 x100)", "Elderly per 100 under-15s"),
    ("share_25_44_pct", "Young Adult Share (Ages 25-44)", "% of total population"),
    ("net_cohort_25_44_3y_ma", "25-44 Cohort Net Attractiveness (3y MA)", "Net inflow per 1,000 pop."),
]):
    s = cesena_vs_peers_series(demo[demo["year"] <= 2025], col, core_only=True)
    ax.plot(s["year"], s["core_peer_median"], color=EU_C, ls="--", lw=1.8, label="Core peer median")
    ax.fill_between(s["year"], s["core_peer_p25"], s["core_peer_p75"], color=PEER_C, alpha=0.3)
    ax.plot(s["year"], s["cesena"], color=ITALY_C, lw=2.3, marker="o", ms=3.5, label="Cesena")
    ax.set_title(title, fontsize=9.5)
    ax.set_ylabel(yl)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()'''),
    (M, """## 4. Cesena and the 16 Surrounding Municipalities (`Comprensorio Cesenate`, a study definition, `2008–2025`)
Population, jobs and income are compared across three rings. Note that the periods differ between series (population from 2008, ASIA jobs from 2012), and that Bertinoro and Forlimpopoli belong to another Unione. Section 8 gives the population change by component in matching periods.
- **Unione Rubicone e Mare (`9` municipalities:** Cesenatico, Gambettola, Gatteo, Longiano, San Mauro Pascoli, Savignano sul Rubicone, Borghi, Roncofreddo, Sogliano al Rubicone);
- **Via Emilia West corridor (`2` municipalities:** Forlimpopoli, Bertinoro);
- **Unione dei Comuni Valle del Savio (`5` municipalities:** Bagno di Romagna, Mercato Saraceno, Montiano, Sarsina, Verghereto)."""),
    (C, '''by_grp, by_muni = summarize_comprensorio_spillovers(belt_df)
print("Comprensorio Cesenate (Cesena + 16 surrounding municipalities) — Population, Net Migration, ASIA Employment, Real Income & Tax Surcharges by Functional Ring:")
display(by_grp[[
    "belt_label", "n_comuni", "pop_2008", "pop_2025", "pop_delta_08_25", "pop_growth_08_25_pct",
    "net_migration_08_24", "natural_balance_08_24",
    "emp_2012", "emp_2024", "emp_delta_12_24", "emp_growth_12_24_pct",
    "emp_per_1k_pop_2012", "emp_per_1k_pop_2024",
    "mean_income_real_2008", "mean_income_real_2024", "real_income_growth_08_24_pct",
    "effective_addizionale_rate_2024_pct", "addizionale_real_pc_2024",
]].round(2))

print("\\nAll 17 Municipalities of the Comprensorio Cesenate (sorted by 2008-2025 population growth %):")
display(by_muni[[
    "comune_id", "comune", "belt_group", "pop_2008", "pop_2025", "pop_growth_08_25_pct",
    "net_migration_08_24", "emp_2012", "emp_2024", "emp_delta_12_24", "emp_growth_12_24_pct",
    "emp_per_1k_pop_2024", "mean_income_real_2024", "real_income_growth_08_24_pct",
    "effective_addizionale_rate_2024_pct", "addizionale_real_pc_2024",
]].round(2))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
ring_sub = by_grp[by_grp["belt_group"].isin(["cesena_core", "rubicone_mare", "via_emilia_west", "unione_valle_savio"])].copy()
short_lbls = ["Cesena\\n(core)", "Rubicone e Mare\\n(9 belt comuni)", "Via Emilia West\\n(Forlimpopoli+Bertinoro)", "Valle del Savio\\n(5 hill/mtn comuni)"]
x = range(len(ring_sub))
axes[0].bar([i - 0.18 for i in x], ring_sub["emp_delta_12_24"], width=0.35, color=ITALY_C, label="ASIA net jobs created (2012-2024)")
axes[0].bar([i + 0.18 for i in x], ring_sub["pop_delta_08_25"], width=0.35, color=EU_C, label="Resident pop. change (2008-2025)")
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_xticks(list(x), short_lbls, fontsize=8)
axes[0].set_title("Job Creation vs Resident Population Growth by Ring", fontsize=9.5)
axes[0].set_ylabel("Persons / jobs")
axes[0].legend(fontsize=8)

axes[1].scatter(by_muni["emp_growth_12_24_pct"], by_muni["pop_growth_08_25_pct"], c=[ITALY_C if c == "040007" else EU_C for c in by_muni["comune_id"]], s=55)
for _, r in by_muni.iterrows():
    if r["comune_id"] in ("040007", "040041", "040016", "040008", "040013", "040003", "040001", "040045"):
        axes[1].annotate(str(r["comune"]), (float(r["emp_growth_12_24_pct"]) + 0.5, float(r["pop_growth_08_25_pct"]) + 0.3), fontsize=7.5)
axes[1].axhline(0, color="gray", lw=0.7, ls=":")
axes[1].axvline(0, color="gray", lw=0.7, ls=":")
axes[1].set_xlabel("ASIA employment growth 2012-2024 (%)")
axes[1].set_ylabel("Resident population growth 2008-2025 (%)")
axes[1].set_title("Workplace vs Residence Split in the Comprensorio", fontsize=9.5)
plt.tight_layout()
plt.show()'''),
    (M, """## 5. Higher Education: UniBo Cesena Campus (MUR USTAT, `2010–2026`)
Students, graduates and subject mix (STEM, ICT, engineering, psychology) across the 13 municipalities, from the **Ministry of University and Research (MUR USTAT)**. Whether graduates stay and work locally is not measured."""),
    (C, '''from cifresignificative.studies.cesena.analysis.data import load_university_summary, load_university_enrolments
from cifresignificative.studies.cesena.analysis.outcomes import summarize_university_benchmarks

uni_sum = load_university_summary()
uni_enr = load_university_enrolments()

print("Emilia-Romagna University Poles — Enrolments, STEM & ICT Shares (2024/2025):")
y24 = uni_sum[uni_sum["calendar_year"] == 2024].sort_values("stem_share", ascending=False)
display(y24[["comune_name", "students_total", "stem_share", "ict_share", "engineering_share", "psychology_share", "graduates_total"]].round(3))

bench_uni = summarize_university_benchmarks(uni_sum, demo, core_only=False, year=2024)
print("\\nCesena vs All-ER Peer Median on University Metrics (2024):")
display(bench_uni[["label", "formatted_cesena", "formatted_peer", "gap_pct"]].rename(columns={"formatted_cesena": "Cesena", "formatted_peer": "ER Peer Median", "gap_pct": "Gap vs Peer (%)"}).round(1))

ces_u = uni_sum[uni_sum["comune_id"] == "040007"].sort_values("calendar_year")
print("\\nCesena University Trajectory (2010-2025):")
display(ces_u[["calendar_year", "students_total", "students_stem", "students_ict", "students_engineering", "students_psychology", "stem_share", "ict_share", "graduates_total"]].round(3))'''),
    (M, """## 6. Housing Prices and the Suburbanisation Hypothesis (Agenzia delle Entrate OMI)

Suburbanisation driven by housing cost is a **hypothesis** here, not a result (research track R2). OMI quotations are one cross-section, averaged across zones without weights; zones with no rental quote are treated as missing.
- **Purchase quotations (€/m²)** across `Abitazioni civili` and `Abitazioni di tipo economico`.
- **Rents (€/m²/month).**
- **Price-to-income and rent burden**, using mean declared income, which is by place of residence.
- **Price differences against Cesena.** Section 8 reports the test: if cheap housing drew residents, cheaper towns should have grown more."""),
    (C, '''from cifresignificative.studies.cesena.analysis.data import (
    load_omi_municipal, load_omi_zones, load_omi_transactions,
)
from cifresignificative.studies.cesena.analysis.housing import (
    compute_housing_affordability, comprensorio_housing_gradient, peer_housing_comparison,
)

omi_muni = load_omi_municipal()
omi_zones = load_omi_zones()
omi_trans = load_omi_transactions()

affordability = compute_housing_affordability(omi_muni, irpef, demo)
gradient = comprensorio_housing_gradient(affordability)
peer_housing = peer_housing_comparison(affordability)

print("1. Comprensorio Cesenate Housing Cost Gradient by Functional Ring:")
display(gradient[[
    "belt_label", "n_municipalities", "compr_mid_median_eur_m2", "loc_mid_median_eur_mo",
    "price_diff_vs_cesena_pct_median", "rent_diff_vs_cesena_pct_median",
    "price_to_income_ratio_median", "rent_burden_pct_median",
    "annual_rent_savings_80m2_median_eur", "purchase_savings_100m2_median_eur",
    "pop_growth_08_25_pct_median", "net_migr_rate_2024_median",
]].round(2))

print("\\n2. Cesena vs. 12 Emilia-Romagna Peer Municipalities on Housing Costs & Affordability:")
display(peer_housing.round(2))

print("\\n3. Housing Costs across All 17 Municipalities of the Comprensorio Cesenate (sorted by purchase price):")
display(affordability[affordability["belt_group"] != "er_peer"][[
    "comune_name", "belt_group", "compr_mid_eur_m2", "diff_vs_cesena_pct",
    "loc_mid_eur_m2_mo", "diff_rent_vs_cesena_pct", "price_to_income_ratio_100m2",
    "rent_burden_pct_80m2", "annual_rent_savings_vs_cesena_eur", "purchase_savings_100m2_vs_cesena_eur",
    "pop_growth_08_25_pct",
]].sort_values("compr_mid_eur_m2", ascending=False).round(2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

# Left panel: Housing price gradient across Comprensorio functional rings
rings = gradient.copy()
x = np.arange(len(rings))
w = 0.35
ax1.bar(x - w/2, rings["compr_mid_median_eur_m2"], width=w, color=ITALY_C, label="Purchase Price (EUR/m²)")
ax1_twin = ax1.twinx()
ax1_twin.bar(x + w/2, rings["pop_growth_08_25_pct_median"], width=w, color=EU_C, label="Pop. Growth 2008-2025 (%)")
ax1.set_xticks(x)
ax1.set_xticklabels([
    "Cesena Core", "Rubicone & Mare\\n(9 belt comuni)", "Via Emilia West\\n(Forlimpopoli+Bertinoro)", "Valle del Savio\\n(5 hill comuni)"
], rotation=10, ha="right", fontsize=8.5)
ax1.set_ylabel("Purchase Price (EUR / m²)", color=ITALY_C)
ax1_twin.set_ylabel("Pop. Growth 2008–2025 (%)", color=EU_C)
ax1.set_title("Housing Price vs. Residential Growth Gradient", fontsize=10)

# Right panel: Cesena intra-municipal zone prices
ces_zones = omi_zones[omi_zones["tipo"].str.contains("civili|economico", case=False)].pivot(
    index="zona_label", columns="tipo", values="compr_mid_eur_m2"
).dropna()
ces_zones = ces_zones.sort_values(ces_zones.columns[0], ascending=True)
y = np.arange(len(ces_zones))
ax2.barh(y - 0.18, ces_zones.iloc[:, 0], height=0.35, color=ITALY_C, label=ces_zones.columns[0])
if len(ces_zones.columns) > 1:
    ax2.barh(y + 0.18, ces_zones.iloc[:, 1], height=0.35, color=PEER_C, label=ces_zones.columns[1])
ax2.set_yticks(y)
ax2.set_yticklabels(ces_zones.index, fontsize=8)
ax2.set_xlabel("Central Quotation (EUR / m²)")
ax2.set_title("Cesena Intra-Municipal Zone Prices (Agenzia Entrate OMI)", fontsize=10)
ax2.legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.show()'''),
    (M, """## 7. Origin-Destination Commuting Matrix (ISTAT Matrice del Pendolarismo, 2011 Census)

Background only: these flows come from the 2011 census, so they predate the 2012–2024 period and cannot support a trend. Student flows cover all school levels.
- **Workplace jobs vs. resident workers** (`jobs_local = internal + inbound`, `resident_workers = internal + outbound`).
- **Self-containment rate** (`internal / jobs_local`) and **retention rate** (`internal / resident_workers`).
- **Corridors:** flows between Cesena and each of the 16 towns.
- **Modal split and commute duration.**"""),
    (C, '''from cifresignificative.studies.cesena.analysis.commuting import (
    load_commuting_summary, peer_commuting_comparison,
    comprensorio_commuter_network, spatial_arbitrage_analysis,
)

df_sum = load_commuting_summary()
peer_comm = peer_commuting_comparison()
net_comm = comprensorio_commuter_network()
arb_comm = spatial_arbitrage_analysis()

print("1. Emilia-Romagna Peer Cities — Commuting Accounting & Self-Containment:")
display(peer_comm[[
    "name", "jobs_local", "resident_workers", "internal_workers", "net_commuter_inflow",
    "job_self_containment_rate", "labor_retention_rate", "commuter_inflow_ratio",
    "net_student_inflow", "pct_active", "pct_car", "pct_commute_le15m",
]].round(3))

print("\\n2. Comprensorio Cesenate Functional Rings — Commuting Exchanges with Cesena:")
display(net_comm["ring_aggregates"][[
    "group_label", "inbound_to_cesena", "outbound_from_cesena", "net_inflow_to_cesena",
    "gross_commuter_flow", "study_inbound_to_cesena", "study_net_inflow_to_cesena",
]])

print("\\n3. Top Commuting Corridors with Cesena (Belt Municipalities + Regional Neighbors):")
display(net_comm["municipal_flows"][[
    "partner_name", "group", "inbound_to_cesena", "outbound_from_cesena", "net_inflow_to_cesena",
    "gross_commuter_flow", "share_partner_resident_workers",
]].head(15).round(3))

print("\\n4. Spatial Arbitrage: Housing Discount vs. Share Commuting to Cesena:")
display(arb_comm[[
    "name", "group", "share_partner_resident_workers", "inbound_to_cesena",
    "compr_mid_eur_m2", "price_discount_eur_m2", "purchase_savings_100m2",
    "loc_mid_eur_m2_mo", "rent_savings_80m2_annual",
]].round(2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

# Left panel: Self-containment vs Net Commuter Inflow across 13 ER peers
x = peer_comm["net_commuter_inflow"]
y = peer_comm["job_self_containment_rate"] * 100.0
colors = [ITALY_C if row["is_cesena"] else (EU_C if row["is_core_peer"] else PEER_C) for _, row in peer_comm.iterrows()]
sizes = peer_comm["jobs_local"] / 1000.0 * 8.0

ax1.scatter(x, y, s=sizes, c=colors, alpha=0.85, edgecolors="black", linewidth=0.8)
for _, row in peer_comm.iterrows():
    ax1.annotate(row["name"], (row["net_commuter_inflow"], row["job_self_containment_rate"] * 100.0),
                 xytext=(5, 4), textcoords="offset points", fontsize=8)

ax1.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
ax1.set_xlabel("Net Commuter Inflow (Workers)", fontsize=9)
ax1.set_ylabel("Job Self-Containment Rate (% of local jobs held by residents)", fontsize=9)
ax1.set_title("13 ER Peers: Job Self-Containment vs. Net Commuter Surplus", fontsize=10)

# Right panel: Bilateral Commuting Flows between Cesena and 16 Belt Municipalities
belt_corr = net_comm["municipal_flows"][net_comm["municipal_flows"]["group"].isin(["rubicone_mare", "via_emilia_west", "unione_valle_savio"])].head(12)
y_pos = np.arange(len(belt_corr))
h = 0.35

ax2.barh(y_pos - h/2, belt_corr["inbound_to_cesena"], height=h, color=ITALY_C, label="Inbound to Cesena")
ax2.barh(y_pos + h/2, belt_corr["outbound_from_cesena"], height=h, color=EU_C, label="Outbound from Cesena")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(belt_corr["partner_name"], fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel("Daily Commuters (Workers)", fontsize=9)
ax2.set_title("Cesena Commuting Flows with Surrounding Belt Municipalities", fontsize=10)
ax2.legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.show()'''),
    (M, """## 8. Composition, Population Balance and Hypothesis Checks (2026-10-30 Review)

Income growth next to taxpayers and aggregate income, by period; population change split into natural balance, internal and international migration and registry adjustment (the four add up exactly); the 25–44 cohort by sub-period; and the housing check: the suburbanisation story predicts negative correlations."""),
    (C, '''P = "data/processed/cesena/"
display(pd.read_parquet(P + "income_composition.parquet").round(2))
bal = pd.read_parquet(P + "population_balance.parquet")
display(bal[bal["area"].isin(["cesena", "belt_16"])].drop(columns=["pop_start"]).round(1))
display(pd.read_parquet(P + "cohort_by_period.parquet").round(2))
display(pd.read_parquet(P + "housing_gradient_test.parquet").round(3))'''),
    (M, """## Income follow-up (6 November 2026)

Frozen national-price calculation; source components do not reconcile to an exhaustive total. Employment-and-assimilated income per recipient is not hourly wages or a longitudinal worker panel. Direct annual HICP levels and the current ECOICOP v2 series are sensitivity checks documented in `reports/income_evidence/`. Descriptive only."""),
    (C, '''from cifresignificative.studies.cesena.analysis.claims import income_followup_claims

price_basis = pd.read_csv("studies/cesena/reports/income_evidence/deflator_audit.csv").set_index("year")["panel_chained_2020_1"]
components = pd.read_parquet("data/processed/cesena/income_followup/income_components.parquet")
display(income_followup_claims(components, price_basis).round(2))'''),
    (M, """## Takeaways (Pillar 4 — Income & Demography; Descriptive)

- **Income is `robust` among the 13 cities.** Real mean and median income growth from 2008 rank first, but most of the lead predates 2019, and taxpayers grew slowly, so aggregate income does not rank first. IRPEF income is by residence, not workplace. Shares below a nominal threshold fall with inflation and are not results.
- **Ageing and weak population growth are `robust`.** Against the 16 surrounding towns, most of the gap comes from natural balance; moves within Italy were positive in Cesena in every period, and the towns' growth is concentrated in 2008–2011.
- **The 25–44 rank is `sensitive`.** Last place comes mostly from 2012–2016, and the measure contains registry revisions.
- **Suburbanisation is a `hypothesis` (research track R2).** Cheaper towns grew less, prices show no relation with the 25–44 cohort across cities, and no data here show who moves. The surcharge is owed where a taxpayer lives, so Cesena's lower rate favours living there.
- **Commuting (`context only`).** 2011 data; student flows cover all school levels.
- **University (`robust` as a description).** A small, specialised campus; graduate retention is not measured.
- **Limits.** All statements are descriptive associations for one municipality, never causal claims. The figures are in the table below and in `reports/findings.md`, both computed by `analysis/claims.py`."""),
    (C, '''from cifresignificative.studies.cesena.analysis.claims import claims_for_notebook

print("Figures behind this notebook, computed from the processed tables (the same ones quoted in reports/findings.md):")
with pd.option_context("display.max_rows", None, "display.max_colwidth", 60):
    display(claims_for_notebook("04"))'''),
]
