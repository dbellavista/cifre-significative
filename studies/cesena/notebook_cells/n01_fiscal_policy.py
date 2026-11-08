"""Cells of notebook 01_fiscal_policy (Pillar 1: Municipal Fiscal Policy & Resource Allocation)."""
from . import NOTEBOOKS, C, M

NOTEBOOKS["01_fiscal_policy"] = [
    (M, """# Pillar 1 — Municipal Fiscal Policy & Resource Allocation: How Does Cesena Collect and Spend Public Resources Compared to Peers?

This notebook examines the municipal balance sheets of **Cesena** (`040007`, Cadastral `C573`, NUTS3 `ITH58`) against two benchmark groups in Emilia-Romagna:
1. **Core Romagna & Mid-Sized Peers (6 peers, excluding Cesena):** Forlì (`040012`), Rimini (`099014`), Ravenna (`039014`), Imola (`037032`), Carpi (`036005`), and Reggio nell'Emilia (`035033`).
2. **All 12 Emilia-Romagna Peers (excluding Cesena):** Core peers plus Faenza (`039010`), Ferrara (`038008`), Modena (`036023`), Parma (`034027`), Piacenza (`033032`), and Bologna (`037006`).

**Methodological rules & accounting break:**
- **No silent splicing across the 2016 accounting harmonization break:** Italian municipalities transitioned from the historical `D.P.R. 194/1996` structure (`2008–2015`, flagged `flag = "break_dlgs118"`) to the harmonized `D.Lgs. 118/2011` Mission/Program classification (`2016+`) with the *Fondo Pluriennale Vincolato* (FPV) and *Fondo Crediti di Dubbia Esigibilità* (FCDE). Growth rates and regime averages are computed strictly *within* each regime, never across the `2015 -> 2016` boundary.
- **Constant 2020 euro per capita:** All nominal euro series are deflated to constant 2020 prices using the Italian national CPI/HICP deflator (`core.deflators.italy_deflator`) and divided by ISTAT Jan-1 resident population.
- **Descriptive only:** Every comparison and rank is descriptive, never causal."""),
    (C, '''from cifresignificative.studies.cesena.analysis.data import (
    load_fiscal, load_missions, load_demography, load_panel_comuni,
    cesena_vs_peers_series, peer_median_ex_cesena,
)
from cifresignificative.studies.cesena.analysis.fiscal import (
    mission_allocation_summary, compare_fiscal_regimes, capital_funding_breakdown,
)

fiscal = load_fiscal()
missions = load_missions()
demo = load_demography()
panel = load_panel_comuni()

cov = fiscal.pivot_table(index="year", columns="budget_type", values="comune_id", aggfunc="count", fill_value=0)
print("Municipal budget coverage across the 13 ER municipalities (rows by year and budget_type):")
display(cov.T)
print("Regime & break flags by year:", fiscal.groupby(["regime", "flag", "budget_type"])["year"].agg(["min", "max", "count"]).to_dict("index"))'''),
    (M, """## 1. Current vs. Capital Expenditure & the 2016 Harmonization Break
We compare Cesena's real per-capita current expenditure (`Titolo I`, *Spese correnti*) and capital expenditure (`Titolo II`, *Spese in conto capitale*, accrual commitments and cash outlays) against the Core Peer median and All-ER Peer median separately in `2008–2015` (`dpr194`) and `2016–2021` (`dlgs118` consuntivo, plus `2022` preventivo)."""),
    (C, '''reg_comp = compare_fiscal_regimes(fiscal, core_only=True)
print("Cesena vs Core Peer median (6 peers ex Cesena) and rank among all 13 ER municipalities, by accounting regime:")
display(reg_comp.round(2))

cons = fiscal[fiscal["budget_type"] == "consuntivo"].copy()
for col, label in [
    ("spese_correnti_comp_real_pc", "Current expenditure (comp., 2020 EUR/pc)"),
    ("spese_capitale_comp_real_pc", "Capital expenditure (comp., 2020 EUR/pc)"),
    ("capital_share_pct", "Capital share of primary expenditure (%)"),
]:
    ser = cesena_vs_peers_series(cons, col, core_only=True)
    print(f"\\n--- {label} ---")
    display(ser[["year", "cesena", "core_peer_median", "er_peer_median", "gap_vs_benchmark", "n_core_peers", "missing_core_peers"]].round(2))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
for ax, (col, title, yl) in zip(axes, [
    ("spese_correnti_comp_real_pc", "Current spending (Titolo I)", "2020 EUR per capita"),
    ("spese_capitale_comp_real_pc", "Capital spending (Titolo II)", "2020 EUR per capita"),
    ("capital_share_pct", "Capital share of primary spending", "% of Titolo I + II"),
]):
    s = cesena_vs_peers_series(cons, col, core_only=True)
    for reg_cond in [s["year"] <= 2015, s["year"] >= 2016]:
        sub = s[reg_cond]
        ax.plot(sub["year"], sub["core_peer_median"], color=EU_C, ls="--", lw=1.8, label="Core peer median" if reg_cond.iloc[0] else None)
        ax.fill_between(sub["year"], sub["core_peer_p25"], sub["core_peer_p75"], color=PEER_C, alpha=0.3)
        ax.plot(sub["year"], sub["cesena"], color=ITALY_C, lw=2.3, marker="o", ms=3.5, label="Cesena" if reg_cond.iloc[0] else None)
    ax.axvline(2015.5, color="k", ls=":", lw=1.0)
    ax.set_title(title, fontsize=9.5)
    ax.set_ylabel(yl)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()'''),
    (M, """## 2. Harmonized Expenditure by Mission (`2016–2021` Consuntivo)
Under `D.Lgs. 118/2011`, municipal expenditure is classified into standardized Missions (`M01` General Institutional Services through `M50` Public Debt). OpenBilanci's parent Mission JSON nodes suffer from a `NaN`-propagation bug whenever a single child Program has zero capital outlays (e.g., Cesena 2021 Mission 05 and Mission 11); our parser (`core.sources.bdap._mission_composition`) reconstructs every Mission from the sum of its child Program nodes (`0` discrepancy against the budget headline)."""),
    (C, '''m_sum = mission_allocation_summary(missions, demo, years=(2016, 2017, 2018, 2019, 2020, 2021), core_only=True)
print("Harmonized expenditure by Mission (2016-2021 consuntivo average, constant 2020 EUR per capita, Cesena vs Core Peer median):")
display(m_sum[[
    "mission_code", "mission_label",
    "cesena_correnti_real_pc", "peer_correnti_real_pc", "gap_correnti_real_pc",
    "cesena_capitale_real_pc", "peer_capitale_real_pc", "gap_capitale_real_pc",
    "cesena_totale_real_pc", "peer_totale_real_pc", "cesena_correnti_share_pct", "cesena_capitale_share_pct",
]].round(2))

top_m = m_sum.head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
y = np.arange(len(top_m))
ax.barh(y - 0.18, top_m["cesena_totale_real_pc"], height=0.35, color=ITALY_C, label="Cesena (2016-2021 mean)")
ax.barh(y + 0.18, top_m["peer_totale_real_pc"], height=0.35, color=EU_C, label="Core peer median")
ax.set_yticks(y, [f"{r.mission_code} - {r.mission_label[:32]}" for _, r in top_m.iterrows()], fontsize=8.5)
ax.set_xlabel("Constant 2020 EUR per capita (current + capital)")
ax.set_title("Top 8 Municipal Missions by Total Per-Capita Spending (2016-2021 Consuntivo)", fontsize=10)
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.show()'''),
    (M, """## 3. Capital Investment Funding Structure: Grants, Asset Sales, Borrowing & Multi-Year FPV (`2008–2021`)
How did Cesena fund its capital expenditure in `2016–2021`?
We break down municipal capital resources across:
1. **Capital Grants & Transfers (`Contributi agli investimenti`, Titolo IV):** State, regional, and EU capital subsidies.
2. **Asset Sales (`Alienazioni di beni patrimoniali`, Titolo IV):** Real estate and asset disposal.
3. **New Borrowing (`Accensione mutui e prestiti`, Titolo VI):** Long-term investment financing (primarily Cassa Depositi e Prestiti).
4. **Capital Multi-Year Committed Fund (`FPV capitale`, Quadro 2.1):** *Fondo Pluriennale Vincolato per spese in conto capitale*, introduced by `D.Lgs. 118/2011` to carry resources for multi-year works between budget years. It is a stock, not a source of money: a high value can reflect planned works as well as slow execution.
5. **Applied Administrative Surplus (`Avanzo di amministrazione applicato`, Quadro 2.1):** Accumulated unencumbered reserves channeled into capital outlays."""),
    (C, '''cap_fund = capital_funding_breakdown(fiscal, core_only=True)
print("Cesena Capital Expenditure vs. Funding Sources per Capita (constant 2020 EUR/pc, Cesena vs. Core Peer median):")
display(cap_fund.round(2))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
# Plot 1: Capital Funding Breakdown for Cesena over time
axes[0].plot(cap_fund["year"], cap_fund["cesena_spese_capitale_pc"], color="black", lw=2.2, label="Capital Expenditure (Titolo II)")
axes[0].plot(cap_fund["year"], cap_fund["cesena_contributi_investimenti_pc"], color=EU_C, lw=1.6, ls="--", label="Capital grants / transfers")
axes[0].plot(cap_fund["year"], cap_fund["cesena_alienazioni_pc"], color="peru", lw=1.6, ls="-.", label="Asset sales (Alienazioni)")
axes[0].plot(cap_fund["year"], cap_fund["cesena_accensione_mutui_pc"], color=ITALY_C, lw=1.6, marker="o", ms=3.0, label="New borrowing (Mutui)")
axes[0].axvline(2015.5, color="k", ls=":", lw=1.0)
axes[0].set_title("Cesena: Capital Outlays vs Funding Channels (EUR/pc)", fontsize=9.5)
axes[0].set_ylabel("Constant 2020 EUR per capita")
axes[0].legend(fontsize=7.5)

# Plot 2: Multi-Year Capital FPV and Applied Surplus in 2016-2021 (D.Lgs. 118)
sub_h = cap_fund[cap_fund["year"] >= 2016]
axes[1].plot(sub_h["year"], sub_h["cesena_fpv_capitale_pc"], color=ITALY_C, lw=2.2, marker="s", ms=4.0, label="Cesena Capital FPV (cushion)")
axes[1].plot(sub_h["year"], sub_h["peer_fpv_capitale_pc"], color=EU_C, lw=1.8, ls="--", label="Core Peer median Capital FPV")
axes[1].plot(sub_h["year"], sub_h["cesena_avanzo_amministrazione_pc"], color="teal", lw=1.6, marker="^", ms=3.5, label="Cesena applied surplus")
axes[1].plot(sub_h["year"], sub_h["peer_avanzo_amministrazione_pc"], color="gray", lw=1.4, ls=":", label="Core Peer median applied surplus")
axes[1].set_title("Capital FPV & Applied Surplus (2016-2021 Consuntivo)", fontsize=9.5)
axes[1].set_ylabel("Constant 2020 EUR per capita")
axes[1].legend(fontsize=7.5)
plt.tight_layout()
plt.show()'''),
    (M, """## 4. Local Tax Pressure: Own Taxes, IMU/ICI, and Addizionale Comunale IRPEF (`2008–2024`)
We inspect total municipal tax revenues (`Titolo I, Tipologia 101 Imposte, tasse e proventi assimilati`), property taxation (`IMU/ICI`), and the municipal personal income tax surcharge (`Addizionale Comunale IRPEF`), combining the municipal balance sheets (`2008–2021`) with the MEF tax declarations panel (`2008–2024`)."""),
    (C, '''tax_mef = cesena_vs_peers_series(panel[panel["year"] <= 2024], "mef_addizionale_real_pc", core_only=True)
print("MEF Addizionale Comunale IRPEF per resident (constant 2020 EUR, 2008-2024):")
display(tax_mef[["year", "cesena", "core_peer_median", "er_peer_median", "gap_vs_benchmark", "n_core_peers"]].round(2))

fisc_ces = fiscal[(fiscal["comune_id"] == "040007") & (fiscal["budget_type"] == "consuntivo")][
    ["year", "regime", "entrate_tributarie_comp_real_pc", "imposte_tasse_comp_real_pc", "imu_ici_comp_real_pc", "addizionale_irpef_comp_real_pc", "mef_addizionale_real_pc", "effective_surcharge_rate_pct"]
]
print("Cesena tax revenue breakdown (2008-2021 consuntivo, constant 2020 EUR per capita):")
display(fisc_ces.round(2))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
s_imp = cesena_vs_peers_series(cons, "imposte_tasse_comp_real_pc", core_only=True)
for reg_cond in [s_imp["year"] <= 2015, s_imp["year"] >= 2016]:
    sub = s_imp[reg_cond]
    axes[0].plot(sub["year"], sub["core_peer_median"], color=EU_C, ls="--", lw=1.8, label="Core peer median" if reg_cond.iloc[0] else None)
    axes[0].plot(sub["year"], sub["cesena"], color=ITALY_C, lw=2.3, marker="o", ms=3.5, label="Cesena" if reg_cond.iloc[0] else None)
axes[0].axvline(2015.5, color="k", ls=":", lw=1.0)
axes[0].set_title("Own Taxes & Fees (Imposte e Tasse, BDAP)", fontsize=9.5)
axes[0].set_ylabel("2020 EUR per capita")
axes[0].legend(fontsize=8)

axes[1].plot(tax_mef["year"], tax_mef["core_peer_median"], color=EU_C, ls="--", lw=1.8, label="Core peer median")
axes[1].fill_between(tax_mef["year"], tax_mef["core_peer_p25"], tax_mef["core_peer_p75"], color=PEER_C, alpha=0.3)
axes[1].plot(tax_mef["year"], tax_mef["cesena"], color=ITALY_C, lw=2.3, marker="o", ms=3.5, label="Cesena")
axes[1].set_title("Addizionale Comunale IRPEF (MEF, 2008-2024)", fontsize=9.5)
axes[1].set_ylabel("2020 EUR per resident")
plt.tight_layout()
plt.show()'''),
    (M, """## 5. Balance Sheet Rigidity & Revenue Collection Capacity
We examine two structural indicators of municipal financial health:
1. **Official Balance Sheet Rigidity (`rigidity_official_pct`):** `(Spesa di personale + Rimborso prestiti Titolo IV) / Entrate correnti (Titoli I + II + III) * 100`. For `2016–2018`, when OpenBilanci's harmonized tree omits the macroaggregato breakdown, personnel expenditure (`Macroaggregato 101 Redditi da lavoro dipendente`) is supplemented directly from the Ministry of the Interior (`Finanza Locale CCOX Quadro 8/10`).
2. **Revenue Collection Capacity (`collection_rate_correnti_pct` & `collection_rate_proprie_pct`):** Cash collections (`Riscossioni in c/competenza + c/residui`) divided by accrual assessments (`Accertamenti`) for total current revenues and own revenues (`Titolo I + Titolo III`)."""),
    (C, '''for col, lbl in [
    ("rigidity_official_pct", "Official rigidity index: (Personnel + Debt principal) / Current revenues (%)"),
    ("collection_rate_correnti_pct", "Current revenue cash-to-accrual collection rate (%)"),
    ("collection_rate_proprie_pct", "Own revenue (Titolo I + III) cash-to-accrual collection rate (%)"),
]:
    s = cesena_vs_peers_series(cons, col, core_only=True)
    print(f"\\n--- {lbl} ---")
    display(s[["year", "cesena", "core_peer_median", "er_peer_median", "gap_vs_benchmark", "n_core_peers", "missing_core_peers"]].round(2))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (col, title) in zip(axes, [
    ("rigidity_official_pct", "Official Rigidity Index (% of Current Revenues)"),
    ("collection_rate_correnti_pct", "Current Revenue Collection Rate (Cash / Accrual %)"),
]):
    s = cesena_vs_peers_series(cons, col, core_only=True)
    for reg_cond in [s["year"] <= 2015, s["year"] >= 2016]:
        sub = s[reg_cond]
        ax.plot(sub["year"], sub["core_peer_median"], color=EU_C, ls="--", lw=1.8, label="Core peer median" if reg_cond.iloc[0] else None)
        ax.fill_between(sub["year"], sub["core_peer_p25"], sub["core_peer_p75"], color=PEER_C, alpha=0.3)
        ax.plot(sub["year"], sub["cesena"], color=ITALY_C, lw=2.3, marker="o", ms=3.5, label="Cesena" if reg_cond.iloc[0] else None)
    ax.axvline(2015.5, color="k", ls=":", lw=1.0)
    ax.set_title(title, fontsize=9.5)
    ax.set_ylabel("%")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()'''),
    (M, """## 6. Consolidated Municipal Perimeter: *Unione dei Comuni Valle del Savio* & SIOPE Cash Accounts (`2016–2024`)
In Emilia-Romagna, municipalities differ sharply in how local public functions (social welfare, local police, civil protection, IT) are organized: some retain services on the direct municipal balance sheet, while others delegate them to an *Unione di Comuni* (*Unione dei Comuni Valle del Savio* for Cesena, *Unione della Romagna Faentina* for Faenza, *Unione Terre d'Argine* for Carpi, *Nuovo Circondario Imolese* for Imola, *Romagna Forlivese* for Forlì).

To test whether Cesena's lower apparent direct spending on social rights (`M12`) and local police (`M03`) reflects true expenditure containment or institutional externalization, we integrate MEF RGS **SIOPE cash expenditure accounts (`2016–2024`)** across all 13 benchmark municipalities and their associated Unioni, constructing a pro-rata **consolidated expenditure perimeter (`spese_consolidate_pc`)**."""),
    (C, '''from cifresignificative.studies.cesena.analysis.data import load_siope_consolidated
from cifresignificative.studies.cesena.analysis.fiscal import summarize_consolidated_benchmarks

siope_cons = load_siope_consolidated()
print("SIOPE Municipal & Unione Consolidated Expenditure in 2024 (EUR per capita):")
tbl_2024 = siope_cons[siope_cons["year"] == 2024][
    ["entity_name", "direct_correnti_pc", "unione_correnti_pc", "consolidated_correnti_pc", "direct_personale_pc", "consolidated_personale_pc", "consolidated_capitale_pc"]
].sort_values("consolidated_correnti_pc", ascending=False)
display(tbl_2024.round(1))

bench_cons = summarize_consolidated_benchmarks(siope_cons, year=2024, core_only=True)
print("\\nCesena vs Core Peers Consolidated Benchmark (2024):")
display(bench_cons[["metric_label", "cesena", "peer_median", "gap_vs_peers", "gap_pct", "rank_desc", "n_comuni"]].round(2))

fig, ax = plt.subplots(figsize=(10, 4.2))
ces_ts = siope_cons[siope_cons["comune_id"] == "040007"].sort_values("year")
peers_ts = siope_cons[siope_cons["comune_id"] != "040007"].groupby("year")[["direct_correnti_pc", "consolidated_correnti_pc"]].median()

ax.plot(ces_ts["year"], ces_ts["direct_correnti_pc"], color=ITALY_C, lw=1.8, ls="--", marker="o", label="Cesena Direct Current Spending (Comune)")
ax.plot(ces_ts["year"], ces_ts["consolidated_correnti_pc"], color=ITALY_C, lw=2.4, marker="s", label="Cesena Consolidated Current Spending (Comune + Unione)")
ax.plot(peers_ts.index, peers_ts["direct_correnti_pc"], color=EU_C, lw=1.6, ls=":", label="ER Peer Median Direct Current Spending")
ax.plot(peers_ts.index, peers_ts["consolidated_correnti_pc"], color=EU_C, lw=2.2, ls="--", label="ER Peer Median Consolidated Current Spending")
ax.set_title("SIOPE Cash Accounts: Direct vs Consolidated Current Spending per Capita (2016-2024)", fontsize=10)
ax.set_ylabel("EUR per capita (cash)")
ax.set_xlabel("Year")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()'''),
    (M, """## 7. Spending Against Standard Need and Service Output: SOSE / MEF OpenCivitas (2023)

OpenCivitas compares each Italian municipality's spending with a standard need and places its service output on a national scale:
1. **Fabbisogni Standard (Standard Expenditure Needs):** the modelled cost of standard services given demography, geography and deprivation.
2. **Spesa Storica (Historical Expenditure):** actual operating expenditure.
3. **Livelli Quantitativi delle Prestazioni (LQP, national deciles 1–10):** quantities of services delivered. The score counts quantities rather than quality, so it does not establish efficiency."""),
    (C, '''from cifresignificative.studies.cesena.analysis.opencivitas import (
    peer_standard_needs_comparison,
    comprensorio_standard_needs_comparison,
    social_and_childcare_benchmarks,
)

# 1. Peer Comparison on Fundamental Functions (OpenCivitas 2023)
peer_needs = peer_standard_needs_comparison()
print("SOSE / MEF OpenCivitas Standard Needs & Service Levels across 13 ER Municipalities (2023):")
display(peer_needs.peers_df[[
    "comune", "tot_spesa_storica_eur_pc", "tot_fabbisogno_standard_eur_pc",
    "tot_diff_eur_pc", "tot_diff_pct", "tot_livello_spesa_0_10",
    "tot_livello_servizi_0_10", "tot_coord_spesa", "tot_coord_servizi", "quadrante_performance"
]].round(2))

# 2. Social Services & Childcare Benchmarks
soc_sum = social_and_childcare_benchmarks()
print("\\nSocial Services & Childcare Provision (OpenCivitas 2023):")
display(soc_sum.social_df[[
    "comune", "soc_spesa_storica_eur_pc", "soc_fabbisogno_standard_eur_pc",
    "copertura_nidi_pct", "deprivazione_indice", "gestione_associata_sociale", "soc_coord_servizi"
]].round(2))

# 3. Dual-Panel Visualization: Efficiency Quadrant & Childcare Coverage
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel 1: OpenCivitas Performance Positioning (-5 to +5 Coordinates)
df_p = peer_needs.peers_df
colors_pts = [ITALY_C if c == "Cesena" else EU_C for c in df_p["comune"]]

axes[0].scatter(df_p["tot_coord_spesa"], df_p["tot_coord_servizi"], color=colors_pts, s=80, zorder=3)
for _, row in df_p.iterrows():
    dx = 0.15 if row["comune"] != "Cesena" else -0.3
    dy = 0.15 if row["comune"] != "Cesena" else 0.25
    axes[0].text(row["tot_coord_spesa"] + dx, row["tot_coord_servizi"] + dy, row["comune"], fontsize=8,
                 fontweight="bold" if row["comune"] == "Cesena" else "normal")

axes[0].axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
axes[0].axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
axes[0].set_xlabel("Expenditure Coordinate (coord_spesa: left = lower spending)")
axes[0].set_ylabel("Service Output Coordinate (coord_servizi: top = higher service)")
axes[0].set_title("OpenCivitas Performance Quadrants (2023)", fontsize=9.5)
axes[0].grid(alpha=0.3)
axes[0].text(-4.5, 4.5, "VIRTUOUS QUADRANT\\n(Low Cost, High Service)", color="darkgreen", fontsize=8, fontweight="bold")
axes[0].text(1.5, 4.5, "High Cost, High Service", color="gray", fontsize=7.5)
axes[0].text(-4.5, -4.5, "Low Cost, Low Service", color="gray", fontsize=7.5)
axes[0].text(1.5, -4.5, "INEFFICIENT QUADRANT\\n(High Cost, Low Service)", color="darkred", fontsize=8, fontweight="bold")

# Panel 2: Childcare Coverage Rate (%) across 13 Peers
df_nidi = soc_sum.social_df.sort_values("copertura_nidi_pct", ascending=True)
y_pos = np.arange(len(df_nidi))
colors_nidi = [ITALY_C if c == "Cesena" else EU_C for c in df_nidi["comune"]]

axes[1].barh(y_pos, df_nidi["copertura_nidi_pct"], color=colors_nidi, alpha=0.85)
axes[1].set_yticks(y_pos, df_nidi["comune"], fontsize=8.5)
axes[1].axvline(33.0, color="red", ls="--", lw=1.2, label="EU 33% Barcelona Target")
axes[1].set_xlabel("Childcare Coverage Rate (%)")
axes[1].set_title("Childcare Coverage (Asili Nido, OpenCivitas 2023)", fontsize=9.5)
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.show()'''),
    (M, """## 5b. Rigidity after 2021: a cash proxy from SIOPE (2017–2024)
The accrual series ends in 2018 for personnel and in 2021 for the rest, and no public per-municipality accrual source exists after 2021 (OpenBilanci stops at 2021; OpenBDAP publishes only national and macro-area aggregates). SIOPE gives cash payments and collections to 2024, so a cash proxy is computed: `(personnel + interest + loan repayment) / current revenue`. It is checked against the official ratio on the years where both exist (2017–2018) and never joined to it. The waste tax (TARI/TARSU) is booked in the municipal budget in some cities and collected through the operator's tariff in others, and Cesena changed in 2023, so the ratio is also shown without it in current revenue."""),
    (C, '''import pandas as pd
from cifresignificative.core.geo import CESENA, CORE_PEER_IDS

cp = pd.read_parquet("data/processed/cesena/siope_cash_panel.parquet")
cp = cp[cp["entity_type"] == "comune"]
core = [c for c in CORE_PEER_IDS if c != CESENA]
for col in ["rigidity_cash_pct", "rigidity_cash_ex_tari_pct"]:
    w = cp.pivot(index="year", columns="comune_id", values=col)
    out = pd.DataFrame({"cesena": w[CESENA], "core_median": w[core].median(axis=1), "all_peer_median": w.drop(columns=CESENA).median(axis=1)}).round(1)
    print(col); display(out)
acc = fiscal[(fiscal["regime"] == "dlgs118") & (fiscal["budget_type"] == "consuntivo") & fiscal["year"].isin([2017, 2018])][["comune_id", "year", "rigidity_official_pct"]]
both = cp[cp["year"].isin([2017, 2018])][["comune_id", "year", "rigidity_cash_pct"]].merge(acc, on=["comune_id", "year"]).dropna()
print(f"Anchor 2017-2018: {len(both)} city-years, correlation of the cash proxy with the official ratio {both.rigidity_cash_pct.corr(both.rigidity_official_pct):.2f}, mean gap {(both.rigidity_cash_pct - both.rigidity_official_pct).mean():+.1f} points")
print("Waste tax collected in the municipal budget (million euro, cash):")
display((cp.pivot(index="comune_id", columns="year", values="tari_tarsu_cassa_eur") / 1e6).round(1))'''),
    (M, """## Takeaways (Pillar 1 — Municipal Fiscal Policy, 13 ER Municipalities; Descriptive)

- **Current spending is low (`robust`).** Cesena ranks near the bottom of the 13 cities on direct current spending in both accounting regimes, which are never compared with each other.
- **Capital spending is middling (`sensitive`).** The average depends on 2019. The city used more loans and fewer grants than most peers; FPV is a stock, not a funding source.
- **Taxes are low (`robust`).** The municipal IRPEF surcharge per resident is the lowest of the 13 in 2024. Budget rigidity in 2016–2018 is not unusually low; in 2019–2024 the cash proxy puts it above the Core Peer median, and whether it rose or fell depends on the waste-tax regime.
- **The Unione narrows the gap but does not close it (`sensitive`).** After netting transfers, consolidated current spending stays low against all 12 peers and close to Core Peers. Unione spending is split by population, and perimeters still differ.
- **OpenCivitas (`sensitive`).** The official low-spend, high-output placement holds but is shared with several peers, and it measures quantities.
- **Limits.** All statements are descriptive associations for one municipality, never causal claims. The figures are in the table below and in `reports/findings.md`, both computed by `analysis/claims.py`."""),
    (C, '''from cifresignificative.studies.cesena.analysis.claims import claims_for_notebook

print("Figures behind this notebook, computed from the processed tables (the same ones quoted in reports/findings.md):")
with pd.option_context("display.max_rows", None, "display.max_colwidth", 60):
    display(claims_for_notebook("01"))'''),
]
