"""Cells of notebook 03_real_economy_shift_share (Pillar 3: Local Real Economy, Sectoral Structure & Shift-Share Decomposition)."""
from . import NOTEBOOKS, C, M

NOTEBOOKS["03_real_economy_shift_share"] = [
    (M, """# Pillar 3 — Local Real Economy, Sectoral Structure & Productivity Proxies: Shift-Share & Firm-Size Decomposition (`2012–2024`)

This notebook examines the productive fabric of **Cesena** (`040007`) against its Emilia-Romagna peers using the **ISTAT ASIA Register of Active Local Units (`Registro statistico delle unità locali delle imprese attive`, `2012–2024`)** across 78 2-digit ATECO 2007 divisions and 7 specialization clusters (`agro_industrial`, `wellness_biomed`, `ict_software`, `tradable_manuf`, `construction_re`, `commerce_hosp`, `business_personal_serv`).

**Methodological Design:**
1. **Endpoint selection before `dropna`:** Following our platform rule (`tests/studies/italy_productivity/test_sectors.py`), every Shift-Share decomposition selects the two compared years (`t0`, `t1`) *before* pivoting/filtering and prints the exact peer count (`n_peers`) and peer list (`peer_ids`).
2. **Exact Classical & Esteban-Marquillas Shift-Share Identity:**
   $$\\Delta E_{i} = \\text{NS}_{i} + \\text{IM}_{i} + \\text{CS}_{i}$$
   where $\\text{NS}_{i}$ is the Benchmark Share (growth at the peer aggregate employment growth rate, strictly excluding Cesena), $\\text{IM}_{i}$ is the Industry Mix effect, and $\\text{CS}_{i}$ is the Local Competitive Shift, further decomposed via Esteban-Marquillas (1972) homothetic employment into pure Competitive Advantage (`em_comp_pp`) and Allocative Effect (`em_alloc_pp`).
3. **Exact Bennet Firm-Size Decomposition (`firm_size_decomposition`):** Decomposes the change in municipal average local unit size (`employees / local_units`) between `t0` and `t1` into `mix_effect` (shifts in unit shares across sectors) and `within_scaling_effect` (firm scaling within sectors). Descriptive only."""),
    (C, '''from cifresignificative.studies.cesena.analysis.data import (
    load_asia, load_asia_size_classes, load_panel_comuni, cesena_vs_peers_series,
    load_frame_sbs_panel, load_frame_sbs_summary,
)
from cifresignificative.studies.cesena.analysis.economy import (
    shift_share, shift_share_all_municipalities, location_quotients,
    firm_size_decomposition, size_class_summary, size_class_by_cluster,
    summarize_frame_sbs_benchmarks, compare_productivity_and_wages,
)

asia = load_asia()
asia_size = load_asia_size_classes()
panel = load_panel_comuni()

print("ASIA municipal 2-digit ATECO dataset coverage (years, municipalities, 2-digit divisions):")
print("Years:", sorted(asia["year"].unique()))
print("Municipalities:", asia["comune_id"].nunique(), "| 2-digit ATECO codes (incl. TOTAL):", asia["ateco_2d"].nunique())

for col, lbl in [
    ("employees", "Total private employees in active local units"),
    ("local_units", "Total active local units"),
    ("avg_firm_size", "Average local unit size (employees / local unit)"),
    ("employees_per_1k_pop", "Private employees per 1,000 residents"),
]:
    s = cesena_vs_peers_series(panel[panel["year"].between(2012, 2024)], col, core_only=True)
    print(f"\\n--- {lbl} (2012-2024) ---")
    display(s[["year", "cesena", "core_peer_median", "er_peer_median", "gap_vs_benchmark", "n_core_peers", "missing_core_peers"]].round(2))'''),
    (M, """## 1. Sectoral Specialization & Location Quotients (`LQ`) in Cesena (`2012` vs `2024`)
We compute the Location Quotient ($\\text{LQ}_i = (E_{i,c}/E_c) / (E_{i,\\text{bench}}/E_{\\text{bench}})$) of Cesena relative to the 12 Emilia-Romagna peers (excluding Cesena) first across the 7 specialization clusters and then across the largest 2-digit ATECO divisions."""),
    (C, '''lq_clust = location_quotients(asia, level="specialization")
lq_12 = lq_clust[lq_clust["year"] == 2012][["specialization", "focal_val", "focal_share_pct", "lq"]].rename(
    columns={"focal_val": "emp_2012", "focal_share_pct": "share_2012_pct", "lq": "lq_2012"}
)
lq_24 = lq_clust[lq_clust["year"] == 2024][["specialization", "sector_label", "focal_val", "focal_share_pct", "bench_share_pct", "lq"]].rename(
    columns={"focal_val": "emp_2024", "focal_share_pct": "share_2024_pct", "lq": "lq_2024"}
)
lq_comp = lq_24.merge(lq_12, on="specialization")
print("Cesena specialization clusters (2012 vs 2024, benchmark = 12 ER peers ex Cesena):")
display(lq_comp[["specialization", "sector_label", "emp_2012", "emp_2024", "share_2012_pct", "share_2024_pct", "bench_share_pct", "lq_2012", "lq_2024"]].round(2))

lq_2d = location_quotients(asia, level="ateco_2d")
lq_2d_24 = lq_2d[lq_2d["year"] == 2024].copy()
lbl_map = dict(zip(asia["ateco_2d"], asia["ateco_label"], strict=False))
lq_2d_24["ateco_label"] = lq_2d_24["ateco_2d"].map(lbl_map)
print("\\nTop 15 2-digit ATECO divisions in Cesena by employment (2024, with LQ vs 12 ER peers):")
display(lq_2d_24.sort_values("focal_val", ascending=False).head(15)[["ateco_2d", "ateco_label", "focal_val", "focal_share_pct", "bench_share_pct", "lq"]].round(2))
print("\\nTop 10 high-specialization 2-digit ATECO divisions in Cesena (LQ > 1.25 and >= 250 employees in 2024):")
display(lq_2d_24[(lq_2d_24["focal_val"] >= 250) & (lq_2d_24["lq"] > 1.25)].sort_values("lq", ascending=False).head(10)[["ateco_2d", "ateco_label", "focal_val", "focal_share_pct", "bench_share_pct", "lq"]].round(2))'''),
    (M, """## 2. Classical & Esteban-Marquillas Shift-Share Decomposition (`2012–2019`, `2019–2024`, `2012–2024`)
We run the exact Shift-Share decomposition across three windows:
- **Pre-pandemic expansion (`2012–2019`)**
- **Pandemic & post-pandemic recovery (`2019–2024`)**
- **Full long-run cycle (`2012–2024`)**

Both against the **6 Core Peers** and against **All 12 ER Peers**, at the specialization-cluster level and at the 77 2-digit ATECO division level."""),
    (C, '''summary_rows = []
for (t0, t1) in [(2012, 2019), (2019, 2024), (2012, 2024)]:
    for core_flag, bench_lbl in [(True, "Core Peers (6)"), (False, "All ER Peers (12)")]:
        for lvl in ["specialization", "ateco_2d"]:
            df_ss, s = shift_share(asia, t0=t0, t1=t1, level=lvl, core_only=core_flag)
            summary_rows.append({
                "period": f"{t0}-{t1}",
                "benchmark": bench_lbl,
                "level": lvl,
                "n_peers": s["n_peers"],
                "emp_t0": s["v0_total"],
                "emp_t1": s["v1_total"],
                "delta_abs": s["delta_total_abs"],
                "cesena_growth_pct": s["local_growth_total_pct"],
                "bench_growth_pct": s["bench_growth_total_pct"],
                "ns_pp": s["ns_total_pp"],
                "im_pp": s["im_total_pp"],
                "cs_pp": s["cs_total_pp"],
                "active_gap_pp": s["active_gap_pp"],
                "identity_err_pp": abs(s["local_growth_total_pct"] - (s["ns_total_pp"] + s["im_total_pp"] + s["cs_total_pp"])),
            })

ss_sum_df = pd.DataFrame(summary_rows)
print("Shift-Share Total Decomposition Summary across Windows, Benchmarks, and Granularities:")
display(ss_sum_df.round(2))

ss_clust_full, s_clust = shift_share(asia, t0=2012, t1=2024, level="specialization", core_only=True)
print("\\nDetailed Cluster Shift-Share (2012-2024, vs 6 Core Peers):")
display(ss_clust_full[[
    "sector", "sector_label", "v0", "v1", "delta_abs", "local_growth_pct", "bench_growth_pct",
    "ns_pp", "im_pp", "cs_pp", "total_contrib_pp", "em_comp_pp", "em_alloc_pp",
]].round(2))

ss_2d_full, _ = shift_share(asia, t0=2012, t1=2024, level="ateco_2d", core_only=True)
print("\\nTop 10 job-creating 2-digit ATECO divisions in Cesena (2012-2024, vs 6 Core Peers):")
display(ss_2d_full.sort_values("delta_abs", ascending=False).head(10)[[
    "sector", "sector_label", "v0", "v1", "delta_abs", "local_growth_pct", "bench_growth_pct", "im_pp", "cs_pp", "total_contrib_pp"
]].round(2))
print("\\nBottom 8 job-shedding 2-digit ATECO divisions in Cesena (2012-2024, vs 6 Core Peers):")
display(ss_2d_full.sort_values("delta_abs", ascending=True).head(8)[[
    "sector", "sector_label", "v0", "v1", "delta_abs", "local_growth_pct", "bench_growth_pct", "im_pp", "cs_pp", "total_contrib_pp"
]].round(2))

all_muni_ss = shift_share_all_municipalities(asia, t0=2012, t1=2024, level="specialization")
print("\\nLeave-one-out Shift-Share Ranking across all 13 ER Municipalities (2012-2024, sorted by Competitive Shift cs_total_pp):")
display(all_muni_ss[[
    "cs_rank", "focal_name", "v0_total", "v1_total", "delta_total_abs",
    "local_growth_total_pct", "bench_growth_total_pct", "ns_total_pp", "im_total_pp", "cs_total_pp", "active_gap_pp"
]].round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
c_plot = ss_clust_full.sort_values("delta_abs")
y = np.arange(len(c_plot))
axes[0].barh(y, c_plot["delta_abs"], color=ITALY_C, alpha=0.85, label="Actual change (emp.)")
axes[0].set_yticks(y, c_plot["sector"], fontsize=8.5)
axes[0].axvline(0, color="k", lw=0.8)
axes[0].set_xlabel("Employment change (2012-2024, heads)")
axes[0].set_title("Employment Change by Cluster (Cesena, 2012-2024)", fontsize=9.5)

axes[1].barh(y - 0.18, c_plot["im_pp"], height=0.35, color=EU_C, label="Industry Mix (im_pp)")
axes[1].barh(y + 0.18, c_plot["cs_pp"], height=0.35, color=ITALY_C, label="Competitive Shift (cs_pp)")
axes[1].set_yticks(y, ["" for _ in c_plot["sector"]])
axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_xlabel("Contribution to total employment growth (pp vs Core Peers)")
axes[1].set_title("Industry Mix vs Local Competitive Shift (pp, 2012-2024)", fontsize=9.5)
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()'''),
    (M, """## 3. Exact Bennet Decomposition of Average Local Unit Size (`employees / local_units`, `2012–2024`)
How much of each municipality's change in average local unit size between `2012` and `2024` comes from shifts in sector unit shares (`mix_effect`) versus within-sector firm scaling (`within_scaling_effect`)?"""),
    (C, '''fs_df = firm_size_decomposition(asia, t0=2012, t1=2024)
print("Exact Bennet Decomposition of Change in Average Local Unit Size (2012-2024) across all 13 ER Municipalities:")
display(fs_df[[
    "comune", "units_t0", "units_t1", "units_growth_pct",
    "employees_t0", "employees_t1", "employees_growth_pct",
    "avg_size_t0", "avg_size_t1", "delta_avg_size", "mix_effect", "within_scaling_effect",
]].round(3))'''),
    (M, """## 4. Employment Size-Class Decomposition (`W0_9`, `W10_49`, `W50_249`, `W_GE250`, `2012–2024`)
Does Cesena's `#1` within-sector firm-size scaling reflect a consolidation of micro-units (`0–9` workers) into small and medium-sized enterprises (`10–49` and `50–249` workers), or employment concentration inside large anchor plants (`250+` workers)? We decompose `asia_size_classes.parquet` across the four ISTAT ASIA employment size classes and across specialization clusters $\\times$ size classes."""),
    (C, '''sc_class, sc_muni = size_class_summary(asia_size, t0=2012, t1=2024, core_only=True)
print("Cesena vs 6 Core Peers by Employment Size Class (2012-2024):")
display(sc_class[[
    "size_class", "size_label", "units_t0", "units_t1", "delta_units", "units_growth_pct",
    "emp_t0", "emp_t1", "delta_emp", "emp_growth_pct", "bench_emp_growth_pct",
    "emp_share_t0_pct", "emp_share_t1_pct", "bench_emp_share_t1_pct",
    "contrib_to_total_growth_pp", "bench_contrib_to_total_growth_pp",
]].round(2))

print("\\nAll 13 ER Municipalities — Employment Growth Contributions (pp) by Size Class (2012-2024, ranked by Medium 50-249 contribution):")
display(sc_muni[[
    "rank_W50_249_contrib", "comune", "emp_growth_pct",
    "contrib_W0_9_pp", "contrib_W10_49_pp", "contrib_W50_249_pp", "contrib_W_GE250_pp",
    "share_W0_9_t0_pct", "share_W0_9_t1_pct", "delta_units_W50_249", "delta_units_W_GE250",
]].round(2))

sc_clust = size_class_by_cluster(asia_size, t0=2012, t1=2024)
print("\\nCesena Specialization Cluster x Size Class Employment Change (2012-2024):")
display(sc_clust[[
    "specialization", "specialization_label",
    "delta_W0_9", "delta_W10_49", "delta_W50_249", "delta_W_GE250", "delta_total",
    "emp_W0_9_2024", "emp_W10_49_2024", "emp_W50_249_2024", "emp_W_GE250_2024", "emp_total_t1",
]].round(2))'''),
    (M, """## 5. ISTAT FRAME-SBS Territorial Economic Accounts: Value Added, Productivity & Wages (`2017–2023`)
How do **Value Added per worker (`VA / addetto`)** and **salaried wages per employee (`retribuzione / dipendente`)** compare with peers? Employment here shares the 2022 food-sector break seen in ASIA, so ratios per worker are unstable.

We examine the **ISTAT FRAME-SBS Territoriale (`Registro Statistico Esteso dei Conti Economici delle Imprese`, `2017–2023`)** municipal accounts across:
- **Total market economy (`totale`)**
- **Industry (`industria`)**
- **Services (`servizi`)**"""),
    (C, '''frame_panel = load_frame_sbs_panel()
frame_sum = load_frame_sbs_summary()

print("FRAME-SBS municipal panel coverage (years, municipalities, macro-sectors):")
print("Years:", sorted(frame_panel["year"].unique()))
print("Municipalities:", frame_panel["comune_id"].nunique(), "| Macro-sectors:", sorted(frame_panel["macro_sector"].unique()))

for sec, lbl in [
    ("totale", "Total Market Economy (Industry + Services)"),
    ("industria", "Industry (Manufacturing, Mining, Energy, Construction)"),
    ("servizi", "Market Services (Commerce, Transport, Tourism, ICT, Business/Personal Services)"),
]:
    b = summarize_frame_sbs_benchmarks(frame_panel, sector=sec)
    print(f"\\n--- {lbl}: Cesena vs Peers (2017-2023) ---")
    display(b[[
        "year", "cesena_addetti", "cesena_va_per_addetto_eur", "core_peer_va_per_addetto_eur",
        "gap_va_per_addetto_core_pct", "rank_va_per_addetto_er",
        "cesena_retrib_per_dipendente_eur", "core_peer_retrib_per_dipendente_eur",
        "gap_retrib_per_dipendente_core_pct", "rank_retrib_per_dipendente_er",
        "cesena_va_su_fatturato_pct", "cesena_retrib_su_va_pct",
    ]].round(2))

# Cumulative growth comparison 2017-2023
growth_tot = compare_productivity_and_wages(frame_panel, sector="totale", t0=2017, t1=2023)
print("\\nCumulative Real Growth across all 13 ER Municipalities (Total Market Economy, 2017-2023):")
display(growth_tot[[
    "comune_nome", "growth_addetti_pct", "rank_growth_addetti",
    "growth_real_va_pct", "rank_growth_real_va",
    "growth_real_vap_pct", "rank_growth_real_vap",
    "growth_real_wage_pct", "rank_growth_real_wage",
]].round(2))

growth_ind = compare_productivity_and_wages(frame_panel, sector="industria", t0=2017, t1=2023)
print("\\nCumulative Real Growth across all 13 ER Municipalities (Industry, 2017-2023):")
display(growth_ind[[
    "comune_nome", "growth_addetti_pct", "rank_growth_addetti",
    "growth_real_va_pct", "rank_growth_real_va",
    "growth_real_vap_pct", "rank_growth_real_vap",
    "growth_real_wage_pct", "rank_growth_real_wage",
]].round(2))

# Plot Cesena vs Core Peers labor productivity and wages
fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
b_tot = summarize_frame_sbs_benchmarks(frame_panel, sector="totale")
b_ind = summarize_frame_sbs_benchmarks(frame_panel, sector="industria")

axes[0].plot(b_tot["year"], b_tot["cesena_va_per_addetto_eur"] / 1000.0, "o-", color=ITALY_C, lw=2, label="Cesena - Total")
axes[0].plot(b_tot["year"], b_tot["core_peer_va_per_addetto_eur"] / 1000.0, "--", color=EU_C, lw=1.8, label="Core Peer Median - Total")
axes[0].plot(b_ind["year"], b_ind["cesena_va_per_addetto_eur"] / 1000.0, "s-", color="darkred", lw=1.5, label="Cesena - Industry")
axes[0].plot(b_ind["year"], b_ind["core_peer_va_per_addetto_eur"] / 1000.0, ":", color="navy", lw=1.5, label="Core Peer Median - Industry")
axes[0].set_title("Value Added per Worker (k EUR, 2017-2023)", fontsize=9.5)
axes[0].set_ylabel("Thousand EUR / worker")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8)

axes[1].plot(b_tot["year"], b_tot["cesena_retrib_per_dipendente_eur"] / 1000.0, "o-", color=ITALY_C, lw=2, label="Cesena - Total")
axes[1].plot(b_tot["year"], b_tot["core_peer_retrib_per_dipendente_eur"] / 1000.0, "--", color=EU_C, lw=1.8, label="Core Peer Median - Total")
axes[1].plot(b_ind["year"], b_ind["cesena_retrib_per_dipendente_eur"] / 1000.0, "s-", color="darkred", lw=1.5, label="Cesena - Industry")
axes[1].plot(b_ind["year"], b_ind["core_peer_retrib_per_dipendente_eur"] / 1000.0, ":", color="navy", lw=1.5, label="Core Peer Median - Industry")
axes[1].set_title("Gross Wages per Employee (k EUR, 2017-2023)", fontsize=9.5)
axes[1].set_ylabel("Thousand EUR / employee")
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()'''),
    (M, """## 7. Primary Agriculture (`ATECO A`) & Provincial Export Specialization (Coeweb & 7° Censimento Agricoltura)

Background at province level, not for the city. Any link between these figures and Cesena's food firms is not tested here, and the land data do not cover poultry.
1. **ISTAT 7° Censimento Generale dell'Agricoltura (2020):** land use (SAT and SAU), fruit orchards (`Frutta`), vineyards (`Vite`), and farm size (`<10 ha` to `>=100 ha`) across the 9 Emilia-Romagna provinces.
2. **ISTAT / Coeweb foreign trade (1991–2025):** provincial exports (`ITH58` Forlì-Cesena), trade balance, coverage ratio and top commodities."""),
    (C, '''from cifresignificative.studies.cesena.analysis.agri_export import (
    provincial_export_comparison,
    agricultural_land_specialization,
    agroindustrial_cluster_integration,
)

# 1. Provincial Export Comparison (2025)
exp_summary = provincial_export_comparison()
print("Provincial Foreign Trade Performance (2025 Coeweb, across 9 Emilia-Romagna provinces):")
display(exp_summary.provinces_df[[
    "provincia", "nuts3", "export_eur", "import_eur", "trade_balance_eur",
    "coverage_ratio_pct", "export_share_pct", "export_per_capita_eur", "balance_per_capita_eur",
    "rank_export", "rank_balance"
]].round(2))

# 2. Agricultural Land Specialization & Fragmentation (2020 Census)
agri_summary = agricultural_land_specialization()
print("\\nAgricultural Land Use Specialization & Farm Fragmentation (7° Censimento Agricoltura 2020):")
display(agri_summary.land_specialization_df[[
    "provincia", "sau_ha", "frutta_ha", "frutta_share_sau_pct", "frutta_share_er_pct",
    "vite_ha", "vite_share_sau_pct", "under10_ha", "under10_share_sau_pct", "over100_share_sau_pct"
]].round(2))

# 3. Dual-Panel Visualization
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

# Panel 1: Provincial Export & Trade Balance (2025)
df_p = exp_summary.provinces_df.sort_values("export_eur", ascending=True)
y_pos = np.arange(len(df_p))
colors_exp = [ITALY_C if p == "Forlì-Cesena" else EU_C for p in df_p["provincia"]]

axes[0].barh(y_pos, df_p["export_eur"] / 1e9, color=colors_exp, alpha=0.85, label="Export (Billion EUR)")
axes[0].scatter(df_p["trade_balance_eur"] / 1e9, y_pos, color="darkgreen", s=50, zorder=3, label="Trade Balance (Billion EUR)")
axes[0].set_yticks(y_pos, df_p["provincia"], fontsize=8.5)
axes[0].set_xlabel("Billion EUR (2025)")
axes[0].set_title("Provincial Foreign Trade (Coeweb 2025)", fontsize=9.5)
axes[0].axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="lower right")

# Panel 2: Fruit Orchards Specialization & Smallholder Share
df_ag = agri_summary.land_specialization_df[agri_summary.land_specialization_df["provincia"] != "Emilia-Romagna"].sort_values("frutta_ha", ascending=True)
y_ag = np.arange(len(df_ag))
colors_ag = [ITALY_C if p == "Forlì-Cesena" else PEER_C for p in df_ag["provincia"]]

axes[1].barh(y_ag, df_ag["frutta_ha"] / 1000.0, color=colors_ag, alpha=0.85, label="Fruit Orchards (k ha)")
axes[1].scatter(df_ag["under10_share_sau_pct"], y_ag, color="purple", s=50, zorder=3, label="Farms <10 ha (% of SAU)")
axes[1].set_yticks(y_ag, df_ag["provincia"], fontsize=8.5)
axes[1].set_xlabel("Thousand Hectares (bars) / % of SAU (dots)")
axes[1].set_title("Fruit Orchards & Smallholder Share (7° Censimento Agricoltura 2020)", fontsize=9.5)
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.show()'''),
    (M, """## 8. Register Breaks and Sensitivity (2026-10-30 Review)

ASIA counts persons employed by local unit, so one large unit can move a municipal sector by thousands in a year. The screen flags yearly moves of at least 300 employees and 20% that are mostly undone the next year (`reversal`) or reach 1,000 (`step`). It finds candidates; it does not establish a cause. The sensitivity table shows Cesena's value, rank and peer median with all sectors, without food (ATECO 10), and without food, temp agencies and cleaning (10, 78, 81)."""),
    (C, '''anom = pd.read_parquet("data/processed/cesena/asia_anomalies.parquet")
print(f"Flagged sector-years across the 13 cities: {len(anom)}; by city:")
display(anom.groupby("comune").size().sort_values(ascending=False).to_frame("flagged"))
display(anom[anom["comune_id"] == "040007"].round(2))
sens = pd.read_parquet("data/processed/cesena/employment_sensitivity.parquet")
display(sens.pivot_table(index=["metric", "variant"], columns="t1", values=["cesena", "rank"]).round(2))

# R1: the food break by province and by Cesena size class
from cifresignificative.core.sources import istat_asia
from cifresignificative.studies.cesena.analysis.economy import sector_size_swing
areas = pd.read_parquet("data/processed/cesena/asia_areas.parquet")
food = areas[(areas["ateco_2d"] == "10") & areas["year"].between(2019, 2024)].pivot(index="nuts", columns="year", values="employees")
print("Food manufacturing (ATECO 10) employees by province, ITD5 = region:")
display(food.round(0))
sizes = istat_asia.load_size_classes()
print("Cesena food by size class:")
display(sector_size_swing(sizes, ("10",)).round(2))
print("Cesena temp agencies + cleaning by size class:")
display(sector_size_swing(sizes, ("78", "81"), (2022, 2023, 2024)).round(2))'''),
    (M, """## Takeaways (Pillar 3 — Local Real Economy; Descriptive)

- **Job growth is `sensitive`.** Employment and competitive shift look strong to 2024 but depend on food manufacturing, temp agencies and cleaning, and on the end year. The food series breaks in 2022, at province level too, and the cause is unknown (research track R1).
- **The food break (research track R1) is an `accepted-limit`.** It is local to Forlì-Cesena among the 9 Emilia-Romagna provinces and sits in Cesena's units of 250 or more employees; the cause is unknown and no series is preferred.
- **Larger units are `sensitive` for the same reasons.** Size classes do not track one set of plants: a firm crossing a threshold moves all its jobs, and agency staff are counted at the agency.
- **FRAME-SBS (`sensitive`).** Real value added grew, but value added per worker swings with the same break. "Real" uses consumer prices; values per worker quoted for 2023 are nominal.
- **Agriculture and exports (`context only`).** Province-level background; links to local firms are not tested.
- **Limits.** All statements are descriptive associations for one municipality, never causal claims. The figures are in the table below and in `reports/findings.md`, both computed by `analysis/claims.py`."""),
    (C, '''from cifresignificative.studies.cesena.analysis.claims import claims_for_notebook

print("Figures behind this notebook, computed from the processed tables (the same ones quoted in reports/findings.md):")
with pd.option_context("display.max_rows", None, "display.max_colwidth", 60):
    display(claims_for_notebook("03"))'''),
]
