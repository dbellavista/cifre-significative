"""Cells of notebook 02_public_investments (Pillar 2: Public Investments, Cohesion/National Projects & Execution Capacity)."""
from . import NOTEBOOKS, C, M

NOTEBOOKS["02_public_investments"] = [
    (M, """# Pillar 2 — Public Investments & Cohesion/National Projects

This notebook analyses the public investment projects recorded in **OpenCoesione** (`2014–2020` programming cycle and national/regional cohesion funds: `FESR`, `FSE`, `FSC`, `POC`, `PAC`, `FEASR`) localized in **Cesena** (`040007`) and the 12 Emilia-Romagna peer municipalities.

**Design & Methodological Checks:**
1. **Multi-municipality allocation rule & single-municipality sensitivity:** OpenCoesione's `localizzazioni.zip` links each project (`COD_LOCALE_PROGETTO`) to one or more Italian municipalities. Following our platform convention (`core.sources.opencoesione`), multi-municipality projects have their allocated and disbursed euro amounts divided equally across the `n_comuni` distinct municipalities on the project. Because large regional or national infrastructure corridors span dozens of municipalities, we explicitly run a **sensitivity check restricting the sample to single-municipality projects (`n_comuni == 1`)** localized exclusively within each municipality.
2. **Execution rates & implementation lags:** We evaluate financial execution (`disbursement_rate_pct = disbursed / allocated * 100`), physical completion (`completion_rate_pct`, share of projects in `"concluso"` or `"liquidato"` status), implementation duration (`lag_days`, median days from start to effective completion), and schedule overrun (`delay_days`, effective minus planned completion date).
3. **Constant 2020 euro per capita:** Project values are deflated using the project's start year (or `2016` when missing) and normalized by 2020 resident population. Descriptive only."""),
    (C, '''from cifresignificative.studies.cesena.analysis.data import load_projects, load_demography
from cifresignificative.studies.cesena.analysis.investments import (
    summarize_investments_by_comune,
    summarize_investments_by_theme,
    annual_investment_flows,
)

projects = load_projects()
demo = load_demography()

print(f"Total OpenCoesione project-municipality rows across the 13 ER municipalities: {len(projects):,}")
print(f"Cesena (040007) projects: {int((projects['comune_id'] == '040007').sum()):,} "
      f"(single-municipality n_comuni==1: {int(((projects['comune_id'] == '040007') & (projects['n_comuni'] == 1)).sum()):,})")
print("Cesena projects by fund source:")
display(projects[projects["comune_id"] == "040007"].groupby("fund_source").agg(
    n_projects=("project_id", "count"),
    single_comune_n=("is_single_comune", "sum"),
    allocated_real_m_eur=("allocated_real_eur", lambda s: s.sum() / 1e6),
    disbursed_real_m_eur=("disbursed_real_eur", lambda s: s.sum() / 1e6),
).round(2))'''),
    (M, """## 1. Cross-Municipality Comparison: All Projects vs. Single-Municipality Projects (`n_comuni == 1`)
We compare Cesena against all 12 Emilia-Romagna peers first across all localized projects (equal-split rule for multi-municipality projects) and then strictly on single-municipality projects (`n_comuni == 1`)."""),
    (C, '''sum_all = summarize_investments_by_comune(projects, demo, single_comune_only=False)
sum_single = summarize_investments_by_comune(projects, demo, single_comune_only=True)
show_cols = [
    "rank_allocated_pc", "comune", "core_peer", "n_projects", "single_comune_share_pct",
    "allocated_real_m_eur", "disbursed_real_m_eur", "allocated_real_pc", "disbursed_real_pc",
    "disbursement_rate_pct", "completion_rate_pct", "median_lag_days", "median_exec_days", "median_delay_days",
    "n_public_works", "public_works_share_pct", "public_works_disb_rate_pct", "comune_actor_share_pct",
]
print("All localized OpenCoesione projects (equal split across n_comuni), ranked by real allocated EUR per capita:")
display(sum_all[[c for c in show_cols if c in sum_all.columns]].round(2))

print("\\nSensitivity check — Single-municipality projects ONLY (n_comuni == 1), ranked by real allocated EUR per capita:")
display(sum_single[[c for c in show_cols if c in sum_single.columns]].round(2))

for label, tbl in [("All projects (equal split)", sum_all), ("Single-comune only (n_comuni == 1)", sum_single)]:
    ces = tbl[tbl["comune_id"] == "040007"].iloc[0]
    peers_core = tbl[(tbl["comune_id"] != "040007") & (tbl["core_peer"])]
    peers_all = tbl[tbl["comune_id"] != "040007"]
    print(f"{label} -> Cesena alloc_pc={ces['allocated_real_pc']:.1f} EUR (Core med={peers_core['allocated_real_pc'].median():.1f}, ER med={peers_all['allocated_real_pc'].median():.1f}, rank={int(ces['rank_allocated_pc'])}/13) | "
          f"disb_rate={ces['disbursement_rate_pct']:.1f}% (Core med={peers_core['disbursement_rate_pct'].median():.1f}%) | "
          f"completion={ces['completion_rate_pct']:.1f}% (Core med={peers_core['completion_rate_pct'].median():.1f}%) | "
          f"median_lag={ces['median_lag_days']:.0f}d (Core med={peers_core['median_lag_days'].median():.0f}d)")'''),
    (M, """## 2. Thematic Breakdown (`OC_TEMA_SINTETICO`): Where Do Cesena's Projects Concentrate?
We compare Cesena's project portfolio against the Core Peer median across OpenCoesione's synthetic policy themes."""),
    (C, '''theme_tbl = summarize_investments_by_theme(projects, demo, core_only=True)
print("Cesena vs Core Peer median by OpenCoesione synthetic policy theme:")
display(theme_tbl.round(2))

top_t = theme_tbl.head(8).iloc[::-1]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
y = np.arange(len(top_t))
axes[0].barh(y - 0.18, top_t["cesena_alloc_real_pc"], height=0.35, color=ITALY_C, label="Cesena")
axes[0].barh(y + 0.18, top_t["peer_alloc_real_pc"], height=0.35, color=EU_C, label="Core peer median")
axes[0].set_yticks(y, [t[:34] for t in top_t["tema"]], fontsize=8.5)
axes[0].set_xlabel("Allocated 2020 EUR per capita")
axes[0].set_title("Allocated Funding per Capita by Theme", fontsize=9.5)
axes[0].legend(fontsize=8)

axes[1].barh(y - 0.18, top_t["cesena_disbursement_rate_pct"], height=0.35, color=ITALY_C, label="Cesena")
axes[1].barh(y + 0.18, top_t["peer_disbursement_rate_pct"], height=0.35, color=EU_C, label="Core peer median")
axes[1].set_yticks(y, ["" for _ in top_t["tema"]])
axes[1].set_xlabel("Disbursement rate (Disbursed / Allocated %)")
axes[1].set_title("Financial Execution Rate by Theme (%)", fontsize=9.5)
plt.tight_layout()
plt.show()'''),
    (M, """## 3. Annual Investment Flows & Largest Individual Projects in Cesena
We inspect the annual profile of project starts (`start_year`) and list the top 15 public investment projects localized in Cesena."""),
    (C, '''flows = annual_investment_flows(projects, demo)
ces_flows = flows[flows["comune_id"] == "040007"]
print("Cesena annual OpenCoesione project starts and real per-capita allocations/disbursements by start_year:")
display(ces_flows.round(2))

ces_top = (
    projects[projects["comune_id"] == "040007"]
    .sort_values("allocated_real_eur", ascending=False)
    [[
        "project_id", "fund_source", "theme_code", "n_comuni", "start_year",
        "allocated_real_eur", "disbursed_real_eur", "status", "lag_days", "title",
    ]]
    .head(15)
    .copy()
)
ces_top["allocated_real_m_eur"] = ces_top["allocated_real_eur"] / 1e6
ces_top["disbursed_real_m_eur"] = ces_top["disbursed_real_eur"] / 1e6
print("Top 15 OpenCoesione projects localized in Cesena by equal-split real allocated funding:")
display(ces_top[["project_id", "fund_source", "theme_code", "n_comuni", "start_year", "allocated_real_m_eur", "disbursed_real_m_eur", "status", "lag_days", "title"]].round(3))'''),
    (M, """## 5. OpenCoesione 2021–2027 Programming Cycle: Early Allocation & Execution
We inspect the ongoing **2021–2027 European and national cohesion programming cycle** (`progetti_esteso_EMR_2021-2027.zip`) across the 13 Emilia-Romagna peer municipalities to evaluate whether Cesena improved its per-capita cohesion funding allocation compared to the `2014–2020` cycle (where it ranked 10th of 13)."""),
    (C, '''from cifresignificative.studies.cesena.analysis.data import load_projects_2127
from cifresignificative.studies.cesena.analysis.investments import summarize_opencoesione_2127_benchmarks

proj_2127 = load_projects_2127()
print(f"Total 2021-2027 OpenCoesione project-municipality rows in ER benchmark cities: {len(proj_2127):,}")
print(f"Cesena 2021-2027 projects: {int((proj_2127['comune_id'] == '040007').sum()):,}")

bench_2127_core = summarize_opencoesione_2127_benchmarks(proj_2127, demo, core_only=True)
bench_2127_all = summarize_opencoesione_2127_benchmarks(proj_2127, demo, core_only=False)

print("\\nOpenCoesione 2021-2027 Benchmark — Cesena vs Core Peers:")
display(bench_2127_core[["label", "formatted_cesena", "formatted_peer", "gap_pct"]].rename(columns={"formatted_cesena": "Cesena", "formatted_peer": "Core Peer Median", "gap_pct": "Gap vs Peer (%)"}).round(1))

ces_21 = proj_2127[proj_2127["comune_id"] == "040007"]
print("\\nCesena 2021-2027 projects by theme:")
display(ces_21.groupby("theme_label").agg(
    n_projects=("project_id", "count"),
    allocated_m_eur=("allocated_real_eur", lambda s: s.sum() / 1e6),
    disbursed_m_eur=("disbursed_real_eur", lambda s: s.sum() / 1e6),
).sort_values("allocated_m_eur", ascending=False).round(2))'''),
    (M, """## Takeaways (Pillar 2 — Public Investments, 13 ER Municipalities; Descriptive)

- **2014–2020 (`robust` as a description).** Allocations per resident are below most peers, and the disbursement rate is the highest of the 13. Public works rest on few, small projects, so the rate is not by itself a measure of administrative capacity.
- **2021–2027 (`sensitive`).** High allocations so far in an open cycle, in nominal euro; three projects make up about half.
- **PNRR procurement (`retracted`).** The file names no contracting authority or location, so tenders cannot be attributed to municipalities.
- **Limits.** All statements are descriptive associations for one municipality, never causal claims. The figures are in the table below and in `reports/findings.md`, both computed by `analysis/claims.py`."""),
    (C, '''from cifresignificative.studies.cesena.analysis.claims import claims_for_notebook

print("Figures behind this notebook, computed from the processed tables (the same ones quoted in reports/findings.md):")
with pd.option_context("display.max_rows", None, "display.max_colwidth", 60):
    display(claims_for_notebook("02"))'''),
]
