"""Dashboard views of the Cesena vs Emilia-Romagna peers study (registered in `studies/cesena/study.toml`)."""

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cifresignificative.core.geo import CESENA, COMUNI_BY_ID
from cifresignificative.dashboard import charts as C
from cifresignificative.dashboard.common import ROOT, chart_with_table, csv_button, tr
from cifresignificative.studies.cesena.analysis import agri_export as AGRI
from cifresignificative.studies.cesena.analysis import commuting as COMM
from cifresignificative.studies.cesena.analysis import data as D
from cifresignificative.studies.cesena.analysis import economy as ECO
from cifresignificative.studies.cesena.analysis import fiscal as FISC
from cifresignificative.studies.cesena.analysis import flood as FLD
from cifresignificative.studies.cesena.analysis import housing as HOU
from cifresignificative.studies.cesena.analysis import investments as INV
from cifresignificative.studies.cesena.analysis import opencivitas as OCIV
from cifresignificative.studies.cesena.analysis import outcomes as OUT


@st.cache_data
def panel_df() -> pd.DataFrame:
    return D.load_panel_comuni()


@st.cache_data
def fiscal_df() -> pd.DataFrame:
    return D.load_fiscal()


@st.cache_data
def missions_df() -> pd.DataFrame:
    return D.load_missions()


@st.cache_data
def projects_df() -> pd.DataFrame:
    return D.load_projects()


@st.cache_data
def asia_df() -> pd.DataFrame:
    return D.load_asia()


@st.cache_data
def asia_size_df() -> pd.DataFrame:
    return D.load_asia_size_classes()


@st.cache_data
def belt_df() -> pd.DataFrame:
    return D.load_belt_comuni()


@st.cache_data
def scorecard_df() -> pd.DataFrame:
    return D.load_scorecard()


@st.cache_data
def ranking_df() -> pd.DataFrame:
    return D.load_placebo_ranking()


@st.cache_data
def ext_ranking_df() -> pd.DataFrame:
    return D.load_extended_placebo_ranking()


@st.cache_data
def synth_series_df() -> pd.DataFrame:
    return D.load_synthetic_cesena()


@st.cache_data
def synth_summary_df() -> pd.DataFrame:
    return D.load_synthetic_summary()


@st.cache_data
def synth_in_space_df() -> pd.DataFrame:
    return D.load_synthetic_in_space()


@st.cache_data
def synth_in_time_df() -> pd.DataFrame:
    return D.load_synthetic_in_time()


@st.cache_data
def lagged_fe_df() -> pd.DataFrame:
    return D.load_lagged_fe()


@st.cache_data
@st.cache_data
def university_summary_df() -> pd.DataFrame:
    return D.load_university_summary()


@st.cache_data
def siope_consolidated_df() -> pd.DataFrame:
    return D.load_siope_consolidated()


@st.cache_data
def projects_2127_df() -> pd.DataFrame:
    return D.load_projects_2127()


@st.cache_data
def frame_sbs_summary_df() -> pd.DataFrame:
    return D.load_frame_sbs_summary()


@st.cache_data
def flood_shock_df() -> pd.DataFrame:
    return D.load_flood_shock()


@st.cache_data
def claims_df() -> dict[str, str]:
    """Printed text of every computed figure, so that no card is typed by hand."""
    return pd.read_parquet(D.PROCESSED_DIR / "claims.parquet").set_index("id")["text"].to_dict()


def flood_tiers_df() -> pd.DataFrame:
    return FLD.compare_flood_tiers(D.load_flood_shock())


@st.cache_data
def omi_municipal_df() -> pd.DataFrame:
    return D.load_omi_municipal()


@st.cache_data
def omi_zones_df() -> pd.DataFrame:
    return D.load_omi_zones()


@st.cache_data
def commuting_summary_df() -> pd.DataFrame:
    return COMM.load_commuting_summary()


@st.cache_data
def commuting_network_df() -> dict[str, pd.DataFrame]:
    return COMM.comprensorio_commuter_network()


@st.cache_data
def coeweb_provinces_df() -> pd.DataFrame:
    return D.load_coeweb_provinces()


@st.cache_data
def agri_land_use_df() -> pd.DataFrame:
    return D.load_agri_land_use()


@st.cache_data
def opencivitas_summary_df() -> pd.DataFrame:
    return D.load_opencivitas_summary()





def _cesena_series_fig(
    s: pd.DataFrame,
    title: str,
    ylabel: str,
    break_year: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    if "core_peer_p25" in s.columns and "core_peer_p75" in s.columns:
        fig.add_trace(
            go.Scatter(
                x=list(s["year"]) + list(s["year"][::-1]),
                y=list(s["core_peer_p75"]) + list(s["core_peer_p25"][::-1]),
                fill="toself",
                fillcolor="rgba(185,184,178,0.25)",
                line={"color": "rgba(0,0,0,0)"},
                name=tr("Core peers IQR (p25-p75)", "IQR pari principali (p25-p75)"),
                hoverinfo="skip",
            )
        )
    if "er_peer_median" in s.columns:
        fig.add_trace(
            go.Scatter(
                x=s["year"],
                y=s["er_peer_median"],
                mode="lines",
                line={"color": C.PEER_C, "width": 1.8, "dash": "dot"},
                name=tr("All 12 ER peers median", "Mediana tutti i 12 pari ER"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=s["year"],
            y=s["core_peer_median"],
            mode="lines",
            line={"color": C.EU_C, "width": 2.4, "dash": "dash"},
            name=tr("6 Core peers median (ex Cesena)", "Mediana 6 pari principali (escl. Cesena)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=s["year"],
            y=s["cesena"],
            mode="lines+markers",
            line={"color": C.ITALY_C, "width": 3.2},
            marker={"size": 6},
            name="Cesena",
        )
    )
    if break_year is not None:
        fig.add_vline(
            x=break_year,
            line_dash="dash",
            line_color=C.INK2,
            annotation_text="D.Lgs. 118/2011",
            annotation_position="top left",
        )
    fig.update_layout(
        **C.LAYOUT,
        title={"text": title, "x": 0, "font": {"color": C.INK, "size": 16}},
        height=430,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text=ylabel)
    return fig


def view_overview(panel: pd.DataFrame) -> None:
    st.header(
        tr(
            "Cesena vs Emilia-Romagna Peers (2008–2025): Overview & Administrative Cycles",
            "Cesena e i Comuni Pari dell'Emilia-Romagna (2008–2025): Panoramica e Cicli Amministrativi",
        )
    )
    st.markdown(
        tr(
            "Descriptive comparison of **Cesena (`040007`)** against the **6 Core Peers** (Forlì, Rimini, Ravenna, Imola, Carpi, Reggio nell'Emilia) and **all 12 Emilia-Romagna peers** across four pillars: municipal balance sheets, OpenCoesione public investments, ISTAT ASIA real-economy Shift-Share & firm size classes, and MEF IRPEF household incomes & ISTAT demography.",
            "Confronto descrittivo di **Cesena (`040007`)** rispetto ai **6 Comuni pari principali** (Forlì, Rimini, Ravenna, Imola, Carpi, Reggio nell'Emilia) e a **tutti i 12 pari dell'Emilia-Romagna** su quattro pilastri: bilanci comunali, investimenti pubblici OpenCoesione, Shift-Share e classi dimensionali ISTAT ASIA, redditi IRPEF MEF e demografia ISTAT.",
        )
    )

    cl = claims_df()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        tr("Real Mean IRPEF Growth (2008–24)", "Crescita Reddito Medio Reale (2008–24)"),
        cl["income_mean_0824"],
        tr(f"{cl['income_mean_0824_rank']} ER; {cl['pool_it83_mean_rank']} in Italy", f"{cl['income_mean_0824_rank']} ER; {cl['pool_it83_mean_rank']} in Italia"),
        delta_color="off",
    )
    c2.metric(
        tr("ASIA Employment Growth (2012–24)", "Crescita Addetti ASIA (2012–24)"),
        f"{cl['emp_all_2024']} ({cl['net_jobs']})",
        tr(f"{cl['emp_all_2024_rank']}; {cl['emp_x3_2024_rank']} without food, agencies, cleaning", f"{cl['emp_all_2024_rank']}; {cl['emp_x3_2024_rank']} senza alimentare, agenzie, pulizie"),
        delta_color="off",
    )
    c3.metric(
        tr("OpenCoesione 2014–20 Disbursement", "Erogazioni OpenCoesione 2014–20"),
        cl["oc1420_disb_rate"],
        tr(f"{cl['oc1420_disb_rate_rank']} ER (Core Peers {cl['oc1420_disb_rate_core']})", f"{cl['oc1420_disb_rate_rank']} ER (pari principali {cl['oc1420_disb_rate_core']})"),
        delta_color="off",
    )
    c4.metric(
        tr("Population Change 2008–24, per 1,000", "Variazione Popolazione 2008–24, per 1.000"),
        cl["balance_total_cesena"],
        tr(f"16 surrounding towns: {cl['balance_total_belt_16']}", f"16 comuni vicini: {cl['balance_total_belt_16']}"),
        delta_color="off",
    )

    rk = ranking_df()
    fig_rk = C.signed_bars_fig(
        rk.set_index("label")["gap_vs_peer_median"],
        tr("Cesena Gap vs ER Peer Median Across Headline Trajectories", "Divario di Cesena rispetto alla Mediana dei Pari ER"),
        tr("Gap vs ER peer median (units of indicator)", "Divario rispetto alla mediana ER"),
    )
    chart_with_table(
        fig_rk,
        rk,
        "cesena_placebo_ranking",
        tr(
            "Placebo ranking of Cesena (`040007`) among the 13 Emilia-Romagna benchmark municipalities (`2/13 = 0.154` two-sided rank floor).",
            "Ordinamento placebo di Cesena (`040007`) tra i 13 comuni dell'Emilia-Romagna (`2/13 = 0.154` soglia minima bilaterale).",
        ),
    )

    st.subheader(tr("Administrative Cycle Scorecard", "Scheda per Ciclo Amministrativo"))
    sc = scorecard_df()
    cycles = list(sc["administration"].unique())
    cyc = st.selectbox(
        tr("Administrative cycle", "Ciclo amministrativo"),
        cycles,
        index=min(2, len(cycles) - 1),
        key="ces_cyc",
    )
    sub_sc = sc[sc["administration"] == cyc][
        ["metric_label", "cesena", "core_peer_median", "er_peer_median", "gap_vs_core", "rank_1_to_13", "n_comuni"]
    ]
    st.dataframe(sub_sc.round(2), use_container_width=True)
    csv_button(sub_sc, "cesena_cycle_scorecard")


def view_fiscal(panel: pd.DataFrame) -> None:
    st.header(
        tr(
            "Pillar 1 — Municipal Fiscal Policy & Balance Sheets (2008–2024)",
            "Pilastro 1 — Politica di Bilancio e Finanza Comunale (2008–2024)",
        )
    )
    st.info(
        tr(
            "**Accounting Break Guard (`break_dlgs118`):** Pre-harmonization budgets (`2008–2015`, `D.P.R. 194/1996`) and harmonized budgets (`2016–2021`, `D.Lgs. 118/2011`) are separated by a vertical dashed line and never spliced across 2015–2016.",
            "**Guardia discontinuità contabile (`break_dlgs118`):** I bilanci pre-armonizzazione (`2008–2015`, `D.P.R. 194/1996`) e armonizzati (`2016–2021`, `D.Lgs. 118/2011`) sono separati dalla linea tratteggiata e mai concatenati tra 2015 e 2016.",
        )
    )
    metrics = {
        "spese_correnti_comp_real_pc": (
            "Current expenditure per capita (2020 EUR)",
            "Spesa corrente pro capite (EUR 2020)",
        ),
        "spese_capitale_comp_real_pc": (
            "Capital expenditure per capita (2020 EUR)",
            "Spesa in conto capitale pro capite (EUR 2020)",
        ),
        "capital_share_pct": (
            "Capital share of primary expenditure (%)",
            "Quota spesa in conto capitale (%)",
        ),
        "imposte_tasse_comp_real_pc": (
            "Own taxes & fees per capita (2020 EUR)",
            "Imposte e tasse proprie pro capite (EUR 2020)",
        ),
        "imu_ici_comp_real_pc": (
            "IMU / ICI property tax per capita (2020 EUR)",
            "Gettito IMU / ICI pro capite (EUR 2020)",
        ),
        "mef_addizionale_real_pc": (
            "MEF municipal IRPEF surcharge per capita (2020 EUR, 2008–2024)",
            "Addizionale comunale IRPEF pro capite (EUR 2020, 2008–2024)",
        ),
        "rigidity_official_pct": (
            "Official balance sheet rigidity (%)",
            "Rigidità strutturale ufficiale di bilancio (%)",
        ),
        "collection_rate_correnti_pct": (
            "Current revenue collection capacity (%)",
            "Capacità di riscossione entrate correnti (%)",
        ),
    }
    col = st.selectbox(
        tr("Fiscal indicator", "Indicatore di bilancio"),
        list(metrics),
        key="ces_fisc_ind",
        format_func=lambda k: tr(*metrics[k]),
    )
    s = D.cesena_vs_peers_series(panel, col, core_only=True)
    brk = None if col == "mef_addizionale_real_pc" else 2015.5
    fig = _cesena_series_fig(s, tr(*metrics[col]), tr(*metrics[col]), break_year=brk)
    chart_with_table(fig, s, f"cesena_fiscal_{col}")

    st.subheader(
        tr(
            "Harmonized Expenditure by Mission (2016–2021 Consuntivo Average)",
            "Spesa Armonizzata per Missione (Media Consuntivo 2016–2021)",
        )
    )
    m_sum = FISC.mission_allocation_summary(missions_df(), D.load_demography())
    fig_m = C.bars_fig(
        m_sum.head(10).set_index("mission_label")[["cesena_totale_real_pc", "peer_totale_real_pc"]].rename(
            columns={
                "cesena_totale_real_pc": "Cesena",
                "peer_totale_real_pc": tr("Core peer median", "Mediana pari principali"),
            }
        ),
        {"Cesena": C.ITALY_C, tr("Core peer median", "Mediana pari principali"): C.EU_C},
        tr("Top 10 Missions by Total Real Expenditure per Capita (2020 EUR)", "Prime 10 Missioni per Spesa Totale Reale Pro Capite (EUR 2020)"),
        xlabel="2020 EUR / resident",
    )
    chart_with_table(fig_m, m_sum, "cesena_missions_2016_2021")

    st.subheader(
        tr(
            "Consolidated Municipal Perimeter: Comune + Unione di Comuni (SIOPE 2024)",
            "Perimetro Consolidato: Comune + Unione di Comuni (SIOPE 2024)",
        )
    )
    siope_c = siope_consolidated_df()
    s24 = siope_c[siope_c["year"] == 2024].sort_values("consolidated_correnti_pc", ascending=False)
    fig_cons = C.bars_fig(
        s24.set_index("entity_name")[["direct_correnti_pc", "consolidated_correnti_pc"]].rename(
            columns={
                "direct_correnti_pc": tr("Direct Municipal Current Spending (EUR/pc)", "Spesa corrente diretta Comune (EUR/ab.)"),
                "consolidated_correnti_pc": tr("Consolidated Current Spending (Comune + Unione, EUR/pc)", "Spesa corrente consolidata (Comune + Unione, EUR/ab.)"),
            }
        ),
        {
            tr("Direct Municipal Current Spending (EUR/pc)", "Spesa corrente diretta Comune (EUR/ab.)"): C.PEER_C,
            tr("Consolidated Current Spending (Comune + Unione, EUR/pc)", "Spesa corrente consolidata (Comune + Unione, EUR/ab.)"): C.ITALY_C,
        },
        tr("Direct vs Consolidated Current Spending per Capita (2024 EUR/pc)", "Spesa Corrente Diretta vs Consolidata Pro Capite (EUR 2024/ab.)"),
        xlabel="EUR / resident",
    )
    chart_with_table(fig_cons, s24, "cesena_siope_consolidated_2024")

    st.subheader(
        tr(
            "Operational Efficiency vs Under-Provision: SOSE OpenCivitas Standard Needs (2023)",
            "Efficienza Operativa vs Sotto-Fornitura: Fabbisogni Standard OpenCivitas SOSE (2023)",
        )
    )
    oc_sum = OCIV.peer_standard_needs_comparison()
    soc_sum = OCIV.social_and_childcare_benchmarks()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        tr("Service Delivery Level", "Livello Quantitativo Servizi"),
        f"{oc_sum.cesena_service_level_0_10} / 10",
        tr("Maximum possible score", "Punteggio massimo"),
    )
    m2.metric(
        tr("Historical Spending", "Spesa Storica"),
        f"{oc_sum.cesena_spesa_storica:.2f} €/ab.",
        tr("Fundamental functions", "Funzioni fondamentali"),
    )
    m3.metric(
        tr("Standard Expenditure Need", "Fabbisogno Standard"),
        f"{oc_sum.cesena_fabbisogno_standard:.2f} €/ab.",
        f"{oc_sum.cesena_gap_pct:+.2f}%",
    )
    m4.metric(
        tr("Childcare Coverage (Nidi)", "Copertura Asili Nido"),
        f"{soc_sum.cesena_childcare_coverage_pct:.2f}%",
        tr("EU target: 33% (2002), 45% for 2030", "Target UE: 33% (2002), 45% per il 2030"),
    )

    peers_chart = oc_sum.peers_df.set_index("comune")[
        ["tot_spesa_storica_eur_pc", "tot_fabbisogno_standard_eur_pc"]
    ].rename(
        columns={
            "tot_spesa_storica_eur_pc": tr("Historical Spending (EUR/pc)", "Spesa Storica (EUR/ab.)"),
            "tot_fabbisogno_standard_eur_pc": tr("Standard Need (EUR/pc)", "Fabbisogno Standard (EUR/ab.)"),
        }
    )
    fig_oc = C.bars_fig(
        peers_chart,
        {
            tr("Historical Spending (EUR/pc)", "Spesa Storica (EUR/ab.)"): C.ITALY_C,
            tr("Standard Need (EUR/pc)", "Fabbisogno Standard (EUR/ab.)"): C.EU_C,
        },
        tr(
            "OpenCivitas Fundamental Functions: Historical Spending vs Standard Need (2023)",
            "OpenCivitas Funzioni Fondamentali: Spesa Storica vs Fabbisogno Standard (2023)",
        ),
        xlabel="EUR / resident",
    )
    display_cols = [
        "comune",
        "tot_spesa_storica_eur_pc",
        "tot_fabbisogno_standard_eur_pc",
        "tot_diff_eur_pc",
        "tot_diff_pct",
        "tot_livello_servizi_0_10",
        "tot_livello_spesa_0_10",
        "quadrante_performance",
        "copertura_nidi_pct",
        "deprivazione_indice",
    ]
    chart_with_table(fig_oc, oc_sum.peers_df[display_cols], "cesena_opencivitas_standard_needs_2023")



def view_investments(panel: pd.DataFrame) -> None:
    st.header(
        tr(
            "Pillar 2 — Public Investments & Financial Execution (OpenCoesione)",
            "Pilastro 2 — Investimenti Pubblici ed Esecuzione Finanziaria (OpenCoesione)",
        )
    )
    projs = projects_df()
    demo = D.load_demography()
    single_only = st.checkbox(
        tr(
            "Restrict to single-municipality projects (`n_comuni == 1` sensitivity)",
            "Limita ai progetti mono-comunali (analisi di sensibilità `n_comuni == 1`)",
        ),
        value=False,
        key="ces_proj_single",
    )
    summary = INV.summarize_investments_by_comune(projs, demo_df=demo, single_comune_only=single_only)
    fig = C.bars_fig(
        summary.set_index("comune")[["allocated_real_pc", "disbursed_real_pc"]].rename(
            columns={
                "allocated_real_pc": tr("Allocated (2020 EUR/pc)", "Finanziato (EUR 2020/ab.)"),
                "disbursed_real_pc": tr("Disbursed (2020 EUR/pc)", "Pagato (EUR 2020/ab.)"),
            }
        ),
        {
            tr("Allocated (2020 EUR/pc)", "Finanziato (EUR 2020/ab.)"): C.PEER_C,
            tr("Disbursed (2020 EUR/pc)", "Pagato (EUR 2020/ab.)"): C.ITALY_C,
        },
        tr(
            "Per-Capita Public Investment Allocations vs Disbursements across 13 ER Municipalities",
            "Finanziamenti e Pagamenti Pro Capite nei 13 Comuni ER",
        ),
        xlabel="2020 EUR / resident",
    )
    chart_with_table(fig, summary, "cesena_opencoesione_summary")

    st.subheader(tr("Thematic Breakdown: Cesena vs Core Peer Median", "Ripartizione Tematica: Cesena vs Mediana Pari Principali"))
    th = INV.summarize_investments_by_theme(projs, demo_df=demo)
    fig_th = C.bars_fig(
        th.set_index("tema")[["cesena_alloc_real_pc", "peer_alloc_real_pc"]].rename(
            columns={
                "cesena_alloc_real_pc": "Cesena",
                "peer_alloc_real_pc": tr("Core peer median", "Mediana pari principali"),
            }
        ),
        {"Cesena": C.ITALY_C, tr("Core peer median", "Mediana pari principali"): C.EU_C},
        tr("Per-Capita Allocations by Theme (2020 EUR/pc)", "Finanziamenti Pro Capite per Tema (EUR 2020/ab.)"),
        xlabel="2020 EUR / resident",
    )
    chart_with_table(fig_th, th, "cesena_opencoesione_themes")

    st.subheader(
        tr(
            "OpenCoesione 2021–2027 Cohesion Cycle: Project Allocations across ER Municipalities",
            "Ciclo di Coesione OpenCoesione 2021–2027: Finanziamenti per Comune ER",
        )
    )
    p21 = projects_2127_df()
    b21 = INV.summarize_investments_by_comune(p21, demo, single_comune_only=False)
    fig_21 = C.bars_fig(
        b21.set_index("comune")[["allocated_real_pc", "disbursed_real_pc"]].rename(
            columns={
                "allocated_real_pc": tr("Allocated funding (2020 EUR/pc)", "Finanziamento assegnato (EUR 2020/ab.)"),
                "disbursed_real_pc": tr("Disbursed funding (2020 EUR/pc)", "Pagamenti erogati (EUR 2020/ab.)"),
            }
        ),
        {
            tr("Allocated funding (2020 EUR/pc)", "Finanziamento assegnato (EUR 2020/ab.)"): C.ITALY_C,
            tr("Disbursed funding (2020 EUR/pc)", "Pagamenti erogati (EUR 2020/ab.)"): C.EU_C,
        },
        tr("OpenCoesione 2021–2027 Cycle Allocations per Capita (EUR/pc)", "Assegnazioni Ciclo 2021–2027 Pro Capite (EUR/ab.)"),
        xlabel="2020 EUR / resident",
    )
    chart_with_table(fig_21, b21, "cesena_opencoesione_2021_2027_summary")




def view_economy(panel: pd.DataFrame) -> None:
    st.header(
        tr(
            "Pillar 3 — Real Economy, Shift-Share & Employment Size Classes (ISTAT ASIA 2012–2024)",
            "Pilastro 3 — Economia Reale, Shift-Share e Classi Dimensionali (ISTAT ASIA 2012–2024)",
        )
    )
    asia = asia_df()
    size_df = asia_size_df()

    windows = {"2012-2024": (2012, 2024), "2012-2019": (2012, 2019), "2019-2024": (2019, 2024)}
    win_key = st.selectbox(
        tr("Shift-Share window", "Finestra temporale Shift-Share"),
        list(windows),
        index=0,
        key="ces_ss_win",
    )
    t0, t1 = windows[win_key]
    ss_all = ECO.shift_share_all_municipalities(asia, t0=t0, t1=t1, level="specialization")
    ss_ces, _ = ECO.shift_share(asia, t0=t0, t1=t1, level="specialization", core_only=True)

    fig_ss = C.bars_fig(
        ss_all.set_index("focal_name")[["ns_total_pp", "im_total_pp", "cs_total_pp"]].rename(
            columns={
                "ns_total_pp": tr("Benchmark Trend (NS)", "Trend Benchmark (NS)"),
                "im_total_pp": tr("Industry Mix (IM)", "Mix Settoriale (IM)"),
                "cs_total_pp": tr("Competitive Shift (CS)", "Effetto Competitivo (CS)"),
            }
        ),
        {
            tr("Benchmark Trend (NS)", "Trend Benchmark (NS)"): C.PEER_C,
            tr("Industry Mix (IM)", "Mix Settoriale (IM)"): C.EU_C,
            tr("Competitive Shift (CS)", "Effetto Competitivo (CS)"): C.ITALY_C,
        },
        tr(
            f"Shift-Share Employment Decomposition across 13 ER Municipalities ({t0}–{t1}, pp)",
            f"Scomposizione Shift-Share dell'Occupazione nei 13 Comuni ER ({t0}–{t1}, pp)",
        ),
        xlabel="Percentage points of initial employment",
        stacked=True,
    )
    chart_with_table(fig_ss, ss_all, f"cesena_shift_share_{win_key}")

    st.subheader(
        tr(
            f"Cesena Cluster Shift-Share & Location Quotients ({t0}–{t1})",
            f"Shift-Share e Quozienti di Localizzazione per Cluster a Cesena ({t0}–{t1})",
        )
    )
    st.dataframe(ss_ces.round(2), use_container_width=True)

    st.subheader(
        tr(
            "Employment Size-Class Decomposition (Micro 0–9, Small 10–49, Medium 50–249, Large 250+)",
            "Scomposizione per Classe Dimensionale (Micro 0–9, Piccole 10–49, Medie 50–249, Grandi 250+)",
        )
    )
    sc_class, sc_muni = ECO.size_class_summary(size_df, t0=2012, t1=2024)
    sc_plot = sc_class.set_index("size_label")[["contrib_to_total_growth_pp", "bench_contrib_to_total_growth_pp"]].rename(
        columns={
            "contrib_to_total_growth_pp": "Cesena",
            "bench_contrib_to_total_growth_pp": tr("Core Peers (ex Cesena)", "Pari Principali (escl. Cesena)"),
        }
    )
    fig_sc = C.bars_fig(
        sc_plot,
        {"Cesena": C.ITALY_C, tr("Core Peers (ex Cesena)", "Pari Principali (escl. Cesena)"): C.EU_C},
        tr(
            "Contribution to Total Employment Growth by Firm Size Class (2012–2024, pp)",
            "Contributo alla Crescita Occupazionale per Classe Dimensionale (2012–2024, pp)",
        ),
        xlabel="Percentage points of total 2012 employment",
    )
    chart_with_table(fig_sc, sc_muni, "cesena_asia_size_classes")

    st.subheader(
        tr(
            "Value Added per Worker & Salaried Wages (ISTAT FRAME-SBS Territoriale 2017–2023)",
            "Valore Aggiunto per Addetto e Retribuzioni (ISTAT FRAME-SBS Territoriale 2017–2023)",
        )
    )
    sbs_sum = frame_sbs_summary_df()
    sector_choices = {
        "totale": tr("Total Market Economy", "Totale Economia di Mercato"),
        "industria": tr("Industry (B–F)", "Industria in senso stretto e Costruzioni (B–F)"),
        "servizi": tr("Market Services (G–S)", "Servizi di Mercato (G–S)"),
    }
    sbs_sec_key = st.selectbox(
        tr("Macro-sector", "Macro-settore"),
        list(sector_choices),
        format_func=lambda k: sector_choices[k],
        key="ces_sbs_sector",
    )
    sbs_sub = sbs_sum[sbs_sum["macro_sector"] == sbs_sec_key].copy()
    plot_sbs = sbs_sub.set_index("year")[["cesena_va_per_addetto_eur", "core_peer_va_per_addetto_eur"]].rename(
        columns={
            "cesena_va_per_addetto_eur": "Cesena",
            "core_peer_va_per_addetto_eur": tr("Core Peers (median)", "Pari Principali (mediana)"),
        }
    )
    fig_sbs = C.line_fig(
        plot_sbs,
        tr(
            f"Value Added per Worker: Cesena vs Core Peers ({sector_choices[sbs_sec_key]}, EUR)",
            f"Valore Aggiunto per Addetto: Cesena vs Pari Principali ({sector_choices[sbs_sec_key]}, EUR)",
        ),
        ylabel="EUR / addetto",
    )
    chart_with_table(fig_sbs, sbs_sub, f"cesena_sbs_va_{sbs_sec_key}")

    st.subheader(
        tr(
            "Primary Agriculture & Provincial Foreign Trade (Coeweb 2025 & 7° Censimento Agricoltura 2020)",
            "Agricoltura Primaria ed Export Provinciale (Coeweb 2025 e 7° Censimento Agricoltura 2020)",
        )
    )
    exp_summary = AGRI.provincial_export_comparison()
    agri_summary = AGRI.agricultural_land_specialization()

    col_ag1, col_ag2, col_ag3 = st.columns(3)
    col_ag1.metric(
        tr("FC Provincial Export (2025)", "Export Forlì-Cesena (2025)"),
        f"{exp_summary.forli_cesena_export_eur / 1e9:.2f} B€",
        tr(f"Rank {exp_summary.forli_cesena_rank_export} of 9 in ER", f"Grado {exp_summary.forli_cesena_rank_export} su 9 in ER"),
    )
    col_ag2.metric(
        tr("FC Trade Balance (2025)", "Saldo Commerciale FC (2025)"),
        f"+{exp_summary.forli_cesena_balance_eur / 1e9:.2f} B€",
        tr(f"Coverage {exp_summary.forli_cesena_coverage_pct:.1f}%", f"Copertura {exp_summary.forli_cesena_coverage_pct:.1f}%"),
    )
    col_ag3.metric(
        tr("FC Fruit Orchards (2020)", "Frutteti Forlì-Cesena (2020)"),
        f"{agri_summary.forli_cesena_frutta_ha:,.0f} ha",
        tr(f"{agri_summary.forli_cesena_frutta_share_er_pct:.1f}% of ER fruit", f"{agri_summary.forli_cesena_frutta_share_er_pct:.1f}% della frutta ER"),
    )

    trade_plot = exp_summary.provinces_df.set_index("provincia")[["export_eur", "trade_balance_eur"]].rename(
        columns={
            "export_eur": tr("Export (EUR)", "Esportazioni (EUR)"),
            "trade_balance_eur": tr("Trade Balance (EUR)", "Saldo Commerciale (EUR)"),
        }
    )
    fig_trade = C.bars_fig(
        trade_plot / 1e9,
        {
            tr("Export (EUR)", "Esportazioni (EUR)"): C.EU_C,
            tr("Trade Balance (EUR)", "Saldo Commerciale (EUR)"): C.ITALY_C,
        },
        tr(
            "Foreign Trade across the 9 Emilia-Romagna Provinces (Coeweb 2025, Billion EUR)",
            "Commercio Estero nelle 9 Province dell'Emilia-Romagna (Coeweb 2025, Miliardi EUR)",
        ),
        xlabel="Billion EUR",
    )
    chart_with_table(fig_trade, exp_summary.provinces_df, "cesena_coeweb_provinces")


    st.subheader(tr("How Much the Results Depend on a Few Sectors", "Quanto i Risultati Dipendono da Pochi Settori"))
    st.caption(
        tr(
            "Cesena against the 13 cities from 2012 to each end year, with all sectors, without food (ATECO 10), and without food, temp agencies and cleaning (10, 78, 81). The food series has a break in 2022 whose cause is unknown (research track R1).",
            "Cesena rispetto alle 13 città dal 2012 a ciascun anno finale, con tutti i settori, senza alimentare (ATECO 10), e senza alimentare, agenzie interinali e pulizie (10, 78, 81). La serie alimentare ha una rottura nel 2022 di causa ignota (linea di ricerca R1).",
        )
    )
    st.dataframe(pd.read_parquet(D.PROCESSED_DIR / "employment_sensitivity.parquet").round(2), hide_index=True)


def view_outcomes(panel: pd.DataFrame) -> None:
    st.header(
        tr(
            "Pillar 4 — Household IRPEF Income, Inequality, Demography & Commuter Belt (2008–2026)",
            "Pilastro 4 — Redditi IRPEF, Disuguaglianza, Demografia e Cintura Pendolare (2008–2026)",
        )
    )
    metrics = {
        "mean_income_real_eur": (
            "Real mean IRPEF income per taxpayer (2020 EUR)",
            "Reddito medio reale IRPEF per contribuente (EUR 2020)",
        ),
        "median_income_real_eur": (
            "Real interpolated median IRPEF income (2020 EUR)",
            "Reddito mediano reale interpolato IRPEF (EUR 2020)",
        ),
        "gini_pareto": (
            "Local IRPEF Gini inequality index (Pareto-adjusted)",
            "Indice di Gini locale sui redditi IRPEF (corretto Pareto)",
        ),
        "aging_index": (
            "Old-age / aging index (65+ per 100 aged 0–14)",
            "Indice di vecchiaia (65+ ogni 100 di 0–14 anni)",
        ),
        "share_25_44_pct": (
            "Young-adult working-age share (25–44, % of population)",
            "Quota popolazione giovane-adulta (25–44 anni, %)",
        ),
        "net_cohort_25_44_per_1k": (
            "25–44 cohort net attractiveness per 1,000 cohort members",
            "Attrattività netta coorte 25–44 per 1.000 residenti 25–44",
        ),
    }
    col = st.selectbox(
        tr("Outcome indicator", "Indicatore di esito"),
        list(metrics),
        key="ces_out_ind",
        format_func=lambda k: tr(*metrics[k]),
    )
    s = D.cesena_vs_peers_series(panel, col, core_only=True)
    fig = _cesena_series_fig(s, tr(*metrics[col]), tr(*metrics[col]))
    chart_with_table(fig, s, f"cesena_outcome_{col}")

    st.subheader(
        tr(
            "Comprensorio Cesenate (Cesena + 16 Surrounding Belt Municipalities): Workplace vs Residence Split",
            "Comprensorio Cesenate (Cesena + 16 Comuni di Cintura): Luogo di Lavoro vs Residenza",
        )
    )
    by_grp, by_muni = OUT.summarize_comprensorio_spillovers(belt_df())
    ring_sub = by_grp[
        by_grp["belt_group"].isin(["cesena_core", "rubicone_mare", "via_emilia_west", "unione_valle_savio"])
    ].set_index("belt_label")
    fig_belt = C.bars_fig(
        ring_sub[["emp_delta_12_24", "pop_delta_08_25"]].rename(
            columns={
                "emp_delta_12_24": tr("Net jobs created (2012–2024)", "Nuovi addetti netti (2012–2024)"),
                "pop_delta_08_25": tr("Resident pop. change (2008–2025)", "Variazione residenti (2008–2025)"),
            }
        ),
        {
            tr("Net jobs created (2012–2024)", "Nuovi addetti netti (2012–2024)"): C.ITALY_C,
            tr("Resident pop. change (2008–2025)", "Variazione residenti (2008–2025)"): C.EU_C,
        },
        tr(
            "Net Jobs (2012–2024) and Resident Change (2008–2025) by Ring; the Periods Differ",
            "Addetti Netti (2012–2024) e Variazione dei Residenti (2008–2025) per Anello; i Periodi Sono Diversi",
        ),
        xlabel="Persons / jobs",
    )
    chart_with_table(fig_belt, by_grp, "cesena_comprensorio_rings")
    st.dataframe(
        by_muni[
            [
                "comune",
                "belt_group",
                "pop_2008",
                "pop_2025",
                "pop_growth_08_25_pct",
                "net_migration_08_24",
                "emp_2012",
                "emp_2024",
                "emp_growth_12_24_pct",
                "emp_per_1k_pop_2024",
                "mean_income_real_2024",
                "real_income_growth_08_24_pct",
            ]
        ].round(2),
        use_container_width=True,
    )

    st.subheader(
        tr(
            "Higher Education & STEM/ICT Specialisation: UniBo Cesena Campus (MUR USTAT)",
            "Istruzione Universitaria e Specializzazione STEM/ICT: Campus di Cesena (MUR USTAT)",
        )
    )
    uni_s = university_summary_df()
    y24 = uni_s[uni_s["calendar_year"] == 2024].sort_values("stem_share", ascending=False)
    fig_uni = C.bars_fig(
        y24.set_index("comune_name")[["stem_share", "ict_share"]].rename(
            columns={
                "stem_share": tr("STEM student share (%)", "Quota studenti STEM (%)"),
                "ict_share": tr("Computer Science / ICT share (%)", "Quota Informatica / ICT (%)"),
            }
        ) * 100.0,
        {
            tr("STEM student share (%)", "Quota studenti STEM (%)"): C.ITALY_C,
            tr("Computer Science / ICT share (%)", "Quota Informatica / ICT (%)"): C.PEER_C,
        },
        tr("STEM and ICT Specialisation across Emilia-Romagna University Poles (2024)", "Specializzazione STEM e ICT nei Poli Universitari ER (2024)"),
        xlabel="Percentage (%)",
    )
    chart_with_table(fig_uni, uni_s, "cesena_mur_university_summary")

    st.subheader(
        tr(
            "Housing Affordability, Real Estate Prices & Suburbanization Gradient (Agenzia delle Entrate OMI)",
            "Accessibilità Abitativa, Prezzi Immobiliari e Gradiente di Periurbanizzazione (Agenzia delle Entrate OMI)",
        )
    )
    omi_muni = omi_municipal_df()
    aff = HOU.compute_housing_affordability(omi_muni, irpef_df=panel, demo_df=panel)
    grad = HOU.comprensorio_housing_gradient(aff)

    fig_grad = C.bars_fig(
        grad.set_index("belt_label")[["compr_mid_median_eur_m2"]].rename(
            columns={"compr_mid_median_eur_m2": tr("Median Purchase Price (€/m²)", "Prezzo Mediano di Acquisto (€/m²)")}
        ),
        {tr("Median Purchase Price (€/m²)", "Prezzo Mediano di Acquisto (€/m²)"): C.ITALY_C},
        tr(
            "Residential Housing Price Gradient across Comprensorio Cesenate Functional Rings (€/m²)",
            "Gradiente dei Prezzi Residenziali nei Cerchi Funzionali del Comprensorio Cesenate (€/m²)",
        ),
        xlabel="EUR / m²",
    )
    chart_with_table(fig_grad, grad, "cesena_omi_housing_gradient")

    st.markdown(
        tr(
            "**Hypothesis, not a result:** the towns around Cesena are mostly cheaper, but among them cheaper towns grew less, and no data here show who moves or from where. See findings 8 and 15 and research track R2.",
            "**Ipotesi, non un risultato:** i comuni intorno a Cesena sono per lo più meno cari, ma tra questi i meno cari sono cresciuti meno, e questi dati non mostrano chi si sposta né da dove. Vedi i risultati 8 e 15 e la linea di ricerca R2.",
        )
    )

    st.subheader(
        tr(
            "Origin-Destination Commuting Matrix & Job Self-Containment (ISTAT 15th Census)",
            "Matrice Origine-Destinazione del Pendolarismo e Autosufficienza Lavorativa (ISTAT)",
        )
    )
    net_c = commuting_network_df()
    rings_comm = net_c["ring_aggregates"].set_index("group_label")

    fig_comm = C.bars_fig(
        rings_comm[["inbound_to_cesena", "outbound_from_cesena"]].rename(
            columns={
                "inbound_to_cesena": tr("Inbound commuters to Cesena", "Pendolari in ingresso a Cesena"),
                "outbound_from_cesena": tr("Outbound commuters from Cesena", "Pendolari in uscita da Cesena"),
            }
        ),
        {
            tr("Inbound commuters to Cesena", "Pendolari in ingresso a Cesena"): C.ITALY_C,
            tr("Outbound commuters from Cesena", "Pendolari in uscita da Cesena"): C.EU_C,
        },
        tr(
            "Commuting Exchanges between Cesena and the 3 Functional Belt Rings (Workers)",
            "Scambi Pendolari tra Cesena e i 3 Cerchi Funzionali di Cintura (Occupati)",
        ),
        xlabel="Daily commuters",
    )
    chart_with_table(fig_comm, net_c["municipal_flows"], "cesena_commuting_corridors")

    st.markdown(
        tr(
            "**Background only:** these flows come from the 2011 census, so they predate the 2012–2024 period and cannot support a trend. Student flows cover all school levels, not just the university.",
            "**Solo contesto:** questi flussi vengono dal censimento 2011, quindi precedono il periodo 2012–2024 e non possono sostenere un andamento. I flussi di studenti comprendono tutti i livelli scolastici, non solo l'università.",
        )
    )



    st.subheader(tr("Population Change by Component", "Variazione della Popolazione per Componente"))
    st.caption(
        tr(
            "Natural balance, moves within Italy, moves from and to abroad, and registry adjustment. The four add up exactly to the population change.",
            "Saldo naturale, spostamenti interni, spostamenti da e per l'estero e rettifiche anagrafiche. Le quattro voci sommano esattamente alla variazione della popolazione.",
        )
    )
    st.dataframe(pd.read_parquet(D.PROCESSED_DIR / "population_balance.parquet").round(1), hide_index=True)


def view_evaluation(panel: pd.DataFrame) -> None:
    st.header(
        tr(
            "Phase 3 — Synthetic Cesena, Extended Municipal Placebo ($N = 36..83$) & Two-Way FE Screen",
            "Fase 3 — Cesena Sintetica, Placebo Comunale Esteso ($N = 36..83$) e Modelli Two-Way FE",
        )
    )
    syn_s = synth_series_df()
    syn_sum = synth_summary_df()
    metrics = {
        "mean_income_real_eur": (
            "Real mean IRPEF income cumulative growth (%, base 2008)",
            "Crescita cumulata reddito medio reale IRPEF (%, base 2008)",
        ),
        "avg_firm_size": (
            "Average firm size cumulative growth (%, base 2012)",
            "Crescita cumulata dimensione media d'impresa (%, base 2012)",
        ),
        "median_income_real_eur": (
            "Real median IRPEF income cumulative growth (%, base 2008)",
            "Crescita cumulata reddito mediano reale IRPEF (%, base 2008)",
        ),
        "employees": (
            "Private employment cumulative growth (%, base 2012)",
            "Crescita cumulata addetti privati (%, base 2012)",
        ),
    }
    m_col = st.selectbox(
        tr("Synthetic Control outcome", "Indicatore Controllo Sintetico"),
        list(metrics),
        key="ces_syn_ind",
        format_func=lambda k: tr(*metrics[k]),
    )
    sub = syn_s[syn_s["metric"] == m_col]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["year"],
            y=sub["peer_median_cum_pct"],
            mode="lines",
            line={"color": C.PEER_C, "width": 1.8, "dash": "dot"},
            name=tr("ER peer median (ex Cesena)", "Mediana pari ER (escl. Cesena)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sub["year"],
            y=sub["synthetic_cesena_cum_pct"],
            mode="lines",
            line={"color": C.EU_C, "width": 2.6, "dash": "dash"},
            name=tr("Synthetic Cesena (pre-2019 match)", "Cesena Sintetica (match pre-2019)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sub["year"],
            y=sub["cesena_cum_pct"],
            mode="lines+markers",
            line={"color": C.ITALY_C, "width": 3.2},
            marker={"size": 6},
            name="Cesena",
        )
    )
    fig.add_vline(x=2018.5, line_dash="dash", line_color=C.INK2, annotation_text="2018")
    fig.update_layout(
        **C.LAYOUT,
        title={"text": tr(*metrics[m_col]), "x": 0, "font": {"color": C.INK, "size": 16}},
        height=430,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="%")
    chart_with_table(fig, syn_sum, "cesena_synthetic_control")

    st.markdown(
        tr(
            "**In-Space & In-Time Synthetic Placebos:** Exact donor-pool permutations across all 13 ER municipalities and counterfactual intervention at 2014.",
            "**Placebo Sintetici nello Spazio e nel Tempo:** Permutazioni esatte dei donatori sui 13 comuni ER e intervento controfattuale al 2014.",
        )
    )
    col_sp, col_ti = st.columns(2)
    with col_sp:
        st.caption(tr("In-Space Placebo (Cesena vs 12 ER Peers)", "Placebo nello Spazio (Cesena vs 12 Pari ER)"))
        sp_df = synth_in_space_df()
        st.dataframe(
            sp_df[sp_df["is_cesena"]][["label", "gap_end_pp", "rank_gap_desc", "perm_p_gap", "ratio_post_pre_rmspe", "rank_ratio_desc"]].round(3),
            use_container_width=True,
        )
    with col_ti:
        st.caption(tr("In-Time Placebo (Pseudo-Treatment Year 2014)", "Placebo nel Tempo (Anno Pseudo-Trattamento 2014)"))
        ti_df = synth_in_time_df()
        st.dataframe(
            ti_df[["label", "pre_rmse_pp", "pseudo_post_mean_gap_pp", "true_post_mean_gap_pp", "end_gap_pp"]].round(2),
            use_container_width=True,
        )

    st.subheader(
        tr(
            "Extended Municipal Placebo Ranking ($N = 36, 45, 73, 83$ Municipal Pools)",
            "Ordinamento Placebo Comunale Esteso ($N = 36, 45, 73, 83$ Comuni)",
        )
    )
    ext = ext_ranking_df()
    st.dataframe(ext.round(3), use_container_width=True)
    csv_button(ext, "cesena_extended_placebo_ranking")

    st.subheader(
        tr(
            "Two-Way Fixed Effects Lagged Screen (140 Specifications, BH-FDR Controlled)",
            "Screening Lagged Two-Way Fixed Effects (140 Specifiche, Controllo BH-FDR)",
        )
    )
    lfe = lagged_fe_df()
    st.dataframe(lfe.head(20).round(3), use_container_width=True)
    csv_button(lfe, "cesena_lagged_fe_results")

    st.subheader(
        tr(
            "May 2023 Flood: Descriptive Comparison by D.L. 61/2023 Listing (2022–2025)",
            "Alluvione di Maggio 2023: Confronto Descrittivo per Elenco del D.L. 61/2023 (2022–2025)",
        )
    )
    fl_tiers = flood_tiers_df()
    fl_df = flood_shock_df()

    st.markdown(
        tr(
            "Empirical multi-tier shock analysis comparing **Severely Impacted Romagna Poles** (Cesena, Forlì, Faenza, Ravenna), **Partially Impacted Poles** (Imola, Bologna, Rimini), and **Unaffected Western Controls** (Reggio Emilia, Parma, Modena, Piacenza, Carpi, Ferrara).",
            "Analisi empirica su tre livelli che confronta i **Poli Romagnoli Gravemente Colpiti** (Cesena, Forlì, Faenza, Ravenna), i **Poli Parzialmente Colpiti** (Imola, Bologna, Rimini) e i **Controlli dell'Emilia Occidentale non Colpiti** (Reggio Emilia, Parma, Modena, Piacenza, Carpi, Ferrara).",
        )
    )
    st.dataframe(
        fl_tiers[[
            "tier_label", "n_municipalities",
            "income_growth_22_23_pct_median", "taxpayer_growth_22_23_pct_median",
            "capitale_cassa_pc_2022_median", "capitale_cassa_pc_2024_median", "capitale_cassa_growth_22_24_pct_median",
            "emp_growth_22_24_pct_median", "units_growth_23_24_pct_median", "real_va_growth_22_23_pct_median",
            "net_migr_rate_2023_median", "pop_growth_22_25_pct_median",
        ]].round(2),
        use_container_width=True,
    )

    sub_sev = fl_df[fl_df["flood_tier"] == "whole_territory"].sort_values("capitale_cassa_pc_2024", ascending=False)
    fig_fld = C.bars_fig(
        sub_sev.set_index("comune_name")[["capitale_cassa_pc_2022", "capitale_cassa_pc_2024"]].rename(
            columns={
                "capitale_cassa_pc_2022": tr("2022 (Pre-flood baseline)", "2022 (Livello pre-alluvione)"),
                "capitale_cassa_pc_2024": tr("2024", "2024"),
            }
        ),
        {
            tr("2022 (Pre-flood baseline)", "2022 (Livello pre-alluvione)"): C.PEER_C,
            tr("2024", "2024"): C.ITALY_C,
        },
        tr(
            "Capital Cash Payments, All Purposes (MEF RGS BDAP SIOPE 2022 vs 2024, EUR/pc)",
            "Pagamenti in Conto Capitale, Tutte le Finalità (MEF RGS SIOPE 2022 vs 2024, EUR/ab.)",
        ),
        xlabel="EUR / resident",
    )
    chart_with_table(fig_fld, fl_df, "cesena_flood_shock_resilience")



def view_about(panel: pd.DataFrame) -> None:
    st.header(tr("About the Cesena Study, Findings & Roadmap", "Informazioni sullo Studio Cesena, Risultati e Sviluppi"))
    tabs = st.tabs(
        [
            tr("Findings", "Risultati"),
            tr("Study Overview", "Descrizione dello Studio"),
        ]
    )
    for tab, path in zip(
        tabs,
        [
            "studies/cesena/reports/findings.md",
            "studies/cesena/README.md",
        ],
        strict=True,
    ):
        with tab:
            p = ROOT / path
            st.markdown(re.sub(r"<!-- c:\w+ -->", "", p.read_text()) if p.exists() else f"`{path}` not found.")


VIEWS = {
    "overview": ("Overview & Scorecard", "Panoramica e Cicli", view_overview),
    "fiscal": ("Fiscal Policy & Missions", "Bilanci e Missioni", view_fiscal),
    "investments": ("Public Investments", "Investimenti Pubblici", view_investments),
    "economy": ("Shift-Share & Firm Size", "Shift-Share e Dimensione Imprese", view_economy),
    "outcomes": ("Income, Demography & Belt", "Redditi, Demografia e Cintura", view_outcomes),
    "evaluation": ("Synthetic Cesena & Econometrics", "Cesena Sintetica ed Econometria", view_evaluation),
    "about": ("About & Findings", "Risultati e Documentazione", view_about),
}

TITLE = ("Cesena vs Emilia-Romagna Peers", "Cesena e i Comuni Pari dell'Emilia-Romagna")
CAPTION = (
    f"Cesena ({COMUNI_BY_ID[CESENA].id}) in blue, 6 Core Peers median in orange (dashed), All 12 ER Peers median in gray. Sources: Ministry of Interior Finanza Locale, OpenBilanci, OpenCoesione, ISTAT ASIA, MEF IRPEF, ISTAT Demo (2008–2026). Descriptive only.",
    f"Cesena ({COMUNI_BY_ID[CESENA].id}) in blu, mediana dei 6 pari principali in arancione (tratteggio), mediana dei 12 pari ER in grigio. Fonti: Finanza Locale, OpenBilanci, OpenCoesione, ISTAT ASIA, MEF IRPEF, ISTAT Demo (2008–2026). Analisi descrittiva.",
)


def data_error() -> str | None:
    """Message to show instead of the views when the processed data are missing."""
    if not (D.PROCESSED_DIR / "panel_comuni.parquet").exists():
        return "Processed data for the Cesena study not found. Run `uv run python -m cifresignificative.studies.cesena.analysis.data` first."
    return None


def load() -> pd.DataFrame:
    return panel_df()
