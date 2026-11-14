"""Dashboard views of the Italy-against-the-EU27 productivity study (registered in studies/italy_productivity/study.toml)."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cifresignificative.core import catalog, glossary
from cifresignificative.dashboard import charts as C
from cifresignificative.dashboard.common import (
    QP,
    ROOT,
    chart_with_table,
    country_label,
    csv_button,
    explain,
    label,
    qp_list,
    tr,
)
from cifresignificative.studies.italy_productivity.analysis import corr
from cifresignificative.studies.italy_productivity.analysis import data as D
from cifresignificative.studies.italy_productivity.analysis import efficiency as EF
from cifresignificative.studies.italy_productivity.analysis import experiments as EXP
from cifresignificative.studies.italy_productivity.analysis import hf as HF
from cifresignificative.studies.italy_productivity.analysis import regional_corr as rc
from cifresignificative.studies.italy_productivity.analysis import regional_south as rs
from cifresignificative.studies.italy_productivity.analysis.scenarios import (
    scenario_inputs,
    simulate_debt,
    stabilising_primary_balance,
)

LAST_FULL_YEAR = 2024
NOTES = {i.name: i.note for lst in (catalog.EUROSTAT, catalog.EUROSTAT_REGIONAL) for i in lst}
SECTOR_LABELS = D.SECTOR_LABELS




@st.cache_data
def panel_df() -> pd.DataFrame:
    return D.load_panel()


@st.cache_data
def correlation_results(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Experiment tables from data/processed/correlations/ (national e*, regional r*), computed on the fly (15-20 s) if missing."""
    out = {}
    for prefix, probe, run in (("e", "e1_levels", lambda: EXP.run_all(panel)), ("r", "r1_levels_within", EXP.run_regional_all),
                               ("h", "hf1_spread", EXP.run_hf_all if HF.HF_PATH.exists() else None)):
        if run is not None and not (EXP.OUT / f"{probe}.parquet").exists():
            with st.spinner(tr("Running the correlation experiments...", "Calcolo delle correlazioni...")):
                run()
        out.update({p.stem: pd.read_parquet(p) for p in EXP.OUT.glob(f"{prefix}*.parquet")})
    return out


@st.cache_data
def highfreq_df() -> pd.DataFrame:
    return HF.load_highfreq()


@st.cache_data
def spending_df() -> pd.DataFrame:
    return D.load_spending()


@st.cache_data
def south_results() -> dict[str, pd.DataFrame]:
    """Tables of the "why the South" round (data/processed/correlations/s*.parquet), computed on the fly (about 2 minutes) if missing."""
    if not (EXP.OUT / "s1_eu_wide.parquet").exists():
        with st.spinner(tr("Running the regional 'why the South' experiments (about 2 minutes)...", "Calcolo degli esperimenti regionali 'perché il Sud' (circa 2 minuti)...")):
            EXP.run_south_all()
    return {p.stem: pd.read_parquet(p) for p in EXP.OUT.glob("s*.parquet")}


@st.cache_data
def regional_df() -> pd.DataFrame:
    return D.load_regional()


@st.cache_data
def sectors_df() -> pd.DataFrame:
    return D.load_sectors()


@st.cache_data
def highered_df() -> pd.DataFrame:
    return D.load_highered()


@st.cache_data
def ageing_df() -> pd.DataFrame:
    return D.load_ageing()


@st.cache_data
def long_df() -> pd.DataFrame:
    return D.load_long()


# ---------------------------------------------------------------- views
def view_overview(panel: pd.DataFrame) -> None:
    st.title(tr("Italy vs the EU27", "L'Italia e l'UE27"))
    st.caption(tr("Where Italy differs from the other EU27 members, 2005-2024. Median = median of the other 26 members; gaps in cross-country standard deviations.",
                  "In cosa l'Italia si distingue dagli altri paesi UE27, 2005-2024. Mediana = mediana degli altri 26 paesi; scarti in deviazioni standard tra paesi."))
    windows = {"2005-2009": (2005, 2009), "2015-2019": (2015, 2019), "2020-2024": (2020, 2024)}
    w = st.radio(tr("Period", "Periodo"), list(windows), index=2, horizontal=True)
    ok = D.gap_table(panel, windows[w]).dropna(subset=["z_gap"])
    cols = st.columns(4)
    for col, (ind, lab, fmt) in zip(cols, [("employment_rate_20_64", tr("Employment rate 20-64", "Tasso di occupazione 20-64"), "{:.1f}%"), ("gov_debt_pct_gdp", tr("Government debt", "Debito pubblico"), "{:.0f}% GDP"),
                                          ("tertiary_attainment_25_34", tr("Tertiary attainment 25-34", "Laureati 25-34"), "{:.1f}%"), ("old_age_dependency", tr("Old-age dependency", "Indice di dipendenza anziani"), "{:.1f}")], strict=True):
        s = D.wide(panel, ind)
        med = D.eu_median_ex_italy(s).loc[LAST_FULL_YEAR]
        col.metric(f"{lab} ({LAST_FULL_YEAR})", fmt.format(s.loc[LAST_FULL_YEAR, D.ITALY]), tr("median of others ", "mediana degli altri ") + fmt.format(med), delta_color="off")
    st.plotly_chart(C.gap_heatmap_fig(ok, f"{tr('Italy minus median of the other EU27', 'Italia meno la mediana degli altri UE27')}, {w}"), use_container_width=True)
    st.caption(tr("Blue = Italy higher than the median, red = lower (whether that is good or bad depends on the indicator). Hover for the underlying values.",
                  "Blu = Italia sopra la mediana, rosso = sotto (se sia un bene o un male dipende dall'indicatore). Passa il mouse per i valori."))
    with st.expander(tr("Table", "Tabella")):
        st.dataframe(ok[["group", "label", "italy", "median_others", "z_gap"]].round(2), use_container_width=True, hide_index=True)


def view_explorer(panel: pd.DataFrame) -> None:
    st.header(tr("Indicator explorer", "Esplora gli indicatori"))
    themes = ["all", *sorted(panel.theme.dropna().unique())]
    c1, c2 = st.columns([1, 3])
    theme = c1.selectbox(tr("Theme", "Tema"), themes)
    has_italy = set(panel[panel.iso3 == D.ITALY].indicator.unique())  # a few series (e.g. central-government debt) have no Italian value
    inds = sorted((set(panel[panel.theme == theme].indicator.unique()) if theme != "all" else set(panel.indicator.unique())) & has_italy, key=lambda k: label(k).lower())
    wanted = QP.get("ind")
    default = inds.index(wanted) if wanted in inds else (inds.index("gdp_real_pc") if "gdp_real_pc" in inds else 0)
    ind = c2.selectbox(tr("Indicator", "Indicatore"), inds, index=default, format_func=label, help=tr("Type to search by name. The definition appears below; the Glossary view lists every indicator.", "Scrivi per cercare per nome. La definizione compare sotto; la vista Glossario elenca tutti gli indicatori."))
    explain(ind)
    wide = D.wide(panel, ind)
    avail = [c for c in [*D.EU27_ISO3, *D.BENCHMARKS] if c in wide and c != D.ITALY and wide[c].notna().any()]
    c3, c4 = st.columns([3, 1])
    default_h = [c for c in qp_list("hl", ["DEU", "FRA", "ESP"]) if c in avail]
    hl = c3.multiselect(tr(f"Highlight countries (up to {C.MAX_HIGHLIGHTS})", f"Evidenzia paesi (fino a {C.MAX_HIGHLIGHTS})"), avail, default=default_h, format_func=country_label, max_selections=C.MAX_HIGHLIGHTS)
    peers = c4.checkbox(tr("Show other EU27 (gray)", "Mostra gli altri UE27 (grigio)"), value=True)
    rebase = c4.checkbox(tr("Rebase first year = 100", "Base 100 al primo anno"), value=False)
    weighted = c4.checkbox(tr("Population-weighted median", "Mediana ponderata per popolazione"), value=False)
    ys = wide[D.ITALY].dropna().index
    lo, hi = int(ys.min()), int(ys.max())
    years = st.slider(tr("Years", "Anni"), min_value=int(wide.index.min()), max_value=int(wide.index.max()), value=(lo, hi)) if lo < hi else (lo, hi)
    QP["view"], QP["ind"], QP["hl"] = "explorer", ind, ",".join(hl)
    pop = D.wide(panel, "population")
    median = D.weighted_median_ex_italy(wide, pop) if weighted else None
    fig = C.line_fig(wide, label(ind), highlights=hl, show_peers=peers, years=years, rebase=rebase, median=median,
                     median_label=tr("Population-weighted median", "Mediana ponderata") if weighted else tr("Median of other EU27", "Mediana altri UE27"))
    med = median if median is not None else D.eu_median_ex_italy(wide)
    table = pd.concat([wide[D.ITALY].rename("Italy"), med.rename("Median of other EU27"), wide[hl]], axis=1).loc[years[0]:years[1]]
    last = wide[D.ITALY].dropna().index.max()
    src = panel[panel.indicator == ind].source.iloc[0]
    r, n = D.italy_rank(wide, int(last))
    medv = med.loc[last] if last in med.index and not np.isnan(med.loc[last]) else float("nan")
    about = (f"**{label(ind)}**\n\n- {tr('Theme', 'Tema')}: {panel[panel.indicator == ind].theme.iloc[0]}\n- {tr('Source', 'Fonte')}: `{src}`\n- {NOTES.get(ind, '')}\n"
             f"- Italy {int(last)}: {wide.loc[last, D.ITALY]:.2f} ({tr('median of others', 'mediana degli altri')} {medv:.2f}); {tr('rank', 'posizione')} {r}/{n} ({tr('1 = highest', '1 = più alto')})\n"
             f"- {tr('Coverage', 'Copertura')}: {panel[panel.indicator == ind].geo.nunique()} {tr('countries/aggregates', 'paesi/aggregati')}, {int(wide.index.min())}-{int(wide.index.max())}.\n\n"
             f"{tr('See docs/data_catalog.md and docs/data_dictionary.md for caveats.', 'Vedi docs/data_catalog.md e docs/data_dictionary.md per le avvertenze.')}")
    chart_with_table(fig, table, ind, about)


def view_scatter(panel: pd.DataFrame) -> None:
    st.header(tr("Compare two indicators across countries", "Confronta due indicatori tra paesi"))
    inds = sorted(panel[panel.iso3.isin(D.EU27_ISO3)].indicator.unique(), key=lambda k: label(k).lower())
    c1, c2 = st.columns(2)
    xi = c1.selectbox(tr("X axis", "Asse X"), inds, index=inds.index("old_age_dependency"), format_func=label)
    yi = c2.selectbox(tr("Y axis", "Asse Y"), inds, index=inds.index("pension_spend_pct_gdp"), format_func=label)
    with c1:
        explain(xi)
    with c2:
        explain(yi)
    years = st.slider(tr("Average over years", "Media sugli anni"), 2005, 2025, (2022, 2024))
    x = D.wide(panel, xi, D.EU27_ISO3).loc[years[0]:years[1]].mean()
    y = D.wide(panel, yi, D.EU27_ISO3).loc[years[0]:years[1]].mean()
    fig = C.scatter_fig(x, y, label(xi), label(yi), f"{label(yi)} vs {label(xi)}, {years[0]}-{years[1]}")
    d = pd.concat([x.rename(label(xi)), y.rename(label(yi))], axis=1).dropna()
    about = tr(f"Correlation across {len(d)} EU27 members: **{d.iloc[:, 0].corr(d.iloc[:, 1]):.2f}** (associations only, no causal reading). The dashed line is fitted on the other members, so Italy's distance from it is not self-influenced.",
               f"Correlazione tra {len(d)} paesi UE27: **{d.iloc[:, 0].corr(d.iloc[:, 1]):.2f}** (solo associazioni, nessuna lettura causale). La linea tratteggiata è stimata sugli altri paesi.")
    chart_with_table(fig, d, f"{xi}_vs_{yi}", about)


def view_correlations(panel: pd.DataFrame) -> None:
    st.header(tr("Correlation discovery", "Ricerca di correlazioni"))
    st.warning(tr("Associations only, not causes. About 30 countries and thousands of pairs tested: results are screened with false-discovery control (BH, q = 5%) and robustness checks, and still need judgement.",
                  "Solo associazioni, non cause. Circa 30 paesi e migliaia di coppie testate: i risultati passano un controllo sulle false scoperte (BH, q = 5%) e verifiche di robustezza, ma richiedono comunque giudizio."))
    res = correlation_results(panel)
    tab_s, tab_i, tab_h, tab_r, tab_f = st.tabs([tr("Screen", "Screening"), tr("Italy's gap", "Gap dell'Italia"), tr("Hypotheses", "Ipotesi"), tr("Regions (NUTS2)", "Regioni (NUTS2)"),
                                                 tr("Monthly / quarterly", "Mensili / trimestrali")])
    with tab_s:
        names = {"e1_levels": tr("Levels 2015-19, income removed", "Livelli 2015-19, al netto del reddito"), "e1_levels_raw": tr("Levels 2015-19, raw", "Livelli 2015-19, grezzi"),
                 "e2_changes": tr("Changes 2005-09 to 2015-19", "Variazioni 2005-09 / 2015-19"), "e3_within": tr("Within-country year-on-year", "Variazioni annuali entro paese")}
        exp = st.selectbox(tr("Experiment", "Esperimento"), list(names), format_func=lambda k: names[k])
        c1, c2 = st.columns(2)
        min_rho = c1.slider(tr("Minimum |rho|", "|rho| minimo"), 0.0, 0.95, 0.5, 0.05)
        robust_only = c2.checkbox(tr("Robust only", "Solo robuste"), value=True, disabled="robust" not in res[exp])
        d = res[exp]
        d = d[d.rho.abs() >= min_rho]
        if robust_only and "robust" in d:
            d = d[d.robust]
        theme = st.multiselect(tr("Indicators mentioning (theme)", "Indicatori del tema"), sorted(panel.theme.unique()))
        if theme:
            keep = set(panel[panel.theme.isin(theme)].indicator)
            d = d[d.x.isin(keep) | d.y.isin(keep)]
        st.caption(tr(f"{len(d)} pairs shown.", f"{len(d)} coppie mostrate."))
        show = d.assign(**{tr("X", "X"): d.x.map(label), tr("Y", "Y"): d.y.map(label)})
        st.dataframe(show[[tr("X", "X"), tr("Y", "Y"), "n", "rho", "q_value"] + (["robust"] if "robust" in d else [])].round(3), use_container_width=True, hide_index=True)
        if exp.startswith(("e1", "e2")) and len(d):
            opts = [f"{k + 1}. {label(r.x)} - {label(r.y)}" for k, r in enumerate(d.itertuples())]
            i = opts.index(st.selectbox(tr("Plot a pair", "Disegna una coppia"), opts))
            xi, yi = d.iloc[i].x, d.iloc[i].y
            M_ = corr.panel_matrix(panel)
            X = corr.cross_section(M_, "level", (2015, 2019)) if exp.startswith("e1") else corr.cross_section(M_, "change", (2015, 2019), (2005, 2009))
            st.plotly_chart(C.scatter_fig(X[xi], X[yi], label(xi), label(yi), f"{label(yi)} vs {label(xi)}"), use_container_width=True)
    with tab_i:
        tgt = st.radio(tr("Target gap", "Gap di riferimento"), ["e5_italy_gap_gdp", "e5_italy_gap_employment"], horizontal=True,
                       format_func=lambda k: label("gdp_real_pc" if k.endswith("gdp") else "employment_rate_20_64"))
        d = res[tgt]
        st.caption(tr("Spearman correlation of year-on-year changes in Italy's gap vs the median of the other EU27.", "Correlazione di Spearman tra variazioni annuali del gap dell'Italia rispetto alla mediana degli altri UE27."))
        st.dataframe(d.assign(indicator=d.indicator.map(label))[["indicator", "n", "rho", "q_value", "significant"]].round(3), use_container_width=True, hide_index=True)
    with tab_h:
        h = res["e7_hypotheses"]
        st.caption(tr("Panel fixed effects (country + year), country-clustered. Sign agrees = matches the literature's expected direction. GDP control = the same test with log real GDP per head held fixed (ratios to GDP move with a country's own cycle). Italy residual = distance from the cross-country line, in SDs.",
                      "Effetti fissi (paese + anno), errori standard per paese. Segno concorde = direzione attesa dalla letteratura. Controllo PIL = lo stesso test a parità di PIL reale per abitante (i rapporti sul PIL si muovono con il ciclo del paese). Residuo Italia = distanza dalla retta tra paesi, in DS."))
        st.dataframe(h.assign(y=h.y.map(label), x=h.x.map(label))[[c for c in ("id", "theme", "label", "y", "x", "coef", "p", "q_value", "sign_agrees", "p_ex_irl", "p_gdp_control", "gdp_control", "italy_resid_z", "n_obs") if c in h.columns]].round(3),
                     use_container_width=True, hide_index=True)
        if "e4_lead_lag" in res:
            with st.expander(tr("Lead-lag: does X at t-k predict Y at t better than the reverse?", "Anticipo-ritardo: X a t-k prevede Y a t meglio del contrario?")):
                ll = res["e4_lead_lag"]
                st.dataframe(ll.assign(y=ll.y.map(label), x=ll.x.map(label)).round(3), use_container_width=True, hide_index=True)
                st.caption(tr("Predictive order only, never causation.", "Solo ordine predittivo, mai causalità."))

    with tab_r:
        view_regional_correlations(res)
    with tab_f:
        view_high_frequency(res)


def view_high_frequency(res: dict[str, pd.DataFrame]) -> None:
    if "hf1_spread" not in res:
        st.info(tr("Monthly and quarterly data not built: run `uv run python -m cifresignificative.core.fetch` and `uv run python -m cifresignificative.core.clean.build`.", "Dati mensili e trimestrali non costruiti: esegui `uv run python -m cifresignificative.core.fetch` e `uv run python -m cifresignificative.core.clean.build`."))
        return
    hf = highfreq_df()
    st.caption(tr("Eurostat monthly and quarterly series, 2005 onwards. All tests run on stationary transformations (changes or growth rates), with prewhitening, own-lag controls, country and month effects, and placebo-based event windows. Predictive order and association only, never causation.",
                  "Serie mensili e trimestrali Eurostat dal 2005. Tutti i test usano trasformazioni stazionarie (variazioni o tassi di crescita), con prewhitening, controllo dei ritardi propri, effetti fissi paese e mese e finestre d'evento con placebo. Solo ordine predittivo e associazioni, mai causalità."))
    inds = sorted(hf.indicator.unique(), key=lambda k: label(k).lower())
    c1, c2 = st.columns([2, 3])
    ind = c1.selectbox(tr("Series", "Serie"), inds, index=inds.index("m_long_term_yield"), format_func=label, key="hf_ind")
    picks = c2.multiselect(tr("Countries", "Paesi"), D.EU27_ISO3, default=["DEU", "FRA", "ESP"], format_func=country_label, key="hf_countries")
    w = HF.wide_hf(hf, ind)
    lo, hi = w.index.min().to_pydatetime(), w.index.max().to_pydatetime()
    a, b = st.slider(tr("Period", "Periodo"), lo, hi, (lo, hi), key="hf_period")
    st.plotly_chart(C.line_fig(w.loc[a:b, [c for c in w.columns if c in D.EU27_ISO3]], label(ind), ylabel=label(ind), highlights=picks), use_container_width=True)
    explain(ind)
    st.subheader(tr("Italy-Germany spread and events", "Spread Italia-Germania ed eventi"))
    from cifresignificative.studies.italy_productivity.analysis.experiments import (
        italian_events,
    )
    ev = italian_events()
    kinds = st.multiselect(tr("Events shown", "Eventi mostrati"), ["government", "ecb", "policy"], default=["government", "ecb"], key="hf_kinds")
    st.plotly_chart(C.events_line_fig(HF.spread_series(hf), ev[ev.type.isin(kinds)], tr("10-year yield spread, Italy minus Germany", "Spread decennale, Italia meno Germania"), "pp"), use_container_width=True)
    sp = res["hf1_spread"]
    cty = st.selectbox(tr("Spread model, country vs Germany", "Modello dello spread, paese vs Germania"), sorted(sp.country.unique()), format_func=country_label, key="hf_spread_country")
    st.dataframe(sp[sp.country == cty][["regime", "regressor", "coef_pp_per_sd", "se", "p", "q_value", "n", "r2"]].round(3), use_container_width=True, hide_index=True)
    st.caption(tr("Monthly change in the spread on last month's standardised differentials with Germany (sentiment, change in unemployment, change in core inflation, industrial growth); Newey-West errors; q-values over all rows.",
                  "Variazione mensile dello spread sui differenziali standardizzati del mese precedente con la Germania (fiducia, variazione della disoccupazione, variazione dell'inflazione di fondo, crescita industriale); errori Newey-West; q-value su tutte le righe."))
    st.subheader(tr("Business-cycle lead-lag", "Anticipo-ritardo del ciclo economico"))
    p = res["hf2_panel"]
    piv = p.pivot_table(index=["x", "y"], columns="lag", values="verdict", aggfunc="first").rename(columns=lambda k: f"lag {k}").reset_index()
    st.dataframe(piv.assign(x=piv.x.map(label), y=piv.y.map(label)), use_container_width=True, hide_index=True)
    it = res["hf2_italy"]
    st.dataframe(it.assign(x=it.x.map(label), y=it.y.map(label)).round(3), use_container_width=True, hide_index=True)
    st.caption(tr("Verdicts: panel Granger tests (country + month effects, own lags controlled); lags in months, or quarters for GDP. Second table: Italy's best lead in prewhitened cross-correlation and its rank among countries.",
                  "Esiti: test di Granger su panel (effetti fissi paese + mese, ritardi propri controllati); ritardi in mesi, o trimestri per il PIL. Seconda tabella: anticipo migliore dell'Italia nella correlazione incrociata prewhitened e sua posizione tra i paesi."))
    st.subheader(tr("Wage pass-through", "Trasmissione ai salari"))
    st.dataframe(res["hf3_passthrough_panel"].round(3), use_container_width=True, hide_index=True)
    st.dataframe(res["hf3_passthrough_italy"].round(3), use_container_width=True, hide_index=True)
    st.subheader(tr("Event windows (placebo)", "Finestre d'evento (placebo)"))
    e = res["hf4_events"]
    t = st.multiselect(tr("Event type", "Tipo di evento"), ["government", "ecb", "policy"], default=["government", "ecb", "policy"], key="hf_ev_type")
    e = e[e.type.isin(t)]
    st.dataframe(e[["label", "event", "type", "date", "change", "z", "p", "q_value"]].round({"change": 3, "z": 3, "p": 3, "q_value": 3}), use_container_width=True, hide_index=True)
    st.caption(tr(f"{len(res['hf4_events'])} tests; {int((res['hf4_events'].q_value < 0.1).sum())} with q < 0.10. Change = mean of the 6 months after minus the 6 months before; p from a placebo distribution of all other months; events cluster at crisis turning points.",
                  f"{len(res['hf4_events'])} test; {int((res['hf4_events'].q_value < 0.1).sum())} con q < 0,10. Variazione = media dei 6 mesi dopo meno i 6 mesi prima; p da una distribuzione placebo di tutti gli altri mesi; gli eventi si concentrano nei punti di svolta delle crisi."))


def view_regional_correlations(res: dict[str, pd.DataFrame]) -> None:
    st.caption(tr("About 230 NUTS2 regions. Screens work within countries (each indicator minus its country mean, income per head removed); robustness drops whole countries; the null shuffles regions inside countries. Neighbouring regions are not independent, so treat results as exploratory.",
                  "Circa 230 regioni NUTS2. Gli screening lavorano entro i paesi (ogni indicatore meno la media del paese, reddito pro capite rimosso); la robustezza esclude interi paesi; il test nullo rimescola le regioni dentro ogni paese. Le regioni vicine non sono indipendenti: risultati esplorativi."))
    names = {"r1_levels_within": tr("Levels 2015-19, within countries", "Livelli 2015-19, entro paese"), "r2_changes_within": tr("Changes 2005-09 to 2015-19, within countries", "Variazioni 2005-09 / 2015-19, entro paese"),
             "r1_levels_between": tr("Levels 2015-19, between countries (n = countries)", "Livelli 2015-19, tra paesi (n = paesi)")}
    exp = st.selectbox(tr("Experiment", "Esperimento"), list(names), format_func=lambda k: names[k], key="reg_exp")
    c1, c2, c3 = st.columns(3)
    min_rho = c1.slider(tr("Minimum |rho|", "|rho| minimo"), 0.0, 0.95, 0.4, 0.05, key="reg_rho")
    robust_only = c2.checkbox(tr("Robust only", "Solo robuste"), value=True, key="reg_robust")
    cross_theme = c3.checkbox(tr("Different themes only", "Solo temi diversi"), value=True, help=tr("Same-theme pairs (labour vs labour) are near-definitional.", "Le coppie dello stesso tema (lavoro-lavoro) sono quasi definitorie."), key="reg_cross")
    d = res[exp]
    d = d[d.rho.abs() >= min_rho]
    if robust_only:
        d = d[d.robust]
    reg = D.load_regional()
    theme_of = reg.assign(indicator=reg.indicator.str.removeprefix("nuts2_")).groupby("indicator").theme.first()
    if cross_theme:
        d = d[d.x.map(theme_of) != d.y.map(theme_of)]
    d = d.reset_index(drop=True)
    st.caption(tr(f"{len(d)} pairs shown.", f"{len(d)} coppie mostrate."))
    lab = lambda k: label(f"nuts2_{k}")
    st.dataframe(d.assign(X=d.x.map(lab), Y=d.y.map(lab))[["X", "Y", "n", "rho", "q_value"]].round(3), use_container_width=True, hide_index=True)
    if exp != "r1_levels_between" and len(d):
        opts = [f"{k + 1}. {lab(r.x)} - {lab(r.y)}" for k, r in enumerate(d.itertuples())]
        i = opts.index(st.selectbox(tr("Plot a pair", "Disegna una coppia"), opts, key="reg_pair"))
        xi, yi = d.iloc[i].x, d.iloc[i].y
        R = rc.regional_matrix(reg, drop_flag_chars="u" if exp == "r1_levels_within" else "ubp")
        X = rc.within_country_table(R, "level", (2015, 2019)) if exp == "r1_levels_within" else rc.within_country_table(R, "change", (2015, 2019), (2005, 2009))
        areas = {g: D.italy_area(g) for g in X.index if g.startswith("IT")}
        st.plotly_chart(C.region_scatter_fig(pd.DataFrame({"x": X[xi], "y": X[yi]}), lab(xi), lab(yi), f"{lab(yi)} vs {lab(xi)}", areas), use_container_width=True)
        st.caption(tr("Values are region minus its country mean, with the income-per-head fit removed.", "Valori: regione meno la media del suo paese, al netto del reddito pro capite."))
    with st.expander(tr("Pre-specified hypotheses (region + year fixed effects, SEs clustered by country)", "Ipotesi predefinite (effetti fissi regione + anno, errori per paese)")):
        h = res["r3_region_panel"]
        st.dataframe(h.assign(y=h.y.map(lab), x=h.x.map(lab))[["id", "label", "y", "x", "coef", "p", "q_value", "sign_agrees", "p_ex_irl", "n_obs", "n_regions"]].round(3), use_container_width=True, hide_index=True)
        st.caption(tr("RH rows were specified in advance (FDR over them only); S rows were picked from the screen, so their p-values are not independent evidence.", "Le righe RH erano definite in anticipo (FDR solo su di esse); le righe S sono scelte dallo screening, quindi i loro p-value non sono evidenza indipendente."))
        ll = res["r3_lead_lag"]
        piv = ll.pivot_table(index=["id", "x", "y"], columns="lag", values="verdict", aggfunc="first").rename(columns=lambda k: f"lag {k}").reset_index()
        st.dataframe(piv.assign(x=piv.x.map(lab), y=piv.y.map(lab)), use_container_width=True, hide_index=True)
    with st.expander(tr("Is Italy's slope different? (placebo ranking across countries)", "La pendenza dell'Italia è diversa? (graduatoria placebo tra paesi)")):
        s = res["r4_italy_slope"]
        st.dataframe(s.assign(y=s.y.map(lab), x=s.x.map(lab))[["id", "x", "y", "slope_italy", "slope_others", "p_diff", "n_countries", "slope_north", "slope_south"]].round(3), use_container_width=True, hide_index=True)
        st.caption(tr("p = share of countries (with at least 5 regions) with an equal or larger slope difference; the smallest possible p is 1 / number of countries.", "p = quota di paesi (con almeno 5 regioni) con differenza di pendenza uguale o maggiore; il p minimo possibile è 1 / numero di paesi."))
    with st.expander(tr("Convergence of regional income", "Convergenza del reddito regionale")):
        st.dataframe(res["r5_convergence"].round(3), use_container_width=True, hide_index=True)
        st.caption(tr("beta < 0: poorer regions catch up (2005-07 to 2017-19, country fixed effects).", "beta < 0: le regioni più povere recuperano (2005-07 / 2017-19, effetti fissi paese)."))


def view_fiscal(panel: pd.DataFrame) -> None:
    st.header(tr("What moved government debt?", "Cosa ha mosso il debito pubblico?"))
    st.caption(tr("Change in debt/GDP split into primary balance, interest, nominal growth and a stock-flow residual (pp of GDP). Interest above growth drives Italy's debt more than primary deficits.",
                  "Variazione del debito/PIL scomposta in saldo primario, interessi, crescita nominale e residuo (punti di PIL). Per l'Italia pesano più gli interessi sopra la crescita che i disavanzi primari."))
    cs = st.multiselect(tr("Countries", "Paesi"), D.EU27_ISO3, default=["ITA", "DEU", "FRA", "ESP", "PRT", "GRC"], format_func=country_label)
    p1 = st.slider(tr("Period 1", "Periodo 1"), 2005, 2024, (2005, 2014))
    p2 = st.slider(tr("Period 2", "Periodo 2"), 2005, 2024, (2014, 2024))
    if not cs:
        st.info(tr("Select at least one country.", "Seleziona almeno un paese."))
        return
    for (a, b) in (p1, p2):
        if a >= b:
            st.warning(tr("Each period needs start < end.", "Ogni periodo richiede inizio < fine."))
            return
        dec = pd.DataFrame({c: D.debt_decomposition(panel, c, a, b) for c in cs}).T
        st.plotly_chart(C.debt_fig(dec, f"{a}-{b}: " + tr("change in debt/GDP and its components", "variazione del debito/PIL e componenti")), use_container_width=True)
        with st.expander(f"{tr('Table', 'Tabella')} {a}-{b}"):
            st.dataframe(dec.assign(total=dec.sum(axis=1)).round(1), use_container_width=True)


def view_ageing(panel: pd.DataFrame) -> None:
    st.header(tr("Ageing costs and debt scenarios", "Costi dell'invecchiamento e scenari sul debito"))
    ar = ageing_df()
    tab_proj, tab_sim = st.tabs([tr("Ageing Report projections", "Proiezioni Ageing Report"), tr("Debt simulation", "Simulazione del debito")])
    with tab_proj:
        st.caption(tr("European Commission / EPC 2024 Ageing Report, baseline 2022-2070.", "Commissione europea / EPC, Ageing Report 2024, scenario base 2022-2070."))
        opts = {tr("Public pension spending, % GDP", "Spesa pensionistica pubblica, % PIL"): ("Baseline as % of GDP", "Public pensions, gross"),
                tr("Total ageing-related spending, % GDP", "Spesa totale legata all'invecchiamento, % PIL"): ("Total spending as % of GDP", "Baseline"),
                tr("Old-age dependency ratio", "Indice di dipendenza degli anziani"): ("Dependency ratios", "Old-age dependency ratio (3)"),
                tr("Employment rate 20-64, %", "Tasso di occupazione 20-64, %"): ("Labour force assumptions", "Employment rate (20-64y)")}
        pick = st.selectbox(tr("Projection", "Proiezione"), list(opts))
        w = D.ageing_series(ar, *opts[pick])
        hl = st.multiselect(tr("Highlight countries", "Evidenzia paesi"), [c for c in D.EU27_ISO3 if c in w and c != D.ITALY], default=[c for c in ("DEU", "FRA", "ESP") if c in w], format_func=country_label, max_selections=C.MAX_HIGHLIGHTS)
        chart_with_table(C.line_fig(w, pick, highlights=hl), w[[D.ITALY, *hl]].rename(columns={D.ITALY: "Italy"}), "ageing_projection")
    with tab_sim:
        st.caption(tr("Illustrative debt arithmetic, not a forecast: debt(t) = debt(t-1) x (1+r)/(1+g) - primary balance + extra ageing cost. Growth, interest and ageing costs follow the Commission's assumptions.",
                      "Aritmetica illustrativa del debito, non una previsione: debito(t) = debito(t-1) x (1+r)/(1+g) - saldo primario + costo aggiuntivo dell'invecchiamento. Crescita, tassi e costi seguono le ipotesi della Commissione."))
        inp = scenario_inputs(panel, ar)
        c1, c2, c3 = st.columns(3)
        pb = c1.slider(tr("Primary balance, % GDP", "Saldo primario, % PIL"), -2.0, 5.0, 2.0, 0.25)
        shift = c2.slider(tr("Interest rate shift, pp", "Variazione dei tassi, punti"), -1.0, 3.0, 0.0, 0.25)
        offset = c3.checkbox(tr("Offset ageing costs", "Compensa i costi dell'invecchiamento"), value=False)
        base = simulate_debt(inp["d0"], inp["years"], inp["pb0"], inp["r"], inp["g"], inp["drift"])
        scen = simulate_debt(inp["d0"], inp["years"], pb, inp["r"] + shift / 100, inp["g"], 0.0 if offset else inp["drift"])
        hist = D.wide(panel, "gov_debt_pct_gdp")[D.ITALY]
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name=tr("history", "storia"), line={"color": C.INK2, "width": 2}))
        fig.add_trace(go.Scatter(x=base.index, y=base.values, mode="lines", name=tr("2024 primary balance held", "saldo primario 2024 mantenuto"), line={"color": C.EU_C, "width": 2.4, "dash": "dash"}))
        fig.add_trace(go.Scatter(x=scen.index, y=scen.values, mode="lines", name=tr("your scenario", "il tuo scenario"), line={"color": C.ITALY_C, "width": 3.2}))
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(C._style(fig, tr("Italy: gross debt, % of GDP", "Italia: debito lordo, % del PIL"), "% GDP", 460), use_container_width=True)
        rg = float(np.mean(inp["r"] + shift / 100 - inp["g"]) * 100)
        st.markdown(tr(f"2024 start: debt {inp['d0']:.1f}% of GDP, primary balance {inp['pb0']:+.1f}%. Average r-g in your scenario: {rg:+.1f} pp. "
                       f"A primary surplus of about **{stabilising_primary_balance(inp['d0'], 0.03 + rg / 100, 0.03):.1f}% of GDP** would hold debt constant at that r-g (before ageing costs).",
                       f"Punto di partenza 2024: debito {inp['d0']:.1f}% del PIL, saldo primario {inp['pb0']:+.1f}%. r-g medio nello scenario: {rg:+.1f} punti. "
                       f"Un avanzo primario di circa **{stabilising_primary_balance(inp['d0'], 0.03 + rg / 100, 0.03):.1f}% del PIL** terrebbe il debito costante a quel r-g (prima dei costi di invecchiamento)."))
        csv_button(pd.DataFrame({"base": base, "scenario": scen}), "debt_scenario")


def view_spending(panel: pd.DataFrame) -> None:
    st.header(tr("Where public money goes", "Dove va la spesa pubblica"))
    sp = spending_df()
    st.caption(tr("General-government expenditure by function (COFOG) and economic nature. 2021-23 contain large one-off capital transfers in Italy.",
                  "Spesa delle amministrazioni pubbliche per funzione (COFOG) e natura economica. Nel 2021-23 l'Italia ha forti trasferimenti in conto capitale una tantum."))
    c1, c2 = st.columns(2)
    year = c1.select_slider(tr("Year", "Anno"), options=list(range(2005, 2025)), value=2024)
    unit = c2.radio(tr("Unit", "Unità"), ["% of GDP", "% of total spending"], horizontal=True)
    u = "PC_GDP" if unit == "% of GDP" else "PC_TOT"
    t = D.cofog_groups(sp, year, u)
    eu = [c for c in t.index if c in D.EU27_ISO3 and c != D.ITALY]
    gap = (t.loc[D.ITALY] - t.loc[eu].median()).dropna()
    tab_gap, tab_series, tab_nature, tab_eur = st.tabs([tr("Italy vs median", "Italia vs mediana"), tr("Over time", "Nel tempo"), tr("By economic nature", "Per natura economica"), tr("Italy in euro", "Italia in euro")])
    with tab_gap:
        st.plotly_chart(C.signed_bars_fig(gap, f"{year}: {tr('Italy minus median of the other EU27', 'Italia meno mediana altri UE27')} ({unit})", "pp"), use_container_width=True)
        comp = pd.DataFrame({"Italy": t.loc[D.ITALY], "Median of other EU27": t.loc[eu].median()}).dropna()
        st.plotly_chart(C.bars_fig(comp, {"Italy": C.ITALY_C, "Median of other EU27": C.EU_C}, f"{year}: " + tr("composition", "composizione") + f" ({unit})", "pp"), use_container_width=True)
        csv_button(comp.assign(gap=gap), f"cofog_{year}")
    with tab_series:
        grp = st.selectbox(tr("Function", "Funzione"), list(D.COFOG_GROUPS), index=list(D.COFOG_GROUPS).index("Old-age pensions"))
        st.plotly_chart(C.line_fig(D.cofog_group_series(sp, grp, u), f"{grp} ({unit})", unit, highlights=["DEU", "FRA", "ESP"]), use_container_width=True)
    with tab_eur:
        _spending_euro_tab(sp, year)
    with tab_nature:
        it = st.selectbox(tr("Item", "Voce"), list(D.NATURE_LABELS), format_func=lambda k: D.NATURE_LABELS[k])
        st.plotly_chart(C.line_fig(D.spend_wide(sp, it, "nature"), D.NATURE_LABELS[it], "% of GDP", highlights=["DEU", "FRA", "ESP"]), use_container_width=True)


def _spending_euro_tab(sp: pd.DataFrame, year: int) -> None:
    st.caption(tr("Million euro at current prices (Italy only: euro levels are not comparable across countries; nominal values include inflation).",
                  "Milioni di euro a prezzi correnti (solo Italia: i livelli in euro non sono confrontabili tra paesi; i valori nominali includono l'inflazione)."))
    if "MIO_EUR" not in set(sp.unit):
        st.info(tr("Rebuild the data to get euro levels.", "Ricostruire i dati per avere i valori in euro."))
        return
    t = D.italy_cofog_euro(sp, [year])
    st.dataframe(t.sort_values(f"{year} MEUR", ascending=False).round(1), use_container_width=True)
    csv_button(t, f"cofog_eur_{year}")


def view_efficiency(panel: pd.DataFrame) -> None:
    st.header(tr("Spending against outcomes", "Spesa e risultati"))
    st.caption(tr("Public spending in an area against an outcome it is meant to improve, 2015-2019 averages. The dashed line is fitted on the other members; Italy's distance from it is a prompt for questions, not a verdict.",
                  "Spesa pubblica in un ambito rispetto a un risultato che dovrebbe migliorare, medie 2015-2019. La linea è stimata sugli altri paesi; la distanza dell'Italia è uno spunto, non un giudizio."))
    data = EF.pair_data(panel, spending_df())
    key = st.selectbox(tr("Pair", "Coppia"), list(EF.PAIRS), format_func=lambda k: EF.PAIRS[k][0])
    title, xl, yl, lower = EF.PAIRS[key]
    x, y = data[key]
    s = EF.residual_summary(x, y, lower)
    chart_with_table(C.scatter_fig(x, y, xl, yl, title), pd.concat([x.rename(xl), y.rename(yl)], axis=1).dropna(), key,
                     tr(f"Italy: {s['Italy y']:.1f} against {s['fitted y']:.1f} predicted from its spending ({s['residual (SD)']:+.2f} SD); rank {s['rank (1 = best)']}/{s['n']} on the outcome relative to spending. Correlation among the other members: {s['corr (others)']:.2f}.",
                        f"Italia: {s['Italy y']:.1f} contro {s['fitted y']:.1f} previsto dalla spesa ({s['residual (SD)']:+.2f} DS); posizione {s['rank (1 = best)']}/{s['n']} sul risultato rispetto alla spesa. Correlazione tra gli altri paesi: {s['corr (others)']:.2f}."))


def view_productivity(panel: pd.DataFrame) -> None:
    st.header(tr("Productivity by sector", "Produttività per settore"))
    st.caption(tr("Real value added per hour worked, nine NACE sections (real estate excluded; public administration, education and health included). Only growth is compared across countries.",
                  "Valore aggiunto reale per ora lavorata, nove sezioni NACE (escluso immobiliare; pubblica amministrazione, istruzione e sanità incluse). Si confrontano solo le crescite."))
    s = sectors_df()
    a, b = st.slider(tr("Period", "Periodo"), 2005, 2023, (2005, 2019))
    agg = D.aggregate_productivity(s)
    cover = agg[[c for c in agg if c in D.EU27_ISO3]].notna().sum(axis=1)
    agg = agg.loc[:cover[cover >= 20].index.max()]
    st.plotly_chart(C.line_fig(agg, tr("Real output per hour, nine sections", "Produzione reale per ora, nove sezioni"), "EUR (2020 prices) per hour", highlights=["DEU", "FRA", "ESP"], rebase=True, years=(a, int(agg.index.max()))), use_container_width=True)
    p = s.pivot_table(index="iso3", columns=["nace_r2", "year"], values="prod")
    g = pd.DataFrame({n: np.log(p[(n, b)] / p[(n, a)]) / (b - a) * 100 for n in D.NINE_SECTIONS})
    eu = [c for c in g.index if c in D.EU27_ISO3 and c != D.ITALY]
    tab = pd.DataFrame({"Italy": g.loc[D.ITALY], "Median of other EU27": g.loc[eu].median(), "Germany": g.loc["DEU"]}).rename(index=SECTOR_LABELS).sort_values("Italy")
    st.plotly_chart(C.bars_fig(tab, {"Italy": C.ITALY_C, "Median of other EU27": C.EU_C, "Germany": C.PEER_C}, f"{tr('Productivity growth by sector', 'Crescita della produttività per settore')}, {a}-{b} (% {tr('per year', 'annuo')})", "%"), use_container_width=True)
    cs = st.multiselect(tr("Shift-share countries", "Paesi per la scomposizione"), D.EU27_ISO3, default=["ITA", "DEU", "FRA", "ESP"], format_func=country_label)
    if cs:
        dec = pd.DataFrame({c: D.shift_share(s, c, a, b) for c in cs}).T
        st.plotly_chart(C.bars_fig(dec, {"within-sector productivity": C.ITALY_C, "reallocation of hours": "#eda100", "interaction": C.PEER_C},
                                   f"{tr('Aggregate productivity change', 'Variazione della produttività aggregata')} {a}-{b}: " + tr("within sector vs reallocation (%)", "entro i settori vs riallocazione (%)"), "%", horizontal=False, stacked=True), use_container_width=True)
        with st.expander(tr("Table", "Tabella")):
            st.dataframe(dec.round(1), use_container_width=True)


def view_regions(panel: pd.DataFrame) -> None:
    st.header(tr("Regions (NUTS2)", "Regioni (NUTS2)"))
    reg = regional_df()
    inds = sorted(reg.indicator.unique())
    ind = st.selectbox(tr("Indicator", "Indicatore"), inds, index=inds.index("nuts2_gdp_pps_pc_eu27") if "nuts2_gdp_pps_pc_eu27" in inds else 0, format_func=lambda k: label(k.replace("nuts2_", "")))
    ys = sorted(reg[reg.indicator == ind].year.unique())
    year = st.select_slider(tr("Year", "Anno"), options=ys, value=ys[-2] if len(ys) > 1 else ys[-1])
    d = reg[(reg.indicator == ind) & (reg.year == year)].dropna(subset=["value"])
    counts = d.groupby("country").size()
    d = d[d.country.isin(counts[counts >= 3].index)].assign(italy=lambda x: x.country == "IT").rename(columns={"geo": "region"})
    st.plotly_chart(C.strip_fig(d, f"{label(ind.replace('nuts2_', ''))} - {year}", "value"), use_container_width=True)
    area = {"ITC": "North-West", "ITH": "North-East", "ITI": "Centre", "ITF": "South", "ITG": "Islands"}
    it = d[d.italy].assign(area=lambda x: x.region.str[:3].map(area)).sort_values("value", ascending=False)[["region", "area", "value"]]
    c1, c2 = st.columns(2)
    c1.subheader(tr("Italian regions", "Regioni italiane"))
    c1.dataframe(it, use_container_width=True, hide_index=True)
    c2.subheader(tr("By macro-area (median)", "Per macro-area (mediana)"))
    c2.dataframe(it.groupby("area").value.median().round(1).reindex(["North-West", "North-East", "Centre", "South", "Islands"]).to_frame("median"), use_container_width=True)


def rlabel(k: str) -> str:
    """Label of a regional variable: the `nuts2_` glossary entry when there is one (the screens strip the prefix), else the plain key."""
    return label("nuts2_" + k if "nuts2_" + k in glossary.load().index else k)


def view_south(panel: pd.DataFrame) -> None:
    st.header(tr("Why the South: governance, firms, migration, spending", "Perché il Sud: governo, imprese, migrazioni, spesa"))
    st.warning(tr("Associations only, not causes. Perceptions of institutions can follow prosperity as well as precede it, migration follows jobs, and spending is attributed to where it is incurred (pensions included).",
                  "Solo associazioni, non cause. Le percezioni delle istituzioni possono seguire la prosperità oltre che precederla, le migrazioni seguono il lavoro e la spesa è attribuita a dove viene sostenuta (pensioni incluse)."))
    res = south_results()
    tab_a, tab_b, tab_c = st.tabs([tr("EU-wide (within countries)", "UE (entro paese)"), tr("Italy only (exploratory)", "Solo Italia (esplorativo)"), tr("Italian regions", "Regioni italiane")])
    with tab_a:
        st.caption(tr("Regions demeaned by country; Spearman rho with within-country permutation p-values; BH q over the family; wave 2017 is the primary test, the other waves replicate.",
                      "Regioni al netto della media del paese; rho di Spearman con p da permutazioni entro paese; q di BH sulla famiglia; l'onda 2017 è il test principale, le altre replicano."))
        c1, c2 = st.columns(2)
        wave = c1.selectbox(tr("EQI wave", "Onda EQI"), ["2017", "2010", "2013", "2021"])
        beyond = c2.checkbox(tr("Beyond income (income fit removed)", "Al netto del reddito"), value=False)
        d = res["s1_eu_wide"]
        d = d[(d.wave == int(wave)) & (d.income_control == beyond)]
        show = d.assign(**{tr("Indicator", "Indicatore"): d.x.map(rlabel), tr("Regional gap", "Divario regionale"): d.y.map(rlabel)})
        st.dataframe(show[[tr("Indicator", "Indicatore"), tr("Regional gap", "Divario regionale"), "n", "n_groups", "rho", "q_value", "loo_min_abs_rho", "loo_sign_flips"]].sort_values("q_value").round(3), use_container_width=True, hide_index=True)
        st.subheader(tr("Is Italy different? Rank of Italy's slope among countries", "L'Italia è diversa? Posizione della pendenza italiana tra i paesi"))
        r2 = res["s2_italy_rank"].sort_values("p_diff").head(10)
        st.dataframe(r2.assign(x=r2.x.map(rlabel), y=r2.y.map(rlabel))[["x", "y", "n_countries", "slope_italy", "slope_others", "p_diff"]].round(3), use_container_width=True, hide_index=True)
    with tab_b:
        st.caption(tr("21 regions, one cluster: p-values come from permuting values across regions, and within North / Centre / South-and-Islands zones (which holds the North-South split fixed).",
                      "21 regioni, un solo gruppo: i p-value vengono da permutazioni dei valori tra regioni e entro le zone Nord / Centro / Sud e Isole (che tengono fisso il divario Nord-Sud)."))
        s3 = res["s3_italy_only"]
        y = st.selectbox(tr("Regional gap", "Divario regionale"), rs.OUTCOMES, index=3, format_func=rlabel)
        only_it = st.checkbox(tr("Italy-only indicators (migration, spending)", "Solo indicatori italiani (migrazioni, spesa)"), value=True)
        d = s3[s3.y == y]
        d = d[d.x.isin(rs.ITALY_ONLY)] if only_it else d
        st.dataframe(d.assign(x=d.x.map(rlabel))[["x", "rho", "p_perm", "rho_within_zone", "p_perm_within_zone", "q_perm_within_zone", "loo_min_rho", "loo_max_rho"]].sort_values("p_perm_within_zone").round(3), use_container_width=True, hide_index=True)
        st.subheader(tr("North-South gap (standard deviations)", "Divario Nord-Sud (deviazioni standard)"))
        g = res["s5_zone_gaps"]
        st.dataframe(g.assign(indicator=g.indicator.map(rlabel)).round(2), use_container_width=True, hide_index=True)
    with tab_c:
        T = res["s_italy_table"].set_index("geo")
        cols = [c for c in [*rs.OUTCOMES, *rs.EU_WIDE, *rs.ITALY_ONLY] if c in T]
        c1, c2 = st.columns(2)
        xi = c1.selectbox(tr("X axis", "Asse X"), cols, index=cols.index("net_migration_per_1000"), format_func=rlabel)
        yi = c2.selectbox(tr("Y axis", "Asse Y"), cols, index=cols.index("employment_rate_20_64"), format_func=rlabel, key="south_y")
        d = T[[xi, yi]].dropna()
        fig = go.Figure()
        for z, col in {"North": C.EU_C, "Centre": C.PEER_C, "South and Islands": C.ITALY_C}.items():
            k = d[[rs.zone_of(i) == z for i in d.index]]
            fig.add_trace(go.Scatter(x=k[xi], y=k[yi], mode="markers+text", text=k.index, textposition="top center", textfont={"size": 9}, marker={"size": 10, "color": col}, name=z))
        fig.update_layout(xaxis_title=rlabel(xi), yaxis_title=rlabel(yi), height=480)
        st.plotly_chart(fig, use_container_width=True)


def view_gender(panel: pd.DataFrame) -> None:
    st.header(tr("Gender and wages", "Genere e salari"))
    preset = st.radio(tr("Panel", "Pannello"), ["Gender equality", "Pay and labour cost"], horizontal=True)
    emp_gap = D.wide(panel, "employment_rate_20_64_male") - D.wide(panel, "employment_rate_20_64_female")
    if preset == "Gender equality":
        items = [(emp_gap, "Employment gap, men minus women (pp)"), (D.wide(panel, "part_time_female_20_64"), "Women in part-time (% of employed women)"),
                 (D.wide(panel, "involuntary_part_time_female_15_64"), "Involuntary part-time, women (% of part-timers)"),
                 (D.wide(panel, "women_managers_share_pct"), "Women among managers (%)"), (D.wide(panel, "gender_pension_gap_65"), "Gender pension gap, 65+ (%)"),
                 (D.wide(panel, "gender_pay_gap"), "Unadjusted gender pay gap (%)")]
        st.caption(tr("The unadjusted pay gap is small in Italy partly because fewer women are employed; the overall earnings gap (pay x hours x employment, 4-yearly, latest 2018) was 43% vs a 28% median.",
                      "Il divario salariale grezzo è basso in Italia anche perché lavorano meno donne; il divario complessivo (paga x ore x occupazione, quadriennale, ultimo 2018) era 43% contro una mediana del 28%."))
    else:
        items = [(D.wide(panel, "gross_earnings_pps"), "Gross annual earnings, average worker (PPS; new basis from 2024)"), (D.wide(panel, "net_earnings_pps"), "Net annual earnings (PPS; new basis from 2024)"),
                 (D.wide(panel, "labour_cost_hourly_eur"), "Hourly labour cost (EUR)"), (D.wide(panel, "tax_wedge_low_wage"), "Tax wedge, low-wage earners (%)"),
                 (D.wide(panel, "inwork_poverty_18_64"), "In-work poverty (% of workers)")]
    st.plotly_chart(C.multiples_fig(items), use_container_width=True)


def view_higher_ed(panel: pd.DataFrame) -> None:
    st.header(tr("Higher education", "Istruzione superiore"))
    he = highered_df()
    tab_over, tab_fields, tab_age = st.tabs([tr("Overview", "Panoramica"), tr("Fields (STEM)", "Campi di studio (STEM)"), tr("Age at graduation", "Età alla laurea")])
    with tab_over:
        gender_gap = D.wide(panel, "tertiary_attainment_25_34_female") - D.wide(panel, "tertiary_attainment_25_34_male")
        items = [(D.wide(panel, "tertiary_attainment_25_34"), "Tertiary attainment, 25-34 (%)"), (gender_gap, "Women minus men, attainment 25-34 (pp)"),
                 (D.wide(panel, "stem_share_grads_pct"), "STEM share of graduates (%)"), (D.wide(panel, "women_share_stem_grads_pct"), "Women among STEM graduates (%)"),
                 (D.wide(panel, "phd_grads_per_1000_pop_25_34"), "Doctoral graduates per 1000 aged 25-34"), (D.wide(panel, "recent_grads_employment_rate"), "Employment of recent graduates (%)")]
        st.plotly_chart(C.multiples_fig(items), use_container_width=True)
    with tab_fields:
        level = st.radio(tr("Level", "Livello"), ["ED5-8", "ED6", "ED7", "ED8"], horizontal=True, format_func=lambda k: {"ED5-8": "All tertiary", "ED6": "Bachelor", "ED7": "Master", "ED8": "Doctoral"}[k])
        year = D.broad_year(he[he.isced11 == level], "he_grads_field")
        sh = D.field_shares(he, "he_grads_field", level, "T", year)
        eu = [c for c in sh.index if c in D.EU27_ISO3 and c != D.ITALY]
        tab = pd.DataFrame({"Italy": sh.loc[D.ITALY], "Median of other EU27": sh.loc[eu].median(), "Germany": sh.loc["DEU"]}).rename(index=D.FIELD_LABELS).sort_values("Italy")
        st.plotly_chart(C.bars_fig(tab, {"Italy": C.ITALY_C, "Median of other EU27": C.EU_C, "Germany": C.PEER_C}, f"{tr('What graduates study', 'Cosa studiano i laureati')}, {year} (% {tr('of graduates', 'dei laureati')})", "%"), use_container_width=True)
        wsh = D.women_share_by_field(he, "he_grads_field", level, year)
        wt = pd.DataFrame({"Italy": wsh.loc[D.ITALY], "Median of other EU27": wsh.loc[[c for c in wsh.index if c in D.EU27_ISO3 and c != D.ITALY]].median()}).rename(index={**D.FIELD_LABELS, "TOTAL": "All fields"})
        st.plotly_chart(C.bars_fig(wt, {"Italy": C.ITALY_C, "Median of other EU27": C.EU_C}, f"{tr('Women as % of graduates by field', 'Donne in % dei laureati per campo')}, {year}", "% women"), use_container_width=True)
    with tab_age:
        lvl = st.radio(tr("Degree", "Titolo"), ["ED6", "ED7"], horizontal=True, format_func=lambda k: {"ED6": "Bachelor", "ED7": "Master"}[k], key="age_level")
        ages = [f"Y{a}" for a in range(20, 36)]
        year = D.broad_year(he[he.isced11 == lvl], "he_grads_age")
        prof = D.age_profile(he, lvl, year, ages).T
        prof.index = [int(a[1:]) for a in prof.index]
        st.plotly_chart(C.line_fig(prof, f"{tr('Graduates by age', 'Laureati per età')}, {year} (% {tr('of graduates', 'dei laureati')})", "%", highlights=["DEU", "FRA", "ESP"]), use_container_width=True)


def view_long(panel: pd.DataFrame) -> None:
    st.header(tr("Long history (1960-2024)", "Lungo periodo (1960-2024)"))
    st.caption(tr("AMECO series in national currency: compare countries through indices or ratios, not raw levels.", "Serie AMECO in valuta nazionale: confronta i paesi con indici o rapporti, non con livelli."))
    lh = long_df()
    names = sorted(lh.variable.unique())
    ratio = {"gdp_real_per_head_ncu", "gdp_real_per_hour_ncu", "gdp_real_ncu_bn", "gdp_nominal_ncu_bn", "gov_debt_ncu_bn", "gfcf_nominal_ncu_bn", "gov_gfcf_nominal_ncu_bn", "comp_per_employee_nominal_th", "population_th", "employment_th", "hours_total_m"}
    var = st.selectbox(tr("Series", "Serie"), names, index=names.index("gdp_real_per_head_ncu"), format_func=label)
    w = D.long_wide(lh, var)
    is_level = var in ratio
    rebase = st.checkbox(tr("Rebase to 100 in year", "Base 100 nell'anno"), value=is_level)
    base = st.slider(tr("Base year", "Anno base"), 1960, 2020, 1995) if rebase else None
    if rebase:
        w = w / w.loc[base] * 100
    hl = st.multiselect(tr("Highlight countries", "Evidenzia paesi"), [c for c in [*D.EU27_ISO3, *D.BENCHMARKS] if c in w and c != D.ITALY], default=["DEU", "FRA", "ESP"], format_func=country_label, max_selections=C.MAX_HIGHLIGHTS)
    st.plotly_chart(C.line_fig(w, label(var) + (f" ({base} = 100)" if rebase else ""), highlights=hl), use_container_width=True)


def view_timeline(panel: pd.DataFrame) -> None:
    from cifresignificative.studies.italy_productivity.analysis.timeline import (
        government_stats,
        load_governments,
        load_policies,
    )

    st.header(tr("Governments and policy milestones", "Governi e interventi normativi"))
    st.caption(tr("Timing only, not causal attribution: policies act with lags and overlap, and euro-area conditions drive some series. Filled diamonds = policy date confirmed in a source; open = partly confirmed or unconfirmed (see the table).",
                  "Solo tempistica, non attribuzione causale: le politiche agiscono con ritardo e si sovrappongono, e il contesto dell'area euro pesa su alcune serie. Rombi pieni = data confermata da una fonte; vuoti = parzialmente o non confermata (vedi tabella)."))
    gov, pol = load_governments("1989-01-01"), load_policies()
    has_italy = sorted(set(panel[panel.iso3 == D.ITALY].indicator.unique()))
    wanted = QP.get("ind")
    ind = st.selectbox(tr("Indicator", "Indicatore"), has_italy, index=has_italy.index(wanted) if wanted in has_italy else has_italy.index("employment_rate_55_64"), format_func=label)
    w = D.wide(panel, ind)
    domains = sorted({d.strip() for ds in pol.domain for d in str(ds).split("/")})
    chosen = st.multiselect(tr("Policy areas", "Aree di intervento"), domains, default=[d for d in domains if d in ("pensions", "labour")] or domains[:2])
    sel = pol[pol.domain.apply(lambda ds: any(d.strip() in chosen for d in str(ds).split("/")))] if chosen else pol.iloc[0:0]
    years = w[D.ITALY].dropna().index
    xlim = st.slider(tr("Years", "Anni"), int(years.min()), int(years.max()), (max(int(years.min()), 2005), int(years.max())))
    show_med = st.checkbox(tr("Show median of other EU27", "Mostra la mediana degli altri UE27"), value=True)
    QP["view"], QP["ind"] = "timeline", ind
    fig = C.timeline_fig(w[D.ITALY].loc[xlim[0]:xlim[1]], D.eu_median_ex_italy(w).loc[xlim[0]:xlim[1]] if show_med else None, gov, sel, label(ind), "", xlim)
    st.plotly_chart(fig, use_container_width=True)
    tab_pol, tab_gov = st.tabs([tr("Policies shown", "Interventi mostrati"), tr("Governments: growth, unemployment, debt", "Governi: crescita, disoccupazione, debito")])
    with tab_pol:
        st.dataframe(sel[["date", "title", "domain", "reference", "what_it_changed", "evidence", "sources"]], use_container_width=True, hide_index=True)
        csv_button(sel.set_index("date"), "italy_policies_selected")
    with tab_gov:
        stats = government_stats(long_df(), load_governments("1989-01-01"), "1989-01-01").round(1)
        st.caption(tr("Weighted by time in office; median of the other EU27 members over the same period. Descriptive only.", "Ponderato per il tempo in carica; mediana degli altri UE27 nello stesso periodo. Solo descrittivo."))
        st.dataframe(stats, use_container_width=True, hide_index=True)
        csv_button(stats.set_index("government"), "italy_government_stats")


def view_glossary(panel: pd.DataFrame) -> None:
    st.header(tr("Glossary", "Glossario"))
    st.caption(tr("What each indicator measures, its unit and the main concepts used in the charts. Search by name, definition or code.",
                  "Cosa misura ogni indicatore, la sua unità e i principali concetti usati nei grafici. Cerca per nome, definizione o codice."))
    lang = st.session_state["lang"]
    g = glossary.load().copy()
    meta = panel.groupby("indicator").agg(theme=("theme", "first"), source=("source", "first"))
    g = g.join(meta, how="left")
    g["theme"] = g.theme.fillna(tr("concept", "concetto"))
    c1, c2 = st.columns([3, 1])
    q = c1.text_input(tr("Search", "Cerca"), value=QP.get("q", ""), placeholder=tr("e.g. productivity, pension, PPS", "es. produttività, pensioni, PPS"))
    themes = [tr("all", "tutti"), *sorted(g.theme.unique())]
    theme = c2.selectbox(tr("Theme", "Tema"), themes)
    if q:
        hay = (g.key + " " + g.title_en + " " + g.title_it + " " + g.definition_en + " " + g.definition_it).str.lower()
        for word in q.lower().split():
            g = g[hay.loc[g.index].str.contains(word, regex=False)]
            hay = hay.loc[g.index]
    if theme != themes[0]:
        g = g[g.theme == theme]
    QP["view"] = "glossary"
    if q:
        QP["q"] = q
    elif "q" in QP:
        del QP["q"]
    st.write(tr(f"{len(g)} entries", f"{len(g)} voci"))
    for r in g.to_dict("records"):
        unit = f" - *{r['unit']}*" if r["unit"] else ""
        st.markdown(f"**{r[f'title_{lang}']}**{unit}  \n{r[f'definition_{lang}']}  \n<small>{tr('Theme', 'Tema')}: {r['theme']} | {tr('Code', 'Codice')}: `{r['key']}`</small>", unsafe_allow_html=True)
    csv_button(g[["key", "kind", f"title_{lang}", "unit", f"definition_{lang}", "theme"]].set_index("key"), "glossary")


def view_about(panel: pd.DataFrame) -> None:
    st.header(tr("About the data and the findings", "Dati e risultati"))
    tabs = st.tabs([tr("Findings", "Risultati"), tr("Data catalog", "Catalogo dei dati"), tr("Data dictionary", "Dizionario dei dati")])
    for tab, path in zip(tabs, ["studies/italy_productivity/reports/findings.md", "docs/data_catalog.md", "docs/data_dictionary.md"], strict=True):
        with tab:
            p = ROOT / path
            st.markdown(p.read_text() if p.exists() else f"`{path}` not found.")


VIEWS = {"overview": ("Overview", "Panoramica", view_overview), "explorer": ("Indicator explorer", "Esplora indicatori", view_explorer),
         "scatter": ("Compare indicators", "Confronta indicatori", view_scatter), "correlations": ("Correlation discovery", "Ricerca di correlazioni", view_correlations), "fiscal": ("Debt and fiscal", "Debito e finanza pubblica", view_fiscal),
         "ageing": ("Ageing and debt scenarios", "Invecchiamento e scenari", view_ageing), "spending": ("Spending", "Spesa pubblica", view_spending),
         "efficiency": ("Spending vs outcomes", "Spesa e risultati", view_efficiency), "productivity": ("Productivity", "Produttività", view_productivity),
         "regions": ("Regions", "Regioni", view_regions), "south": ("Why the South", "Perché il Sud", view_south), "gender": ("Gender and wages", "Genere e salari", view_gender),
         "higher_ed": ("Higher education", "Istruzione superiore", view_higher_ed), "long": ("Long history", "Lungo periodo", view_long), "timeline": ("Governments and policies", "Governi e politiche", view_timeline), "glossary": ("Glossary", "Glossario", view_glossary), "about": ("About", "Informazioni", view_about)}


TITLE = ("Italy vs EU27", "Italia e UE27")
CAPTION = (
    "Italy in blue, median of the other EU27 in orange (dashed), other countries in gray. Data: Eurostat, World Bank, IMF, AMECO, OECD, European Commission, Istat. 2025 values are provisional. The link in the address bar reproduces this view.",
    "Italia in blu, mediana degli altri UE27 in arancione (tratteggio), altri paesi in grigio. Dati: Eurostat, Banca mondiale, FMI, AMECO, OCSE, Commissione europea, Istat. I valori 2025 sono provvisori. Il link nella barra degli indirizzi riproduce questa vista.",
)


def data_error() -> str | None:
    """Message to show instead of the views when the processed data are missing."""
    if not Path("data/processed/panel.parquet").exists():
        return "Processed data not found. Run `uv run python -m cifresignificative.core.fetch && uv run python -m cifresignificative.core.clean.build` first."
    return None


def load() -> pd.DataFrame:
    return panel_df()
