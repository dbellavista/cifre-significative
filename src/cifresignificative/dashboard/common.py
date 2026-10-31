"""Helpers shared by the dashboard shell and the views of every study: language, query parameters, labels, tables."""
from pathlib import Path

import pandas as pd
import streamlit as st

from cifresignificative.core import glossary
from cifresignificative.core.countries import NAMES

ROOT = Path(__file__).resolve().parents[3]
QP = st.query_params


def tr(en: str, it: str | None = None) -> str:
    """English or Italian text depending on the language selector (falls back to English)."""
    return it if (st.session_state.get("lang") == "it" and it) else en


def label(ind: str) -> str:
    return glossary.title(ind, st.session_state.get("lang", "en"))


def explain(ind: str) -> None:
    """Definition and unit of an indicator, shown under its selector."""
    text = glossary.describe(ind, st.session_state.get("lang", "en"))
    if text:
        st.caption(f"**{label(ind)}**: {text}")


def country_label(iso: str) -> str:
    return f"{NAMES.get(iso, iso)} ({iso})"


def csv_button(df: pd.DataFrame, name: str) -> None:
    st.download_button(tr("Download table (CSV)", "Scarica la tabella (CSV)"), df.to_csv().encode(), file_name=f"{name}.csv", mime="text/csv")


def chart_with_table(fig, table: pd.DataFrame, name: str, about: str = "") -> None:
    tab_chart, tab_table, tab_about = st.tabs([tr("Chart", "Grafico"), tr("Table", "Tabella"), tr("About", "Informazioni")])
    with tab_chart:
        st.plotly_chart(fig, use_container_width=True)
    with tab_table:
        st.dataframe(table.round(2), use_container_width=True)
        csv_button(table, name)
    with tab_about:
        st.markdown(about or tr("No additional notes.", "Nessuna nota aggiuntiva."))


def qp_list(key: str, default: list[str]) -> list[str]:
    raw = QP.get(key)
    return [x for x in raw.split(",") if x] if raw else default
