"""Streamlit dashboard shell. Run with `uv run streamlit run src/cifresignificative/dashboard/app.py`.

Each study that registers a `dashboard` module in its study.toml contributes its views (`VIEWS`, `TITLE`, `CAPTION`, `load()`,
`data_error()`); the shell owns the language selector, the study selector (shown when there is more than one), the theme and the URL.
English and Italian interface (indicator names stay in English), charts follow the Streamlit light/dark theme, and the selected
study, view, indicator, language and highlighted countries are kept in the URL so a view can be shared as a link.
"""
import importlib
import os

from cifresignificative.dashboard.common import QP, ROOT

os.chdir(ROOT)  # data paths in the study analysis code are relative to the project root

import streamlit as st

from cifresignificative.dashboard import charts as C
from cifresignificative.dashboard.common import tr
from cifresignificative.study import load_studies

MODULES = {s.name: importlib.import_module(s.dashboard) for s in load_studies().values() if s.dashboard}
requested = QP.get("study")
default_study = "italy_productivity" if "italy_productivity" in MODULES else next(iter(MODULES))
chosen = requested if requested is not None and requested in MODULES else default_study
st.set_page_config(page_title=MODULES[chosen].TITLE[0], layout="wide", page_icon=":bar_chart:")


def main() -> None:
    lang_default = 1 if QP.get("lang") == "it" else 0
    lang = st.sidebar.selectbox("Language / Lingua", ["English", "Italiano"], index=lang_default, key="lang_choice")
    st.session_state["lang"] = "it" if lang == "Italiano" else "en"
    QP["lang"] = st.session_state["lang"]
    name = chosen
    if len(MODULES) > 1:
        name = st.sidebar.selectbox(tr("Study", "Studio"), list(MODULES), index=list(MODULES).index(chosen), key="study", format_func=lambda n: tr(*MODULES[n].TITLE))
        QP["study"] = name
    mod = MODULES[name]
    error = mod.data_error()
    if error:
        st.error(error)
        return
    try:  # follow the Streamlit theme (Settings menu): charts switch palettes with it
        C.apply_theme(st.context.theme.type or "light")
    except Exception:  # noqa: BLE001 - older Streamlit or no theme info: keep the light palette
        C.apply_theme("light")
    VIEWS = mod.VIEWS
    keys = list(VIEWS)
    if st.session_state.get("view") not in keys:  # another study's view left in the widget state
        st.session_state.pop("view", None)
    qp_view = QP.get("view")
    default_view = keys.index(qp_view) if qp_view in keys else 0
    view = st.sidebar.radio(tr("View", "Vista"), keys, index=default_view, key="view", format_func=lambda k: tr(VIEWS[k][0], VIEWS[k][1]))
    if view != "explorer":
        QP["view"] = view
        for k in ("ind", "hl") + (() if view == "glossary" else ("q",)):
            if k in QP:
                del QP[k]
    st.sidebar.markdown("---")
    st.sidebar.caption(tr(*mod.CAPTION))
    VIEWS[view][2](mod.load())


main()
