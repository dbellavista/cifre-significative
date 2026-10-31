"""Smoke tests: the Streamlit app renders every view without exceptions (needs the processed data)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path("src/cifresignificative/dashboard/app.py").resolve())
pytestmark = pytest.mark.skipif(not Path("data/processed/panel.parquet").exists(), reason="data not built")
VIEWS = ["overview", "explorer", "scatter", "correlations", "fiscal", "ageing", "spending", "efficiency", "productivity", "regions", "south", "gender", "higher_ed", "long", "timeline", "glossary", "about"]


@pytest.mark.parametrize("view", VIEWS)
def test_view_renders(view):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    at.sidebar.radio(key="view").set_value(view).run()
    assert not at.exception, [e.value for e in at.exception]


def test_italian_interface_and_deep_link():
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["lang"] = "it"
    at.query_params["view"] = "explorer"
    at.query_params["ind"] = "gov_debt_pct_gdp"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.sidebar.radio(key="view").value == "explorer"
    assert any("Esplora" in h.value for h in at.header)


def test_weighted_median_toggle_runs():
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["view"] = "explorer"
    at.run()
    at.checkbox[2].check().run()
    assert not at.exception, [e.value for e in at.exception]


def test_glossary_search_and_coverage():
    import pandas as pd

    from cifresignificative.core import glossary

    g = glossary.load()
    ind = set(pd.read_parquet("data/processed/panel.parquet").indicator)
    assert not ind - set(g.index), sorted(ind - set(g.index))
    assert g.title_en[g.kind == "indicator"].is_unique and g.title_it[g.kind == "indicator"].is_unique
    assert (g[["title_en", "title_it", "definition_en", "definition_it"]] != "").all().all()
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["view"] = "glossary"
    at.run()
    at.text_input[0].set_value("pension").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("entries" in m.value for m in at.markdown) or any("entries" in str(t.value) for t in at.text)


def test_regional_correlation_tab_renders_in_both_languages():
    for lang in ("en", "it"):
        at = AppTest.from_file(APP, default_timeout=180)
        at.query_params["lang"] = lang
        at.query_params["view"] = "correlations"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        if lang == "en":  # AppTest re-evaluates format_func outside the script (session language unset), so only interact in English
            at.selectbox(key="reg_exp").set_value("r2_changes_within").run()
            assert not at.exception, [e.value for e in at.exception]


def test_high_frequency_tab_renders_in_both_languages():
    for lang in ("en", "it"):
        at = AppTest.from_file(APP, default_timeout=240)
        at.query_params["lang"] = lang
        at.query_params["view"] = "correlations"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        if lang == "en":
            at.selectbox(key="hf_ind").set_value("m_esi").run()
            assert not at.exception, [e.value for e in at.exception]
