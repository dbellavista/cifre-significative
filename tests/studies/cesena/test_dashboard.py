"""Smoke tests: the Streamlit app renders every view of the `cesena` study without exceptions."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path("src/cifresignificative/dashboard/app.py").resolve())
pytestmark = pytest.mark.skipif(not Path("data/processed/cesena/panel_comuni.parquet").exists(), reason="cesena data not built")
VIEWS = ["overview", "fiscal", "investments", "economy", "outcomes", "evaluation", "about"]


@pytest.mark.parametrize("view", VIEWS)
def test_cesena_view_renders(view: str) -> None:
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["study"] = "cesena"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.sidebar.radio(key="view").set_value(view).run()
    assert not at.exception, [e.value for e in at.exception]


def test_cesena_italian_interface() -> None:
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["study"] = "cesena"
    at.query_params["lang"] = "it"
    at.query_params["view"] = "outcomes"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.sidebar.radio(key="view").value == "outcomes"
    assert any("Comprensorio" in s.value for s in at.subheader)
