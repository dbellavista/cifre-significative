"""Docs and registries stay in sync (offline: no processed data needed)."""
import importlib.util
from pathlib import Path

import pytest

from cifresignificative import catalog, glossary

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_data_catalog", ROOT / "scripts" / "make_data_catalog.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_static_sections_of_the_data_catalog_are_current():
    gen = _load_generator()
    assert gen.static_sections() in (ROOT / "docs" / "data_catalog.md").read_text(), "run: uv run inv docs.catalog"


def test_check_mode_without_processed_data_checks_static_sections_only(monkeypatch, tmp_path, capsys):
    gen = _load_generator()
    monkeypatch.setattr(gen, "PANEL_PATH", tmp_path / "missing.parquet")
    assert gen.main(check=True) == 0 and "full check skipped" in capsys.readouterr().out
    monkeypatch.setattr(gen, "DOC", tmp_path / "doc.md")
    (tmp_path / "doc.md").write_text("# stale\n")
    assert gen.main(check=True) == 1
    assert gen.main(check=False) == 1  # writing needs the built panel


@pytest.mark.parametrize("indicators", [catalog.EUROSTAT_REGIONAL, catalog.EUROSTAT_HIGHFREQ], ids=["regional", "highfreq"])
def test_every_regional_and_high_frequency_indicator_has_a_glossary_row(indicators):
    g = glossary.load()
    missing = [i.name for i in indicators if i.name not in g.index]
    assert not missing, f"add bilingual glossary rows for {missing}"
    assert all(g.at[i.name, "title_it"] and g.at[i.name, "definition_it"] for i in indicators)


def test_indicator_names_are_unique_and_prefixed():
    names = [i.name for i in catalog.EUROSTAT_REGIONAL + catalog.EUROSTAT_HIGHFREQ]
    assert len(names) == len(set(names))
    assert all(i.name.startswith("nuts2_") for i in catalog.EUROSTAT_REGIONAL)
    assert all(i.name[:2] in ("m_", "q_") and i.freq == i.name[0].upper() for i in catalog.EUROSTAT_HIGHFREQ)
