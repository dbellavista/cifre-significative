"""The study registry reads study.toml files and rejects a name that differs from its folder."""
import pytest

from cifresignificative.study import load_studies, load_study


def _write(folder, name):
    folder.mkdir(parents=True)
    (folder / "study.toml").write_text(f'name = "{name}"\ntitle = "T"\ndescription = "D"\nstatus = "draft"\nnotebooks = "nb"\nreports = "rep"\n')


def test_load_studies_skips_the_template(tmp_path):
    _write(tmp_path / "a", "a")
    _write(tmp_path / "_template", "template")
    studies = load_studies(tmp_path)
    assert list(studies) == ["a"] and studies["a"].notebooks == (tmp_path / "a" / "nb").resolve() and studies["a"].publications is None


def test_name_must_match_folder(tmp_path):
    _write(tmp_path / "a", "b")
    with pytest.raises(ValueError):
        load_study(tmp_path / "a")


def test_cells_folder_is_optional(tmp_path):
    _write(tmp_path / "a", "a")
    assert load_study(tmp_path / "a").cells is None
    (tmp_path / "a" / "study.toml").write_text((tmp_path / "a" / "study.toml").read_text() + 'cells = "nc"\n')
    assert load_study(tmp_path / "a").cells == (tmp_path / "a" / "nc").resolve()


def test_registered_dashboards_expose_the_shell_interface():
    pytest.importorskip("streamlit")
    import importlib

    for s in load_studies().values():
        if s.dashboard:
            m = importlib.import_module(s.dashboard)
            assert m.VIEWS and all(len(v) == 3 and callable(v[2]) for v in m.VIEWS.values()) and len(m.TITLE) == 2 and len(m.CAPTION) == 2
            assert callable(m.load) and callable(m.data_error)


def test_new_study_scaffold_registers_and_rejects_bad_names(tmp_path):
    import importlib.util
    import shutil
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("new_study", ROOT / "scripts/new_study.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scaffold = mod.scaffold

    (tmp_path / "studies").mkdir()
    shutil.copytree(ROOT / "studies/_template", tmp_path / "studies/_template")
    scaffold("sample", 'Sample "city"', "Question.", root=tmp_path)
    s = load_study(tmp_path / "studies/sample")
    assert s.name == "sample" and s.title == 'Sample "city"' and s.status == "draft"
    assert (tmp_path / "src/cifresignificative/studies/sample/analysis/__init__.py").exists() and (tmp_path / "tests/studies/sample/test_registry.py").exists()
    with pytest.raises(FileExistsError):
        scaffold("sample", "T", "D", root=tmp_path)
    with pytest.raises(ValueError):
        scaffold("Bad-Name", "T", "D", root=tmp_path)
