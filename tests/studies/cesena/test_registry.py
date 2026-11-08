"""The study is registered and its code package imports."""
from cifresignificative.study import load_studies


def test_cesena_is_registered():
    s = load_studies()["cesena"]
    assert s.title and s.notebooks.exists()


def test_cesena_package_imports():
    import cifresignificative.studies.cesena.analysis  # noqa: F401
