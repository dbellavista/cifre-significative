"""Generate the analysis notebooks (nbformat) so they stay reproducible and diff-able.

Usage: uv run python scripts/make_notebooks.py [--force] [--study NAME] [name ...] && uv run jupyter nbconvert --to notebook --execute --inplace studies/<study>/notebooks/<name>.ipynb

The cells of a study live in studies/<study>/notebook_cells/ (one module per notebook, `NOTEBOOKS[name]`). A notebook is rewritten only
when its cell sources differ from the file on disk (or with --force), so executed outputs of unchanged notebooks are never wiped.
Pass notebook names to restrict the run. Re-execute the ones reported as written.
"""
import importlib.util
import sys

import nbformat as nbf

from cifresignificative.study import Study, load_studies

DEFAULT_SETUP = '''import os, warnings
from pathlib import Path
while not Path("pyproject.toml").exists() and Path.cwd() != Path.cwd().parent:
    os.chdir(Path.cwd().parent)  # run from the project root so data/ paths resolve (the notebook sits in studies/<name>/notebooks)
warnings.filterwarnings("ignore")
'''


def load_cells(study: Study):
    """Import studies/<name>/notebook_cells as a package (it is not on the import path); None when the study has no cells."""
    if study.cells is None or not (study.cells / "__init__.py").exists():
        return None
    name = f"notebook_cells_{study.name}"
    spec = importlib.util.spec_from_file_location(name, study.cells / "__init__.py", submodule_search_locations=[str(study.cells)])
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def build(study: Study, name: str, cells: list[tuple[str, str]], setup: str, force: bool = False) -> bool:
    """Write <study>/notebooks/<name>.ipynb; returns False (leaving the executed file untouched) when the sources are unchanged."""
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_code_cell(setup)]
    for kind, text in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    path = study.notebooks / f"{name}.ipynb"
    if path.exists() and not force:
        old = nbf.read(path, 4)
        if [(c.cell_type, c.source) for c in old.cells] == [(c.cell_type, c.source) for c in nb.cells]:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    return True


argv = sys.argv[1:]
only = argv[argv.index("--study") + 1] if "--study" in argv else None
args = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--study")]
if only and only not in load_studies():
    sys.exit(f"unknown study: {only}")
studies = load_studies()
registry = {s.name: m for s in studies.values() if (not only or s.name == only) and (m := load_cells(s)) is not None}
known = {n for m in registry.values() for n in m.NOTEBOOKS}
if set(args) - known:
    sys.exit(f"unknown notebook(s): {', '.join(sorted(set(args) - known))}")
written = [f"{sname}/{n}" for sname, mod in registry.items() for n, c in mod.NOTEBOOKS.items()
           if (not args or n in args) and build(studies[sname], n, c, getattr(mod, "SETUP", DEFAULT_SETUP), "--force" in argv)]
print("wrote (re-execute these):", ", ".join(written) or "nothing, all notebooks up to date")
