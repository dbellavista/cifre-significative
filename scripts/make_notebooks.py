"""Generate the analysis notebooks (nbformat) so they stay reproducible and diff-able.

Usage: uv run python scripts/make_notebooks.py [--force] [name ...] && uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb

A notebook is rewritten only when its cell sources differ from the file on disk (or with --force), so executed outputs of
unchanged notebooks are never wiped. Pass notebook names to restrict the run. Re-execute the ones reported as written.
"""
import sys
from pathlib import Path

import nbformat as nbf
from notebook_cells import (
    NOTEBOOKS,  # sibling module (scripts/ is on sys.path when run as a script)
)

SETUP = '''import os, warnings
from pathlib import Path
if Path.cwd().name == "notebooks":
    os.chdir(Path.cwd().parent)  # run from the project root so data/ paths resolve
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
from cifresignificative.analysis.data import *
from cifresignificative.viz.style import *
apply_style()
panel = load_panel()
'''


def build(name: str, cells: list[tuple[str, str]], force: bool = False) -> bool:
    """Write notebooks/<name>.ipynb; returns False (leaving the executed file untouched) when the sources are unchanged."""
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_code_cell(SETUP)]
    for kind, text in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    path = Path("notebooks") / f"{name}.ipynb"
    if path.exists() and not force:
        old = nbf.read(path, 4)
        if [(c.cell_type, c.source) for c in old.cells] == [(c.cell_type, c.source) for c in nb.cells]:
            return False
    path.parent.mkdir(exist_ok=True)
    nbf.write(nb, path)
    return True


args = [a for a in sys.argv[1:] if not a.startswith("--")]
unknown = set(args) - set(NOTEBOOKS)
if unknown:
    sys.exit(f"unknown notebook(s): {', '.join(sorted(unknown))}")
written = [n for n, c in NOTEBOOKS.items() if (not args or n in args) and build(n, c, "--force" in sys.argv)]
print("wrote (re-execute these):", ", ".join(written) or "nothing, all notebooks up to date")
