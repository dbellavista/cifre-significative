"""Cell definitions of the notebooks of the `cesena` study: NOTEBOOKS[name] = [("md"|"code", text), ...], one module per notebook.

Read by `scripts/make_notebooks.py`; the first cell of every notebook is `SETUP`."""
import importlib

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {}
M, C = "md", "code"
SETUP = '''import os, warnings
from pathlib import Path
while not Path("pyproject.toml").exists() and Path.cwd() != Path.cwd().parent:
    os.chdir(Path.cwd().parent)  # run from the project root so data/ paths resolve (the notebook sits in studies/<name>/notebooks)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
from cifresignificative.core.viz.style import apply_style, ITALY_C, EU_C, PEER_C
apply_style()
'''

for _name in [
    "01_fiscal_policy",
    "02_public_investments",
    "03_real_economy_shift_share",
    "04_income_demography",
    "05_policy_evaluation",
]:
    importlib.import_module(f"{__name__}.n{_name}")
