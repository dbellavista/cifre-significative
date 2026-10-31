"""Cell definitions of the notebooks of this study: NOTEBOOKS[name] = [("md"|"code", text), ...], one module per notebook.

Read by scripts/make_notebooks.py. Add `nNN_name.py` with `from . import NOTEBOOKS, C, M` and `NOTEBOOKS["NN_name"] = [...]`, then import it at the bottom of this file.
`SETUP` is the first cell of every notebook (optional: the default only sets the project root and the warnings)."""
NOTEBOOKS: dict[str, list[tuple[str, str]]] = {}
M, C = "md", "code"
