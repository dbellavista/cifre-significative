"""Execute every notebook in `studies/cesena/notebooks/` top to bottom (opt in with `RUN_SLOW=1`)."""
import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_SLOW") != "1" or not Path("data/processed/cesena/panel_comuni.parquet").exists(),
        reason="set RUN_SLOW=1 with built cesena data",
    ),
]


@pytest.mark.parametrize("path", sorted(Path("studies/cesena/notebooks").glob("*.ipynb")), ids=lambda p: p.stem)
def test_cesena_notebook_runs(path: Path):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(Path.cwd())}}).execute()
