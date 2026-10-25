"""Execute every notebook top to bottom (slow: ~2-4 minutes). Opt in with RUN_SLOW=1; needs the processed data."""
import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

pytestmark = [pytest.mark.slow, pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1" or not Path("data/processed/panel.parquet").exists(), reason="set RUN_SLOW=1 with built data")]


@pytest.mark.parametrize("path", sorted(Path("notebooks").glob("*.ipynb")), ids=lambda p: p.stem)
def test_notebook_runs(path):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(Path.cwd())}}).execute()
