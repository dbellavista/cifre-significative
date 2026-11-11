"""Every decimal figure in a notebook's markdown must be printed by one of its cells (a hand-typed number can match no result).

Reads the committed notebooks only (no data needed). A figure matches when a number in the text outputs rounds to it (also as a share
against a percentage, or millions against billions). Figures that come from a source document, a method parameter or an earlier data vintage are listed in UNPRINTED
with the reason; add a print to the notebook rather than a new entry whenever the figure is computed from the data."""
import bisect
import json
import re
from pathlib import Path

import pytest

NB_DIR = Path("studies/italy_productivity/notebooks")
NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*\.?\d*(?:e-?\d+)?")
DEC = re.compile(r"(?<![\w.])[-+]?\d[\d,]*\.\d+")
UNPRINTED: dict[str, set[str]] = json.loads((Path(__file__).parent / "takeaways_unprinted.json").read_text())


def _texts(path: Path) -> tuple[str, str]:
    md, out = [], []
    for c in json.loads(path.read_text())["cells"]:
        if c["cell_type"] == "markdown":
            md.append("".join(c["source"]))
        for o in c.get("outputs", []):
            out.append("".join(o.get("text", [])) + "".join(o.get("data", {}).get("text/plain", [])))
    return re.sub(r"`[^`]*`", "", "\n".join(md)), "\n".join(out)


def unmatched(path: Path) -> list[str]:
    md, out = _texts(path)
    pool = sorted({abs(float(m.replace(",", "").rstrip("."))) for m in NUM.findall(out) if re.search(r"\d", m)})
    miss = []
    for m in DEC.finditer(md):
        s = m.group(0).replace(",", ""); v, d = abs(float(s)), len(s.split(".")[1])
        for scale in (1, 100, 0.01, 1000, 0.001):  # shares against percentages, millions against billions
            x, tol = v * scale, (0.5 * 10 ** -d + 1e-9) * scale
            i = bisect.bisect_left(pool, x - tol)
            if i < len(pool) and pool[i] <= x + tol:
                break
        else:
            miss.append(m.group(0))
    return miss


@pytest.mark.parametrize("path", sorted(NB_DIR.glob("*.ipynb")), ids=lambda p: p.stem[:2])
def test_takeaway_figures_are_printed(path):
    allowed = set(UNPRINTED.get(path.stem[:2], []))
    extra = sorted(set(unmatched(path)) - allowed)
    assert not extra, f"{path.stem}: figures in the text that no cell prints: {extra}"
