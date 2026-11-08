"""`findings.md` must quote only computed figures: every number carries a tag, and every tag matches `claims.parquet`."""
import re
from pathlib import Path

import pandas as pd
import pytest

FINDINGS = Path("studies/cesena/reports/findings.md")
CLAIMS = Path("data/processed/cesena/claims.parquet")
TAG = re.compile(r"<!-- c:(\w+) -->")
# A figure a reader would take for a result: decimals, percentages, euro, thousands, ranks.
FIGURE = re.compile(r"\d+\.\d+|\d+%|€\s?\d|\d,\d{3}|#\d+")
CAUSAL = re.compile(r"\b(prov(?:es?|ing|en)|proof|explains?|explained by|driven by|natural experiment|confirms?|confirmed|resolves?|refut\w+|caused? by)\b", re.IGNORECASE)

needs_claims = pytest.mark.skipif(not CLAIMS.exists(), reason="run the cesena build first")


def _strip_tagged(text: str, claims: dict[str, str]) -> tuple[str, list[str]]:
    """Remove each tagged figure from the text; return what is left and the mismatches."""
    errors: list[str] = []
    out, pos = [], 0
    for m in TAG.finditer(text):
        before = text[pos:m.start()]
        expected = claims.get(m.group(1))
        if expected is None:
            errors.append(f"unknown tag {m.group(1)}")
            out.append(before)
        elif not before.endswith(expected):
            errors.append(f"{m.group(1)}: expected {expected!r}, text has {before[-25:]!r}")
            out.append(before)
        else:
            out.append(before[: -len(expected)])
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), errors


@needs_claims
def test_every_tag_matches_the_computed_claim():
    claims = pd.read_parquet(CLAIMS).set_index("id")["text"].to_dict()
    _, errors = _strip_tagged(FINDINGS.read_text(), claims)
    assert not errors, errors


@needs_claims
def test_no_untagged_figures():
    claims = pd.read_parquet(CLAIMS).set_index("id")["text"].to_dict()
    rest, _ = _strip_tagged(FINDINGS.read_text(), claims)
    assert not FIGURE.findall(rest), FIGURE.findall(rest)


def test_no_causal_wording():
    assert not CAUSAL.findall(FINDINGS.read_text()), CAUSAL.findall(FINDINGS.read_text())


def test_strip_tagged_catches_a_wrong_number():
    _, errors = _strip_tagged("grew +4.70%<!-- c:a --> and <!-- c:b -->", {"a": "+4.76%"})
    assert len(errors) == 2


def test_untagged_figure_is_found():
    rest, errors = _strip_tagged("grew +4.76%<!-- c:a --> and +3.1% more", {"a": "+4.76%"})
    assert not errors
    assert FIGURE.findall(rest) == ["3.1", "3.1%"] or FIGURE.findall(rest) == ["3.1"]
