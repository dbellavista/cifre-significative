import json

import pytest

from cifresignificative.core.clean import jsonstat


def _doc(extra_dim=False):
    dims = ["unit", "geo", "time"]
    units = ["PC", "EUR"] if extra_dim else ["PC"]
    return {
        "id": dims,
        "size": [len(units), 2, 2],
        "dimension": {
            "unit": {"category": {"index": {u: i for i, u in enumerate(units)}}},
            "geo": {"category": {"index": {"IT": 0, "DE": 1}}},
            "time": {"category": {"index": {"2005": 0, "2006": 1}}},
        },
        "value": {"0": 1.0, "1": 2.0, "3": 4.0},
        "status": {"1": "p"},
    }


def test_parse_tidy(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps(_doc()))
    df = jsonstat.parse(f)
    assert len(df) == 3  # position 2 (DE, 2005) is missing
    assert set(df.columns) == {"geo", "year", "value", "flag"}
    assert df.loc[(df.geo == "IT") & (df.year == 2006), "flag"].item() == "p"
    assert df.loc[(df.geo == "DE") & (df.year == 2006), "value"].item() == 4.0


def test_parse_rejects_unfiltered_dimension(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps(_doc(extra_dim=True)))
    with pytest.raises(ValueError, match="unfiltered"):
        jsonstat.parse(f)
