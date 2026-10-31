"""Shift-share of sector productivity: a year missing outside the two compared years must not drop a sector."""
import numpy as np
import pandas as pd
import pytest

from cifresignificative.studies.italy_productivity.analysis.data import (
    PROCESSED,
    load_sectors,
    shift_share,
)


def _frame(drop_year_2025: bool) -> pd.DataFrame:
    rows = [("ITA", y, n, 100.0 + 5 * (y - 2005) * (1 + k), 1000.0 + 10 * k) for k, n in enumerate(["A", "F"]) for y in (2005, 2019, 2025)]
    s = pd.DataFrame(rows, columns=["iso3", "year", "nace_r2", "prod", "hours_ths"])
    if drop_year_2025:
        s.loc[(s.nace_r2 == "F") & (s.year == 2025), ["prod", "hours_ths"]] = np.nan
    return s


def test_missing_other_year_does_not_drop_sector():
    full, gap = shift_share(_frame(False), "ITA", 2005, 2019), shift_share(_frame(True), "ITA", 2005, 2019)
    pd.testing.assert_series_equal(full, gap)


@pytest.mark.skipif(not (PROCESSED / "sectors.parquet").exists(), reason="sectors not built")
def test_every_member_has_a_shift_share():
    from cifresignificative.studies.italy_productivity.analysis.data import EU27_ISO3

    s = load_sectors()
    out = {c: shift_share(s, c, 2005, 2019) for c in EU27_ISO3 if c in s.iso3.unique()}
    assert len(out) == 27 and all(v.notna().all() for v in out.values())
