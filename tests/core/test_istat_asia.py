"""Unit tests for the ISTAT ASIA loaders (`core.sources.istat_asia`), on synthetic SDMX CSVs."""
from pathlib import Path

import pandas as pd

from cifresignificative.core.sources import istat_asia

COLS = ["DATAFLOW", "FREQ", "REF_AREA", "DATA_TYPE", "ECON_ACTIVITY_NACE_2007", "PERS_EMPL_SIZE_CLASS", "TIME_PERIOD", "OBS_VALUE"]


def _write(path: Path, rows: list[tuple]) -> None:
    pd.DataFrame([("f", "A", *r) for r in rows], columns=COLS).to_csv(path, index=False)


def test_load_areas_keeps_total_apart_from_food_and_translates_nuts(tmp_path: Path) -> None:
    """With numeric divisions only, a naive read would turn '0010' (total) into 10 (food) and average them."""
    rows = [
        ("ITD58", "LUEMPDAA", "0010", "TOTAL", 2021, 100000.0),
        ("ITD58", "LUEMPDAA", "10", "TOTAL", 2021, 7310.0),
        ("ITD58", "LUEMPDAA", "10", "TOTAL", 2022, 4443.38),
        ("ITD58", "LU", "10", "TOTAL", 2022, 700.0),
        ("ITD58", "LUEMPDAA", "10", "W0_9", 2022, 999.0),  # other size class: dropped
        ("ITD58", "LUEMPDAA", "A", "TOTAL", 2022, 5.0),  # section: dropped
    ]
    _write(istat_asia.areas_raw_path(tmp_path), rows)
    d = istat_asia.load_areas(tmp_path)
    food = d[d.ateco_2d == "10"].set_index("year")
    assert food.loc[2021, "employees"] == 7310.0
    assert food.loc[2022, "employees"] == 4443.38 and food.loc[2022, "local_units"] == 700.0
    assert d[d.ateco_2d == "TOTAL"].employees.item() == 100000.0
    assert set(d.nuts) == {"ITH58"} and set(d.area) == {"ITD58"}
    assert set(d.ateco_2d) == {"10", "TOTAL"}


def test_area_codes_are_nuts_2010() -> None:
    assert "ITD58" in istat_asia.AREA_CODES and "ITH58" not in istat_asia.AREA_CODES
    assert len([c for c in istat_asia.AREA_CODES if len(c) == 5]) == 9
