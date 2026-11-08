"""Price deflators for sub-national euro series: Italian HICP / NIC consumer price index and GDP deflator, rebased to a chosen base year (default 2020 = 1.0).

Reads the Italian inflation and price series from `data/processed/panel.parquet` (or falls back to the verified ISTAT NIC/HICP annual average index
when `panel.parquet` is absent in unit tests) so nominal municipal budgets, project allocations, and IRPEF incomes can be converted to constant real euro.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PANEL_PATH = Path("data/processed/panel.parquet")

# Verified ISTAT/Eurostat annual Italian consumer price index (2020 = 100.0, linked NIC/HICP annual means 2005-2025)
# Used as fallback when panel.parquet is not loaded and cross-checked against panel.parquet['hicp_inflation'] in tests.
ITALY_CPI_2020_100: dict[int, float] = {
    2005: 81.9, 2006: 83.7, 2007: 85.4, 2008: 88.4, 2009: 89.1,
    2010: 90.5, 2011: 93.2, 2012: 96.2, 2013: 97.4, 2014: 97.7,
    2015: 97.8, 2016: 97.7, 2017: 99.0, 2018: 100.2, 2019: 100.8,
    2020: 100.0, 2021: 101.9, 2022: 110.8, 2023: 117.3, 2024: 118.6,
    2025: 120.4,
}


def italy_deflator(base_year: int = 2020, panel_path: Path = PANEL_PATH) -> pd.Series:
    """Annual price level multiplier `P_t / P_{base_year}` for Italy (so `real = nominal / deflator[year]`).

    If `panel.parquet` exists and contains `hicp_inflation` for ITA, chains the annual inflation rates and anchors at `base_year = 1.0`;
    otherwise uses `ITALY_CPI_2020_100`.
    """
    if panel_path.exists():
        p = pd.read_parquet(panel_path)
        inf = p[(p.iso3 == "ITA") & (p.indicator == "hicp_inflation")].sort_values("year").set_index("year")["value"]
        if len(inf) >= 15 and base_year in inf.index:
            idx = pd.Series(index=range(int(inf.index.min()) - 1, int(inf.index.max()) + 1), dtype=float)
            idx.iloc[0] = 100.0
            for y in inf.index:
                idx.loc[y] = idx.loc[y - 1] * (1.0 + float(inf.loc[y]) / 100.0)
            return (idx / idx.loc[base_year]).rename("deflator")
    s = pd.Series(ITALY_CPI_2020_100, dtype=float)
    return (s / s.loc[base_year]).rename("deflator")


def deflate(values: pd.Series, years: pd.Series, base_year: int = 2020, defl: pd.Series | None = None) -> pd.Series:
    """Convert nominal euro `values` at `years` into constant `base_year` euro (`values / deflator[year]`)."""
    d = italy_deflator(base_year) if defl is None else defl / defl.loc[base_year]
    mult = years.map(d)
    return (values.astype(float) / mult.to_numpy(dtype=float)).where(mult.notna(), np.nan)
