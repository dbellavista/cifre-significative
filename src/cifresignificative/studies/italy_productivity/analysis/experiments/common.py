"""Shared paths and settings of the experiment runners."""


from cifresignificative.studies.italy_productivity.analysis.data import (
    PROCESSED,
)

OUT = PROCESSED / "correlations"
INCOME = "gdp_real_pc"  # development control for cross-sections (logged)
WINDOWS = {"recent": (2015, 2019), "early": (2005, 2009)}
