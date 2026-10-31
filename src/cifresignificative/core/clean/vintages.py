"""Vintages: keep a dated snapshot whenever a built table changes, and diff two snapshots (panel, regional, regional_extra, highfreq).

Statistical agencies revise recent years; snapshots make it possible to see what moved between data pulls.
"""
import hashlib
from datetime import date
from pathlib import Path

import pandas as pd

VINTAGE_DIR = Path("data/processed/vintages")
KEEP = 12


# table -> (key columns, columns kept in the snapshot)
TABLES = {
    "panel": (["indicator", "geo", "year"], ["iso3", "geo", "year", "indicator", "value", "flag", "source"]),
    "regional": (["indicator", "geo", "year"], ["geo", "country", "year", "indicator", "value", "flag", "source"]),
    "regional_extra": (["indicator", "geo", "year"], ["geo", "country", "year", "indicator", "value", "flag", "source"]),
    "regional_nuts2021": (["indicator", "geo", "year"], ["geo", "geo_old", "geo_method", "country", "year", "indicator", "value", "flag", "source"]),
    "highfreq": (["indicator", "geo", "period"], ["iso3", "geo", "period", "date", "freq", "indicator", "value", "flag", "source"]),
}


def _digest(df: pd.DataFrame, key: list[str]) -> str:
    k = df.sort_values(key)[[*key, "value"]].round(6)
    return hashlib.sha1(pd.util.hash_pandas_object(k, index=False).values.tobytes()).hexdigest()[:12]


def snapshot(df: pd.DataFrame, name: str = "panel") -> Path | None:
    """Write data/processed/vintages/<name>_<date>_<hash>.parquet unless the latest snapshot has the same content. Returns the path or None."""
    key, cols = TABLES[name]
    VINTAGE_DIR.mkdir(parents=True, exist_ok=True)
    digest = _digest(df, key)
    existing = sorted(VINTAGE_DIR.glob(f"{name}_2*.parquet"))  # "2*" keeps regional and regional_extra apart
    if existing and existing[-1].stem.endswith(digest):
        return None
    path = VINTAGE_DIR / f"{name}_{date.today():%Y%m%d}_{digest}.parquet"
    df[cols].to_parquet(path, index=False)
    for old in sorted(VINTAGE_DIR.glob(f"{name}_2*.parquet"))[:-KEEP]:
        old.unlink()
    return path


def diff(old: pd.DataFrame, new: pd.DataFrame, rel_tol: float = 0.005, abs_tol: float = 0.05, key: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Compare two snapshots: revised points, added points and dropped points (default keys: indicator, geo, year; highfreq uses period)."""
    key = key or ["indicator", "geo", "year"]
    m = old[[*key, "value"]].merge(new[[*key, "value"]], on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
    both = m[m._merge == "both"].assign(change=lambda d: d.value_new - d.value_old)
    thr = abs_tol + rel_tol * both.value_old.abs()
    return {"revised": both[both.change.abs() > thr].drop(columns="_merge").sort_values("change", key=abs, ascending=False),
            "added": m[m._merge == "right_only"].drop(columns=["_merge", "value_old"]),
            "dropped": m[m._merge == "left_only"].drop(columns=["_merge", "value_new"])}
