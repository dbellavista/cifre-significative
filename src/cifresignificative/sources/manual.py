"""Hand-entered series (data/external/*.csv) that no API serves: PISA 2022, PIAAC. See docs/manual_data.md."""
from pathlib import Path

import pandas as pd

from cifresignificative.countries import ISO3

EXTERNAL = Path("data/external")
REQUIRED = ["iso3", "year", "indicator", "value", "theme", "source"]


def load() -> pd.DataFrame:
    """Concatenate all filled-in rows; rows with an empty `value` are template placeholders and are skipped."""
    geo_of = {v: k for k, v in ISO3.items()}
    frames = []
    for path in sorted(EXTERNAL.glob("*.csv")):
        df = pd.read_csv(path, dtype={"source": str})
        if "indicator" not in df.columns:  # reference tables (governments, policies) live here too but are not panel series
            continue
        missing = set(REQUIRED) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        df = df.dropna(subset=["value"])
        unknown = set(df.iso3) - set(geo_of)
        if unknown:
            raise ValueError(f"{path.name}: unknown iso3 codes {sorted(unknown)}")
        if df.source.isna().any():
            raise ValueError(f"{path.name}: every filled row needs a `source` (table name or URL)")
        frames.append(df.assign(geo=df.iso3.map(geo_of), flag=None, source="manual:" + df.source)[
            ["iso3", "geo", "year", "indicator", "value", "flag", "theme", "source"]
        ])
    if not frames:
        return pd.DataFrame(columns=["iso3", "geo", "year", "indicator", "value", "flag", "theme", "source"])
    return pd.concat(frames, ignore_index=True)
