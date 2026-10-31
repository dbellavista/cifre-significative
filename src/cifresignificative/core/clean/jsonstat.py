"""Parse Eurostat JSON-stat 2.0 responses into tidy long frames."""
import itertools
import json
from pathlib import Path
from typing import cast

import pandas as pd

from cifresignificative.core.countries import EU27, ISO3

# Eurostat geo prefixes of EU27 countries (EL for Greece), used to keep NUTS2 codes
NUTS_COUNTRIES = set(EU27)


def _records(path: Path, keep: tuple[str, ...]):
    """Yield (record, value, flag) for every non-empty cell of a JSON-stat file, after verifying no other free dimension remains."""
    js = json.loads(Path(path).read_text())
    ids, sizes = js["id"], js["size"]
    cats = {}
    for dim in ids:
        idx = js["dimension"][dim]["category"]["index"]
        cats[dim] = [k for k, _ in sorted(idx.items(), key=lambda kv: kv[1])] if isinstance(idx, dict) else list(idx)
    extra = [d for d, s in zip(ids, sizes) if s > 1 and d not in ("geo", "time", *keep)]
    if extra:
        raise ValueError(f"{Path(path).name}: unfiltered dimensions {extra}: {[cats[d] for d in extra]}")
    values, status = js.get("value", {}), js.get("status", {})
    for pos, combo in enumerate(itertools.product(*(cats[d] for d in ids))):
        key = str(pos)
        if key in values:
            yield dict(zip(ids, combo)), values[key], status.get(key)


def parse(path: Path, keep: tuple[str, ...] = ()) -> pd.DataFrame:
    """Return DataFrame[geo, year, value, flag, *keep] after verifying no other free dimension remains."""
    rows = [{"geo": rec["geo"], "year": int(rec["time"][:4]), "value": v, "flag": f, **{k: rec[k] for k in keep}} for rec, v, f in _records(path, keep)]
    return pd.DataFrame(rows, columns=["geo", "year", "value", "flag", *keep])


def period_end(label: str) -> tuple[pd.Timestamp, str]:
    """('2023-01' -> (2023-01-31, 'M'), '2023-Q2' -> (2023-06-30, 'Q'), '2023' -> (2023-12-31, 'A'))."""
    freq = "Q" if "Q" in label else ("M" if "-" in label else "A")
    return cast(pd.Period, pd.Period(label, freq={"A": "Y"}.get(freq, freq))).end_time.normalize(), freq


def parse_periodic(path: Path, keep: tuple[str, ...] = ()) -> pd.DataFrame:
    """Like `parse` for monthly/quarterly files: DataFrame[geo, period, date, freq, value, flag, *keep]; `date` is the period end."""
    rows = []
    for rec, v, f in _records(path, keep):
        date, freq = period_end(rec["time"])
        rows.append({"geo": rec["geo"], "period": rec["time"], "date": date, "freq": freq, "value": v, "flag": f, **{k: rec[k] for k in keep}})
    return pd.DataFrame(rows, columns=["geo", "period", "date", "freq", "value", "flag", *keep])


def to_panel(name: str, theme: str, dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(indicator=name, theme=theme, source=f"eurostat:{dataset}")
    df["iso3"] = df["geo"].map(ISO3)
    return df
