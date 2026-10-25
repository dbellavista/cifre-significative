"""Plain-language glossary of the indicators and concepts used in the panel (data/external/glossary.csv, English and Italian)."""
from functools import lru_cache
from pathlib import Path

import pandas as pd

PATH = Path(__file__).resolve().parents[2] / "data" / "external" / "glossary.csv"


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    return pd.read_csv(PATH, keep_default_na=False).set_index("key", drop=False)


def title(key: str, lang: str = "en") -> str:
    g = load()
    if key in g.index and g.at[key, f"title_{lang}"]:
        return g.at[key, f"title_{lang}"]
    return key.replace("_", " ")


def describe(key: str, lang: str = "en") -> str:
    """One line for a caption: definition followed by the unit, or '' when the key has no entry."""
    g = load()
    if key not in g.index:
        return ""
    r = g.loc[key]
    unit = f" *({r.unit})*" if r.unit else ""
    return f"{r[f'definition_{lang}']}{unit}"
