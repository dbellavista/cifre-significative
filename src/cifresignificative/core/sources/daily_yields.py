"""Daily 10-year government bond yields (Italy, Germany): manual Investing.com exports in data/external/source/daily_yields/. Daily table, kept apart from the monthly/quarterly `highfreq` contract."""
from pathlib import Path

import pandas as pd

SRC_DIR = Path("data/external/source/daily_yields")
COUNTRIES = {"italy": "ITA", "germany": "DEU", "spain": "ESP", "france": "FRA", "portugal": "PRT", "greece": "GRC"}
INDICATOR = "d_long_term_yield"


def _read_one(path: Path) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig")  # the export starts with a BOM
    s = pd.Series(df["Price"].to_numpy(float), index=pd.to_datetime(df["Date"], format="%m/%d/%Y"), name="value")
    return s.sort_index()


def load(src_dir: Path = SRC_DIR, countries: dict[str, str] = COUNTRIES) -> pd.DataFrame:
    """Daily close (percent), long format `iso3, date, indicator, value, source`. Weekend rows are dropped. Files of one country are chunks of the same series (the site caps exports at 5,000 rows): overlapping dates must agree exactly, otherwise we raise rather than pick one."""
    frames = []
    for stem, iso3 in countries.items():
        parts = [_read_one(p) for p in sorted(src_dir.glob(f"{stem}_10y_*.csv"))]
        if not parts:
            raise FileNotFoundError(f"no {stem}_10y_*.csv in {src_dir} (see {src_dir / 'README.md'})")
        s = pd.concat(parts)
        dup = s[s.index.duplicated(keep=False)]
        if (dup.groupby(level=0).nunique() > 1).any():
            raise ValueError(f"{iso3}: overlapping chunks disagree on {int((dup.groupby(level=0).nunique() > 1).sum())} dates")
        s = s[~s.index.duplicated()].sort_index()
        s = s[s.index.dayofweek < 5]  # the export carries indicative Saturday/Sunday quotes (6% of the Italian and 3% of the German rows, mostly 2013-2023); bond markets are closed, so they are not observations
        frames.append(pd.DataFrame({"iso3": iso3, "date": s.index, "indicator": INDICATOR, "value": s.to_numpy(), "source": "manual:investing.com"}))
    return pd.concat(frames, ignore_index=True)
