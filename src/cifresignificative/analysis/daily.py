"""Daily sovereign-spread dynamics: event windows in trading days, volatility-normalised, against a placebo of non-event days.
Descriptive only: a spread move around a dated event is an association (other news arrives the same day), never a causal effect."""
from pathlib import Path

import numpy as np
import pandas as pd

from cifresignificative.analysis import corr

DAILY = Path("data/processed/daily_yields.parquet")
HORIZONS = (1, 5, 10)  # trading days after the event day (the event day itself is in the window: announcements may come after the close)
VOL_WINDOW = 60  # trailing days used to scale the change by the recent volatility of daily spread changes
MAX_SNAP_DAYS = 5  # an event further than this from the next trading day is dropped (holiday gaps)


def load_daily(path: Path = DAILY) -> pd.DataFrame:
    return pd.read_parquet(path)


def spread_daily(dy: pd.DataFrame, country: str = "ITA", ref: str = "DEU") -> pd.Series:
    """Yield spread over `ref` in percentage points, on the days both countries have a quote (no filling across holidays)."""
    w = dy.pivot(index="date", columns="iso3", values="value").sort_index()[[country, ref]].dropna()
    return (w[country] - w[ref]).rename(f"spread_{country}_{ref}")


def _stats(x: pd.Series, h: int, vol_window: int) -> tuple[np.ndarray, np.ndarray]:
    """For every day i: raw change x[i+h] - x[i-1], and the same divided by the trailing volatility of daily changes (information up to i-1)
    times sqrt(h + 1), the number of daily changes in the window. The scaled version makes the 2011-12 and 2020 events comparable with calm periods."""
    sd = x.diff().rolling(vol_window, min_periods=vol_window // 2).std().shift(1)
    raw = x.shift(-h) - x.shift(1)
    return raw.to_numpy(), (raw / (sd * np.sqrt(h + 1))).to_numpy()


def _positions(x: pd.Series, dates) -> list[int | None]:
    """Index of the first trading day on or after each date (None when outside the sample or too far from a trading day)."""
    idx = pd.DatetimeIndex(x.index)
    out = []
    for d in dates:
        d = pd.Timestamp(d)
        i = int(np.searchsorted(idx.to_numpy(), d.to_datetime64()))
        out.append(i if i < len(idx) and (idx[i] - d).days <= MAX_SNAP_DAYS and d >= idx[0] else None)
    return out


def _blocked(n: int, positions, h: int) -> np.ndarray:
    """Days whose [i-1, i+h] window overlaps an event window [p-1, p+h], i.e. |i - p| <= h + 1: not usable as placebo days."""
    ind = np.zeros(n)
    ind[[p for p in positions if p is not None]] = 1
    return np.convolve(ind, np.ones(2 * h + 3), mode="same") > 0


def event_study_daily(x: pd.Series, dates, horizons=HORIZONS, vol_window: int = VOL_WINDOW, min_null: int = 250, exclude=None) -> pd.DataFrame:
    """Per event and horizon: raw change in `x` over [t-1, t+h] (t = first trading day on or after the event) and its volatility-normalised
    version z, against the placebo distribution of z over all days whose window overlaps no event in `dates` (and none in `exclude`).
    p is two-sided and exact over the placebo days (+1 smoothing). Input: iterable of (name, date)."""
    named = list(dates)
    pos = _positions(x, [d for _, d in named])
    all_pos = [p for p in pos if p is not None] + [p for p in _positions(x, [] if exclude is None else exclude) if p is not None]
    rows = []
    for h in horizons:
        raw, z = _stats(x, h, vol_window)
        pool = z[~_blocked(len(x), all_pos, h) & np.isfinite(z)]
        if len(pool) < min_null:
            continue
        for (name, when), i in zip(named, pos, strict=True):
            if i is None or not np.isfinite(z[i]):
                continue
            rows.append({"event": name, "date": pd.Timestamp(when), "trading_day": x.index[i], "h": h, "change_pp": float(raw[i]), "z": float(z[i]),
                         "p": float((1 + (np.abs(pool) >= abs(z[i])).sum()) / (1 + len(pool))), "n_placebo": len(pool)})
    return pd.DataFrame(rows)


def pooled_signed(x: pd.Series, events: pd.DataFrame, horizons=HORIZONS, vol_window: int = VOL_WINDOW, n_draws: int = 20000, seed: int = 0, exclude=None) -> pd.DataFrame:
    """Direction-aligned pooled test. `events` has `date` and `direction` (+1: the event should push the spread up, -1: down). Statistic: mean over
    events of direction * z. Null: the same statistic for as many random placebo days with the same directions. One-sided p (H1: mean > 0), Monte Carlo with +1 smoothing."""
    rng = np.random.default_rng(seed)
    pos = _positions(x, events.date)
    keep = [k for k, p in enumerate(pos) if p is not None]
    ev_pos = np.array([p for p in pos if p is not None], dtype=int)
    sign = events.direction.to_numpy(float)[keep]
    all_pos = list(ev_pos) + [p for p in _positions(x, [] if exclude is None else exclude) if p is not None]
    rows = []
    for h in horizons:
        raw, z = _stats(x, h, vol_window)
        ok = np.isfinite(z[ev_pos])
        if ok.sum() < 3:
            continue
        pool = z[~_blocked(len(x), all_pos, h) & np.isfinite(z)]
        s = sign[ok]
        obs = float(np.mean(s * z[ev_pos[ok]]))
        null = (pool[rng.integers(0, len(pool), size=(n_draws, len(s)))] * s).mean(axis=1)
        rows.append({"h": h, "n_events": int(ok.sum()), "mean_signed_z": obs, "mean_signed_change_pp": float(np.mean(s * raw[ev_pos[ok]])),
                     "share_expected_sign": float(np.mean(s * z[ev_pos[ok]] > 0)), "p_one_sided": float((1 + (null >= obs).sum()) / (1 + n_draws)),
                     "null_sd": float(null.std())})
    return pd.DataFrame(rows)


def placebo_discoveries(x: pd.Series, n_events: int = 60, horizons=HORIZONS, n_rep: int = 100, seed: int = 0, q: float = 0.05, vol_window: int = VOL_WINDOW) -> dict:
    """Shuffled-date null: draw `n_events` random dates, run the event study with BH control over all (event, horizon) tests, count discoveries.
    Calibration check: the mean count must be about zero (the exact placebo p-values are uniform under the null, so BH should almost never fire)."""
    rng = np.random.default_rng(seed)
    eligible = x.index[vol_window:-max(horizons) - 1]
    counts, min_p = [], []
    for _ in range(n_rep):
        dates = rng.choice(eligible, size=n_events, replace=False)
        r = event_study_daily(x, [(str(d), d) for d in dates], horizons, vol_window)
        r = corr.fdr(r, q)
        counts.append(int(r.significant.sum()))
        min_p.append(float(r.p.min()))
    return {"n_rep": n_rep, "n_events": n_events, "mean_discoveries": float(np.mean(counts)), "share_reps_with_discovery": float(np.mean(np.array(counts) > 0)),
            "median_min_p": float(np.median(min_p))}
