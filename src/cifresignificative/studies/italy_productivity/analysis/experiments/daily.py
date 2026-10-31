"""Daily spreads and rating events: event windows, pooled tests, peers and Greece."""
from pathlib import Path

import pandas as pd
from scipy import stats

from cifresignificative.studies.italy_productivity.analysis import corr
from cifresignificative.studies.italy_productivity.analysis import daily as DY

from .common import OUT
from .highfreq import evidence_levels, italian_events, rating_actions


def daily_events() -> pd.DataFrame:
    """All dated events of the daily study: government, policy, ECB (`italian_events`) plus Italian rating changes. Events on the same date give an
    identical statistic (the same ECB decision sits in both the ECB and policy files), so they are merged into one test: names joined, types joined."""
    ra = rating_actions()
    ev = pd.concat([italian_events(), ra[["event", "date", "type"]]], ignore_index=True)
    return ev.groupby("date", as_index=False).agg(event=("event", lambda e: " / ".join(dict.fromkeys(e))), type=("type", lambda t: "+".join(sorted(set(t))))).reset_index(drop=True)


def run_daily_events(dy: pd.DataFrame, horizons=DY.HORIZONS) -> pd.DataFrame:
    """Event study of the daily ITA-DEU spread for every dated event (one test per date), one BH family over all (event, horizon) tests."""
    ev = daily_events()
    r = DY.event_study_daily(DY.spread_daily(dy), list(zip(ev.event, ev.date, strict=True)), horizons)
    r = r.join(ev.set_index("date").type, on="date")
    return r.assign(q_value=corr.fdr(r, 0.05).q_value).sort_values("p").reset_index(drop=True)


def run_daily_pooled(dy: pd.DataFrame) -> pd.DataFrame:
    """Direction-aligned pooled test for rating changes (downgrade / worse outlook should widen the spread), with robustness over the volatility
    window and over evidence quality; one BH family over all rows. Placebo days avoid every dated event, not only the rating ones."""
    x = DY.spread_daily(dy)
    allev = daily_events()
    rows = []
    for evidence, ra in evidence_levels().items():
        for vol in (60, 20):
            rows.append(DY.pooled_signed(x, ra, vol_window=vol, exclude=allev.date).assign(evidence=evidence, vol_window=vol))
    out = pd.concat(rows, ignore_index=True)
    return out.assign(q_value=corr.fdr(out, 0.05, p_col="p_one_sided").q_value)


def run_daily_rating_support(dy: pd.DataFrame, vol_window: int = 60) -> pd.DataFrame:
    """Sensitivity of the pooled rating test to the evidence level of each date (descriptive, no new FDR family): the distinct levels of `evidence_levels()` x three horizons. With every date verified against the agencies' own records there is a single level."""
    x, allev = DY.spread_daily(dy), daily_events()
    return pd.concat([DY.pooled_signed(x, v, vol_window=vol_window, exclude=allev.date).assign(subset=k) for k, v in evidence_levels().items()], ignore_index=True)


def run_daily_placebo(dy: pd.DataFrame, n_rep: int = 200) -> pd.DataFrame:
    return pd.DataFrame([DY.placebo_discoveries(DY.spread_daily(dy), n_events=len(daily_events()), n_rep=n_rep)])


PEERS = ("ESP", "FRA", "PRT")  # Italy's spread over each peer; Germany stays the primary reference (d1-d3)
GREECE_PSI = (pd.Timestamp("2011-03-01"), pd.Timestamp("2013-03-31"))  # yields above 30%, a different benchmark bond and the March 2012 PSI exchange; the end leaves the 60-day volatility window clear of 2012


def run_daily_peer_events(dy: pd.DataFrame, horizons=DY.HORIZONS) -> pd.DataFrame:
    """Event study of the Italy-minus-peer spread (Spain, France, Portugal) for the same dated events as `run_daily_events`. `q_peer` = BH within each peer's family (279 tests), `q_value` = BH over all peers together (837 tests)."""
    ev = daily_events()
    rows = []
    for peer in PEERS:
        r = DY.event_study_daily(DY.spread_daily(dy, "ITA", peer), list(zip(ev.event, ev.date, strict=True)), horizons).assign(peer=peer)
        rows.append(r.join(ev.set_index("date").type, on="date").assign(q_peer=lambda d: corr.fdr(d, 0.05).q_value))
    out = pd.concat(rows, ignore_index=True)
    return out.assign(q_value=corr.fdr(out, 0.05).q_value).sort_values("p").reset_index(drop=True)


def run_daily_peer_pooled(dy: pd.DataFrame) -> pd.DataFrame:
    """Direction-aligned pooled test for the rating changes on each Italy-minus-peer spread (same events, horizons and volatility windows as `run_daily_pooled`). `q_peer`: BH within a peer (6 rows), `q_value`: over the three peers (18 rows);
    Germany is repeated as `reference` (not part of either family). `mde_z` = smallest mean signed z reachable with one-sided p < 0.05 (1.645 placebo sd), `mde_z_family` the same at Bonferroni over the 18 peer rows."""
    allev, ra = daily_events(), list(evidence_levels().values())[-1]
    rows = []
    for peer in (*PEERS, "DEU"):
        x = DY.spread_daily(dy, "ITA", peer)
        for vol in (60, 20):
            rows.append(DY.pooled_signed(x, ra, vol_window=vol, exclude=allev.date).assign(peer=peer, vol_window=vol, reference=peer == "DEU"))
    out = pd.concat(rows, ignore_index=True)
    fam = out[~out.reference]
    q_all = corr.fdr(fam, 0.05, p_col="p_one_sided").q_value
    q_peer = pd.concat([corr.fdr(g, 0.05, p_col="p_one_sided").q_value for _, g in fam.groupby("peer", sort=False)])
    out["q_value"], out["q_peer"] = q_all.reindex(out.index), q_peer.reindex(out.index)
    return out.assign(mde_z=stats.norm.isf(0.05) * out.null_sd, mde_z_family=stats.norm.isf(0.05 / len(fam)) * out.null_sd)


def run_daily_peer_placebo(dy: pd.DataFrame, n_rep: int = 200) -> pd.DataFrame:
    """Shuffled-date null for every peer spread (93 random dates, BH over the peer's 279 tests): the mean number of discoveries must be about zero."""
    n = len(daily_events())
    return pd.DataFrame([DY.placebo_discoveries(DY.spread_daily(dy, "ITA", peer), n_events=n, n_rep=n_rep) | {"peer": peer} for peer in PEERS])


def run_daily_greece(dy: pd.DataFrame, horizons=DY.HORIZONS) -> pd.DataFrame:
    """Descriptive block, no test family: the Greece-minus-Germany spread around the same Italian events and the pooled rating test, with and without the restructuring years
    (variant `after_psi`: events from 2013-04-01 only and no placebo day within `GREECE_PSI`, so the unstable 2011-12 volatility cannot enter the placebo). Greek quotes in 2011-12 are another benchmark bond (see `data/external/source/daily_yields/README.md`)."""
    x, ev, ra = DY.spread_daily(dy, "GRC", "DEU"), daily_events(), list(evidence_levels().values())[-1]
    psi_days = list(x.index[(x.index >= GREECE_PSI[0]) & (x.index <= GREECE_PSI[1])])
    rows = []
    for variant, keep, excl in (("full", ev, None), ("after_psi", ev[ev.date > GREECE_PSI[1]], psi_days)):
        r = DY.event_study_daily(x, list(zip(keep.event, keep.date, strict=True)), horizons, exclude=excl)
        rows.append(r.assign(variant=variant, block="events"))
        k = ra if variant == "full" else ra[ra.date > GREECE_PSI[1]]
        rows.append(DY.pooled_signed(x, k, exclude=[*ev.date, *(excl or [])]).assign(variant=variant, block="ratings_pooled"))
    return pd.concat(rows, ignore_index=True)


def run_daily_all(dy: pd.DataFrame | None = None, out: Path = OUT) -> dict[str, pd.DataFrame]:
    dy = DY.load_daily() if dy is None else dy
    out.mkdir(parents=True, exist_ok=True)
    results = {"d1_events": run_daily_events(dy), "d2_ratings_pooled": run_daily_pooled(dy), "d3_placebo": run_daily_placebo(dy), "d4_peer_events": run_daily_peer_events(dy),
               "d5_peer_pooled": run_daily_peer_pooled(dy), "d6_peer_placebo": run_daily_peer_placebo(dy), "d7_greece": run_daily_greece(dy)}
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results
