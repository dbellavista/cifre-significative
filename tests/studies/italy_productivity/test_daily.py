"""Daily yields: manual-export loader (chunk merge, overlap check) and a cross-check against the monthly Eurostat yield."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.core.sources import daily_yields
from cifresignificative.studies.italy_productivity.analysis import daily as D

HF = Path("data/processed/highfreq.parquet")
TWO = {"italy": "ITA", "germany": "DEU"}  # the synthetic trees hold two countries


def _csv(path: Path, rows):
    path.write_text('﻿"Date","Price","Open","High","Low","Change %"\n' + "".join(f'"{d}","{p}","0","0","0","0%"\n' for d, p in rows))


def _tree(tmp_path, it_overlap=2.0):
    _csv(tmp_path / "italy_10y_a.csv", [("01/04/2005", 3.5), ("01/03/2005", 3.4), ("01/05/2005", 2.0)])  # unsorted on purpose
    _csv(tmp_path / "italy_10y_b.csv", [("01/05/2005", it_overlap), ("01/06/2005", 3.6)])
    _csv(tmp_path / "germany_10y_a.csv", [("01/03/2005", 3.0), ("01/04/2005", 3.1)])
    return tmp_path


def test_load_merges_chunks_and_dedups_overlap(tmp_path):
    df = daily_yields.load(_tree(tmp_path), TWO)
    it = df[df.iso3 == "ITA"]
    assert list(it.date.dt.strftime("%Y-%m-%d")) == ["2005-01-03", "2005-01-04", "2005-01-05", "2005-01-06"]
    assert it.value.tolist() == [3.4, 3.5, 2.0, 3.6]
    assert set(df.indicator) == {daily_yields.INDICATOR} and set(df.iso3) == {"ITA", "DEU"}


def test_load_drops_weekend_quotes(tmp_path):
    _csv(tmp_path / "italy_10y_a.csv", [("01/07/2005", 3.4), ("01/08/2005", 3.9), ("01/10/2005", 3.5)])  # Friday, Saturday, Monday
    _csv(tmp_path / "germany_10y_a.csv", [("01/07/2005", 3.0), ("01/09/2005", 3.1), ("01/10/2005", 3.2)])  # Friday, Sunday, Monday
    df = daily_yields.load(tmp_path, TWO)
    assert df.date.dt.dayofweek.max() < 5 and len(df[df.iso3 == "ITA"]) == 2 and len(df[df.iso3 == "DEU"]) == 2


def test_load_rejects_disagreeing_overlap(tmp_path):
    with pytest.raises(ValueError, match="disagree"):
        daily_yields.load(_tree(tmp_path, it_overlap=2.1), TWO)


def test_load_missing_country_raises(tmp_path):
    _csv(tmp_path / "italy_10y_a.csv", [("01/03/2005", 3.4)])
    with pytest.raises(FileNotFoundError):
        daily_yields.load(tmp_path)


def test_real_files_cover_2005_to_present_without_holes():
    if not list(daily_yields.SRC_DIR.glob("*_10y_*.csv")): pytest.skip("the manual daily-yield exports are not redistributed (see the README in their folder)")
    df = daily_yields.load()
    for iso3, g in df.groupby("iso3"):
        assert g.date.min() == pd.Timestamp("2005-01-03") and g.date.max() >= pd.Timestamp("2026-10-01") and g.date.dt.dayofweek.max() < 5
        # known source holes: Greece has no quotes from late Feb to 23 Jun 2005, Portugal has one 7-day gap in 2007
        assert g[g.date >= "2005-07-01"].date.diff().dt.days.max() <= 7 and g.value.notna().all() and not g.date.duplicated().any()
    assert df[df.iso3 == "ITA"].value.max() > 7  # the Nov 2011 peak is in the sample


@pytest.mark.skipif(not HF.exists(), reason="processed highfreq table missing")
@pytest.mark.skipif(not any(daily_yields.SRC_DIR.glob("*_10y_*.csv")), reason="the manual daily-yield exports are not redistributed")
def test_monthly_mean_of_daily_matches_eurostat_convergence_yield():
    hf = pd.read_parquet(HF)
    dy = daily_yields.load()
    for iso3 in ("ITA", "DEU"):
        m = hf[(hf.indicator == "m_long_term_yield") & (hf.iso3 == iso3)].assign(month=lambda d: pd.to_datetime(d.date).dt.to_period("M")).set_index("month").value
        d = dy[dy.iso3 == iso3].assign(month=lambda x: x.date.dt.to_period("M")).groupby("month").value.mean()
        j = pd.concat([m, d], axis=1, keys=["eurostat", "daily"]).dropna()
        assert len(j) > 200 and j.corr().iloc[0, 1] > 0.99
        assert (j.eurostat - j.daily).abs().median() < 0.1  # different bond/benchmark, same level


def test_rating_actions_file_is_consistent():
    from cifresignificative.studies.italy_productivity.analysis import experiments as ex
    raw = pd.read_csv("data/external/rating_actions.csv", parse_dates=["date"], keep_default_na=False)
    assert set(raw.evidence) <= {"verified", "partly verified", "unverified"} and set(raw.agency) == {"S&P", "Moody's", "Fitch", "DBRS"}
    assert set(raw.action) == {"upgrade", "downgrade", "outlook"} and not raw.duplicated(["date", "agency"]).any() and (raw.sources != "").all()
    assert ((raw.action == "outlook") == (raw.outlook_from.ne(raw.outlook_to) & raw.rating_from.eq(raw.rating_to))).all()  # an "outlook" row changes the outlook only
    assert (raw[raw.action != "outlook"].rating_from != raw[raw.action != "outlook"].rating_to).all()
    ev = ex.rating_actions()
    assert len(ev) == len(raw) and set(ev.type) == {"rating"} and ev.date.is_monotonic_increasing and ev.date.min() == pd.Timestamp("2011-09-19")
    assert len(ex.rating_actions("verified")) <= len(ex.rating_actions("partly verified")) <= len(ev)
    assert (raw[raw.evidence == "verified"].sources.str.contains("moodys.com|fitchratings.com|dbrs.morningstar.com|spglobal.com")).all()  # `verified` means an agency's own record is among the sources
    levels = ex.evidence_levels()
    assert next(iter(levels)) == "all" and all(len(a) > len(b) for a, b in zip(list(levels.values()), list(levels.values())[1:]))  # distinct levels only, broadest first


# --- daily event study: planted effects and null calibration --------------------------------------------------------


def _spread(n=3000, seed=1, jumps=()):
    """Random-walk spread with a volatile crisis block (volatility x5), business-day index; `jumps` = [(position, size)] added to the daily change."""
    rng = np.random.default_rng(seed)
    vol = np.full(n, 0.02)
    vol[800:1000] = 0.10
    dx = rng.normal(0, vol)
    for i, s in jumps:
        dx[i] += s
    return pd.Series(100 + np.cumsum(dx), index=pd.bdate_range("2005-01-03", periods=n), name="spread")


def test_event_snaps_to_next_trading_day_and_drops_out_of_sample_dates():
    x = _spread()
    pos = D._positions(x, ["2005-01-08", "2005-01-03", "1999-01-01", "2100-01-01"])  # a Saturday, the first day, before and after the sample
    assert pos[0] == 5 and pos[1] == 0 and pos[2] is None and pos[3] is None


def test_planted_jump_is_found_in_calm_and_in_volatile_periods():
    x = _spread(jumps=[(1500, 0.3), (900, 1.5)])  # 15 and 15 daily sd of the respective regime
    r = D.event_study_daily(x, [("calm", x.index[1500]), ("crisis", x.index[900]), ("quiet day", x.index[2200])], horizons=(1, 5))
    p = r.pivot(index="event", columns="h", values="p")
    assert (p.loc["calm"] < 0.01).all() and (p.loc["crisis"] < 0.01).all() and (p.loc["quiet day"] > 0.01).all()
    z = r.set_index(["event", "h"]).z
    assert z[("calm", 1)] > 5 and abs(z[("crisis", 1)] - z[("calm", 1)]) < 8  # normalised: a 1.5 move in a x5 vol regime is comparable to 0.3 in a calm one
    assert r.change_pp.iloc[0] > 0.2


def test_placebo_days_exclude_event_windows():
    x = _spread()
    b = D._blocked(len(x), [100], h=5)  # windows [i-1, i+5] and [99, 105] overlap for i in 94..106
    assert b[94:107].all() and not b[[93, 107]].any() and b.sum() == 13


def test_shuffled_dates_give_about_zero_discoveries():
    out = D.placebo_discoveries(_spread(seed=7), n_events=40, n_rep=60, seed=3)
    assert out["mean_discoveries"] < 0.2 and out["share_reps_with_discovery"] < 0.15  # BH at 5%: a null that fires on >15% of reps would flag a bug
    assert 0.002 < out["median_min_p"] < 0.02  # the minimum of 40 events x 3 horizons uniform p-values has median 1 - 0.5**(1/120) = 0.0058


def test_pooled_signed_test_detects_aligned_effect_and_not_noise():
    rng = np.random.default_rng(5)
    pos = np.sort(rng.choice(np.arange(100, 2800, 40), 20, replace=False))
    sign = rng.choice([-1, 1], 20)
    ev = pd.DataFrame({"date": 0, "direction": sign})
    x0 = _spread(seed=11)
    ev["date"] = x0.index[pos]
    planted = _spread(seed=11, jumps=[(p, 0.04 * s) for p, s in zip(pos, sign, strict=True)])  # ~2 daily sd, aligned with direction
    hit = D.pooled_signed(planted, ev, horizons=(1,), n_draws=5000, seed=0).iloc[0]
    null = D.pooled_signed(x0, ev, horizons=(1,), n_draws=5000, seed=0).iloc[0]
    assert hit.p_one_sided < 0.01 and hit.mean_signed_z > 1 and hit.share_expected_sign > 0.7
    assert null.p_one_sided > 0.02 and abs(null.mean_signed_z) < 1


def test_spread_daily_uses_common_days_only():
    dy = pd.DataFrame({"iso3": ["ITA"] * 3 + ["DEU"] * 2, "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-02", "2020-01-03"]),
                       "indicator": "d_long_term_yield", "value": [2.0, 2.5, 2.6, 0.5, 0.4], "source": "x"})
    s = D.spread_daily(dy)
    assert list(pd.DatetimeIndex(s.index).strftime("%m-%d")) == ["01-02", "01-03"] and s.round(2).tolist() == [2.0, 2.2]


def test_daily_events_one_test_per_date_and_ratings_included():
    from cifresignificative.studies.italy_productivity.analysis import experiments as ex
    ev = ex.daily_events()
    assert not ev.date.duplicated().any() and ev.type.str.contains("rating").sum() >= 20 and ev.type.str.contains("ecb").any()
    pepp = ev[ev.date == "2020-03-18"].iloc[0]
    assert "ecb" in pepp.type and "policy" in pepp.type and " / " in pepp.event  # same decision listed in two files, now one test


DY_PARQUET = Path("data/processed/daily_yields.parquet")


@pytest.mark.skipif(not DY_PARQUET.exists(), reason="daily_yields not built")
def test_real_daily_study_runs_fdr_applied_and_null_is_calibrated():
    from cifresignificative.studies.italy_productivity.analysis import experiments as ex
    dy = D.load_daily()
    e = ex.run_daily_events(dy)
    assert len(e) > 150 and {"q_value", "type"} <= set(e) and (e.q_value >= e.p - 1e-12).all() and set(e.h) == set(D.HORIZONS)
    assert e[e.event.str.contains("COVID-19 lockdown")].z.abs().max() > 3  # the March 2020 shock is visible, so the normalisation does not hide real moves
    pl = ex.run_daily_placebo(dy, n_rep=40).iloc[0]
    assert pl.mean_discoveries < 0.3 and pl.share_reps_with_discovery < 0.2
    pooled = ex.run_daily_pooled(dy)
    assert len(pooled) == 6 * len(ex.evidence_levels()) and set(pooled.evidence) == set(ex.evidence_levels()) and (pooled.q_value >= pooled.p_one_sided - 1e-12).all()


@pytest.mark.skipif(not Path("data/processed/daily_yields.parquet").exists(), reason="processed daily table missing")
def test_rating_support_subsets_are_consistent():
    from cifresignificative.studies.italy_productivity.analysis import experiments as ex
    r = ex.run_daily_rating_support(D.load_daily())
    n = r[r.h == 1].set_index("subset").n_events
    assert set(n.index) == set(ex.evidence_levels()) and n.iloc[0] == len(ex.rating_actions()) and set(r.h) == set(D.HORIZONS)


@pytest.mark.skipif(not DY_PARQUET.exists(), reason="daily_yields not built")
def test_peer_spreads_families_and_greece_block():
    from cifresignificative.studies.italy_productivity.analysis import experiments as ex
    dy = D.load_daily()
    n_ev = len(ex.daily_events())
    e = ex.run_daily_peer_events(dy)
    assert set(e.peer) == set(ex.PEERS) and (e.groupby("peer").size() > 0.9 * 3 * n_ev).all() and (e.q_peer >= e.p - 1e-12).all() and (e.q_value >= e.p - 1e-12).all()
    po = ex.run_daily_peer_pooled(dy)
    fam, ref = po[~po.reference], po[po.reference]
    assert len(fam) == 3 * 6 and set(ref.peer) == {"DEU"} and ref.q_value.isna().all() and fam.q_value.notna().all() and (fam.q_value >= fam.p_one_sided - 1e-12).all()
    assert np.allclose(po.mde_z, 1.6449 * po.null_sd, rtol=1e-3) and (po.mde_z_family > po.mde_z).all()
    assert np.allclose(ref[ref.vol_window == 60].set_index("h").mean_signed_z, ex.run_daily_pooled(dy).query("vol_window == 60").set_index("h").mean_signed_z)  # Germany row = the primary result
    pl = ex.run_daily_peer_placebo(dy, n_rep=30)
    assert (pl.mean_discoveries < 0.3).all() and set(pl.peer) == set(ex.PEERS)
    g = ex.run_daily_greece(dy)
    after = g[(g.variant == "after_psi") & (g.block == "events")]
    assert set(g.variant) == {"full", "after_psi"} and (after.date > ex.GREECE_PSI[1]).all() and (g[(g.variant == "full") & (g.block == "events")].date <= ex.GREECE_PSI[1]).any()
