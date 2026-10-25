"""High-frequency framework: periodic parsing, request parameters, planted lead-lag and event effects, spurious-regression guard."""
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from cifresignificative.analysis import experiments as ex
from cifresignificative.analysis import hf as H
from cifresignificative.catalog import EUROSTAT_HIGHFREQ, Indicator
from cifresignificative.clean import jsonstat
from cifresignificative.sources import eurostat

HF = Path("data/processed/highfreq.parquet")


def _doc(times, geos=("IT", "DE")):
    n = len(times)
    return {"id": ["geo", "time"], "size": [len(geos), n],
            "dimension": {"geo": {"category": {"index": {g: i for i, g in enumerate(geos)}}}, "time": {"category": {"index": {t: i for i, t in enumerate(times)}}}},
            "value": {str(i): float(i) for i in range(len(geos) * n) if i != 1}, "status": {"2": "p"}}


def test_parse_periodic_monthly_and_quarterly(tmp_path):
    f = tmp_path / "m.json"
    f.write_text(json.dumps(_doc(["2023-01", "2023-02"])))
    m = jsonstat.parse_periodic(f)
    assert len(m) == 3 and set(m.freq) == {"M"} and not m.duplicated(["geo", "period"]).any()
    assert m.date.min() == pd.Timestamp("2023-01-31") and m.loc[m.flag == "p", "geo"].item() == "DE"
    f.write_text(json.dumps(_doc(["2023-Q1", "2023-Q2"])))
    q = jsonstat.parse_periodic(f)
    assert set(q.freq) == {"Q"} and q.date.max() == pd.Timestamp("2023-06-30")
    assert jsonstat.period_end("2023")[1] == "A"


def test_request_starts_at_first_period_of_each_frequency():
    for freq, first in [("A", "2005"), ("M", "2005-01"), ("Q", "2005-Q1")]:
        params = dict(eurostat.build_params(Indicator("x", "t", "ds", {"unit": "PC"}, freq=freq)))
        assert params["sinceTimePeriod"] == first
    assert all(i.freq in ("M", "Q") and i.name[:2] in ("m_", "q_") for i in EUROSTAT_HIGHFREQ)
    assert len({i.name for i in EUROSTAT_HIGHFREQ}) == len(EUROSTAT_HIGHFREQ)


def ar1(n, phi, rng):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x


def test_cross_corr_recovers_planted_lead_not_reverse():
    rng = np.random.default_rng(1)
    x = ar1(300, 0.8, rng)
    y = np.r_[np.zeros(3), 0.8 * x[:-3]] + rng.normal(scale=0.5, size=300)
    idx = pd.date_range("2000-01-31", periods=300, freq="ME")
    cc = H.cross_corr(pd.Series(x, idx), pd.Series(y, idx), max_lag=8)
    assert H.best_lead(cc)["lag"] == 3 and H.best_lead(cc)["significant"]
    assert H.best_lead(cc, min_lag=-8)["lag"] == 3  # no stronger relation with y leading x
    g = H.granger(pd.Series(x, idx), pd.Series(y, idx))
    assert g["p_x_leads_y"] < 0.001 < g["p_y_leads_x"]


def test_independent_random_walks_are_not_discoveries_after_transform():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2000-01-31", periods=240, freq="ME")
    hits = 0
    for _ in range(40):
        a, b = (pd.Series(rng.normal(size=240).cumsum(), idx) for _ in range(2))
        g = H.granger(H.transform(a, "diff"), H.transform(b, "diff"))
        hits += g["p_x_leads_y"] < 0.05
    assert hits <= 8  # about 5% expected; raw levels would give most pairs 'significant'
    raw = sum(abs(pd.concat([pd.Series(rng.normal(size=240).cumsum()), pd.Series(rng.normal(size=240).cumsum())], axis=1).corr().iloc[0, 1]) > 0.3 for _ in range(40))
    assert raw > 8  # the levels are the trap the transform avoids


def test_event_study_detects_jump_and_is_calibrated_on_random_dates():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2000-01-31", periods=260, freq="ME")
    x = pd.Series(ar1(260, 0.5, rng), idx)
    x.iloc[130:] += 6  # a permanent jump after month 130
    hit = H.event_study(x, [("jump", idx[130] - pd.offsets.MonthEnd(0))]).iloc[0]
    assert hit.p < 0.02 and hit.change > 4
    noise = pd.Series(ar1(260, 0.5, rng), idx)
    p = H.event_study(noise, [(f"e{i}", d) for i, d in enumerate(idx[20:240:5])]).p
    assert (p < 0.05).mean() < 0.2 and p.median() > 0.25


def _panel(lead: int, n_countries=14, n=150, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    for c in range(n_countries):
        x = ar1(n, 0.6, rng)
        y = 0.5 * np.r_[np.zeros(lead), x[:-lead]] + 0.3 * ar1(n, 0.4, rng) + rng.normal(scale=0.5, size=n)
        rows.append(pd.DataFrame({"iso3": f"C{c:02d}", "date": idx, "x": x, "y": y}))
    return pd.concat(rows).set_index(["iso3", "date"])


def test_panel_granger_finds_x_leads_y():
    t = H.lead_lag_table(_panel(2), [("x", "y")], lags=(2,)).iloc[0]
    assert t.verdict == "x leads" and t.coef_x_leads_y == pytest.approx(0.5, abs=0.15)


def test_hf_panel_keeps_gaps_so_lags_do_not_jump():
    d = pd.date_range("2020-01-31", periods=6, freq="ME")
    rows = [{"iso3": "ITA", "geo": "IT", "period": str(t)[:7], "date": t, "freq": "M", "indicator": "m_esi", "value": float(i), "flag": None, "theme": "s", "source": "t"}
            for i, t in enumerate(d) if i != 2]  # March missing
    M = H.hf_panel(pd.DataFrame(rows), ["m_esi"], how={"m_esi": "level"})
    assert len(M) == 6 and np.isnan(M.loc[("ITA", d[2]), "m_esi"])


def test_italy_rank():
    df = pd.DataFrame({"iso3": ["ITA", "DEU", "FRA", "ESP"], "r": [0.3, -0.5, 0.1, 0.2]})
    assert H.italy_rank(df) == {"italy_value": 0.3, "rank": 2, "n_countries": 4}


def test_events_file_covers_government_policy_and_ecb():
    ev = ex.italian_events()
    assert set(ev.type) == {"government", "policy", "ecb"} and ev.date.min() >= pd.Timestamp("2005-01-01") and len(ev) > 40


@pytest.mark.skipif(not HF.exists(), reason="highfreq not built")
def test_real_highfreq_table_keys_and_known_values():
    hf = pd.read_parquet(HF)
    assert not hf.duplicated(["iso3", "indicator", "period"]).any()
    spread = H.spread_series(hf)
    assert 4.5 < spread.max() < 6.5 and cast(pd.Timestamp, spread.idxmax()).year == 2011  # sovereign-debt crisis peak, Nov 2011
    u = H.wide_hf(hf, "m_unemployment_rate")["ITA"]
    assert u.idxmax().year == 2014 and 12.5 < u.max() < 14
    assert hf[hf.indicator == "m_hicp_inflation"].period.max() >= "2026-01"  # ECOICOP-2 dataset carries the full history to the present


@pytest.mark.skipif(not HF.exists(), reason="highfreq not built")
def test_real_hf_event_study_runs_and_fdr_is_applied():
    hf = pd.read_parquet(HF)
    ev = ex.run_hf_events(hf)
    assert len(ev) > 100 and {"q_value", "p", "z", "series", "type"} <= set(ev.columns) and (ev.q_value >= ev.p).all()
