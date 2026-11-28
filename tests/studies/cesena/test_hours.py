"""Hours-versus-pay decomposition: exact identity on a planted case, unit ranking, and the local data when present."""
import numpy as np
import pandas as pd
import pytest

from cifresignificative.core.sources import inps_osservatorio as inps
from cifresignificative.studies.cesena.analysis import hours


def _long(spec: dict) -> pd.DataFrame:
    """spec[(year, province)] = (workers_no, pay_no, days_no, workers_si, pay_si, days_si)"""
    rows = []
    for (y, p), (wn, pn, dn, ws, ps, ds) in spec.items():
        rows += [(y, p, "No", wn, pn, dn), (y, p, "Si", ws, ps, ds), (y, p, "Totale", wn + ws, pn + ps, dn + ds)]
    d = pd.DataFrame(rows, columns=["year", "province", "part_time", "workers", "pay_eur", "paid_days"])
    d["kind"] = "province"
    return d


def test_decomposition_is_exact_and_signs_planted():
    # group pay per worker fixed (No 30000, Si 12000); only the part-time share moves 20% -> 40%: within 0, mix negative
    spec = {(2008, "A"): (80, 80 * 30000, 80 * 250, 20, 20 * 12000, 20 * 210), (2024, "A"): (60, 60 * 30000, 60 * 250, 40, 40 * 12000, 40 * 210)}
    p = hours.province_panel(_long(spec))
    dec = hours.mix_decomposition(p, 2008, 2024)
    assert dec.loc["A", "within_pct"] == pytest.approx(0, abs=1e-9)
    assert dec.loc["A", "mix_pct"] == pytest.approx(dec.loc["A", "pay_per_worker_change_pct"]) and dec.loc["A", "mix_pct"] < 0
    assert dec.loc["A", "share_pt_change_pp"] == pytest.approx(20)
    # the identity (1+within)(1+mix) = 1+total holds with moving group pay too
    spec[(2024, "A")] = (60, 60 * 36000, 60 * 252, 40, 40 * 14000, 40 * 215)
    dec = hours.mix_decomposition(hours.province_panel(_long(spec)), 2008, 2024).loc["A"]
    assert (1 + dec.within_pct / 100) * (1 + dec.mix_pct / 100) == pytest.approx(1 + dec.pay_per_worker_change_pct / 100)


def test_real_changes_deflate_and_keep_identity():
    # No pay +20%, Si pay +20%, share fixed; prices +20%: real changes are zero, mix is zero
    spec = {(2008, "A"): (80, 80 * 30000, 80 * 250, 20, 20 * 12000, 20 * 210), (2024, "A"): (80, 80 * 36000, 80 * 250, 20, 20 * 14400, 20 * 210)}
    p = hours.province_panel(_long(spec))
    r = hours.real_pay_changes(p, 2008, 2024, pd.Series({2008: 1.0, 2024: 1.2})).loc["A"]
    for c in ("pay_per_worker_real_pct", "within_real_pct", "mix_pct", "pay_per_worker_no_real_pct", "pay_per_worker_si_real_pct", "days_per_worker_no_change_pct"):
        assert r[c] == pytest.approx(0, abs=1e-9)
    spec[(2024, "A")] = (60, 60 * 33000, 60 * 252, 40, 40 * 13000, 40 * 215)
    r = hours.real_pay_changes(hours.province_panel(_long(spec)), 2008, 2024, pd.Series({2008: 1.0, 2024: 1.2})).loc["A"]
    assert (1 + r.within_real_pct / 100) * (1 + r.mix_pct / 100) == pytest.approx(1 + r.pay_per_worker_real_pct / 100)
    assert r.pay_per_worker_no_real_pct == pytest.approx((33000 / 30000 / 1.2 - 1) * 100)


def test_panel_rejects_inconsistent_totals():
    d = _long({(2008, "A"): (80, 1000, 100, 20, 500, 50)})
    d.loc[(d.part_time == "Totale"), "workers"] += 1
    with pytest.raises(ValueError, match="differs"):
        hours.province_panel(d)


def test_unit_vs_others_rank_and_median():
    dec = pd.DataFrame({"x": [5.0, 1.0, 2.0, 3.0, 4.0]}, index=["U", "a", "b", "c", "d"])
    r = hours.unit_vs_others(dec, "U").loc["x"]
    assert r["rank"] == 1 and r.n_others == 4 and r.median_others == 2.5 and r.share_others_below_pct == 100


def test_comparable_provinces_drops_redrawn():
    idx = pd.MultiIndex.from_product([[2008, 2024], ["Torino", "Milano"]], names=["year", "province"])
    p = pd.DataFrame({"x": 1}, index=idx)
    assert hours.comparable_provinces(p, (2008, 2024)) == ["Torino"]
    assert hours.comparable_provinces(p, (2008, 2024), strict=False) == ["Milano", "Torino"]


def test_real_data_unit_present():
    if not list(inps.RAW.glob(inps.GLOB)):
        pytest.skip("INPS exports not in data/raw/inps_osservatorio")
    p = hours.province_panel(inps.load())
    keep = hours.comparable_provinces(p, (2008, 2024))
    dec = hours.mix_decomposition(p, 2008, 2024, keep)
    assert hours.UNIT in dec.index and len(dec) > 80 and np.isfinite(dec.mix_pct).all() and (dec.mix_pct < 0).all()


def test_totals_growth_identity_and_masked_provinces_dropped():
    t = pd.DataFrame({"year": [2019, 2020] * 3, "province": ["A", "A", "B", "B", "C", "C"], "kind": "province",
                      "workers": [100, 90, 50, 50, 10, 10], "pay_eur": [3000, 2400, 1000, 1100, 100, 100], "paid_days": [25000, 21600, 12000, 12000, 2000, 2000],
                      "masked_cells": [0, 0, 0, 0, 1, 0]})
    p = hours.totals_panel(t)
    assert set(p.index.get_level_values("province")) == {"A", "B"}  # C has a hidden section in 2019
    g = hours.totals_growth(p, 2019, 2020)
    assert g.loc["A", "workers_change_pct"] == pytest.approx(-10)
    assert np.allclose((1 + g.days_per_worker_change_pct / 100) * (1 + g.pay_per_day_change_pct / 100), 1 + g.pay_per_worker_change_pct / 100)


def test_real_totals_cover_2019_2025_when_present():
    if not list(inps.RAW.glob(inps.GLOB)):
        pytest.skip("INPS exports not in data/raw/inps_osservatorio")
    p = hours.totals_panel(inps.load_totals())
    g = hours.totals_growth(p, 2019, 2020, hours.comparable_provinces(p, (2019, 2020)))
    assert hours.UNIT in g.index and len(g) > 80 and g.workers_change_pct.loc[hours.UNIT] == pytest.approx((116575 / 120305 - 1) * 100)


def test_working_time_panel_shares_and_change():
    w = pd.DataFrame({"year": [2019, 2025] * 2, "province": ["A", "A", "B", "B"], "kind": "province", "Totale": [100, 120, 50, 50], "part_time": [30, 30, 10, 10],
                      "part_time_share_pct": [30, 25, 20, 20], "Tempo parziale orizzontale": [25, 25, 8, 8], "Tempo parziale verticale": [2, 2, 1, 1], "Tempo parziale misto": [3, 3, 1, 1]})
    p = hours.working_time_panel(w)
    ch = hours.working_time_change(p, 2019, 2025)
    assert ch.loc["A", "share_change_pp"] == -5 and ch.loc["A", "workers_month_change_pct"] == pytest.approx(20) and ch.loc["B", "share_change_pp"] == 0
    assert p.loc[(2019, "A"), "share_horizontal_pct"] == 25
