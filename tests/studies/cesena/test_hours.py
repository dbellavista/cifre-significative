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
