"""Industry panel: planted slope recovered, shuffled intensity calibrated, Italy's placebo rank."""
import numpy as np
import pandas as pd

from cifresignificative.studies.italy_productivity.analysis import industry_panel as IP

SECS = ["A", "B", "C", "D", "F", "G", "H", "I", "J", "K", "M", "N", "R", "S"]


def _G(slope_of: dict[str, float]):
    rng = np.random.default_rng(5)
    x = pd.Series(np.linspace(0, 1, 11), index=["A", "B", "C", "D-E", "F", "G", "H", "I", "J", "K", "M-N"])
    inds = [f"{s}{i}" for s in SECS[:11] for i in range(3)]
    rows = []
    for c, b in slope_of.items():
        for n in inds:
            rows.append({"iso3": c, "nace_r2": n, "g": b * x[IP.section_of(n)] + rng.normal(scale=0.2)})
    return pd.DataFrame(rows), x


def test_section_mapping():
    assert [IP.section_of(n) for n in ("A01", "C10-C12", "E36", "N80-N82", "J62_J63")] == ["A", "C", "D-E", "M-N", "J"]


def test_slopes_recover_plant_and_italy_rank():
    G, x = _G({"ITA": 0.0, "AUT": 1.0, "DNK": 1.0, "ESP": 1.0, "FIN": -0.5})
    S = IP.country_slopes(G, x, n_perm=199)
    s = S.set_index("iso3")
    assert abs(s.slope["AUT"] - 1.0) < 0.3 and s.p_perm["AUT"] < 0.05 and s.p_perm["ITA"] > 0.05
    r = IP.italy_rank(S)
    assert r["rank_from_top"] == 4 and r["n_countries"] == 5


def test_permutation_p_is_calibrated_under_the_null():
    rng = np.random.default_rng(9)
    G, x = _G({f"C{i:02d}": 0.0 for i in range(30)})
    share = np.mean([(IP.country_slopes(G, pd.Series(rng.permutation(x.to_numpy()), index=x.index), n_perm=99, seed=r).p_perm < 0.05).mean() for r in range(10)])
    assert share < 0.12
