"""Firm size and productivity: the mix effect and the composition identity on synthetic data with known answers, empty classes, break flags, and a data-dependent check on the processed table."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.studies.italy_productivity.analysis import size_class as SC
from cifresignificative.studies.italy_productivity.analysis.data import EU27_ISO3

PROCESSED = Path("data/processed/size_class.parquet")
OTHERS = [c for c in EU27_ISO3 if c != "ITA"]  # real codes: the loaders keep EU27 members only
LP = np.array([30.0, 45.0, 55.0, 70.0, 80.0])  # thousand euro per person by class, the same in every country


def _long(shares: dict[str, list[float]], year=2019, vintage="old", lp=LP, flags=None, lp_by=None, pli=None):
    rows = []
    for c, s in shares.items():
        for k, cl in enumerate(SC.CLASSES):
            persons = 1000.0 * s[k]
            rows.append({"iso3": c, "year": year, "vintage": vintage, "nace_r2": SC.ECONOMY[vintage], "size_emp": cl, "persons": persons, "va_meur": persons * (lp_by or {}).get(c, lp)[k] / 1000, "enterprises": persons / 5 + 1,
                         "pli": (pli or {}).get(c, 100.0), "flag": (flags or {}).get(c)})
    return pd.DataFrame(rows)


def _shares():
    typical = [0.30, 0.10, 0.10, 0.20, 0.30]
    out = {c: typical for c in OTHERS[:8]}
    out["ITA"] = [0.45, 0.10, 0.10, 0.15, 0.20]  # more micro
    return out


def test_mix_effect_is_zero_for_the_typical_mix_and_exact_for_a_planted_one():
    st = SC.structure(SC.class_table(_long(_shares()), "old")).set_index("iso3")
    assert st.mix_effect.drop("ITA").abs().max() < 0.01
    it = np.array(_shares()["ITA"])
    typ = np.array([0.30, 0.10, 0.10, 0.20, 0.30])
    assert st.mix_effect["ITA"] == pytest.approx(np.log((typ * LP).sum() / (it * LP).sum()))
    assert st.mix_effect["ITA"] > 0.1 and st.gradient["ITA"] == pytest.approx(np.log(80 / 30))


def test_position_ranks_italy_and_reports_the_two_sided_placebo_p():
    st = SC.structure(SC.class_table(_long(_shares() | {c: [0.3, 0.1, 0.1, 0.2, 0.3] for c in OTHERS[8:14]}), "old"))
    p = SC.position(st, 2019, ["micro_share", "large_share"]).set_index("statistic")
    assert p.loc["micro_share", "rank"] == 1 and p.loc["large_share", "rank"] == p.loc["large_share", "n"]
    assert p.loc["micro_share", "p"] == pytest.approx(2 / p.loc["micro_share", "n"])


def test_empty_class_gives_nan_not_a_warning():
    s = _shares()
    s[OTHERS[0]] = [0.5, 0.2, 0.1, 0.2, 0.0]
    st = SC.structure(SC.class_table(_long(s), "old")).set_index("iso3")
    assert np.isnan(st.loc[OTHERS[0], "gradient"]) and np.isnan(st.loc[OTHERS[0], "mix_effect"]) and st.loc[OTHERS[0], "large_share"] == 0
    assert not np.isnan(st.loc["ITA", "mix_effect"])


def test_break_flag_is_carried_and_counted():
    st = SC.structure(SC.class_table(_long(_shares() | {c: [0.3, 0.1, 0.1, 0.2, 0.3] for c in OTHERS[8:12]}, flags={OTHERS[1]: "b", OTHERS[8]: "bp"}), "old"))
    assert st.set_index("iso3").broken.to_dict()[OTHERS[1]] and not st.set_index("iso3").broken["ITA"]
    p = SC.position(st, 2019, ["micro_share"])
    assert p.n_break.iloc[0] == 2 and not p.italy_break.iloc[0]


def test_incomplete_class_sets_are_dropped():
    d = _long(_shares())
    d = d[~((d.iso3 == OTHERS[0]) & (d.size_emp == "GE250"))]
    assert OTHERS[0] not in set(SC.class_table(d, "old").iso3)


def _sections(micro: dict[str, list[float]], weights: dict[str, list[float]]):
    rows = []
    secs = list(SC.SECTIONS)[:3]
    for c in micro:
        for k, sec in enumerate(secs):
            persons = 1000.0 * weights[c][k]
            for cl, v in (("TOTAL", persons), ("0-9", persons * micro[c][k])):
                rows.append({"iso3": c, "year": 2019, "vintage": "old", "nace_r2": sec, "size_emp": cl, "persons": v})
    return pd.DataFrame(rows)


def test_composition_identity_and_attribution():
    ctr = {c: ([0.2, 0.3, 0.4], [0.5, 0.3, 0.2]) for c in OTHERS[:6]}
    ctr["ITA"] = ([0.4, 0.5, 0.6], [0.5, 0.3, 0.2])  # same sector mix, higher micro share in every sector
    ctr[OTHERS[6]] = ([0.2, 0.3, 0.4], [0.2, 0.3, 0.5])  # different sector mix, same micro shares
    d = SC.composition(_sections({c: v[0] for c, v in ctr.items()}, {c: v[1] for c, v in ctr.items()}), sections=list(SC.SECTIONS)[:3]).set_index("iso3")
    assert (d.composition + d.within).to_numpy() == pytest.approx(d.total_gap.to_numpy())
    assert d.loc["ITA", "composition"] == pytest.approx(0.0, abs=1e-9) and d.loc["ITA", "within"] == pytest.approx(0.2)
    assert d.loc[OTHERS[6], "within"] == pytest.approx(0.0, abs=1e-9) and d.loc[OTHERS[6], "composition"] > 0.01


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed data missing")
def test_processed_size_table_is_consistent():
    size = SC.load_size()
    old = size[(size.vintage == "old") & (size.nace_r2 == SC.ECONOMY["old"]) & (size.year == 2019) & (size.iso3 == "ITA")].set_index("size_emp")
    parts = old.loc[SC.CLASSES, "persons"].sum()
    assert abs(parts / old.loc["TOTAL", "persons"] - 1) < 0.01  # the five classes add up to the published total
    assert 0.35 < old.loc["0-9", "persons"] / parts < 0.5
    pos = SC.position_all(size)
    main = pos[pos.primary].set_index("statistic")
    assert main.loc["micro_share", "n"] >= 15 and main.loc["micro_share", "rank"] <= 5 and main.q_value.notna().all()
    assert not (pos.vintage.eq("new") & pos.year.lt(2021)).any() and not (pos.vintage.eq("old") & pos.year.gt(2020)).any()


def _panel(hours: dict[str, float]):
    return pd.DataFrame([{"iso3": c, "year": 2019, "indicator": "ameco_hours_per_worker", "value": h} for c, h in hours.items()])


def test_gap_decomposition_is_exact_and_the_size_mix_does_not_depend_on_price_or_hours():
    shares = _shares() | {c: [0.3, 0.1, 0.1, 0.2, 0.3] for c in OTHERS[8:14]}
    lp_by = {"ITA": LP * 0.9, OTHERS[0]: LP * 1.3}  # Italy 10% below the others in every class; the first member 30% above
    pli = {"ITA": 95.0, OTHERS[0]: 120.0}
    size = _long(shares, lp_by=lp_by, pli=pli)
    hours = {c: 1600.0 for c in shares} | {"ITA": 1800.0, OTHERS[0]: 1400.0}
    d = {b: SC.gap_decomposition(size, _panel(hours), basis=b, peers=[OTHERS[0]]).set_index("reference") for b in ("hour", "head")}
    for t in d.values():
        assert (t.size_mix + t.within).to_numpy() == pytest.approx(t.total.to_numpy())
    assert d["hour"].size_mix.to_numpy() == pytest.approx(d["head"].size_mix.to_numpy())  # a country-wide price or hours factor cancels in the mix term
    assert d["head"].loc["typical", "size_mix"] < -0.05
    ratio = np.log((0.9 / 0.95) / (1.3 / 1.2))  # every class of Italy against the first member, on the common price scale: independent of the mix
    assert d["head"].loc[OTHERS[0], "within"] == pytest.approx(ratio)
    assert d["hour"].loc[OTHERS[0], "within"] == pytest.approx(ratio + np.log(1400 / 1800))


def test_planted_within_gap_has_no_size_mix_and_vice_versa():
    same_mix = {c: [0.3, 0.1, 0.1, 0.2, 0.3] for c in OTHERS[:12]} | {"ITA": [0.3, 0.1, 0.1, 0.2, 0.3]}
    d = SC.gap_decomposition(_long(same_mix, lp_by={"ITA": LP * 0.8}), _panel({c: 1600.0 for c in same_mix}), basis="head", peers=[OTHERS[0]]).set_index("reference")
    assert d.loc[OTHERS[0], "size_mix"] == pytest.approx(0.0, abs=1e-12) and d.loc[OTHERS[0], "within"] == pytest.approx(np.log(0.8))
    d2 = SC.gap_decomposition(_long(_shares() | {c: [0.3, 0.1, 0.1, 0.2, 0.3] for c in OTHERS[8:14]}), _panel({c: 1600.0 for c in OTHERS[:14]} | {"ITA": 1600.0}), basis="head", peers=[OTHERS[0]]).set_index("reference")
    assert d2.loc[OTHERS[0], "within"] == pytest.approx(0.0, abs=1e-12) and d2.loc[OTHERS[0], "size_mix"] < -0.05


def test_level_gap_ratios_to_a_peer():
    shares = {c: [0.3, 0.1, 0.1, 0.2, 0.3] for c in OTHERS[:12]} | {"ITA": [0.3, 0.1, 0.1, 0.2, 0.3]}
    g = SC.level_gap(_long(shares, lp_by={"ITA": LP * 0.8}), _panel({c: 1600.0 for c in shares}), peers=[OTHERS[0]]).set_index(["basis", "size_emp"])
    assert g.loc[("head", "GE250"), f"ratio_{OTHERS[0]}"] == pytest.approx(0.8) and g.loc[("head", "total"), "rank"] == g.loc[("head", "total"), "n"]
