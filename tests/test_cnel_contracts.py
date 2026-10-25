"""CNEL renewal cycles (offline: the cycle rules on toy agreements, the committed map and cycle files)."""
import pandas as pd

from cifresignificative.sources import cnel_contracts as cc


def agr(rows):
    d = pd.DataFrame(rows, columns=["code", "id", "kind", "signed", "start", "expiry", "confederal"]).assign(title="t")
    for c in ("signed", "start", "expiry"):
        d[c] = pd.to_datetime(d[c])
    return d


R, T = "Accordo di rinnovo", "Testo definitivo"


def test_private_cycle_uses_renewal_not_definitive_text_and_earliest_confederal_signing():
    d = agr([("A1", 1, R, "2020-03-01", "2020-01-01", "2022-12-31", False), ("A1", 2, R, "2020-06-01", "2020-01-01", "2022-12-31", True), ("A1", 3, T, "2020-09-01", "2020-01-01", "2022-12-31", True), ("A1", 4, R, "2020-07-01", "2020-01-01", "2022-12-31", True)])
    c = cc.cycles(d)
    assert len(c) == 1 and c.signed[0] == pd.Timestamp("2020-06-01") and c.kind[0] == R and bool(c.confederal[0]) and c.n_rows[0] == 3


def test_public_code_uses_the_definitive_text_and_private_falls_back_to_it_without_renewal_row():
    d = agr([("S9", 1, R, "2017-12-23", "2016-01-01", "2018-12-31", True), ("S9", 2, T, "2018-02-12", "2016-01-01", "2018-12-31", True), ("P1", 3, T, "2019-03-11", "2019-01-01", "2022-12-31", False)])
    c = cc.cycles(d).set_index("code")
    assert c.loc["S9", "signed"] == pd.Timestamp("2018-02-12") and c.loc["S9", "kind"] == T
    assert c.loc["P1", "kind"] == T and not c.loc["P1", "confederal"]


def test_wage_only_agreements_are_not_renewals_and_rows_without_period_stand_alone():
    d = agr([("A1", 1, "Accordo economico", "2020-01-01", "2020-01-01", "2020-12-31", True), ("A1", 2, R, "2023-11-14", None, None, True), ("A1", 3, R, "2024-02-01", None, None, True)])
    c = cc.cycles(d)
    assert len(c) == 2 and c.expiry.isna().all()


def test_primary_cycles_drop_side_signings_only_when_a_confederal_cycle_exists():
    c = pd.DataFrame({"code": ["H", "H", "C"], "confederal": [True, False, False], "signed": pd.to_datetime(["2015-03-30", "2019-07-30", "2019-03-11"])})
    assert cc.primary_cycles(c).signed.dt.year.tolist() == [2015, 2019]  # H keeps 2015, C (no confederal cycle) keeps 2019


def test_committed_files_are_consistent():
    m, c = cc.load_map(), cc.load()
    leaves = m.istat_group
    assert leaves.is_unique and len(leaves) == 78
    assert set(m.evidence) <= {"verified", "partly verified", "unverified"}
    assert ((m.cnel_code == "") == (m.evidence == "unverified")).all()
    assert set(m.cnel_code) - {""} <= set(c.code)
    assert cc.check(c).empty
    assert c.signed.notna().all() and c.signed.min() >= pd.Timestamp(cc.FIRST_YEAR)


def test_known_renewals_of_the_committed_cycles():
    p = cc.primary_cycles(cc.load())
    h = p[p.code == "H011"].signed.dt.strftime("%Y-%m").tolist()
    assert "2015-03" in h and "2024-03" in h  # Confcommercio commerce: the CGIL-CISL-UIL renewals
    f = p[p.code == "F012"].signed.dt.strftime("%Y-%m").tolist()
    assert f[-3:] == ["2018-07", "2022-03", "2025-02"]
