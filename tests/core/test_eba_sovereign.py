"""EBA sovereign loader: label selection, Italy-only filter, bank sum and cross-file agreement."""
import pandas as pd
import pytest

from cifresignificative.core.sources import eba_sovereign as E

G, N = E.MEASURES["gross"], E.MEASURES["net"]


def _write(path, rows):
    pd.DataFrame(rows, columns=["LEI_Code", "NSA", "Period", "Label", "Country", "Maturity", "Amount"]).to_csv(path, index=False)


def test_load_sums_italian_banks_and_filters(tmp_path):
    _write(tmp_path / "2024_tr_sov.csv", [
        ("A", "IT", "202312", N, "15", "8", "1000"), ("B", "IT", "202312", N, "15", "8", "500"),
        ("A", "IT", "202312", N, "15", "3", "999"),  # maturity bucket
        ("A", "IT", "202312", N + " of which: Financial assets at amortised cost", "15", "8", "999"),  # sub-row
        ("A", "IT", "202312", N, "2", "8", "999"),  # other sovereign
        ("C", "DE", "202312", N, "15", "8", "999"),  # other country's bank
        ("A", "IT", "202312", G, "15", "8", "2000"),
    ])
    df = E.load(tmp_path).set_index("measure")
    assert df.loc["net", "value"] == 1.5 and df.loc["net", "n_banks"] == 2 and df.loc["gross", "value"] == 2.0
    assert df.period.iloc[0] == pd.Timestamp("2023-12-31")


def test_load_rejects_disagreeing_overlap(tmp_path):
    _write(tmp_path / "2023_tr_sov.csv", [("A", "IT", "202312", N, "15", "8", "1000")])
    _write(tmp_path / "2024_tr_sov.csv", [("A", "IT", "202312", N, "15", "8", "1100")])
    with pytest.raises(ValueError):
        E.load(tmp_path)
