"""Tests for MEF municipal IRPEF parser (`core.sources.mef_comuni`)."""
import io
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from cifresignificative.core.sources import mef_comuni


def _make_mock_mef_zip(tmp_path: Path, year: int = 2023) -> Path:
    zpath = tmp_path / f"irpef_comuni_{year}.zip"
    header = [
        "Anno di imposta", "Codice catastale", "Codice Istat Comune", "Denominazione Comune",
        "Sigla Provincia", "Regione", "Codice Istat Regione", "Numero contribuenti",
        "Reddito imponibile - Frequenza", "Reddito imponibile - Ammontare in euro",
        "Imposta netta - Frequenza", "Imposta netta - Ammontare in euro",
        "Addizionale comunale dovuta - Frequenza", "Addizionale comunale dovuta - Ammontare in euro",
        "Reddito da lavoro dipendente e assimilati - Frequenza", "Reddito da lavoro dipendente e assimilati - Ammontare in euro",
        "Reddito da pensione - Frequenza", "Reddito da pensione - Ammontare in euro",
        "Reddito da lavoro autonomo (comprensivo dei valori nulli) - Frequenza", "Reddito da lavoro autonomo (comprensivo dei valori nulli) - Ammontare in euro",
        "Reddito spettanza imprenditore ordinaria (compresi nulli) - Frequenza", "Reddito spettanza imprenditore ordinaria (compresi nulli) - Ammontare",
        "Reddito da partecipazione (compresi nulli) - Frequenza", "Reddito da partecipazione (compresi nulli) - Ammontare",
        "Reddito complessivo - Frequenza", "Reddito complessivo - Ammontare in euro",
        "Reddito complessivo minore o uguale a zero euro - Frequenza", "Reddito complessivo minore o uguale a zero euro - Ammontare in euro",
        "Reddito complessivo da 0 a 10000 euro - Frequenza", "Reddito complessivo da 0 a 10000 euro - Ammontare in euro",
        "Reddito complessivo da 10000 a 15000 euro - Frequenza", "Reddito complessivo da 10000 a 15000 euro - Ammontare in euro",
        "Reddito complessivo da 15000 a 26000 euro - Frequenza", "Reddito complessivo da 15000 a 26000 euro - Ammontare in euro",
        "Reddito complessivo da 26000 a 55000 euro - Frequenza", "Reddito complessivo da 26000 a 55000 euro - Ammontare in euro",
        "Reddito complessivo da 55000 a 75000 euro - Frequenza", "Reddito complessivo da 55000 a 75000 euro - Ammontare in euro",
        "Reddito complessivo da 75000 a 120000 euro - Frequenza", "Reddito complessivo da 75000 a 120000 euro - Ammontare in euro",
        "Reddito complessivo oltre 120000 euro - Frequenza", "Reddito complessivo oltre 120000 euro - Ammontare in euro",
    ]
    cesena_row = [
        str(year), "C573", "040007", "CESENA", "FC", "Emilia Romagna", "08", "73500",
        "71000", "1750000000", "62000", "350000000", "58000", "11500000",
        "40000", "980000000", "25000", "510000000", "3200", "145000000",
        "12", "123456", "", "",
        "73000", "1850000000",
        "300", "0",
        "14000", "70000000",
        "8500", "106000000",
        "24000", "490000000",
        "21500", "740000000",
        "2200", "140000000",
        "1600", "144000000",
        "900", "160000000",
    ]
    buf = io.StringIO()
    buf.write(";".join(header) + "\n")
    buf.write(";".join(cesena_row) + "\n")
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr(f"irpef_{year}.csv", buf.getvalue().encode("latin1"))
    return zpath


def test_parse_mef_comuni_zip(tmp_path: Path):
    zp = _make_mock_mef_zip(tmp_path, 2023)
    brackets, summary = mef_comuni.parse_year(zp)
    assert len(summary) == 1
    assert summary.iloc[0]["comune_id"] == "040007"
    assert summary.iloc[0]["tax_year"] == 2023
    assert np.isclose(summary.iloc[0]["addizionale_comunale_eur"], 11_500_000.0)
    assert summary.iloc[0]["reddito_impresa_ordinaria_freq"] == 12
    assert summary.iloc[0]["reddito_impresa_ordinaria_eur"] == 123456
    assert np.isnan(summary.iloc[0]["reddito_partecipazione_eur"])
    assert len(brackets) == 8
    assert np.isclose(brackets[brackets["class_order"] == 7]["income_eur"].iloc[0], 160_000_000.0)
    assert np.isclose(brackets[brackets["class_order"] >= 1]["taxpayers"].sum(), 72_700.0)


def test_business_and_participation_header_variants():
    patterns = {name: pattern for name, pattern, _ in mef_comuni.SUMMARY_PATTERNS}
    for name, label in [
        ("reddito_impresa_ordinaria", "Reddito spettanza imprenditore ordinaria (compresi nulli) - Frequenza"),
        ("reddito_impresa_semplificata", "Reddito di spettanza dell'imprenditore in contabilità semplificata (compresi valori nulli) - Ammontare"),
        ("reddito_partecipazione", "Reddito da partecipazione (compresi nulli) - Frequenza"),
    ]:
        assert patterns[name].match(label)


def _row(total_f: float, total_a: float, cells: Mapping[str, tuple[float | None, float | None]]) -> pd.Series:
    labels = {
        "zero": "minore o uguale a zero euro", "b1": "da 0 a 10000 euro", "b2": "da 10000 a 15000 euro", "b3": "da 15000 a 26000 euro",
        "b4": "da 26000 a 55000 euro", "b5": "da 55000 a 75000 euro", "b6": "da 75000 a 120000 euro", "top": "oltre 120000 euro",
    }
    out = {"Reddito complessivo - Frequenza": str(total_f), "Reddito complessivo - Ammontare in euro": str(total_a)}
    for key, lbl in labels.items():
        f_val, a_val = cells.get(key, (None, None))
        out[f"Reddito complessivo {lbl} - Frequenza"] = None if f_val is None else str(f_val)
        out[f"Reddito complessivo {lbl} - Ammontare in euro"] = None if a_val is None else str(a_val)
    return pd.Series(out)


def test_parse_brackets_recovers_suppressed_top_bracket():
    """A blank `>120k` cell next to a blank `<=0` cell is total minus published brackets, not zero income."""
    cells = {"b1": (10, 50_000), "b2": (10, 125_000), "b3": (10, 200_000), "b4": (10, 400_000), "b5": (5, 300_000), "b6": (5, 450_000)}
    row = _row(52, 2_525_000, cells)
    out = {c["class_order"]: c for c in mef_comuni.parse_brackets(row, row.index)}
    assert out[7]["imputed"] and out[7]["taxpayers"] == 2 and out[7]["income_eur"] == 1_000_000
    assert np.isnan(out[0]["taxpayers"]) and not out[0]["imputed"]
    assert sum(c["income_eur"] for k, c in out.items() if k >= 1) == 2_525_000


def test_parse_brackets_leaves_two_blank_brackets_missing():
    """With two positive cells blank the residual cannot be split, so both stay missing."""
    row = _row(30, 400_000, {"b1": (10, 50_000), "b2": (10, 125_000), "b3": (10, 200_000)})
    out = {c["class_order"]: c for c in mef_comuni.parse_brackets(row, row.index)}
    assert np.isnan(out[7]["income_eur"]) and not out[7]["imputed"]


def test_parse_brackets_without_totals_keeps_blank():
    """Before 2023 the files have no `Reddito complessivo` total: nothing is recovered."""
    row = _row(0, 0, {"b1": (10, 50_000)}).drop(["Reddito complessivo - Frequenza", "Reddito complessivo - Ammontare in euro"])
    out = {c["class_order"]: c for c in mef_comuni.parse_brackets(row, row.index)}
    assert np.isnan(out[7]["income_eur"]) and not out[7]["imputed"]
