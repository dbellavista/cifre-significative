"""MEF tables by region, sector, age, turnover and twentile: parser details and the analysis helpers on synthetic frames, plus checks on the processed table."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cifresignificative.core.sources import mef_irpef as M
from cifresignificative.studies.italy_productivity.analysis import irpef_dims as ID

PROCESSED = Path("data/processed/mef_irpef_dims.parquet")
HEAD = "﻿Analisi statistiche - Dichiarazioni 2025 - Anno d'imposta 2024\nAnno presentazione:;2025\nClassificazione:;Sezione di attivita'\nData ultimo aggiornamento:;Maggio 2026\n\n"


def test_labels_with_semicolons_stay_in_the_label(tmp_path):
    body = ("Sezione di attivita';Codice;Numero titolari di partita IVA - Frequenza;Numero titolari di partita IVA - Percentuale;\n"
            "Costruzioni;06;300;30,00;\nCommercio all'ingrosso e al dettaglio; riparazione di autoveicoli e motocicli;07;700;70,00;\nTOTALE;0;1.000;100,00;\n")
    p = tmp_path / "2025ABPFTIT0105.csv"
    p.write_bytes((HEAD + body).encode())
    d = M.parse_file(p)
    r = d[(d.variable == "taxpayers") & (d.measure == "freq")]
    assert set(d.dimension) == {"activity"} and r[r.class_label.str.startswith("Commercio")].value.iloc[0] == 700 and r[r.is_total].value.iloc[0] == r[~r.is_total].value.sum()


def test_ventile_bound_column_and_turnover_labels(tmp_path):
    head = HEAD.replace("Sezione di attivita'", "Ventili di reddito complessivo")
    body = "Ventili di reddito complessivo;Codice;Reddito complessivo all'estremo del ventile in euro;Reddito complessivo - Frequenza;Reddito complessivo - Ammontare;Reddito complessivo - Media;\nI ventile;1;5.000;10;40;4,00;\nTOTALE;;;10;40;4,00;\n"
    p = tmp_path / "2025AAPFTOT0203.csv"
    p.write_bytes((head + body).encode())
    d = M.parse_file(p)
    assert set(d.dimension) == {"ventile"} and d[d.variable == "ventile_upper_eur"].value.iloc[0] == 5000
    assert M.class_bounds("Assenza dichiarazione Iva")[0] != M.class_bounds("Assenza dichiarazione Iva")[0] and M.class_bounds("minore di zero") == (-np.inf, 0.0) and M.class_bounds("0 - 14") == (0.0, 14.0)


def test_sector_key_maps_ateco_2025_names_back():
    assert ID.sector_key("Commercio all'ingrosso e al dettaglio; riparazione di autoveicoli e motocicli") == ID.sector_key("Commercio all'ingrosso e al dettaglio")
    assert ID.sector_key("Istruzione e formazione") == "Istruzione" and ID.sector_key("Telecomunicazioni, programmazione e consulenza informatica, infrastrutture informatiche e altre attivita' dei servizi d'informazione") == "Servizi di informazione e comunicazione"


def _frame(rows):
    cols = ["dimension", "code", "tax_year", "class_label", "is_total", "variable", "measure", "value"]
    return pd.DataFrame(rows, columns=cols)


def test_by_class_sums_trentino_provinces_into_one_region():
    rows = [("region", "T", 2020, "Trentino Alto Adige (P.A. Trento)", False, "taxpayers", "freq", 10.0), ("region", "T", 2020, "Trentino Alto Adige (P.A. Bolzano)", False, "taxpayers", "freq", 15.0),
            ("region", "T", 2020, "Veneto", False, "taxpayers", "freq", 100.0)]
    x = ID.by_class(_frame(rows), "region", "T")
    assert x.loc["Trentino Alto Adige", 2020] == 25 and x.loc["Veneto", 2020] == 100


def test_turnover_bands_put_the_threshold_class_in_its_own_band():
    labels = [("00", "Assenza dichiarazione Iva", np.nan, np.nan), ("01", "minore di zero", -np.inf, 0.0), ("11", "da 30.000 a 40.000", 30000.0, 40000.0), ("14", "da 60.000 a 75.000", 60000.0, 75000.0), ("20", "da 200.000 a 250.000", 200000.0, 250000.0)]
    rows = [{"dimension": "turnover", "code": "X", "tax_year": 2020, "class_label": lab, "class_code": c, "class_lo": lo, "class_hi": hi, "is_total": False, "variable": "taxpayers", "measure": "freq", "value": 1.0} for c, lab, lo, hi in labels]
    b = ID.turnover_bands(pd.DataFrame(rows), "X")
    assert b.loc["no VAT declaration", 2020] == 1 and b.loc["<=0", 2020] == 1 and b.loc["30-60k", 2020] == 1 and b.loc["60-75k", 2020] == 1 and b.loc[">100k", 2020] == 1


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed data missing")
def test_processed_dims_table_is_consistent():
    d = pd.read_parquet(PROCESSED)
    assert set(d.dimension) == {"region", "activity", "age", "turnover", "ventile"} and not d.duplicated(["dimension", "code", "tax_year", "class_label", "variable_raw", "measure"]).any()
    q = M.qa(d)
    ok = q[(q.suppressed == 0) & q.total.notna()]
    assert ((ok.class_sum_ratio - 1).abs() > 0.005).sum() <= 2  # rows add up to the TOTALE row (the ';' bug made sector tables fall far short)
    r, t = ID.by_class(d, "region", "AAPFTOT0201"), ID.by_class(d, "region", "AAPFTOT0201", total=True)
    assert ((r.sum() / t.iloc[0]).round(4) == 1).all() and len(ID.regional_mean(d, "AAPFTOT0201")) == 20
    sh = ID.ventile_shares(d)
    assert sh.shape[1] == 20 and abs(sh.loc[2024].sum() - 1) < 1e-9 and 0.2 < sh.loc[2024, 20] < 0.25
