"""INPS Osservatorio parser: both export layouts on synthetic text; the real exports only when they are local."""
import pytest

from cifresignificative.core.sources import inps_osservatorio as inps

TOTALS = '''"Osservatorio: x"
"Filtri:"
"Anno";"Provincia";"";"Numero lavoratori nell'anno ";"Retribuzione nell'anno ";"Numero giornate retribuite nell'anno "
"2008";"Forli' -Cesena";"";"114.328 ";"2.070.940.000 ";"27.974.884 "
" ";"Non ripartibili";"";"12 ";"1.000 ";"100 "
"2008";"Totale";"";"126.340 ";"2.071.941.000 ";"27.974.984 "
'''
PT = '''"Anno";"Provincia";"Presenza tempo parziale nell'anno";"";"Numero lavoratori nell'anno ";"Retribuzione nell'anno ";"Numero giornate retribuite nell'anno "
"2008";"Forli' -Cesena";"No";"";"86.436 ";"1.797.926.000 ";"22.017.073 "
"                                ";" ";"Si";"";"27.892 ";"273.014.000 ";"5.957.811 "
" ";"Forli' -Cesena";"Totale";"";"114.328 ";"2.070.940.000 ";"27.974.884 "
'''


def test_parse_totals_and_names():
    d = inps.parse_export(TOTALS)
    assert list(d.province) == ["Forlì-Cesena", "Non ripartibili", "Totale"]
    assert list(d.kind) == ["province", "other", "total"]
    assert d.loc[0, "workers"] == 114328 and d.loc[0, "part_time"] == "Totale"


def test_parse_part_time_layout_carries_province_and_year():
    d = inps.parse_export(PT)
    assert list(d.part_time) == ["No", "Si", "Totale"] and set(d.province) == {"Forlì-Cesena"} and set(d.year) == {2008}
    assert d.workers.iloc[:2].sum() == d.workers.iloc[2]


def test_load_merges_layouts_and_rejects_disagreement(tmp_path):
    (tmp_path / "inps_a.csv").write_text(TOTALS)
    (tmp_path / "inps_b.csv").write_text(PT)
    d = inps.load(tmp_path)
    assert len(d[(d.province == "Forlì-Cesena") & (d.part_time == "Totale")]) == 1
    (tmp_path / "inps_c.csv").write_text(PT.replace("114.328 ", "114.329 "))
    with pytest.raises(ValueError, match="differs"):
        inps.load(tmp_path)


def test_missing_files_and_bad_header(tmp_path):
    with pytest.raises(FileNotFoundError):
        inps.load(tmp_path)
    with pytest.raises(ValueError):
        inps.parse_export('"Anno";"Provincia"\n')


def test_clean_name():
    assert inps.clean_name("Verbano -Cusio-Ossola") == "Verbano-Cusio-Ossola"
    assert inps.clean_name("Massa -Carrara") == "Massa-Carrara"


def test_real_exports_when_present():
    if not list(inps.RAW.glob(inps.GLOB)):
        pytest.skip("INPS exports not in data/raw/inps_osservatorio")
    d = inps.load()
    fc = d[(d.province == "Forlì-Cesena") & (d.year == 2008)].set_index("part_time")
    assert fc.loc["No", "workers"] + fc.loc["Si", "workers"] == fc.loc["Totale", "workers"] == 114328
    assert d[d.kind == "total"].groupby("year").size().eq(3).all()
