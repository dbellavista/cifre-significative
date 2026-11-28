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
    tot = d[d.kind == "total"].groupby("year").size()
    assert tot.loc[[2008, 2018, 2024, 2025]].eq(3).all() and tot.loc[2019:2023].eq(1).all()  # the part-time cut is missing for 2019-2023


OPENDATA = """TERRITORIO,Territorio di lavoro,INDICATORI,Indicatori (DESC),ATECO_2007,Ateco 2007,Anno (TIME_PERIOD),Osservazione,MIS_PRIVACY,Gestione misura sotto soglia di sicurezza
ITH58,Forlì-Cesena,NUM_LAVORATORI,'Numero lavoratori nell"'anno',C,'Attivita"' manifatturiere',2019,100,,
ITH58,Forlì-Cesena,NUM_LAVORATORI,'Numero lavoratori nell"'anno',D,'Fornitura di energia elettrica, gas, vapore e aria condizionata',2019,5,,
ITH58,Forlì-Cesena,RETR_ANNO,'Retribuzione nell"'anno',C,'Attivita"' manifatturiere',2019,2000,,
ITH58,Forlì-Cesena,RETR_ANNO,'Retribuzione nell"'anno',D,'Fornitura di energia elettrica, gas, vapore e aria condizionata',2019,90,,
ITH58,Forlì-Cesena,NUM_GG_RETR,'Numero giornate retribuite nell"'anno',C,'Attivita"' manifatturiere',2019,30,,
ITH58,Forlì-Cesena,NUM_GG_RETR,'Numero giornate retribuite nell"'anno',D,'Fornitura di energia elettrica, gas, vapore e aria condizionata',2019,2,,
ITC,'L"'Aquila',NUM_LAVORATORI,'Numero lavoratori nell"'anno',T,'Attivita"' di famiglie',2019,(*),(*),_
ITC,'L"'Aquila',NUM_LAVORATORI,'Numero lavoratori nell"'anno',C,'Attivita"' manifatturiere',2019,7,,
"""


def test_parse_opendata_handles_quotes_commas_and_masked_cells():
    s = inps.parse_opendata(OPENDATA)
    assert set(s.province) == {"Forlì-Cesena", "L'Aquila"} and set(s.section) == {"C", "D", "T"}
    assert s.loc[(s.province == "Forlì-Cesena") & (s.section == "D"), ["workers", "pay_eur", "paid_days"]].iloc[0].tolist() == [5, 90, 2]
    t = inps.opendata_totals(s).set_index("province")
    assert t.loc["Forlì-Cesena", ["workers", "pay_eur", "paid_days", "masked_cells"]].tolist() == [105, 2090, 32, 0]
    assert t.loc["L'Aquila", "workers"] == 7 and t.loc["L'Aquila", "masked_cells"] == 1


def test_opendata_rejects_other_layouts():
    with pytest.raises(ValueError):
        inps.parse_opendata("Anno;Provincia\n")
    with pytest.raises(ValueError):
        inps.parse_opendata(OPENDATA.replace("RETR_ANNO", "OTHER"))


def test_load_totals_appends_new_years_and_checks_overlap(tmp_path):
    (tmp_path / "inps_a.csv").write_text(TOTALS)
    (tmp_path / "opendata_x.csv").write_text(OPENDATA.replace("2019", "2008").replace("100,", "114323,").replace(",2000,", ",2070940000,"))
    with pytest.raises(ValueError):  # the Open Data 2008 names do not all exist in the Osservatorio block
        inps.load_totals(tmp_path)
    (tmp_path / "opendata_x.csv").write_text(OPENDATA)
    t = inps.load_totals(tmp_path)
    assert set(t.source) == {"osservatorio", "opendata"} and t[t.year == 2019].workers.sum() == 112


def test_real_opendata_agrees_with_osservatorio_when_present():
    if not list(inps.RAW.glob(inps.OPENDATA_GLOB)) or not list(inps.RAW.glob(inps.GLOB)):
        pytest.skip("INPS exports not in data/raw/inps_osservatorio")
    t = inps.load_totals()  # raises if 2014-2018 differ beyond masked cells
    fc = t[t.province == "Forlì-Cesena"].set_index("year")
    assert fc.loc[2016, "workers"] == 108365 and fc.loc[2016, "pay_eur"] == 2241994750 and fc.loc[2016, "paid_days"] == 26637620
    assert fc.loc[2019, "workers"] == 120305 and fc.loc[2020, "paid_days"] == 26434442
    assert fc.loc[2021, "workers"] == 122340 and fc.loc[2021, "pay_eur"] == 2549048759  # revised 2026-11-24 (Open Data: 122085)
    assert fc.loc[2022, "workers"] == 127939 and fc.loc[2023, "workers"] == 131274 and fc.loc[2025, "workers"] == 135183
    assert fc.source.eq("osservatorio").all() and t[t.year == 2019].province.nunique() == 104
    assert t[t.year == 2021].workers.sum() == 16274677


WT = '''"Osservatorio: Osservatorio sui Lavoratori Dipendenti - Anno 2021"
"Filtri:"
"Mese:Media"

"Provincia";"Tipologia orario di lavoro";"";"Numero lavoratori "
"Forli' -Cesena";"Totale";"";"1.000 "
"Forli' -Cesena";"Tempo parziale misto";"";"30 "
" ";"Tempo parziale orizzontale";"";"230 "
" ";"Tempo parziale verticale";"";"10 "
" ";"Tempo pieno";"";"730 "
'''


def test_parse_working_time_and_load(tmp_path):
    d = inps.parse_working_time(WT)
    assert set(d.year) == {2021} and set(d.province) == {"Forlì-Cesena"} and d.set_index("working_time").workers["Tempo pieno"] == 730
    (tmp_path / "tipo_orario_media_2021.csv").write_text(WT)
    w = inps.load_working_time(tmp_path)
    assert w.loc[0, "part_time_share_pct"] == pytest.approx(27.0)
    (tmp_path / "tipo_orario_media_2021.csv").write_text(WT.replace("730 ", "720 "))
    with pytest.raises(ValueError, match="add up"):
        inps.load_working_time(tmp_path)


def test_real_working_time_when_present():
    if not list(inps.RAW.glob(inps.WORKING_TIME_GLOB)):
        pytest.skip("INPS working-time exports not in data/raw/inps_osservatorio")
    w = inps.load_working_time()
    fc = w[w.province == "Forlì-Cesena"].set_index("year")
    assert fc.loc[2019, "Totale"] == 99717 and fc.loc[2012, "Totale"] == 92848 and 2016 not in fc.index and fc.loc[2025, "Totale"] == 113647 and fc.loc[2025, "Tempo pieno"] == 84418 and set(range(2019, 2026)) <= set(fc.index) and fc.loc[2008, "Totale"] == 95651


def test_parse_working_time_type_first_column_order():
    swapped = '''"Osservatorio: Osservatorio sui Lavoratori Dipendenti - Anno 2012"
"Filtri:"
"Mese:Media"

"Tipologia orario di lavoro";"Provincia";"";"Numero lavoratori "
"Tempo pieno";"Torino";"";"417.406 "
" ";"Forli' -Cesena";"";"60.000 "
"Tempo parziale verticale";"Torino";"";"1.000 "
'''
    d = inps.parse_working_time(swapped).set_index(["province", "working_time"]).workers
    assert d["Torino", "Tempo pieno"] == 417406 and d["Forlì-Cesena", "Tempo pieno"] == 60000 and d["Torino", "Tempo parziale verticale"] == 1000
