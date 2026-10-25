"""MEF IRPEF return statistics by region, sector, age, turnover class and income twentile (table `data/processed/mef_irpef_dims.parquet`, loader `sources/mef_irpef.py`).

Same caveats as `irpef.py`: income of the tax year (file prefix minus one), amounts in thousand euro in the source (euro here), taxpayers classed by reddito complessivo, which excludes income taxed under the flat-rate regimes.
Codes repeat across classifications: every selection is by (dimension, code). `MIN` = flat-rate filers (minimi, regime di vantaggio, regimi agevolati). Descriptive only.
"""
import itertools

import pandas as pd

AREA = {"Piemonte": "North-West", "Valle d'Aosta": "North-West", "Lombardia": "North-West", "Liguria": "North-West", "Trentino Alto Adige": "North-East", "Veneto": "North-East",
        "Friuli Venezia Giulia": "North-East", "Emilia Romagna": "North-East", "Toscana": "Centre", "Umbria": "Centre", "Marche": "Centre", "Lazio": "Centre", "Abruzzo": "South", "Molise": "South",
        "Campania": "South", "Puglia": "South", "Basilicata": "South", "Calabria": "South", "Sicilia": "Islands", "Sardegna": "Islands"}
INCOME_TYPES = {"employment": ["Reddito da lavoro dipendente e assimilati"], "pensions": ["Reddito da pensione"], "self-employment": ["Reddito da lavoro autonomo"],
                "business": ["Reddito di spettanza dell'imprenditore in contabilita' ordinaria", "Reddito di spettanza dell'imprenditore in contabilita' semplificata"],
                "buildings": ["Reddito da fabbricati"], "capital": ["Reddito di capitale (sez. IA e IB)"]}


def _sel(d: pd.DataFrame, dimension: str, code: str, variable: str, measure: str) -> pd.DataFrame:
    return d[(d.dimension == dimension) & (d.code == code) & (d.variable == variable) & (d.measure == measure)]


def by_class(d: pd.DataFrame, dimension: str, code: str, variable: str = "taxpayers", measure: str = "freq", total: bool = False) -> pd.DataFrame:
    """class_label x tax_year values of one variable (the TOTALE row only with `total=True`); Trentino and Alto Adige (two rows, one region code) are summed."""
    x = _sel(d, dimension, code, variable, measure)
    x = x[x.is_total] if total else x[~x.is_total]
    if dimension == "region":
        x = x.assign(class_label=x.class_label.str.replace(r"\s*\(P\.A\. .*\)", "", regex=True))
        return x.groupby(["class_label", "tax_year"]).value.sum(min_count=1).unstack("tax_year")
    return x.pivot_table(index="class_label", columns="tax_year", values="value", sort=False)


def regional_mean(d: pd.DataFrame, code: str, variable: str = "Reddito complessivo") -> pd.DataFrame:
    """Region x tax year: euro of `variable` per taxpayer (amount / taxpayers of the same table), regions only (no 'Non indicata')."""
    a = by_class(d, "region", code, variable, "amount_keur") * 1000
    n = by_class(d, "region", code, "taxpayers", "freq")
    m = (a / n).reindex([r for r in AREA if r in a.index])
    return m


def area_mean(d: pd.DataFrame, code: str, variable: str = "Reddito complessivo") -> pd.DataFrame:
    """Macro-area x tax year: euro of `variable` per taxpayer pooled over the regions of the area (population-weighted, not a mean of regional means)."""
    a = by_class(d, "region", code, variable, "amount_keur") * 1000
    n = by_class(d, "region", code, "taxpayers", "freq")
    keep = [r for r in AREA if r in a.index]
    area = pd.Series({r: AREA[r] for r in keep})
    return a.loc[keep].groupby(area).sum(min_count=1) / n.loc[keep].groupby(area).sum(min_count=1)


def ventile_shares(d: pd.DataFrame) -> pd.DataFrame:
    """tax_year x ventile (1-20): share of total reddito complessivo held by each twentile of taxpayers (ranked by reddito complessivo). The first ventile also holds zero and negative incomes."""
    a = by_class(d, "ventile", "AAPFTOT0203", "Reddito complessivo", "amount_keur")
    return (a / a.sum()).T.set_axis(range(1, 21), axis=1)


def ventile_mix(d: pd.DataFrame, year: int) -> pd.DataFrame:
    """Ventile x income type: share of the ventile's reddito complessivo coming from each source in `INCOME_TYPES`, plus `other` (the remainder) for one tax year."""
    tot = by_class(d, "ventile", "AAPFTOT0203", "Reddito complessivo", "amount_keur")[year]
    out = {}
    for k, vs in INCOME_TYPES.items():
        s = sum(by_class(d, "ventile", "AAPFTOT0203", v, "amount_keur").reindex(tot.index)[year].fillna(0) for v in vs)
        out[k] = s / tot
    out = pd.DataFrame(out)
    out["other"] = 1 - out.sum(axis=1)
    return out.set_axis(range(1, 21))


def flat_share(d: pd.DataFrame, by: str, tit: str, mn: str) -> pd.DataFrame:
    """Flat-rate filers (MIN tables) as a share of individuals with a VAT number (TIT tables) by class of a dimension (region, activity), per tax year."""
    return (by_class(d, by, mn) / by_class(d, by, tit)).dropna(how="all")


SECTOR_RENAME = {"Attivita' estrattive": "Estrazione di minerali da cave e miniere", "Telecomunicazioni, programmazione e consulenza informatica, infrastrutture informatiche e altre attivita' dei servizi d'informazione": "Servizi di informazione e comunicazione",
                 "Attivita' editoriali, trasmissioni radiofoniche e produzione e distribuzione di contenuti": "Servizi di informazione e comunicazione", "Attivita' amministrative e di servizi di supporto": "Noleggio, agenzie di viaggio, servizi di supporto alle imprese",
                 "Istruzione e formazione": "Istruzione", "Attivita' per la salute umana e di assistenza sociale": "Sanita' e assistenza sociale", "Attivita' artistiche, sportive e di divertimento": "Attivita' artistiche, sportive, di intrattenimento e divertimento",
                 "Attivita' di organizzazioni e organismi extraterritoriali": "Organizzazioni ed organismi extraterritoriali"}
SECTOR_BREAK_YEAR = 2024  # tax year 2024 (filing 2025) uses the ATECO 2025 section list, which splits and renames some sections


def sector_key(label: str) -> str:
    """Section label on the ATECO 2007 list used through tax year 2023: text before a `;` (the long names with a second part), and the 2024 names mapped back where the correspondence is one to one (information and communication is the sum of two 2024 sections)."""
    stem = label.split(";")[0].strip()
    return SECTOR_RENAME.get(stem, stem)


def by_sector(d: pd.DataFrame, code: str, variable: str = "taxpayers", measure: str = "freq") -> pd.DataFrame:
    """Section (ATECO 2007 names, see `sector_key`) x tax year; sections merged by the mapping are summed. Tax year 2024 follows another list: use it with care."""
    x = by_class(d, "activity", code, variable, measure)
    return x.groupby(x.index.map(sector_key)).sum(min_count=1)


def turnover_bands(d: pd.DataFrame, code: str, edges=(0, 30000, 60000, 75000, 100000)) -> pd.DataFrame:
    """Taxpayers of a turnover table by VAT-turnover band (band `no VAT declaration` = class code 00: flat-rate filers and others exempt from VAT; `<=0` = negative or zero turnover; classes are put in the first band whose upper edge is not below their upper bound; the
    class containing 65,000 euro (60-75k) is its own band). Rows: bands; columns: tax year."""
    x = d[(d.dimension == "turnover") & (d.code == code) & (d.variable == "taxpayers") & (d.measure == "freq") & ~d.is_total]
    names = ["<=0"] + [f"{a // 1000}-{b // 1000}k" for a, b in itertools.pairwise(edges)] + [f">{edges[-1] // 1000}k"]

    def band(r):
        if r.class_code == "00":
            return "no VAT declaration"
        if r.class_label.startswith(("minore", "zero")):
            return "<=0"
        if r.class_hi <= edges[0]:
            return "<=0"
        for (_, b), n in zip(itertools.pairwise(edges), names[1:-1], strict=True):
            if r.class_hi <= b:
                return n
        return names[-1]
    return x.assign(band=x.apply(band, axis=1)).groupby(["band", "tax_year"]).value.sum(min_count=1).unstack("tax_year").reindex(["no VAT declaration", *names])
