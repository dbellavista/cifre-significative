"""MUR USTAT: University student enrolments and graduates by campus municipality (2010–2026).

Provides course-level and disciplinary-group student enrolments and graduation figures from the open-data releases
of the Italian Ministry of University and Research (MUR - Ufficio Statistica e Studi, https://dati-ustat.mur.gov.it).
Enables sub-national university benchmarking across the Emilia-Romagna campus network (UniBo multifold campus:
Cesena, Forlì, Ravenna, Rimini, Bologna; UNIFE Ferrara; UNIPR Parma; UNIMORE Modena & Reggio Emilia; UCSC/PoliMi Piacenza).
"""
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.geo import ALL_COMUNI_IDS, COMUNI_BY_ID, format_comune_id
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/mur_ustat")
PROCESSED_DIR = Path("data/processed/cesena")

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}

FILES = {
    "iscritti_2019_2025.csv": (
        "https://dati-ustat.mur.gov.it/dataset/3dd9ca7f-9cc9-4a1a-915c-e569b181dbd5/resource/ad5a1516-ccd3-4eb8-a889-f584f04de578/download/13_iscrittixcorso.csv",
        "Iscritti",
    ),
    "iscritti_2016_2018.csv": (
        "https://dati-ustat.mur.gov.it/dataset/3dd9ca7f-9cc9-4a1a-915c-e569b181dbd5/resource/d40ca09b-9da8-47e7-8b6c-481f000f4e0c/download/13bis_iscrittixcorso_.csv",
        "Isc",
    ),
    "iscritti_2010_2015.csv": (
        "https://dati-ustat.mur.gov.it/dataset/3dd9ca7f-9cc9-4a1a-915c-e569b181dbd5/resource/3dbdc324-545a-4cc5-83e1-40fbfc415039/download/13ter_iscrittixcorso.csv",
        "Isc",
    ),
    "laureati_2018_2025.csv": (
        "https://dati-ustat.mur.gov.it/dataset/ca0b9831-c9ef-49d4-b792-6bd963e4d837/resource/5046c89e-ccd1-4436-93be-daa92b24f8a1/download/12c_laureati_corso_2018-2025.csv",
        "laureati",
    ),
    "laureati_2010_2017.csv": (
        "https://dati-ustat.mur.gov.it/dataset/ca0b9831-c9ef-49d4-b792-6bd963e4d837/resource/21a6312b-29c0-45e0-aa0a-8904653ecc65/download/12b_laureati_corso_2010-2017.csv",
        "Lau",
    ),
}

DISCIPLINARY_GROUPS: dict[int, str] = {
    1: "Educazione e Formazione",
    2: "Arte e Design",
    3: "Letterario-Umanistico",
    4: "Linguistico",
    5: "Politico-Sociale e Comunicazione",
    6: "Psicologico",
    7: "Economico",
    8: "Giuridico",
    9: "Scientifico",
    10: "Informatica e Tecnologie ICT",
    11: "Architettura e Ingegneria civile",
    12: "Ingegneria industriale e dell'informazione",
    13: "Agrario-Forestale e Veterinario",
    14: "Medico-Sanitario e Farmaceutico",
    15: "Scienze motorie e sportive",
}

# Macro categories
STEM_GROUP_IDS = {9, 10, 11, 12, 13}
ICT_GROUP_IDS = {10}
ENGINEERING_GROUP_IDS = {11, 12}
PSYCHOLOGY_GROUP_IDS = {6}


def _canon_name(s: str) -> str:
    norm = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")
    return norm.upper().strip()


# Canonical mapping for municipality names in the graduates tables
MUNI_NAME_TO_ID: dict[str, str] = {
    "CESENA": "040007",
    "FORLI": "040012",
    "FORLI'": "040012",
    "RIMINI": "099014",
    "RAVENNA": "039014",
    "IMOLA": "037032",
    "CARPI": "036005",
    "REGGIO NELL'EMILIA": "035033",
    "REGGIO EMILIA": "035033",
    "FERRARA": "038008",
    "MODENA": "036023",
    "PARMA": "034027",
    "PIACENZA": "033032",
    "FAENZA": "039010",
    "BOLOGNA": "037006",
}


def fetch(force: bool = False) -> Path:
    """Download the 5 MUR USTAT raw CSV files if missing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname, (url, _) in FILES.items():
        dest = RAW_DIR / fname
        if dest.exists() and not force:
            continue
        resp = requests.get(url, headers=UA, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        record(f"mur_ustat_{fname}", source="MUR_USTAT", dataset=fname, url=url)
    return RAW_DIR


def load_enrolments(comuni_ids: set[str] | tuple[str, ...] | list[str] | None = None, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load course-level student enrolments for target municipalities (academic years 2010/2011 to 2025/2026)."""
    target = set(ALL_COMUNI_IDS) if comuni_ids is None else set(comuni_ids)
    frames = []

    for fname in ("iscritti_2019_2025.csv", "iscritti_2016_2018.csv", "iscritti_2010_2015.csv"):
        p = raw_dir / fname
        if not p.exists():
            continue
        count_col = FILES[fname][1]
        df = pd.read_csv(p, sep=";", encoding="latin1", low_memory=False)
        df["comune_id"] = df["SedeC"].astype(str).str.strip().apply(format_comune_id)
        sub = df[df["comune_id"].isin(target)].copy()
        if sub.empty:
            continue

        sub["students"] = pd.to_numeric(sub[count_col], errors="coerce").fillna(0)
        sub["group_id"] = pd.to_numeric(sub["GruppoCODICE"], errors="coerce").fillna(0).astype(int)
        sub["group_label"] = sub["group_id"].map(DISCIPLINARY_GROUPS).fillna("Altro")
        sub["calendar_year"] = sub["AnnoA"].astype(str).str[:4].astype(int)
        sub["comune_name"] = sub["comune_id"].map(lambda cid: COMUNI_BY_ID[cid].name if cid in COMUNI_BY_ID else cid)

        is_ict = sub["group_id"].isin(ICT_GROUP_IDS)
        is_eng = sub["group_id"].isin(ENGINEERING_GROUP_IDS)
        is_stem = sub["group_id"].isin(STEM_GROUP_IDS)
        is_psy = sub["group_id"].isin(PSYCHOLOGY_GROUP_IDS)

        sub["macro_group"] = "humanities_social"
        sub.loc[is_psy, "macro_group"] = "psychology"
        sub.loc[is_stem, "macro_group"] = "stem_other"
        sub.loc[is_eng, "macro_group"] = "stem_engineering"
        sub.loc[is_ict, "macro_group"] = "stem_ict"

        frames.append(
            sub.rename(
                columns={
                    "AnnoA": "academic_year",
                    "AteneoNOME": "university_name",
                    "CorsoNOME": "course_name",
                    "ClasseNUMERO": "degree_class",
                }
            )[
                [
                    "academic_year", "calendar_year", "comune_id", "comune_name",
                    "university_name", "course_name", "degree_class",
                    "group_id", "group_label", "macro_group", "students",
                ]
            ]
        )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_graduates(comuni_ids: set[str] | tuple[str, ...] | list[str] | None = None, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load graduate counts by municipality for target municipalities (calendar years 2010 to 2025)."""
    target = set(ALL_COMUNI_IDS) if comuni_ids is None else set(comuni_ids)
    frames = []

    # 1. 2018-2025 file
    p1 = raw_dir / "laureati_2018_2025.csv"
    if p1.exists():
        df1 = pd.read_csv(p1, sep=";", encoding="latin1", low_memory=False)
        muni_id_col = df1["COMUNE"].astype(str).apply(_canon_name).map(MUNI_NAME_TO_ID)
        df1["comune_id"] = muni_id_col
        sub1 = df1[df1["comune_id"].isin(target)].copy()
        if not sub1.empty:
            sub1["graduates"] = pd.to_numeric(sub1["laureati"], errors="coerce").fillna(0)
            sub1["calendar_year"] = pd.to_numeric(sub1["ANNO_LAUREA"], errors="coerce").fillna(0).astype(int)
            sub1["comune_name"] = sub1["comune_id"].map(lambda cid: COMUNI_BY_ID[cid].name if cid in COMUNI_BY_ID else cid)
            frames.append(
                sub1.rename(
                    columns={
                        "NomeOperativo": "university_name",
                        "NOME_CORSO": "course_name",
                        "NUMERO": "degree_class",
                    }
                )[
                    ["calendar_year", "comune_id", "comune_name", "university_name", "course_name", "degree_class", "graduates"]
                ]
            )

    # 2. 2010-2017 file
    p2 = raw_dir / "laureati_2010_2017.csv"
    if p2.exists():
        df2 = pd.read_csv(p2, sep=";", encoding="latin1", low_memory=False)
        muni_id_col2 = df2["SedeC"].astype(str).apply(_canon_name).map(MUNI_NAME_TO_ID)
        df2["comune_id"] = muni_id_col2
        sub2 = df2[df2["comune_id"].isin(target)].copy()
        if not sub2.empty:
            sub2["graduates"] = pd.to_numeric(sub2["Lau"], errors="coerce").fillna(0)
            sub2["calendar_year"] = pd.to_numeric(sub2["AnnoS"], errors="coerce").fillna(0).astype(int)
            sub2["comune_name"] = sub2["comune_id"].map(lambda cid: COMUNI_BY_ID[cid].name if cid in COMUNI_BY_ID else cid)
            frames.append(
                sub2.rename(
                    columns={
                        "AteneoNOME": "university_name",
                        "CorsoNOME": "course_name",
                        "ClasseNUMERO": "degree_class",
                    }
                )[
                    ["calendar_year", "comune_id", "comune_name", "university_name", "course_name", "degree_class", "graduates"]
                ]
            )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_university_poles(comuni_ids: set[str] | tuple[str, ...] | list[str] | None = None, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Summarise annual university metrics per municipality: enrolments, STEM/ICT specialisation, and graduates."""
    enr = load_enrolments(comuni_ids, raw_dir=raw_dir)
    gra = load_graduates(comuni_ids, raw_dir=raw_dir)

    if enr.empty:
        return pd.DataFrame()

    enr_agg = enr.groupby(["calendar_year", "comune_id", "comune_name"]).agg(
        students_total=("students", "sum"),
        students_stem=("students", lambda s: s[enr.loc[s.index, "group_id"].isin(STEM_GROUP_IDS)].sum()),
        students_ict=("students", lambda s: s[enr.loc[s.index, "group_id"].isin(ICT_GROUP_IDS)].sum()),
        students_engineering=("students", lambda s: s[enr.loc[s.index, "group_id"].isin(ENGINEERING_GROUP_IDS)].sum()),
        students_psychology=("students", lambda s: s[enr.loc[s.index, "group_id"].isin(PSYCHOLOGY_GROUP_IDS)].sum()),
    ).reset_index()

    enr_agg["stem_share"] = enr_agg["students_stem"] / enr_agg["students_total"].replace(0, float("nan"))
    enr_agg["ict_share"] = enr_agg["students_ict"] / enr_agg["students_total"].replace(0, float("nan"))
    enr_agg["engineering_share"] = enr_agg["students_engineering"] / enr_agg["students_total"].replace(0, float("nan"))
    enr_agg["psychology_share"] = enr_agg["students_psychology"] / enr_agg["students_total"].replace(0, float("nan"))

    if not gra.empty:
        gra_agg = gra.groupby(["calendar_year", "comune_id"]).agg(graduates_total=("graduates", "sum")).reset_index()
        enr_agg = enr_agg.merge(gra_agg, on=["calendar_year", "comune_id"], how="left")
    else:
        enr_agg["graduates_total"] = float("nan")

    return enr_agg.sort_values(["calendar_year", "comune_id"], ignore_index=True)


def build(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build processed university parquet tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    enr = load_enrolments()
    gra = load_graduates()
    summary = summarize_university_poles()

    enr.to_parquet(out_dir / "mur_university_enrolments.parquet", index=False)
    gra.to_parquet(out_dir / "mur_university_graduates.parquet", index=False)
    summary.to_parquet(out_dir / "mur_university_summary.parquet", index=False)
    return summary


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "fetch":
        fetch("--force" in sys.argv)
    else:
        df = build()
        print(f"Built university summary: {len(df)} rows across {df['comune_id'].nunique()} municipalities.")
        print(df[df["comune_id"] == "040007"].tail(6).to_string())
