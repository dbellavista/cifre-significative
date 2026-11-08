"""OpenCoesione at municipal level: project-level rows for a set of Italian municipalities, cycles 2014-2020 and 2021-2027 (the regional tables are in `opencoesione`)."""
import zipfile
from pathlib import Path

import pandas as pd

from cifresignificative.core.geo import COMUNI_BY_ID, from_prov_comune
from cifresignificative.core.sources.opencoesione import FUNDS, RAW_DIR, _read

THEME_MAP: dict[str, str] = {
    "Ricerca e innovazione": "ricerca_innovazione",
    "Reti e servizi digitali": "ricerca_innovazione",
    "Competitività delle imprese": "imprese_competitivita",
    "Trasporti e mobilità": "infrastrutture_mobilita",
    "Ambiente e risorse naturali": "ambiente_energia",
    "Energia": "ambiente_energia",
    "Istruzione e formazione": "istruzione_cultura",
    "Cultura e turismo": "istruzione_cultura",
    "Inclusione sociale e salute": "inclusione_rigenerazione",
    "Occupazione e lavoro": "inclusione_rigenerazione",
    "Capacità amministrativa": "altro",
}
MUNI_COLS = [
    "COD_LOCALE_PROGETTO", "CUP", "OC_TITOLO_PROGETTO", "OC_COD_TEMA_SINTETICO", "OC_TEMA_SINTETICO",
    "CUP_DESCR_NATURA", "CUP_DESCR_SETTORE", "OC_DATA_INIZIO_PROGETTO", "OC_DATA_FINE_PROGETTO_PREVISTA",
    "OC_DATA_FINE_PROGETTO_EFFETTIVA", "OC_STATO_PROGETTO", "OC_FINANZ_UE_NETTO", "OC_FINANZ_COMUNE_NETTO",
    "OC_FINANZ_TOT_PUB_NETTO", "IMPEGNI", "TOT_PAGAMENTI",
    "OC_DENOM_ATTUATORE", "OC_DENOM_BENEFICIARIO", "OC_STATO_PROCEDURALE",
    "DATA_INIZIO_EFF_ESECUZIONE", "DATA_FINE_EFF_ESECUZIONE",
    "DATA_INIZIO_EFF_STIP_ATTRIB", "DATA_FINE_EFF_STIP_ATTRIB",
]


def project_municipalities(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Return `(loc_df, n_comuni)` mapping each `COD_LOCALE_PROGETTO` to its distinct 6-digit Italian `comune_id`s and total municipality count."""
    L = _read(raw_dir / "localizzazioni.zip", ["COD_LOCALE_PROGETTO", "COD_PROVINCIA", "COD_COMUNE"])
    L["comune_id"] = from_prov_comune(L["COD_PROVINCIA"], L["COD_COMUNE"])
    L = L.dropna(subset=["comune_id"]).drop_duplicates(["COD_LOCALE_PROGETTO", "comune_id"])
    n_comuni = pd.Series(L.groupby("COD_LOCALE_PROGETTO").size(), name="n_comuni")
    return L[["COD_LOCALE_PROGETTO", "comune_id"]], n_comuni


def _load_municipal_projects_2127(comuni_ids: set[str], raw_dir: Path) -> pd.DataFrame:
    """Extract project-level OpenCoesione rows for `comuni_ids` from the 2021-2027 cycle (`progetti_esteso_EMR_2021-2027.zip`)."""
    zpath = raw_dir / "progetti_esteso_EMR_2021-2027.zip"
    if not zpath.exists():
        return pd.DataFrame()
    z = zipfile.ZipFile(zpath)
    fname = z.namelist()[0]
    first_line = z.open(fname).readline().decode("utf-8", errors="ignore")
    available_cols = [c for c in MUNI_COLS + ["COD_COMUNE", "FONDO_COMUNITARIO", "OC_COD_FONTE"] if c in first_line]
    raw_df = pd.read_csv(z.open(fname), sep=";", usecols=available_cols, low_memory=False)

    records: list[dict] = []
    for _, row in raw_df.iterrows():
        cod_c = row.get("COD_COMUNE")
        if pd.isna(cod_c):
            continue
        parts = [p.strip()[-6:] for p in str(cod_c).split(":::") if len(p.strip()) >= 6]
        n_c = max(len(parts), 1)
        for c in parts:
            if c in comuni_ids:
                rec = row.to_dict()
                rec["comune_id"] = c
                rec["n_comuni"] = n_c
                records.append(rec)
    if not records:
        return pd.DataFrame()

    res = pd.DataFrame(records)
    for c in ("OC_FINANZ_UE_NETTO", "OC_FINANZ_COMUNE_NETTO", "OC_FINANZ_TOT_PUB_NETTO", "IMPEGNI", "TOT_PAGAMENTI"):
        res[c] = pd.to_numeric(res[c].astype(str).str.replace(",", "."), errors="coerce").fillna(0.0)
    for dcol in (
        "OC_DATA_INIZIO_PROGETTO", "OC_DATA_FINE_PROGETTO_PREVISTA", "OC_DATA_FINE_PROGETTO_EFFETTIVA",
        "DATA_INIZIO_EFF_ESECUZIONE", "DATA_FINE_EFF_ESECUZIONE",
        "DATA_INIZIO_EFF_STIP_ATTRIB", "DATA_FINE_EFF_STIP_ATTRIB",
    ):
        res[dcol] = pd.to_datetime(res[dcol].astype(str).str.strip(), format="%Y%m%d", errors="coerce")

    res["allocated_eur"] = res["OC_FINANZ_TOT_PUB_NETTO"] / res["n_comuni"]
    res["eu_eur"] = res["OC_FINANZ_UE_NETTO"] / res["n_comuni"]
    res["comune_cofin_eur"] = res["OC_FINANZ_COMUNE_NETTO"] / res["n_comuni"]
    res["committed_eur"] = res["IMPEGNI"] / res["n_comuni"]
    res["disbursed_eur"] = res["TOT_PAGAMENTI"] / res["n_comuni"]
    res["theme_code"] = res["OC_TEMA_SINTETICO"].map(lambda x: THEME_MAP.get(str(x).strip(), "altro"))
    res["start_year"] = res["OC_DATA_INIZIO_PROGETTO"].dt.year
    res["lag_days"] = (res["OC_DATA_FINE_PROGETTO_EFFETTIVA"] - res["OC_DATA_INIZIO_PROGETTO"]).dt.days
    res["delay_days"] = (res["OC_DATA_FINE_PROGETTO_EFFETTIVA"] - res["OC_DATA_FINE_PROGETTO_PREVISTA"]).dt.days
    res["exec_lag_days"] = (res["DATA_FINE_EFF_ESECUZIONE"] - res["DATA_INIZIO_EFF_ESECUZIONE"]).dt.days
    res["award_lag_days"] = (res["DATA_FINE_EFF_STIP_ATTRIB"] - res["DATA_INIZIO_EFF_STIP_ATTRIB"]).dt.days
    res["status"] = res["OC_STATO_PROGETTO"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    res["procedural_status"] = res["OC_STATO_PROCEDURALE"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    res["actor_name"] = res["OC_DENOM_ATTUATORE"].astype(str).str.strip()
    res["beneficiary_name"] = res["OC_DENOM_BENEFICIARIO"].astype(str).str.strip()
    res["fund_source"] = res["FONDO_COMUNITARIO"].fillna(res.get("OC_COD_FONTE", "")).astype(str).str.strip()
    res["dataset"] = "opencoesione_2127"

    return res.rename(
        columns={
            "COD_LOCALE_PROGETTO": "project_id",
            "CUP": "cup",
            "OC_TITOLO_PROGETTO": "title",
            "OC_TEMA_SINTETICO": "theme_label",
            "CUP_DESCR_NATURA": "cup_nature",
            "CUP_DESCR_SETTORE": "cup_sector",
            "OC_DATA_INIZIO_PROGETTO": "start_date",
            "OC_DATA_FINE_PROGETTO_PREVISTA": "end_date_planned",
            "OC_DATA_FINE_PROGETTO_EFFETTIVA": "end_date_actual",
            "DATA_INIZIO_EFF_ESECUZIONE": "exec_start_date",
            "DATA_FINE_EFF_ESECUZIONE": "exec_end_date",
            "DATA_INIZIO_EFF_STIP_ATTRIB": "award_start_date",
            "DATA_FINE_EFF_STIP_ATTRIB": "award_end_date",
        }
    )[
        [
            "project_id", "cup", "comune_id", "n_comuni", "dataset", "fund_source", "theme_code", "theme_label",
            "cup_nature", "cup_sector", "title", "start_date", "end_date_planned", "end_date_actual", "start_year",
            "exec_start_date", "exec_end_date", "award_start_date", "award_end_date",
            "allocated_eur", "eu_eur", "comune_cofin_eur", "committed_eur", "disbursed_eur",
            "status", "procedural_status", "actor_name", "beneficiary_name",
            "lag_days", "delay_days", "exec_lag_days", "award_lag_days",
        ]
    ]


def load_municipal_projects(
    comuni_ids: set[str] | None = None,
    raw_dir: Path = RAW_DIR,
    cycle: str = "2014-2020",
) -> pd.DataFrame:
    """Extract project-level OpenCoesione rows for `comuni_ids` (defaults to the 13 ER benchmark municipalities in `core.geo.COMUNI_BY_ID`).

    `cycle` can be:
    - `"2014-2020"` (default): Cohesion projects from the 2014-2020 programming cycle (`FESR`, `FSE`, `FSC`, `POC`, `PAC`, `FEASR`).
    - `"2021-2027"`: Cohesion projects from the 2021-2027 programming cycle (`progetti_esteso_EMR_2021-2027.zip`).
    - `"all"`: Concatenation of both cycles.

    Multi-municipality projects have their financial amounts (`allocated_eur`, `disbursed_eur`, `eu_eur`, `comune_cofin_eur`) divided equally
    by `n_comuni` (the number of distinct Italian municipalities on the project), matching `build()`'s regional convention, while keeping
    `n_comuni` so single-municipality projects (`n_comuni == 1`) can be analysed separately.
    """
    target = set(COMUNI_BY_ID) if comuni_ids is None else set(comuni_ids)
    frames: list[pd.DataFrame] = []

    if cycle in ("2014-2020", "all"):
        loc, n_comuni = project_municipalities(raw_dir)
        sub_loc = loc[loc["comune_id"].isin(target)].merge(n_comuni, left_on="COD_LOCALE_PROGETTO", right_index=True, how="left")
        target_projs = set(sub_loc["COD_LOCALE_PROGETTO"])
        for f in FUNDS:
            zpath = raw_dir / f"progetti_esteso_{f}_2014-2020.zip"
            if not zpath.exists():
                continue
            P = _read(zpath, MUNI_COLS)
            P = P[P["COD_LOCALE_PROGETTO"].isin(target_projs)].copy()
            if P.empty:
                continue
            for c in ("OC_FINANZ_UE_NETTO", "OC_FINANZ_COMUNE_NETTO", "OC_FINANZ_TOT_PUB_NETTO", "IMPEGNI", "TOT_PAGAMENTI"):
                P[c] = pd.to_numeric(P[c].str.replace(",", "."), errors="coerce").fillna(0.0)
            for dcol in (
                "OC_DATA_INIZIO_PROGETTO", "OC_DATA_FINE_PROGETTO_PREVISTA", "OC_DATA_FINE_PROGETTO_EFFETTIVA",
                "DATA_INIZIO_EFF_ESECUZIONE", "DATA_FINE_EFF_ESECUZIONE",
                "DATA_INIZIO_EFF_STIP_ATTRIB", "DATA_FINE_EFF_STIP_ATTRIB",
            ):
                P[dcol] = pd.to_datetime(P[dcol].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
            m = sub_loc.merge(P, on="COD_LOCALE_PROGETTO", how="inner")
            m["allocated_eur"] = m["OC_FINANZ_TOT_PUB_NETTO"] / m["n_comuni"]
            m["eu_eur"] = m["OC_FINANZ_UE_NETTO"] / m["n_comuni"]
            m["comune_cofin_eur"] = m["OC_FINANZ_COMUNE_NETTO"] / m["n_comuni"]
            m["committed_eur"] = m["IMPEGNI"] / m["n_comuni"]
            m["disbursed_eur"] = m["TOT_PAGAMENTI"] / m["n_comuni"]
            m["theme_code"] = m["OC_TEMA_SINTETICO"].map(lambda x: THEME_MAP.get(str(x).strip(), "altro"))
            m["start_year"] = m["OC_DATA_INIZIO_PROGETTO"].dt.year
            m["lag_days"] = (m["OC_DATA_FINE_PROGETTO_EFFETTIVA"] - m["OC_DATA_INIZIO_PROGETTO"]).dt.days
            m["delay_days"] = (m["OC_DATA_FINE_PROGETTO_EFFETTIVA"] - m["OC_DATA_FINE_PROGETTO_PREVISTA"]).dt.days
            m["exec_lag_days"] = (m["DATA_FINE_EFF_ESECUZIONE"] - m["DATA_INIZIO_EFF_ESECUZIONE"]).dt.days
            m["award_lag_days"] = (m["DATA_FINE_EFF_STIP_ATTRIB"] - m["DATA_INIZIO_EFF_STIP_ATTRIB"]).dt.days
            m["status"] = m["OC_STATO_PROGETTO"].astype(str).str.strip().str.lower().str.replace(" ", "_")
            m["procedural_status"] = m["OC_STATO_PROCEDURALE"].astype(str).str.strip().str.lower().str.replace(" ", "_")
            m["actor_name"] = m["OC_DENOM_ATTUATORE"].astype(str).str.strip()
            m["beneficiary_name"] = m["OC_DENOM_BENEFICIARIO"].astype(str).str.strip()
            frames.append(
                m.rename(
                    columns={
                        "COD_LOCALE_PROGETTO": "project_id",
                        "CUP": "cup",
                        "OC_TITOLO_PROGETTO": "title",
                        "OC_TEMA_SINTETICO": "theme_label",
                        "CUP_DESCR_NATURA": "cup_nature",
                        "CUP_DESCR_SETTORE": "cup_sector",
                        "OC_DATA_INIZIO_PROGETTO": "start_date",
                        "OC_DATA_FINE_PROGETTO_PREVISTA": "end_date_planned",
                        "OC_DATA_FINE_PROGETTO_EFFETTIVA": "end_date_actual",
                        "DATA_INIZIO_EFF_ESECUZIONE": "exec_start_date",
                        "DATA_FINE_EFF_ESECUZIONE": "exec_end_date",
                        "DATA_INIZIO_EFF_STIP_ATTRIB": "award_start_date",
                        "DATA_FINE_EFF_STIP_ATTRIB": "award_end_date",
                    }
                ).assign(fund_source=f, dataset="opencoesione_1420")[
                    [
                        "project_id", "cup", "comune_id", "n_comuni", "dataset", "fund_source", "theme_code", "theme_label",
                        "cup_nature", "cup_sector", "title", "start_date", "end_date_planned", "end_date_actual", "start_year",
                        "exec_start_date", "exec_end_date", "award_start_date", "award_end_date",
                        "allocated_eur", "eu_eur", "comune_cofin_eur", "committed_eur", "disbursed_eur",
                        "status", "procedural_status", "actor_name", "beneficiary_name",
                        "lag_days", "delay_days", "exec_lag_days", "award_lag_days",
                    ]
                ]
            )

    if cycle in ("2021-2027", "all"):
        df_2127 = _load_municipal_projects_2127(target, raw_dir)
        if not df_2127.empty:
            frames.append(df_2127)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
