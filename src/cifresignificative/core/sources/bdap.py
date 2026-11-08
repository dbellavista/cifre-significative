"""Municipal balance sheets ('Bilanci comunali'): D.P.R. 194/1996 (2008-2015) and D.Lgs. 118/2011 harmonized accounts (2016-2022).

Combines three official/open-data municipal budget repositories into `data/raw/bdap/` (gitignored):
1. **OpenBilanci Storico OpenData ZIPs (`2008-2014`)**: `http://storico.openbilanci.it/opendata/<cod_finloc>.zip` (CSVs of Quadri 02, 03, 04 of D.P.R. 194/1996 Certificati Consuntivi).
2. **Ministero dell'Interno DAIT Finanza Locale (`2015` CCOU and `2016-2018` CCOX)**:
   - `2015` `CCOU` (`Quadro 02` Entrate, `Quadro 03` Spese, `Quadro 04` Spese correnti per intervento incl. `Personale`).
   - `2016-2018` `CCOX` (`Quadro 09` Accertamenti tributari IMU/Addizionale/TARI, `Quadro 10` Spese correnti per macroaggregati incl. `Redditi da lavoro dipendente`, and `Quadro 20` Piano degli Indicatori di Bilancio `1.1 Incidenza spese rigide`).
3. **OpenBilanci Bilanci Armonizzati JSON trees (`2016-2021` consuntivo, `2022` preventivo)**:
   - `https://openbilanci.it/armonizzati/bilanci/<slug>/{entrate,spese}/dettaglio.json?year=<Y>&type=<type>&cas_com_type=<competenza|cassa>`
   - Provides full breakdown of revenues by Title/Category and expenditures by Mission (`M01`..`M20`, `M50`, `M60`, `M99`) and Program, decomposed into `Titolo I` (`correnti_value`), `Titolo II` (`investimenti_value`), `Titolo III` (`incremento_fin_value`), and `Titolo IV` (`altro_value` rimborso prestiti).
   - **Pitfall avoided**: at the parent Mission node, OpenBilanci's backend propagates `NaN` when any child Program has an empty/zero capital or current cell (e.g., Cesena Mission 05 Education and Mission 11 Transport in 2021). We always sum `correnti_value`, `investimenti_value`, `incremento_fin_value`, and `altro_value` across the child Program nodes (falling back to the Mission node only when it has no children).

All pre-2016 rows are explicitly stamped with `regime = "dpr194"` and `flag = "break_dlgs118"` so downstream analyses never splice pre- and post-harmonization accounting series silently.
"""
import io
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import truststore

from cifresignificative.core.geo import COMUNI_BY_ID
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/bdap")
EXTERNAL_DIR = Path("data/external/source/bdap_comuni")
STORICO_URL = "http://storico.openbilanci.it/opendata"
OB_ARM_URL = "https://openbilanci.it/armonizzati/bilanci"
FLOC_URL = "https://finanzalocale.interno.gov.it/apps/floc.php/certificati/index"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}

STORICO_YEARS = tuple(range(2008, 2015))
HARMONIZED_CONS_YEARS = tuple(range(2016, 2022))
HARMONIZED_PREV_YEARS = (2022,)
FLOC_CCOX_YEARS = (2016, 2017, 2018)


@dataclass(frozen=True)
class BudgetMeta:
    comune_id: str
    slug: str
    cod_finloc: str
    finloc_id: str


BUDGET_ENTITIES: dict[str, BudgetMeta] = {
    "040007": BudgetMeta("040007", "cesena-comune-fc", "CESENA--2080320060", "2080320060"),
    "040012": BudgetMeta("040012", "forli-comune-fc", "FORLI--2080320110", "2080320110"),
    "099014": BudgetMeta("099014", "rimini-comune-rn", "RIMINI--2081010140", "2081010140"),
    "039014": BudgetMeta("039014", "ravenna-comune-ra", "RAVENNA--2080660140", "2080660140"),
    "037032": BudgetMeta("037032", "imola-comune-bo", "IMOLA--2080130320", "2080130320"),
    "036005": BudgetMeta("036005", "carpi-comune-mo", "CARPI--2080500050", "2080500050"),
    "035033": BudgetMeta("035033", "reggio-nellemilia-comune-re", "REGGIO-NELL-EMILIA--2080680330", "2080680330"),
    "038008": BudgetMeta("038008", "ferrara-comune-fe", "FERRARA--2080290080", "2080290080"),
    "036023": BudgetMeta("036023", "modena-comune-mo", "MODENA--2080500230", "2080500230"),
    "034027": BudgetMeta("034027", "parma-comune-pr", "PARMA--2080560270", "2080560270"),
    "033032": BudgetMeta("033032", "piacenza-comune-pc", "PIACENZA--2080610320", "2080610320"),
    "039010": BudgetMeta("039010", "faenza-comune-ra", "FAENZA--2080660100", "2080660100"),
    "037006": BudgetMeta("037006", "bologna-comune-bo", "BOLOGNA--2080130060", "2080130060"),
}

MISSION_MAP: dict[str, tuple[str, str]] = {
    "ccox-quadro-4-2": ("M01", "Servizi istituzionali, generali e di gestione"),
    "ccox-quadro-4-3": ("M02", "Giustizia"),
    "ccox-quadro-4-4": ("M03", "Ordine pubblico e sicurezza"),
    "ccox-quadro-4-5": ("M04", "Istruzione e diritto allo studio"),
    "ccox-quadro-4-6": ("M05", "Tutela e valorizzazione dei beni e attività culturali"),
    "ccox-quadro-4-7": ("M06", "Politiche giovanili, sport e tempo libero"),
    "ccox-quadro-4-8": ("M07", "Turismo"),
    "ccox-quadro-4-9": ("M08", "Assetto del territorio ed edilizia abitativa"),
    "ccox-quadro-4-10": ("M09", "Sviluppo sostenibile e tutela del territorio e dell'ambiente"),
    "ccox-quadro-4-11": ("M10", "Trasporti e diritto alla mobilità"),
    "ccox-quadro-4-12": ("M11", "Soccorso civile"),
    "ccox-quadro-4-13": ("M12", "Diritti sociali, politiche sociali e famiglia"),
    "ccox-quadro-4-14": ("M13", "Tutela della salute"),
    "ccox-quadro-4-15": ("M14", "Sviluppo economico e competitività"),
    "ccox-quadro-4-16": ("M15", "Politiche per il lavoro e la formazione professionale"),
    "ccox-quadro-4-17": ("M16", "Agricoltura, politiche agroalimentari e pesca"),
    "ccox-quadro-4-18": ("M17", "Energia e diversificazione delle fonti energetiche"),
    "ccox-quadro-4-19": ("M18", "Relazioni con le altre autonomie territoriali e locali"),
    "ccox-quadro-4-20": ("M19", "Relazioni internazionali"),
    "ccox-quadro-4-21": ("M20", "Fondi e accantonamenti"),
    "ccox-quadro-4-22": ("M50", "Debito pubblico"),
    "ccox-quadro-4-23": ("M60", "Anticipazioni finanziarie"),
    "ccox-quadro-4-24": ("M99", "Servizi per conto terzi"),
}


def _to_num(val: object) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(".", "").replace(",", ".")
    if not s or s.lower() in ("nan", "none", "-", "n.c.", "( n.c.)"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _clean_num(val: object) -> float:
    v = _to_num(val)
    return 0.0 if np.isnan(v) else v


def storico_zip_path(cid: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / "storico" / f"{cid}_{BUDGET_ENTITIES[cid].cod_finloc}.zip"


def armonizzati_json_path(cid: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / "armonizzati" / f"{cid}_openbilanci_2016_2022.json"


def floc_bundle_path(cid: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / "floc" / f"{cid}_floc_2015_2018.json"


def fetch(force: bool = False, raw_dir: Path = RAW_DIR, comuni_ids: tuple[str, ...] | None = None) -> Path:
    """Download OpenBilanci storico ZIPs (2008-2014), Finanza Locale HTML tables (2015-2018), and OpenBilanci harmonized JSONs (2016-2022)."""
    targets = tuple(BUDGET_ENTITIES) if comuni_ids is None else comuni_ids
    (raw_dir / "storico").mkdir(parents=True, exist_ok=True)
    (raw_dir / "armonizzati").mkdir(parents=True, exist_ok=True)
    (raw_dir / "floc").mkdir(parents=True, exist_ok=True)

    sess = requests.Session()
    sess.headers.update(UA)

    for cid in targets:
        meta = BUDGET_ENTITIES[cid]

        # 1. Storico ZIP (2005-2014); Reggio nell'Emilia ('035033') returns 404 due to apostrophe in batch slug -> fallback to Finanza Locale CCOU
        zp = storico_zip_path(cid, raw_dir)
        has_storico = zp.exists()
        if force or not zp.exists():
            url = f"{STORICO_URL}/{meta.cod_finloc}.zip"
            r = sess.get(url, timeout=60)
            if r.status_code == 200 and r.content.startswith(b"PK"):
                zp.write_bytes(r.content)
                has_storico = True
                record(f"bdap_storico_{cid}", source="OpenBilanci / DAIT", dataset=f"storico_{meta.cod_finloc}", url=url)
            elif r.status_code != 404:
                r.raise_for_status()

        # 2. Harmonized JSON bundle (2016-2021 consuntivo, 2022 preventivo)
        ap = armonizzati_json_path(cid, raw_dir)
        if force or not ap.exists():
            bundle: dict[str, dict] = {}
            year_specs = [(y, "consuntivo") for y in HARMONIZED_CONS_YEARS] + [(y, "preventivo") for y in HARMONIZED_PREV_YEARS]
            for y, btype in year_specs:
                y_key = f"{y}_{btype}"
                bundle[y_key] = {}
                for side in ("entrate", "spese"):
                    for cas_com in ("competenza", "cassa"):
                        u = f"{OB_ARM_URL}/{meta.slug}/{side}/dettaglio.json?year={y}&type={btype}&cas_com_type={cas_com}"
                        resp = sess.get(u, timeout=30)
                        resp.raise_for_status()
                        bundle[y_key][f"{side}_{cas_com}"] = resp.json()
            ap.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
            record(f"bdap_armonizzati_{cid}", source="OpenBilanci / BDAP", dataset=f"armonizzati_{meta.slug}", url=f"{OB_ARM_URL}/{meta.slug}")

        # 3. Finanza Locale HTML tables (2015 CCOU Q02,03,04 + 2008-2014 CCOU when storico zip is absent + 2016-2018 CCOX Q09,10,20)
        fp = floc_bundle_path(cid, raw_dir)
        if force or not fp.exists():
            fl_data: dict[str, str] = {}
            ccou_years = (2015,) if has_storico else (*STORICO_YEARS, 2015)
            for y_ccou in ccou_years:
                for q in ("02", "03", "04"):
                    u = f"{FLOC_URL}/codice_ente/{meta.finloc_id}/cod/4/anno/{y_ccou}/md/0/cod_modello/CCOU/tipo_modello/U/cod_quadro/{q}"
                    resp = sess.get(u, timeout=30)
                    resp.raise_for_status()
                    fl_data[f"{y_ccou}_CCOU_{q}"] = resp.text
            for y in FLOC_CCOX_YEARS:
                for q in ("09", "10", "20"):
                    u = f"{FLOC_URL}/codice_ente/{meta.finloc_id}/cod/4/anno/{y}/md/0/cod_modello/CCOX/tipo_modello/X/cod_quadro/{q}"
                    resp = sess.get(u, timeout=30)
                    resp.raise_for_status()
                    fl_data[f"{y}_CCOX_{q}"] = resp.text
            fp.write_text(json.dumps(fl_data, ensure_ascii=False), encoding="utf-8")
            record(f"bdap_floc_{cid}", source="DAIT Finanza Locale", dataset=f"floc_{meta.finloc_id}", url=FLOC_URL)

    return raw_dir


def _parse_html_table_rows(html: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE):
        cells = [
            re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip()
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.DOTALL | re.IGNORECASE)
        ]
        if any(cells):
            rows.append(cells)
    return rows


def _extract_dpr194_row_metrics(
    q2_rows: list[list[str]],
    q3_rows: list[list[str]],
    q4_impegni_df: pd.DataFrame | None,
) -> dict[str, float]:
    """Extract standardized revenue and expenditure metrics from D.P.R. 194/1996 Quadri 02, 03, and 04."""
    out: dict[str, float] = {
        "entrate_tributarie_comp_eur": np.nan,
        "entrate_tributarie_cassa_eur": np.nan,
        "imposte_tasse_comp_eur": 0.0,
        "imposte_tasse_cassa_eur": 0.0,
        "fondo_perequativo_comp_eur": 0.0,
        "imu_ici_comp_eur": 0.0,
        "addizionale_irpef_comp_eur": np.nan,
        "tari_tarsu_comp_eur": 0.0,
        "trasferimenti_correnti_comp_eur": np.nan,
        "trasferimenti_correnti_cassa_eur": np.nan,
        "entrate_extratributarie_comp_eur": np.nan,
        "entrate_extratributarie_cassa_eur": np.nan,
        "entrate_capitale_comp_eur": np.nan,
        "entrate_capitale_cassa_eur": np.nan,
        "contributi_investimenti_comp_eur": 0.0,
        "contributi_investimenti_cassa_eur": 0.0,
        "alienazioni_comp_eur": 0.0,
        "alienazioni_cassa_eur": 0.0,
        "accensione_prestiti_comp_eur": np.nan,
        "accensione_prestiti_cassa_eur": np.nan,
        "accensione_mutui_comp_eur": 0.0,
        "accensione_mutui_cassa_eur": 0.0,
        "fpv_corrente_comp_eur": np.nan,
        "fpv_capitale_comp_eur": np.nan,
        "avanzo_amministrazione_comp_eur": np.nan,
        "entrate_totali_comp_eur": np.nan,
        "entrate_totali_cassa_eur": np.nan,
        "spese_correnti_comp_eur": np.nan,
        "spese_correnti_cassa_eur": np.nan,
        "spese_capitale_comp_eur": np.nan,
        "spese_capitale_cassa_eur": np.nan,
        "rimborso_prestiti_comp_eur": np.nan,
        "rimborso_prestiti_cassa_eur": np.nan,
        "spese_totali_comp_eur": np.nan,
        "spese_totali_cassa_eur": np.nan,
        "spesa_personale_comp_eur": np.nan,
        "interessi_passivi_comp_eur": np.nan,
    }

    for r in q2_rows:
        if len(r) < 4:
            continue
        lbl = r[0].strip().lower()
        acc = _clean_num(r[1])
        cassa = _clean_num(r[2]) + _clean_num(r[3])
        if lbl == "totale entrate tributarie":
            out["entrate_tributarie_comp_eur"] = acc
            out["entrate_tributarie_cassa_eur"] = cassa
        elif lbl in ("categoria 1 - imposte", "categoria 2 - tasse"):
            out["imposte_tasse_comp_eur"] += acc
            out["imposte_tasse_cassa_eur"] += cassa
        elif "fondo di solidariet" in lbl or "fondo sperimentale di riequilibrio" in lbl:
            out["fondo_perequativo_comp_eur"] += acc
        elif lbl.startswith(("addizionale irpef", "addizionale comunale irpef")):
            out["addizionale_irpef_comp_eur"] = acc
        elif (
            lbl.startswith(("i.m.u.", "imposta municipale propria", "i.c.i.", "imposta comunale sugli immobili"))
            and "di cui" not in lbl
        ) or lbl.startswith(("- i.m.u. fattispecie", "- entrata a titolo di i.m.u.", "- i.c.i. su abitazione", "- i.c.i. su altri")):
            out["imu_ici_comp_eur"] += acc
        elif lbl.startswith((
            "tassa sui rifiuti (tari)",
            "tributo comunale sui rifiuti",
            "tassa per la raccolta e lo smaltimento dei rifiuti",
            "tributo sui servizi indivisibili (tasi)",
        )):
            out["tari_tarsu_comp_eur"] += acc
        elif "totale entrate derivanti da contributi e trasferimenti correnti" in lbl:
            out["trasferimenti_correnti_comp_eur"] = acc
            out["trasferimenti_correnti_cassa_eur"] = cassa
        elif "totale entrate extratributarie" in lbl:
            out["entrate_extratributarie_comp_eur"] = acc
            out["entrate_extratributarie_cassa_eur"] = cassa
        elif "totale entrate derivanti da alienazione" in lbl:
            out["entrate_capitale_comp_eur"] = acc
            out["entrate_capitale_cassa_eur"] = cassa
        elif lbl.startswith(("categoria 1 - alienazione", "categoria 1° - alienazione")):
            out["alienazioni_comp_eur"] += acc
            out["alienazioni_cassa_eur"] += cassa
        elif lbl.startswith((
            "categoria 2 - trasferimenti di capitali",
            "categoria 2° - trasferimenti di capitali",
            "categoria 3 - trasferimenti di capitali",
            "categoria 3° - trasferimenti di capitali",
            "categoria 4 - trasferimenti di capitali",
            "categoria 4° - trasferimenti di capitali",
            "categoria 5 - trasferimenti di capitali",
            "categoria 5° - trasferimenti di capitali",
        )):
            out["contributi_investimenti_comp_eur"] += acc
            out["contributi_investimenti_cassa_eur"] += cassa
        elif lbl.startswith(("categoria 3 - assunzioni di mutui", "categoria 3° - assunzioni di mutui")):
            out["accensione_mutui_comp_eur"] += acc
            out["accensione_mutui_cassa_eur"] += cassa
        elif "totale entrate derivanti da accensione di prestiti" in lbl or "totale entrate derivanti da accensioni di prestiti" in lbl:
            out["accensione_prestiti_comp_eur"] = acc
            out["accensione_prestiti_cassa_eur"] = cassa
        elif lbl == "totale generale delle entrate":
            out["entrate_totali_comp_eur"] = acc
            out["entrate_totali_cassa_eur"] = cassa

    for r in q3_rows:
        if len(r) < 4:
            continue
        lbl = r[0].strip().lower()
        imp = _clean_num(r[1])
        pag = _clean_num(r[2]) + _clean_num(r[3])
        if lbl.startswith("titolo i - spese correnti"):
            out["spese_correnti_comp_eur"] = imp
            out["spese_correnti_cassa_eur"] = pag
        elif lbl.startswith(("titolo ii - spese in c/capitale", "titolo ii - spese in conto capitale")):
            out["spese_capitale_comp_eur"] = imp
            out["spese_capitale_cassa_eur"] = pag
        elif lbl.startswith("titolo iii - spese per rimborso di prestiti"):
            out["rimborso_prestiti_comp_eur"] = imp
            out["rimborso_prestiti_cassa_eur"] = pag
        elif lbl == "totale generale delle spese":
            out["spese_totali_comp_eur"] = imp
            out["spese_totali_cassa_eur"] = pag

    if q4_impegni_df is not None and len(q4_impegni_df):
        lbl_col = q4_impegni_df.columns[0]
        tot_rows = q4_impegni_df[q4_impegni_df[lbl_col].astype(str).str.strip().str.lower() == "totale"]
        if len(tot_rows):
            row0 = tot_rows.iloc[0]
            pers_col = next((c for c in q4_impegni_df.columns if "personale" in c.lower()), None)
            int_col = next((c for c in q4_impegni_df.columns if "interessi passivi" in c.lower()), None)
            if pers_col:
                out["spesa_personale_comp_eur"] = _to_num(row0[pers_col])
            if int_col:
                out["interessi_passivi_comp_eur"] = _to_num(row0[int_col])

    out["entrate_correnti_comp_eur"] = (
        _clean_num(out["entrate_tributarie_comp_eur"])
        + _clean_num(out["trasferimenti_correnti_comp_eur"])
        + _clean_num(out["entrate_extratributarie_comp_eur"])
    )
    out["entrate_correnti_cassa_eur"] = (
        _clean_num(out["entrate_tributarie_cassa_eur"])
        + _clean_num(out["trasferimenti_correnti_cassa_eur"])
        + _clean_num(out["entrate_extratributarie_cassa_eur"])
    )
    return out


def parse_storico_zip(zpath: Path, cid: str, years: tuple[int, ...] = STORICO_YEARS) -> list[dict]:
    """Parse D.P.R. 194/1996 Consuntivo CSVs (2008-2014) from an OpenBilanci `storico` ZIP."""
    rows_out: list[dict] = []
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        for y in years:
            y_prefix = f"/{y}/"
            cons_files = [n for n in names if y_prefix in n and "consuntivo" in n.lower()]
            if not cons_files:
                continue
            q2_rows: list[list[str]] = []
            q3_rows: list[list[str]] = []
            q4_df: pd.DataFrame | None = None

            for fn in sorted(cons_files):
                if "/02/quadro-2-" in fn:
                    df = pd.read_csv(io.BytesIO(z.read(fn)), sep=";", dtype=str)
                    q2_rows.extend(df.fillna("").values.tolist())
                elif "/03/quadro-3-riepilogo-generale-delle-spese" in fn:
                    df = pd.read_csv(io.BytesIO(z.read(fn)), sep=";", dtype=str)
                    q3_rows.extend(df.fillna("").values.tolist())
                elif "/04/quadro-4-a-impegni" in fn:
                    q4_df = pd.read_csv(io.BytesIO(z.read(fn)), sep=";", dtype=str)

            m = _extract_dpr194_row_metrics(q2_rows, q3_rows, q4_df)
            if np.isnan(m["spese_correnti_comp_eur"]) or m["entrate_correnti_comp_eur"] <= 0:
                continue
            if np.isnan(m["entrate_totali_comp_eur"]):
                m["entrate_totali_comp_eur"] = (
                    m["entrate_correnti_comp_eur"]
                    + _clean_num(m["entrate_capitale_comp_eur"])
                    + _clean_num(m["accensione_prestiti_comp_eur"])
                )
                m["entrate_totali_cassa_eur"] = (
                    m["entrate_correnti_cassa_eur"]
                    + _clean_num(m["entrate_capitale_cassa_eur"])
                    + _clean_num(m["accensione_prestiti_cassa_eur"])
                )
            rows_out.append({
                "comune_id": cid,
                "comune": COMUNI_BY_ID[cid].name,
                "year": y,
                "regime": "dpr194",
                "budget_type": "consuntivo",
                "flag": "break_dlgs118",
                **m,
                "missione_01_correnti_comp_eur": np.nan,
                "missione_50_debito_comp_eur": np.nan,
                "rigidity_quadro20_pct": np.nan,
            })
    return rows_out


def parse_floc_ccou_year(fl_bundle: dict[str, str], cid: str, year: int = 2015) -> dict | None:
    """Parse a D.P.R. 194/1996 CCOU Consuntivo year (`2008-2015`) from Finanza Locale HTML tables."""
    k2, k3, k4 = f"{year}_CCOU_02", f"{year}_CCOU_03", f"{year}_CCOU_04"
    if k2 not in fl_bundle or k3 not in fl_bundle:
        return None
    q2_rows = _parse_html_table_rows(fl_bundle[k2])
    q3_rows = _parse_html_table_rows(fl_bundle[k3])
    q4_rows = _parse_html_table_rows(fl_bundle.get(k4, ""))

    q4_df: pd.DataFrame | None = None
    for i, r in enumerate(q4_rows):
        if len(r) >= 9 and "personale" in r[1].lower():
            hdr = r
            body = [row for row in q4_rows[i + 1 :] if len(row) == len(hdr)]
            if body:
                q4_df = pd.DataFrame(body, columns=hdr)
            break

    m = _extract_dpr194_row_metrics(q2_rows, q3_rows, q4_df)
    if np.isnan(m["spese_correnti_comp_eur"]) or m["entrate_correnti_comp_eur"] <= 0:
        return None
    return {
        "comune_id": cid,
        "comune": COMUNI_BY_ID[cid].name,
        "year": year,
        "regime": "dpr194",
        "budget_type": "consuntivo",
        "flag": "break_dlgs118",
        **m,
        "missione_01_correnti_comp_eur": np.nan,
        "missione_50_debito_comp_eur": np.nan,
        "rigidity_quadro20_pct": np.nan,
    }


def parse_floc_ccox_supplements(fl_bundle: dict[str, str]) -> dict[int, dict[str, float]]:
    """Extract 2016-2018 CCOX Quadro 09 (IMU/Addizionale/TARI), Quadro 10 (Personale/Interessi), and Quadro 20 (Rigidity)."""
    sup: dict[int, dict[str, float]] = {}
    for y in FLOC_CCOX_YEARS:
        d: dict[str, float] = {
            "imu_ici_comp_eur": np.nan,
            "addizionale_irpef_comp_eur": np.nan,
            "tari_tarsu_comp_eur": np.nan,
            "spesa_personale_comp_eur": np.nan,
            "interessi_passivi_comp_eur": np.nan,
            "rigidity_quadro20_pct": np.nan,
        }
        q9 = _parse_html_table_rows(fl_bundle.get(f"{y}_CCOX_09", ""))
        for r in q9:
            if len(r) < 2:
                continue
            lbl = r[0].strip().lower()
            acc = _to_num(r[1])
            if lbl.startswith(("imposta municipale propria", "imposta comunale sugli immobili")):
                d["imu_ici_comp_eur"] = _clean_num(d["imu_ici_comp_eur"]) + _clean_num(acc)
            elif lbl.startswith("addizionale comunale irpef"):
                d["addizionale_irpef_comp_eur"] = acc
            elif lbl.startswith((
                "tassa sui rifiuti",
                "tassa per la raccolta e lo smaltimento dei rifiuti",
                "tassa smaltimento rifiuti",
                "tributo comunale sui rifiuti",
                "tributo sui servizi indivisibili",
                "tributo per i servizi indivisibili",
                "tassa sui servizi comunali",
            )):
                d["tari_tarsu_comp_eur"] = _clean_num(d["tari_tarsu_comp_eur"]) + _clean_num(acc)

        q10 = _parse_html_table_rows(fl_bundle.get(f"{y}_CCOX_10", ""))
        for r in q10:
            if len(r) >= 6 and r[0].strip().upper() == "TOTALE":
                # First TOTALE row in Quadro 10 is always SPESE CORRENTI - IMPEGNI
                d["spesa_personale_comp_eur"] = _to_num(r[1])
                d["interessi_passivi_comp_eur"] = _to_num(r[5])
                break

        q20 = _parse_html_table_rows(fl_bundle.get(f"{y}_CCOX_20", ""))
        for r in q20:
            if len(r) >= 2 and r[0].strip().startswith("1.1 "):
                d["rigidity_quadro20_pct"] = _to_num(r[1])
                break
        sup[y] = d
    return sup


def _node_abs(node: dict, year_str: str) -> float:
    for v in node.get("values", []):
        if year_str in v:
            return _clean_num(v[year_str].get("abs"))
    return 0.0


def _mission_composition(node: dict, year_str: str) -> tuple[float, float, float, float, float]:
    """Return `(correnti, capitale, att_fin, rimborso_prestiti, totale)` for a Mission node by summing child Programs."""
    slug = node.get("slug", "")
    tot = _node_abs(node, year_str)
    if slug == "ccox-quadro-4-24":
        # Missione 99 - Servizi per conto terzi
        return 0.0, 0.0, 0.0, 0.0, tot
    children = node.get("children", [])
    if children:
        corr = sum(_clean_num(ch.get("composition", {}).get("correnti_value")) for ch in children)
        cap = sum(_clean_num(ch.get("composition", {}).get("investimenti_value")) for ch in children)
        fin = sum(_clean_num(ch.get("composition", {}).get("incremento_fin_value")) for ch in children)
        prest = sum(_clean_num(ch.get("composition", {}).get("altro_value")) for ch in children)
        return corr, cap, fin, prest, tot
    comp = node.get("composition", {}) or {}
    return (
        _clean_num(comp.get("correnti_value")),
        _clean_num(comp.get("investimenti_value")),
        _clean_num(comp.get("incremento_fin_value")),
        _clean_num(comp.get("altro_value")),
        tot,
    )


def _e_val(tree: dict[str, dict], y_str: str, slug: str, child_slug: str | None = None) -> float:
    node = tree.get(slug)
    if not node:
        return 0.0
    if child_slug is None:
        return _node_abs(node, y_str)
    for ch in node.get("children", []):
        if ch.get("slug") == child_slug:
            return _node_abs(ch, y_str)
    return 0.0


def parse_armonizzati_bundle(
    arm_bundle: dict[str, dict],
    cid: str,
    ccox_sup: dict[int, dict[str, float]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Parse OpenBilanci harmonized JSON bundle (2016-2022) into `(budget_rows, mission_rows)`."""
    budget_rows: list[dict] = []
    mission_rows: list[dict] = []
    sup_map = ccox_sup or {}

    for y_key, payload in sorted(arm_bundle.items()):
        y_str, btype = y_key.split("_", 1)
        year = int(y_str)

        ent_comp = {item["slug"]: item for item in payload.get("entrate_competenza", [])}
        ent_cassa = {item["slug"]: item for item in payload.get("entrate_cassa", [])}
        spe_comp = {item["slug"]: item for item in payload.get("spese_competenza", [])}
        spe_cassa = {item["slug"]: item for item in payload.get("spese_cassa", [])}

        t1_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-2")
        t1_cassa = _e_val(ent_cassa, y_str, "ccox-quadro-2-2")
        imp_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-2", "ccox-quadro-2-2-41")
        imp_cassa = _e_val(ent_cassa, y_str, "ccox-quadro-2-2", "ccox-quadro-2-2-41")
        fsc_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-2", "ccox-quadro-2-2-70") + _e_val(ent_comp, y_str, "ccox-quadro-2-2", "ccox-quadro-2-2-80")

        t2_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-3")
        t2_cassa = _e_val(ent_cassa, y_str, "ccox-quadro-2-3")
        t3_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-4")
        t3_cassa = _e_val(ent_cassa, y_str, "ccox-quadro-2-4")
        t4_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-5")
        t4_cassa = _e_val(ent_cassa, y_str, "ccox-quadro-2-5")
        contr_inv_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-5", "ccox-quadro-2-5-230")
        contr_inv_cassa = _e_val(ent_cassa, y_str, "ccox-quadro-2-5", "ccox-quadro-2-5-230")
        alien_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-5", "ccox-quadro-2-5-250")
        alien_cassa = _e_val(ent_cassa, y_str, "ccox-quadro-2-5", "ccox-quadro-2-5-250")
        t6_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-7")
        t6_cassa = _e_val(ent_cassa, y_str, "ccox-quadro-2-7")
        mutui_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-7", "ccox-quadro-2-7-340")
        mutui_cassa = _e_val(ent_cassa, y_str, "ccox-quadro-2-7", "ccox-quadro-2-7-340")
        fpv_corr_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-1", "ccox-quadro-2-1-10")
        fpv_cap_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-1", "ccox-quadro-2-1-20")
        avanzo_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-1", "ccox-quadro-2-1-30")
        tot_e_comp = _e_val(ent_comp, y_str, "ccox-quadro-2-11")
        tot_e_cassa = _e_val(ent_cassa, y_str, "ccox-quadro-2-11")

        corr_comp_sum, cap_comp_sum, prest_comp_sum = 0.0, 0.0, 0.0
        corr_cassa_sum, cap_cassa_sum, prest_cassa_sum = 0.0, 0.0, 0.0
        m01_corr_comp, m50_tot_comp = 0.0, 0.0

        for slug, (m_code, m_lbl) in MISSION_MAP.items():
            nc = spe_comp.get(slug, {})
            ns = spe_cassa.get(slug, {})
            c_corr, c_cap, _, c_prest, c_tot = _mission_composition(nc, y_str) if nc else (0.0, 0.0, 0.0, 0.0, 0.0)
            s_corr, s_cap, _, s_prest, s_tot = _mission_composition(ns, y_str) if ns else (0.0, 0.0, 0.0, 0.0, 0.0)

            corr_comp_sum += c_corr
            cap_comp_sum += c_cap
            prest_comp_sum += c_prest
            corr_cassa_sum += s_corr
            cap_cassa_sum += s_cap
            prest_cassa_sum += s_prest

            if m_code == "M01":
                m01_corr_comp = c_corr
            elif m_code == "M50":
                m50_tot_comp = c_corr + c_prest

            mission_rows.append({
                "comune_id": cid,
                "comune": COMUNI_BY_ID[cid].name,
                "year": year,
                "budget_type": btype,
                "mission_code": m_code,
                "mission_label": m_lbl,
                "correnti_comp_eur": c_corr,
                "capitale_comp_eur": c_cap,
                "rimborso_prestiti_comp_eur": c_prest,
                "totale_comp_eur": c_tot,
                "correnti_cassa_eur": s_corr,
                "capitale_cassa_eur": s_cap,
                "rimborso_prestiti_cassa_eur": s_prest,
                "totale_cassa_eur": s_tot,
            })

        tot_s_comp = _node_abs(spe_comp.get("ccox-quadro-4-26", {}), y_str)
        tot_s_cassa = _node_abs(spe_cassa.get("ccox-quadro-4-26", {}), y_str)
        sup_y = sup_map.get(year, {})

        budget_rows.append({
            "comune_id": cid,
            "comune": COMUNI_BY_ID[cid].name,
            "year": year,
            "regime": "dlgs118",
            "budget_type": btype,
            "flag": "" if btype == "consuntivo" else "preventivo",
            "entrate_tributarie_comp_eur": t1_comp,
            "entrate_tributarie_cassa_eur": t1_cassa,
            "imposte_tasse_comp_eur": imp_comp,
            "imposte_tasse_cassa_eur": imp_cassa,
            "fondo_perequativo_comp_eur": fsc_comp,
            "imu_ici_comp_eur": sup_y.get("imu_ici_comp_eur", np.nan),
            "addizionale_irpef_comp_eur": sup_y.get("addizionale_irpef_comp_eur", np.nan),
            "tari_tarsu_comp_eur": sup_y.get("tari_tarsu_comp_eur", np.nan),
            "trasferimenti_correnti_comp_eur": t2_comp,
            "trasferimenti_correnti_cassa_eur": t2_cassa,
            "entrate_extratributarie_comp_eur": t3_comp,
            "entrate_extratributarie_cassa_eur": t3_cassa,
            "entrate_correnti_comp_eur": t1_comp + t2_comp + t3_comp,
            "entrate_correnti_cassa_eur": t1_cassa + t2_cassa + t3_cassa,
            "entrate_capitale_comp_eur": t4_comp,
            "entrate_capitale_cassa_eur": t4_cassa,
            "contributi_investimenti_comp_eur": contr_inv_comp,
            "contributi_investimenti_cassa_eur": contr_inv_cassa,
            "alienazioni_comp_eur": alien_comp,
            "alienazioni_cassa_eur": alien_cassa,
            "accensione_prestiti_comp_eur": t6_comp,
            "accensione_prestiti_cassa_eur": t6_cassa,
            "accensione_mutui_comp_eur": mutui_comp,
            "accensione_mutui_cassa_eur": mutui_cassa,
            "fpv_corrente_comp_eur": fpv_corr_comp,
            "fpv_capitale_comp_eur": fpv_cap_comp,
            "avanzo_amministrazione_comp_eur": avanzo_comp,
            "entrate_totali_comp_eur": tot_e_comp,
            "entrate_totali_cassa_eur": tot_e_cassa,
            "spese_correnti_comp_eur": corr_comp_sum,
            "spese_correnti_cassa_eur": corr_cassa_sum,
            "spese_capitale_comp_eur": cap_comp_sum,
            "spese_capitale_cassa_eur": cap_cassa_sum,
            "rimborso_prestiti_comp_eur": prest_comp_sum,
            "rimborso_prestiti_cassa_eur": prest_cassa_sum,
            "spese_totali_comp_eur": tot_s_comp,
            "spese_totali_cassa_eur": tot_s_cassa,
            "spesa_personale_comp_eur": sup_y.get("spesa_personale_comp_eur", np.nan),
            "interessi_passivi_comp_eur": sup_y.get("interessi_passivi_comp_eur", np.nan),
            "missione_01_correnti_comp_eur": m01_corr_comp,
            "missione_50_debito_comp_eur": m50_tot_comp,
            "rigidity_quadro20_pct": sup_y.get("rigidity_quadro20_pct", np.nan),
        })

    return budget_rows, mission_rows


def _compute_fiscal_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Attach standardized fiscal ratios (rigidity, collection capacity, capital share, autonomy)."""
    out = df.copy()
    ec = out["entrate_correnti_comp_eur"].replace(0, np.nan)
    et = out["entrate_tributarie_comp_eur"].replace(0, np.nan)
    ep = (out["entrate_tributarie_comp_eur"] + out["entrate_extratributarie_comp_eur"]).replace(0, np.nan)
    sp_prim = (out["spese_correnti_comp_eur"] + out["spese_capitale_comp_eur"]).replace(0, np.nan)

    # 1. Exact personnel + debt service rigidity (available 2008-2018)
    rig_calc = (
        out["spesa_personale_comp_eur"]
        + out["interessi_passivi_comp_eur"].fillna(0.0)
        + out["rimborso_prestiti_comp_eur"].fillna(0.0)
    ) / ec * 100.0
    out["rigidity_official_pct"] = out["rigidity_quadro20_pct"].combine_first(rig_calc)
    out["rigidity_pers_debt_pct"] = rig_calc

    # 2. Harmonized structural rigidity proxy (Mission 01 Institutional/General Management + Mission 50 Public Debt / Current Revenues, 2016-2022)
    out["rigidity_struct_pct"] = (out["missione_01_correnti_comp_eur"] + out["missione_50_debito_comp_eur"]) / ec * 100.0

    # 3. Current expenditure absorption (Spese Correnti / Entrate Correnti, continuous within each accounting regime)
    out["current_absorption_pct"] = out["spese_correnti_comp_eur"] / ec * 100.0

    # 4. Collection capacity ratios (Riscossioni Cassa / Accertamenti Competenza, consuntivo years only)
    is_cons = out["budget_type"] == "consuntivo"
    out["collection_rate_correnti_pct"] = np.where(is_cons, out["entrate_correnti_cassa_eur"] / ec * 100.0, np.nan)
    out["collection_rate_tributarie_pct"] = np.where(is_cons, out["entrate_tributarie_cassa_eur"] / et * 100.0, np.nan)
    out["collection_rate_proprie_pct"] = np.where(
        is_cons,
        (out["entrate_tributarie_cassa_eur"] + out["entrate_extratributarie_cassa_eur"]) / ep * 100.0,
        np.nan,
    )

    # 5. Capital investment share of primary spending & fiscal autonomy
    out["capital_share_pct"] = out["spese_capitale_comp_eur"] / sp_prim * 100.0
    out["autonomia_finanziaria_pct"] = (out["entrate_tributarie_comp_eur"] + out["entrate_extratributarie_comp_eur"]) / ec * 100.0
    return out


def load_all(raw_dir: Path = RAW_DIR, comuni_ids: tuple[str, ...] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load `(budgets_df, missions_df)` across all cached ER municipalities (`2008-2022`)."""
    targets = tuple(BUDGET_ENTITIES) if comuni_ids is None else comuni_ids
    b_rows: list[dict] = []
    m_rows: list[dict] = []

    for cid in targets:
        zp = storico_zip_path(cid, raw_dir)
        ap = armonizzati_json_path(cid, raw_dir)
        fp = floc_bundle_path(cid, raw_dir)
        if not ap.exists() or not fp.exists():
            raise FileNotFoundError(
                f"Missing BDAP/OpenBilanci cache for {cid} in {raw_dir}; run `uv run python -m cifresignificative.core.sources.bdap fetch`"
            )
        fl_bundle = json.loads(fp.read_text(encoding="utf-8"))
        arm_bundle = json.loads(ap.read_text(encoding="utf-8"))

        if zp.exists():
            b_rows.extend(parse_storico_zip(zp, cid))
        else:
            for y_ccou in STORICO_YEARS:
                r_y = parse_floc_ccou_year(fl_bundle, cid, year=y_ccou)
                if r_y:
                    b_rows.append(r_y)
        r2015 = parse_floc_ccou_year(fl_bundle, cid, year=2015)
        if r2015:
            b_rows.append(r2015)

        ccox_sup = parse_floc_ccox_supplements(fl_bundle)
        b_arm, m_arm = parse_armonizzati_bundle(arm_bundle, cid, ccox_sup=ccox_sup)
        b_rows.extend(b_arm)
        m_rows.extend(m_arm)

    b_df = _compute_fiscal_ratios(pd.DataFrame(b_rows)).sort_values(["comune_id", "year"], ignore_index=True)
    m_df = pd.DataFrame(m_rows).sort_values(["comune_id", "year", "mission_code"], ignore_index=True)
    return b_df, m_df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        print(fetch("--force" in sys.argv))
    b, m = load_all()
    print(f"budgets: {len(b)} rows ({b.year.min()}-{b.year.max()}), missions: {len(m)} rows, comuni: {b.comune_id.nunique()}")
