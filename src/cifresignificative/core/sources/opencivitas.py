"""SOSE / MEF Ragioneria Generale dello Stato: OpenCivitas Standard Expenditure Needs (Fabbisogni Standard).

Downloads and processes official municipal standard expenditure needs ('Fabbisogni Standard'),
historical expenditures ('Spesa Storica'), and quantitative service delivery levels
('Livelli Quantitativi delle Prestazioni', LQP) published on OpenCivitas (2023).

Covers all 13 Emilia-Romagna benchmark municipalities and 16 Comprensorio Cesenate belt municipalities
across fundamental functions:
- Total fundamental services (TOT)
- Social services and childcare (SOCNID)
- Local police (POLIZIA)
- General administration (AMMINISTR)
- Waste collection and disposal (RIFIUTI)
- Education (ISTRUZ)

Produces 2 harmonised parquet tables under `data/processed/cesena/`:
1. `opencivitas_municipal_summary.parquet`: Cross-sectional performance and efficiency metrics
   (standard needs, historical spending, gaps, service output levels 0-10, positioning coordinates,
   childcare coverage rates, and associated governance flags).
2. `opencivitas_indicators_detailed.parquet`: Tidy long-form indicator matrix.
"""
import urllib.request
from pathlib import Path

import pandas as pd
import truststore

from cifresignificative.core.geo import BELT_BY_CATASTALE, COMUNI_BY_CATASTALE
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/opencivitas")
PROCESSED_DIR = Path("data/processed/cesena")

OPENCIVITAS_BASE = "https://www.opencivitas.it/portale/documents/20118"
URLS: dict[str, str] = {
    "meta": f"{OPENCIVITAS_BASE}/0/Metadati_Enti_2023_IT.xlsx/9af2749a-d3a5-cc6f-c78d-08593c7dbc32",
    "tot": f"{OPENCIVITAS_BASE}/0/2023_Ind_TOT_01_IT.csv/1086db31-bf42-8678-0520-c77eacce2a23",
    "socnid": f"{OPENCIVITAS_BASE}/60275/2023_Ind_SOCNID_01_IT.csv/cfedc2c9-b05d-07cb-da97-decaf9190f5f",
    "polizia": f"{OPENCIVITAS_BASE}/0/2023_Ind_POLIZIA_01_IT.csv/175d849b-35f1-c439-c869-fe48959a24ce",
    "amministr": f"{OPENCIVITAS_BASE}/0/2023_Ind_AMMINISTR_01_IT.csv/45e7ec1b-d282-278d-de7b-73af1b374255",
    "rifiuti": f"{OPENCIVITAS_BASE}/0/2023_Ind_RIFIUTI_01_IT.csv/34fd224a-5cac-7676-3016-7c55fe5d6450",
    "istruz": f"{OPENCIVITAS_BASE}/0/2023_Ind_ISTRUZ_01_IT.csv/c9b4ea9e-d2c1-9ae4-3587-ceea43985da3",
    "viabilita": f"{OPENCIVITAS_BASE}/60275/2023_Ind_TERR_VIAB_01_IT.csv/5565663a-6434-27e1-1b7a-273e87b02635",
}


def fetch(raw_dir: Path = RAW_DIR, force: bool = False) -> dict[str, Path]:
    """Download and cache all OpenCivitas metadata and indicator files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, url in URLS.items():
        ext = ".xlsx" if key == "meta" else ".csv"
        dest = raw_dir / f"2023_Ind_{key}{ext}" if key != "meta" else raw_dir / "Metadati_Enti_2023_IT.xlsx"
        if not dest.exists() or force:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            dest.write_bytes(data)
            record(f"opencivitas_{key}", path=str(dest), url=url, size_bytes=len(data))
        paths[key] = dest
    return paths


def _build_user_mapping(raw_dir: Path) -> dict[str, dict[str, object]]:
    """Build mapping from OpenCivitas USERNAME to our standard municipal metadata."""
    meta_path = raw_dir / "Metadati_Enti_2023_IT.xlsx"
    if not meta_path.exists():
        fetch(raw_dir)

    meta_df = pd.read_excel(meta_path)
    user_to_meta: dict[str, dict[str, object]] = {}

    for _, row in meta_df.iterrows():
        cat = str(row["COMUNE_CATASTO_COD"])
        username = str(row["USERNAME"]).strip()
        if cat in COMUNI_BY_CATASTALE:
            c = COMUNI_BY_CATASTALE[cat]
            user_to_meta[username] = {
                "comune_id": c.id,
                "comune": c.name,
                "catastale": cat,
                "is_peer": True,
                "core_peer": c.core_peer,
                "is_belt": False,
                "belt_group": "er_peer",
            }
        elif cat in BELT_BY_CATASTALE:
            b = BELT_BY_CATASTALE[cat]
            user_to_meta[username] = {
                "comune_id": b.id,
                "comune": b.name,
                "catastale": cat,
                "is_peer": False,
                "core_peer": False,
                "is_belt": True,
                "belt_group": b.belt_group,
            }

    return user_to_meta


def build_opencivitas_summary(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build unified cross-sectional summary of standard needs and service levels (2023)."""
    user_to_meta = _build_user_mapping(raw_dir)
    base_rows = {u: dict(m) for u, m in user_to_meta.items()}

    service_mappings = {
        "tot": {
            "SPESA_STORICA_PROAB": "tot_spesa_storica_eur_pc",
            "FST_RIPROPORZIONATO_BI_PROAB": "tot_fabbisogno_standard_eur_pc",
            "POSIZIONE_SPESA_PERC_TOT": "tot_livello_spesa_0_10",
            "POSIZIONE_OUTPUT_PERC_TOT": "tot_livello_servizi_0_10",
            "COORD_SPESA": "tot_coord_spesa",
            "COORD_OUT": "tot_coord_servizi",
            "DIPENDENTI_X1000AB": "tot_dipendenti_per_1k",
            "D03": "tot_costo_lavoro_annuo_eur",
        },
        "socnid": {
            "SPESA_STORICA_PROAB": "soc_spesa_storica_eur_pc",
            "FST_RIPROPORZIONATO_BI_PROAB": "soc_fabbisogno_standard_eur_pc",
            "COORD_SPESA": "soc_coord_spesa",
            "COORD_OUT": "soc_coord_servizi",
            "COPERTURA_NID_NEW": "copertura_nidi_pct",
            "DEPRIVAZIONE_MEAN": "deprivazione_indice",
            "DUMMY_SOC_ASSOCIATA": "gestione_associata_sociale",
        },
        "polizia": {
            "SPESA_STORICA_PROAB": "pol_spesa_storica_eur_pc",
            "FST_RIPROPORZIONATO_BI_PROAB": "pol_fabbisogno_standard_eur_pc",
            "COORD_SPESA": "pol_coord_spesa",
            "COORD_OUT": "pol_coord_servizi",
        },
        "amministr": {
            "SPESA_STORICA_PROAB": "amm_spesa_storica_eur_pc",
            "FST_RIPROPORZIONATO_BI_PROAB": "amm_fabbisogno_standard_eur_pc",
            "COORD_SPESA": "amm_coord_spesa",
            "COORD_OUT": "amm_coord_servizi",
        },
        "rifiuti": {
            "SPESA_STORICA_PROAB": "rif_spesa_storica_eur_pc",
            "FST_RIPROPORZIONATO_BI_PROAB": "rif_fabbisogno_standard_eur_pc",
            "COORD_SPESA": "rif_coord_spesa",
            "COORD_OUT": "rif_coord_servizi",
        },
        "istruz": {
            "SPESA_STORICA_PROAB": "istr_spesa_storica_eur_pc",
            "FST_RIPROPORZIONATO_BI_PROAB": "istr_fabbisogno_standard_eur_pc",
            "COORD_SPESA": "istr_coord_spesa",
            "COORD_OUT": "istr_coord_servizi",
        },
        "viabilita": {
            "SPESA_STORICA_PROAB": "viab_spesa_storica_eur_pc",
            "FST_RIPROPORZIONATO_BI_PROAB": "viab_fabbisogno_standard_eur_pc",
            "COORD_SPESA": "viab_coord_spesa",
            "COORD_OUT": "viab_coord_servizi",
        },
    }

    for s_key, ind_map in service_mappings.items():
        csv_file = raw_dir / f"2023_Ind_{s_key}.csv"
        if not csv_file.exists():
            fetch(raw_dir)

        df = pd.read_csv(csv_file, sep=";", low_memory=False)
        sub = df[df["USERNAME"].isin(user_to_meta)]
        piv = sub.pivot(index="USERNAME", columns="Indicatore/Determinante", values="Valore")

        for ind_col, target_col in ind_map.items():
            if ind_col in piv.columns:
                s_vals = pd.to_numeric(piv[ind_col].astype(str).str.replace(",", "."), errors="coerce")
                for u, val in s_vals.items():
                    if u in base_rows:
                        base_rows[str(u)][target_col] = val

    summary_df = pd.DataFrame(list(base_rows.values()))
    summary_df["tot_diff_eur_pc"] = (summary_df["tot_spesa_storica_eur_pc"] - summary_df["tot_fabbisogno_standard_eur_pc"]).round(2)
    summary_df["tot_diff_pct"] = ((summary_df["tot_diff_eur_pc"] / summary_df["tot_fabbisogno_standard_eur_pc"]) * 100).round(2)
    summary_df["fl_virtuoso"] = (summary_df["tot_diff_eur_pc"] <= 0) & (summary_df["tot_livello_servizi_0_10"] >= 8)

    # Classify into standard performance quadrants
    # 1. Virtuoso: Lower expenditure, High service output (spesa <= 0, out >= 0 in coordinates)
    # 2. Alto servizio / Alta spesa: High service output, Higher expenditure
    # 3. Basso servizio / Bassa spesa: Lower expenditure, Lower service output
    # 4. Inefficiente: Higher expenditure, Lower service output
    def assign_quadrant(row: pd.Series) -> str:
        csp = float(row.get("tot_coord_spesa", 0))
        cout = float(row.get("tot_coord_servizi", 0))
        if csp <= 0 and cout >= 0:
            return "virtuoso"
        elif csp > 0 and cout >= 0:
            return "alto_servizio_alta_spesa"
        elif csp <= 0 and cout < 0:
            return "basso_servizio_bassa_spesa"
        else:
            return "critico_inefficiente"

    summary_df["quadrante_performance"] = summary_df.apply(assign_quadrant, axis=1)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "opencivitas_municipal_summary.parquet"
    summary_df.to_parquet(out_path, index=False)
    return summary_df


def build_opencivitas_detailed(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build long-form detailed indicator matrix for all 29 target municipalities."""
    user_to_meta = _build_user_mapping(raw_dir)
    frames = []

    services = ["tot", "socnid", "polizia", "amministr", "rifiuti", "istruz", "viabilita"]
    for s in services:
        csv_file = raw_dir / f"2023_Ind_{s}.csv"
        if not csv_file.exists():
            fetch(raw_dir)
        df = pd.read_csv(csv_file, sep=";", low_memory=False)
        sub = df[df["USERNAME"].isin(user_to_meta)].copy()
        sub["servizio"] = s
        sub["comune_id"] = sub["USERNAME"].map(lambda u: user_to_meta[u]["comune_id"])
        sub["comune"] = sub["USERNAME"].map(lambda u: user_to_meta[u]["comune"])
        sub["catastale"] = sub["USERNAME"].map(lambda u: user_to_meta[u]["catastale"])
        sub["valore_num"] = pd.to_numeric(sub["Valore"].astype(str).str.replace(",", "."), errors="coerce")
        frames.append(sub[["comune_id", "comune", "catastale", "servizio", "Indicatore/Determinante", "Valore", "valore_num"]])

    all_det = pd.concat(frames, ignore_index=True).rename(columns={"Indicatore/Determinante": "indicatore_codice"})
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "opencivitas_indicators_detailed.parquet"
    all_det.to_parquet(out_path, index=False)
    return all_det


def build_all(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Build both OpenCivitas summary and detailed tables."""
    return {
        "summary": build_opencivitas_summary(raw_dir, processed_dir),
        "detailed": build_opencivitas_detailed(raw_dir, processed_dir),
    }


if __name__ == "__main__":
    dfs = build_all()
    print("OpenCivitas tables built successfully:")
    for k, v in dfs.items():
        print(f"  {k}: shape {v.shape}")
