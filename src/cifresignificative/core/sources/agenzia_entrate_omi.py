"""Agenzia delle Entrate OMI (Osservatorio del Mercato Immobiliare) real estate quotations & transactions.

Extracts:
1. Municipal-level residential property purchase quotations (€/m²) and rental rates (€/m²/month)
   for the 13 Emilia-Romagna benchmark municipalities and 16 Comprensorio Cesenate belt municipalities (N = 29)
   from official OMI micro-quotation datasets (covering civil dwellings, economic dwellings, villas, and boxes).
2. Zone-level intra-municipal detail for Cesena (Centro Storico, Semicentro, Periferia, Frazioni/Zone Agricole).
3. Residential transaction volumes (Numero di Transazioni Normalizzate, NTN) and Market Intensity (IMI)
   from Agenzia delle Entrate definitive time series (2011–2024).

Outputs:
- `data/processed/cesena/omi_quotations_municipal.parquet`: municipal residential prices, rents, yield, and differentials vs Cesena.
- `data/processed/cesena/omi_zones_cesena.parquet`: intra-municipal zone detail for Cesena.
- `data/processed/cesena/omi_transactions_panel.parquet`: annual transaction volumes (NTN) and intensity across ER provinces and capoluoghi.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import truststore

from cifresignificative.core.geo import (
    ALL_COMUNI_IDS,
    BELT_BY_ID,
    BELT_COMUNI_IDS,
    CESENA,
    COMUNI_BY_ID,
)
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/agenzia_entrate_omi")
PROCESSED_DIR = Path("data/processed/cesena")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}

PROV_JSON_URLS: dict[str, str] = {
    "Bologna": "https://raw.githubusercontent.com/dedoitalia/omi-data/main/data/Bologna.json",
    "Ferrara": "https://raw.githubusercontent.com/dedoitalia/omi-data/main/data/Ferrara.json",
    "Forli-Cesena": "https://raw.githubusercontent.com/dedoitalia/omi-data/main/data/FO.json",
    "Modena": "https://raw.githubusercontent.com/dedoitalia/omi-data/main/data/Modena.json",
    "Parma": "https://raw.githubusercontent.com/dedoitalia/omi-data/main/data/Parma.json",
    "Piacenza": "https://raw.githubusercontent.com/dedoitalia/omi-data/main/data/Piacenza.json",
    "Ravenna": "https://raw.githubusercontent.com/dedoitalia/omi-data/main/data/Ravenna.json",
    "Reggio Emilia": "https://raw.githubusercontent.com/dedoitalia/omi-data/main/data/Reggio%20Emilia.json",
    "Rimini": "https://raw.githubusercontent.com/dedoitalia/omi-data/main/data/Rimini.json",
}

TRANSACTIONS_URL = (
    "https://www.agenziaentrate.gov.it/portale/documents/20143/9812824/"
    "RESIDENZIALE_DEFINITIVO_2011_2024.zip/8cb254ca-e71c-da1b-2d7a-56d859dce93d?t=1772716377382"
)


def fetch_all(raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Download OMI provincial JSON quotation files and transactions ZIP archive."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    # 1. Provincial quotation JSON files
    for prov, url in PROV_JSON_URLS.items():
        fname = f"{prov.lower().replace(' ', '_')}_omi.json"
        p = raw_dir / fname
        if not p.exists() or p.stat().st_size == 0:
            print(f"[omi] Downloading {prov} quotations from {url}...")
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            p.write_bytes(r.content)
            record(f"agenzia_entrate_omi_{prov.lower()}", filename=fname, size_bytes=len(r.content), url=url, note=f"OMI quotations for {prov}")
        out[f"prov_{prov}"] = p

    # 2. Residential transactions ZIP (2011-2024)
    zip_path = raw_dir / "RESIDENZIALE_DEFINITIVO_2011_2024.zip"
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print("[omi] Downloading residential transactions archive from Agenzia delle Entrate...")
        try:
            r = requests.get(TRANSACTIONS_URL, headers=UA, timeout=60)
            if r.status_code == 200:
                zip_path.write_bytes(r.content)
                record("agenzia_entrate_omi_transactions", filename=zip_path.name, size_bytes=len(r.content), url=TRANSACTIONS_URL, note="NTN residential transactions 2011-2024")
                out["transactions_zip"] = zip_path
        except (requests.RequestException, OSError) as e:
            print(f"[omi] Warning: could not download transactions ZIP: {e}")
    else:
        out["transactions_zip"] = zip_path

    return out


def _load_raw_quotations(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and unify all provincial OMI quotation records."""
    fetch_all(raw_dir)
    frames: list[pd.DataFrame] = []
    for prov in PROV_JSON_URLS:
        fname = f"{prov.lower().replace(' ', '_')}_omi.json"
        p = raw_dir / fname
        if p.exists():
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            df = pd.DataFrame(d)
            df["prov_source"] = prov
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No provincial OMI JSONs found in {raw_dir}")
    merged = pd.concat(frames, ignore_index=True)
    merged["comune_clean"] = merged["comune"].str.upper().str.replace("`", "'").str.strip()
    return merged


def build_omi_quotations(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build municipal residential quotation dataset across all 13 peers and 16 belt municipalities."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df = _load_raw_quotations(raw_dir)

    target_ids = list(ALL_COMUNI_IDS) + list(BELT_COMUNI_IDS)
    records: list[dict[str, Any]] = []

    for cid in target_ids:
        if cid in COMUNI_BY_ID:
            c = COMUNI_BY_ID[cid]
            name = c.name
            nuts3 = c.nuts3
            prov_sigla = c.prov_sigla
            is_core = c.core_peer
            is_peer = True
            belt_grp = "cesena_core" if cid == CESENA else "er_peer"
        else:
            b = BELT_BY_ID[cid]
            name = b.name
            nuts3 = "ITH58"
            prov_sigla = "FC"
            is_core = False
            is_peer = False
            belt_grp = b.belt_group

        target_clean = name.upper().replace("`", "'").replace("Ì", "I").strip()
        sub = raw_df[raw_df["comune_clean"].str.replace("`", "").str.replace("'", "") == target_clean.replace("'", "")]
        if sub.empty:
            w1 = target_clean.split()[0].replace("'", "")
            sub = raw_df[raw_df["comune_clean"].str.replace("`", "").str.replace("'", "").str.startswith(w1)]

        if sub.empty:
            continue

        # OMI files a zone without a rental quote as 0 (35% of Bologna's and 52% of Imola's residential rows): that is a missing quote, not a free rent.
        sub = sub.copy()
        for col in ("loc_min", "loc_max"):
            sub[col] = sub[col].where(sub[col] > 0)
        # Filter to residential (civil & economic dwellings)
        civ = sub[sub["tipo"].str.contains("civili", case=False, na=False)]
        econ = sub[sub["tipo"].str.contains("economico", case=False, na=False)]
        res = sub[sub["tipo"].str.contains("civili|economico", case=False, na=False)]

        compr_min = float(res["compr_min"].mean()) if not res.empty else None
        compr_max = float(res["compr_max"].mean()) if not res.empty else None
        compr_mid = (compr_min + compr_max) / 2.0 if (compr_min and compr_max) else None

        loc_min = float(res["loc_min"].mean()) if not res.empty else None
        loc_max = float(res["loc_max"].mean()) if not res.empty else None
        loc_mid = (loc_min + loc_max) / 2.0 if (loc_min and loc_max) else None

        # Typology-specific mid prices
        c_civ_mid = ((civ["compr_min"].mean() + civ["compr_max"].mean()) / 2.0) if not civ.empty else None
        c_eco_mid = ((econ["compr_min"].mean() + econ["compr_max"].mean()) / 2.0) if not econ.empty else None
        l_civ_mid = ((civ["loc_min"].mean() + civ["loc_max"].mean()) / 2.0) if not civ.empty else None
        l_eco_mid = ((econ["loc_min"].mean() + econ["loc_max"].mean()) / 2.0) if not econ.empty else None

        # Gross rental yield = (monthly rent * 12) / purchase price * 100, on the rows that have both quotes
        quoted = res[res["loc_min"].notna() & res["loc_max"].notna()]
        compr_mid_quoted = float(quoted["compr_min"].mean() + quoted["compr_max"].mean()) / 2.0 if not quoted.empty else None
        gross_yield = (loc_mid * 12.0 / compr_mid_quoted * 100.0) if (loc_mid and compr_mid_quoted) else None
        rent_quote_share = float(len(quoted) / len(res) * 100.0) if not res.empty else None

        records.append({
            "comune_id": cid,
            "comune_name": name,
            "nuts3": nuts3,
            "prov_sigla": prov_sigla,
            "is_cesena": cid == CESENA,
            "is_core_peer": is_core,
            "is_er_peer": is_peer,
            "belt_group": belt_grp,
            "n_omi_zones": len(sub["zona"].unique()),
            "n_records": len(res),
            "compr_min_eur_m2": compr_min,
            "compr_max_eur_m2": compr_max,
            "compr_mid_eur_m2": compr_mid,
            "compr_civili_mid_eur_m2": c_civ_mid,
            "compr_economico_mid_eur_m2": c_eco_mid,
            "loc_min_eur_m2_mo": loc_min,
            "loc_max_eur_m2_mo": loc_max,
            "loc_mid_eur_m2_mo": loc_mid,
            "loc_civili_mid_eur_m2_mo": l_civ_mid,
            "loc_economico_mid_eur_m2_mo": l_eco_mid,
            "gross_rental_yield_pct": gross_yield,
            "rent_quote_share_pct": rent_quote_share,
        })

    df = pd.DataFrame(records)

    # Compute differentials relative to Cesena core
    ces_row = df[df["is_cesena"]].iloc[0]
    ces_price = ces_row["compr_mid_eur_m2"]
    ces_rent = ces_row["loc_mid_eur_m2_mo"]

    df["diff_vs_cesena_eur_m2"] = df["compr_mid_eur_m2"] - ces_price
    df["diff_vs_cesena_pct"] = (df["compr_mid_eur_m2"] / ces_price - 1.0) * 100.0
    df["diff_rent_vs_cesena_eur_mo"] = df["loc_mid_eur_m2_mo"] - ces_rent
    df["diff_rent_vs_cesena_pct"] = (df["loc_mid_eur_m2_mo"] / ces_rent - 1.0) * 100.0

    p_out = out_dir / "omi_quotations_municipal.parquet"
    df.to_parquet(p_out, index=False)
    print(f"[omi] Saved {len(df)} municipal quotation rows to {p_out}")
    return df


def build_omi_zones_cesena(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Extract and aggregate detailed intra-municipal OMI zones for Cesena ('040007')."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df = _load_raw_quotations(raw_dir)
    ces_raw = raw_df[raw_df["comune"] == "CESENA"].copy()

    records: list[dict[str, Any]] = []
    for (zona, tipo), sub in ces_raw.groupby(["zona", "tipo"]):
        c_min = float(sub["compr_min"].mean())
        c_max = float(sub["compr_max"].mean())
        l_min = float(sub["loc_min"].mean())
        l_max = float(sub["loc_max"].mean())
        records.append({
            "comune_id": CESENA,
            "comune_name": "Cesena",
            "zona": zona,
            "zona_label": _format_cesena_zone_label(zona),
            "tipo": tipo,
            "n_records": len(sub),
            "compr_min_eur_m2": c_min,
            "compr_max_eur_m2": c_max,
            "compr_mid_eur_m2": (c_min + c_max) / 2.0,
            "loc_min_eur_m2_mo": l_min,
            "loc_max_eur_m2_mo": l_max,
            "loc_mid_eur_m2_mo": (l_min + l_max) / 2.0,
        })

    df = pd.DataFrame(records).sort_values(["compr_mid_eur_m2"], ascending=False).reset_index(drop=True)
    p_out = out_dir / "omi_zones_cesena.parquet"
    df.to_parquet(p_out, index=False)
    print(f"[omi] Saved {len(df)} Cesena zone quotation rows to {p_out}")
    return df


def _format_cesena_zone_label(zona: str) -> str:
    """Provide concise labels for Cesena's 7 OMI zones."""
    z_up = zona.upper()
    if "CENTRO STORICO" in z_up:
        return "B1 Centro Storico"
    if "CENTRO URBANO" in z_up:
        return "C1 Centro Urbano / Semicentro (S. Egidio, Vigne, Martorano)"
    if "PIEVESETINA" in z_up:
        return "D1 Nord / Dismano (Pievesetina, Torre del Moro, Calisese)"
    if "BORELLO" in z_up:
        return "D2 Valle Savio Urbana (Borello, S. Vittore, S. Carlo)"
    if "SANT`ANDREA" in z_up or "BAGNOLO" in z_up:
        return "E1 Pianura Nord (Ronta, S. Giorgio, Macerone, Calabrina)"
    if "COLLINA LATO EST" in z_up:
        return "E2 Collina Est (Montevecchio, Saiano, Carpineta)"
    if "COLLINA LATO OVEST" in z_up:
        return "E3 Collina Ovest (Formignano, Tessello, Paderno)"
    return zona[:45]


def build_omi_transactions(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build annual residential transactions (NTN) panel across ER provinces and capoluoghi (2011-2024)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "RESIDENZIALE_DEFINITIVO_2011_2024.zip"
    if not zip_path.exists():
        fetch_all(raw_dir)

    rows: list[dict[str, Any]] = []
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            if "RES.csv" in z.namelist():
                with z.open("RES.csv") as f:
                    df = pd.read_csv(f, sep=";", encoding="latin1")
                    # Filter to Emilia-Romagna
                    er = df[df["Regione"].str.contains("Emilia", case=False, na=False)].copy()
                    quarter_cols = [c for c in er.columns if re_match_year_quarter(c)]
                    years = sorted({int(c.split("_")[0]) for c in quarter_cols})

                    for _, r in er.iterrows():
                        prov = r["prov"]
                        cap_type = r["Cap"]
                        for yr in years:
                            y_cols = [f"{yr}_{q}" for q in range(1, 5) if f"{yr}_{q}" in er.columns]
                            ntn_sum = sum(clean_num(r[col]) for col in y_cols)
                            rows.append({
                                "regione": "Emilia-Romagna",
                                "prov_sigla": prov,
                                "area_type": cap_type,
                                "year": yr,
                                "ntn_residential": ntn_sum,
                            })

    out_df = pd.DataFrame(rows)
    p_out = out_dir / "omi_transactions_panel.parquet"
    out_df.to_parquet(p_out, index=False)
    print(f"[omi] Saved {len(out_df)} transaction panel rows to {p_out}")
    return out_df


def re_match_year_quarter(col: str) -> bool:
    parts = col.split("_")
    return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()


def clean_num(val: Any) -> float:
    if pd.isna(val):
        return 0.0
    s = str(val).replace(",", ".").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def build(out_dir: Path = PROCESSED_DIR) -> dict[str, Path]:
    """Execute complete OMI build and return generated paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    build_omi_quotations(out_dir=out_dir)
    build_omi_zones_cesena(out_dir=out_dir)
    build_omi_transactions(out_dir=out_dir)
    return {
        "omi_quotations_municipal": out_dir / "omi_quotations_municipal.parquet",
        "omi_zones_cesena": out_dir / "omi_zones_cesena.parquet",
        "omi_transactions_panel": out_dir / "omi_transactions_panel.parquet",
    }


if __name__ == "__main__":
    build()
