"""ISTAT FRAME-SBS Territoriale: Municipal economic accounts, value added, and wages (`2017–2023`).

Extracts the official ISTAT FRAME-SBS Territoriale (FST - Registro Statistico Esteso dei Conti Economici delle Imprese)
municipal tables across Industry, Services, and Total non-farm market economy from:
- `2017`: `4.-Tavole-2017-protette-comuni.xlsx`
- `2018`: `Tavole-2018.xlsx`
- `2019`: `tavole-risultati-economici-multinazionali-2019.xlsx`
- `2020`: `Appendice_Frame_Terr_multinaz_2020_FINALE.xlsx`
- `2021`: `tavole.zip` -> `1. Appendice Statistica Report Territoriale 2021.xlsx`
- `2022`: `Tavole-1.zip` -> `1. Appendice statistica Frame territoriale 2022.xlsx`
- `2023`: `Tavole.zip` -> `1. Appendice Statistica Frame territoriale.xlsx`

Provides:
1. `frame_sbs_municipal.parquet`: Municipal panel (`2017–2023`) across 3 macro-sectors (`totale`, `industria`, `servizi`)
   for the 13 Emilia-Romagna benchmark municipalities and 16 Comprensorio Cesenate belt municipalities (`N = 29`).
   Variables:
   - `unita_locali`: Active local units
   - `addetti`: Total persons employed (workers)
   - `dipendenti`: Salaried employees
   - `retribuzioni_eur`: Total gross wages (EUR)
   - `costo_lavoro_eur`: Total labor cost including social contributions (EUR, 2021+)
   - `valore_aggiunto_eur`: Value Added at basic prices (EUR)
   - `fatturato_eur`: Turnover (EUR)
   - `acquisti_eur`: Purchases of intermediate goods and services (EUR)
   - `va_per_addetto_eur`: Value Added per worker (EUR/worker, nominal & real 2020 EUR)
   - `retribuzione_per_dipendente_eur`: Gross wage per employee (EUR/employee, nominal & real 2020 EUR)
   - `va_su_fatturato_pct`: Value added share of turnover (%)
   - `retribuzioni_su_va_pct`: Wage share of Value Added (%)
2. `frame_sbs_summary.parquet`: Benchmark summary comparing Cesena against Core Peers and All ER Peers.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import requests
import truststore

from cifresignificative.core.deflators import deflate
from cifresignificative.core.geo import (
    ALL_COMUNI_IDS,
    BELT_COMUNI_IDS,
    CESENA,
    COMUNI_BY_ID,
    CORE_PEER_IDS,
    format_comune_id,
)
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/istat_frame_sbs")
PROCESSED_DIR = Path("data/processed/cesena")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}

SOURCES: dict[int, tuple[str, str, str | None, dict[str, str]]] = {
    2017: (
        "xlsx",
        "https://www.istat.it/wp-content/uploads/2020/06/4.-Tavole-2017-protette-comuni.xlsx",
        None,
        {"totale": "Tav. 28 Comuni", "industria": "Tav. 29 Comuni Industria", "servizi": "Tav. 30 Comuni Servizi"},
    ),
    2018: (
        "xlsx",
        "https://www.istat.it/wp-content/uploads/2020/12/Tavole-2018.xlsx",
        None,
        {"totale": "Tav. 28 Comuni", "industria": "Tav. 29 Comuni Industria", "servizi": "Tav. 30 Comuni Servizi"},
    ),
    2019: (
        "xlsx",
        "https://www.istat.it/wp-content/uploads/2021/12/tavole-risultati-economici-multinazionali-2019.xlsx",
        None,
        {"totale": "Tav. 28 Comuni", "industria": "Tav. 29 Comuni Industria", "servizi": "Tav. 30 Comuni Servizi"},
    ),
    2020: (
        "xlsx",
        "https://www.istat.it/wp-content/uploads/2023/01/Appendice_Frame_Terr_multinaz_2020_FINALE.xlsx",
        None,
        {"totale": "Tav. 28 Comuni", "industria": "Tav. 29 Comuni Industria", "servizi": "Tav. 30 Comuni Servizi"},
    ),
    2021: (
        "zip",
        "https://www.istat.it/wp-content/uploads/2024/01/tavole.zip",
        "tav_sostituire/1. Appendice Statistica Report Territoriale 2021.xlsx",
        {"totale": "Tav.39 Comuni_Totale", "industria": "Tav.40 Comuni_Industria", "servizi": "Tav.41 Comuni_Servizi"},
    ),
    2022: (
        "zip",
        "https://www.istat.it/wp-content/uploads/2024/12/Tavole-1.zip",
        "Tavole/1. Appendice statistica Frame territoriale 2022.xlsx",
        {"totale": "Tav.39_Comuni_Totale", "industria": "Tav.40_Comuni_Industria", "servizi": "Tav.41_Comuni_Servizi"},
    ),
    2023: (
        "zip",
        "https://www.istat.it/wp-content/uploads/2026/01/Tavole.zip",
        "Tavole/1. Appendice Statistica Frame territoriale.xlsx",
        {"totale": "Tav.39_Comuni_Totale", "industria": "Tav.40_Comuni_Industria", "servizi": "Tav.41_Comuni_Servizi"},
    ),
}


def _clean_num(val: Any) -> float | None:
    if val is None or val == "*" or val == "-" or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def fetch(force: bool = False, raw_dir: Path = RAW_DIR) -> dict[int, Path]:
    """Download and cache raw Excel and Zip packages from ISTAT."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for yr, (kind, url, _, _) in sorted(SOURCES.items()):
        filename = f"frame_sbs_{yr}.{kind}"
        dest = raw_dir / filename
        if not dest.exists() or force:
            print(f"[istat_frame_sbs] Downloading {yr} from {url} ...")
            resp = requests.get(url, headers=UA, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            record(f"istat_frame_sbs_{yr}", source="istat", dataset="FRAME-SBS", url=url)
        paths[yr] = dest
    return paths


def parse_annual_workbook(year: int, raw_path: Path, target_ids: set[str]) -> list[dict[str, Any]]:
    """Parse the municipal sheets for a single year from the cached raw file."""
    kind, _, inner, sheets = SOURCES[year]
    if kind == "zip":
        if inner is None:
            raise ValueError(f"Inner zip file path missing for {year}")
        with zipfile.ZipFile(raw_path) as z:
            data = z.read(inner)
    else:
        data = raw_path.read_bytes()

    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    rows: list[dict[str, Any]] = []

    for macro_sector, sheet_name in sheets.items():
        if sheet_name not in wb.sheetnames:
            print(f"[istat_frame_sbs] Warning: sheet '{sheet_name}' not found in {year}")
            continue
        sh = wb[sheet_name]

        # Detect header row (row where "aggiunto" appears in text)
        header_row = 3
        for r_idx in range(1, 6):
            sample = [sh.cell(r_idx, c).value for c in range(1, 15)]
            if any(v and "aggiunto" in str(v).lower() for v in sample):
                header_row = r_idx
                break

        # Map column headers to column indices (1-indexed)
        h_map: dict[str, int] = {}
        for c in range(1, 26):
            cell_v = sh.cell(header_row, c).value
            if not cell_v:
                continue
            v_low = str(cell_v).lower().strip()
            if "codice comune" in v_low:
                h_map["comune_id"] = c
            elif "denominazione comune" in v_low:
                h_map["comune_nome"] = c
            elif "numero unità locali" in v_low or "numero unita locali" in v_low:
                h_map["unita_locali"] = c
            elif "addetti" in v_low and "per addetto" not in v_low:
                h_map["addetti"] = c
            elif "dipendenti" in v_low and "per dipendente" not in v_low:
                h_map["dipendenti"] = c
            elif "costo del lavoro" in v_low:
                h_map["costo_lavoro_keur"] = c
            elif "retribuzioni" in v_low and "valore aggiunto" not in v_low and "per dipendente" not in v_low:
                h_map["retribuzioni_keur"] = c
            elif "retribuzione per dipendente" in v_low:
                h_map["retribuzione_per_dipendente_keur"] = c
            elif "retribuzione sul valore aggiunto" in v_low:
                h_map["retribuzioni_su_va_pct"] = c
            elif "valore aggiunto per addetto" in v_low:
                h_map["va_per_addetto_keur"] = c
            elif "valore aggiunto sul fatturato" in v_low:
                h_map["va_su_fatturato_pct"] = c
            elif "valore aggiunto" in v_low and "per addetto" not in v_low and "sul fatturato" not in v_low and "retribuzion" not in v_low:
                h_map["valore_aggiunto_keur"] = c
            elif "fatturato" in v_low and "sul fatturato" not in v_low:
                h_map["fatturato_keur"] = c
            elif "acquisti di beni" in v_low and "sul fatturato" not in v_low:
                h_map["acquisti_keur"] = c

        cod_col = h_map.get("comune_id")
        if not cod_col:
            raise ValueError(f"Header 'Codice Comune Istat' not found in {year} {sheet_name}")

        for r_vals in sh.iter_rows(min_row=header_row + 1, values_only=True):
            raw_cod = r_vals[cod_col - 1]
            if raw_cod is None:
                continue
            cod = format_comune_id(raw_cod)
            if cod in target_ids:
                u_locali = _clean_num(r_vals[h_map["unita_locali"] - 1]) if "unita_locali" in h_map else None
                add = _clean_num(r_vals[h_map["addetti"] - 1]) if "addetti" in h_map else None
                dip = _clean_num(r_vals[h_map["dipendenti"] - 1]) if "dipendenti" in h_map else None

                retrib_k = _clean_num(r_vals[h_map["retribuzioni_keur"] - 1]) if "retribuzioni_keur" in h_map else None
                costo_k = _clean_num(r_vals[h_map["costo_lavoro_keur"] - 1]) if "costo_lavoro_keur" in h_map else None
                va_k = _clean_num(r_vals[h_map["valore_aggiunto_keur"] - 1]) if "valore_aggiunto_keur" in h_map else None
                fatt_k = _clean_num(r_vals[h_map["fatturato_keur"] - 1]) if "fatturato_keur" in h_map else None
                acq_k = _clean_num(r_vals[h_map["acquisti_keur"] - 1]) if "acquisti_keur" in h_map else None
                va_add_k = _clean_num(r_vals[h_map["va_per_addetto_keur"] - 1]) if "va_per_addetto_keur" in h_map else None
                retrib_dip_k = _clean_num(r_vals[h_map["retribuzione_per_dipendente_keur"] - 1]) if "retribuzione_per_dipendente_keur" in h_map else None

                # Convert thousands of EUR to EUR
                va_eur = va_k * 1000.0 if va_k is not None else (va_add_k * 1000.0 * add if (va_add_k and add) else None)
                retrib_eur = retrib_k * 1000.0 if retrib_k is not None else (retrib_dip_k * 1000.0 * dip if (retrib_dip_k and dip) else None)
                costo_eur = costo_k * 1000.0 if costo_k is not None else None
                fatt_eur = fatt_k * 1000.0 if fatt_k is not None else None
                acq_eur = acq_k * 1000.0 if acq_k is not None else None

                # Direct ratios per worker / per employee
                va_per_add = (va_eur / add) if (va_eur and add and add > 0) else (va_add_k * 1000.0 if va_add_k else None)
                retrib_per_dip = (retrib_eur / dip) if (retrib_eur and dip and dip > 0) else (retrib_dip_k * 1000.0 if retrib_dip_k else None)
                costo_per_dip = (costo_eur / dip) if (costo_eur and dip and dip > 0) else None

                row_dict = {
                    "year": int(year),
                    "macro_sector": macro_sector,
                    "comune_id": cod,
                    "comune_nome": COMUNI_BY_ID[cod].name if cod in COMUNI_BY_ID else str(r_vals[h_map["comune_nome"] - 1]),
                    "unita_locali": u_locali,
                    "addetti": add,
                    "dipendenti": dip,
                    "valore_aggiunto_eur": va_eur,
                    "retribuzioni_eur": retrib_eur,
                    "costo_lavoro_eur": costo_eur,
                    "fatturato_eur": fatt_eur,
                    "acquisti_eur": acq_eur,
                    "va_per_addetto_eur": va_per_add,
                    "retribuzione_per_dipendente_eur": retrib_per_dip,
                    "costo_lavoro_per_dipendente_eur": costo_per_dip,
                    "va_su_fatturato_pct": _clean_num(r_vals[h_map["va_su_fatturato_pct"] - 1]) if "va_su_fatturato_pct" in h_map else None,
                    "retribuzioni_su_va_pct": _clean_num(r_vals[h_map["retribuzioni_su_va_pct"] - 1]) if "retribuzioni_su_va_pct" in h_map else None,
                }
                rows.append(row_dict)
    return rows


def build_frame_sbs_panel(force: bool = False, raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build the harmonized municipal FRAME-SBS panel (2017-2023) and save to Parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frame_sbs_municipal.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    paths = fetch(force=force, raw_dir=raw_dir)
    target_ids = set(ALL_COMUNI_IDS) | set(BELT_COMUNI_IDS)

    all_rows: list[dict[str, Any]] = []
    for yr in sorted(paths):
        yr_rows = parse_annual_workbook(yr, paths[yr], target_ids)
        all_rows.extend(yr_rows)

    df = pd.DataFrame(all_rows)

    # Compute constant 2020 EUR series using Italy CPI / GDP deflator
    df["valore_aggiunto_real_eur"] = deflate(df["valore_aggiunto_eur"], df["year"], base_year=2020)
    df["retribuzioni_real_eur"] = deflate(df["retribuzioni_eur"], df["year"], base_year=2020)
    df["fatturato_real_eur"] = deflate(df["fatturato_eur"], df["year"], base_year=2020)
    df["acquisti_real_eur"] = deflate(df["acquisti_eur"], df["year"], base_year=2020)
    df["va_per_addetto_real_eur"] = df["valore_aggiunto_real_eur"] / df["addetti"]
    df["retribuzione_per_dipendente_real_eur"] = df["retribuzioni_real_eur"] / df["dipendenti"]

    # Compute labor cost real where available
    df["costo_lavoro_real_eur"] = deflate(df["costo_lavoro_eur"], df["year"], base_year=2020)
    df["costo_lavoro_per_dipendente_real_eur"] = df["costo_lavoro_real_eur"] / df["dipendenti"]

    # Tag peer classification
    df["is_cesena"] = df["comune_id"] == CESENA
    df["is_core_peer"] = df["comune_id"].isin(CORE_PEER_IDS)
    df["is_er_benchmark"] = df["comune_id"].isin(ALL_COMUNI_IDS)
    df["is_belt"] = df["comune_id"].isin(BELT_COMUNI_IDS)

    df = df.sort_values(["macro_sector", "comune_id", "year"]).reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    print(f"[istat_frame_sbs] Saved {len(df)} rows to {out_path}")
    return df


def build_frame_sbs_summary(panel: pd.DataFrame | None = None, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build benchmark comparison summary for Cesena vs Core Peers and All ER Peers."""
    if panel is None:
        panel = build_frame_sbs_panel()

    out_path = out_dir / "frame_sbs_summary.parquet"
    records: list[dict[str, Any]] = []

    er_panel = panel[panel["is_er_benchmark"]].copy()

    for sector in ("totale", "industria", "servizi"):
        sec_df = er_panel[er_panel["macro_sector"] == sector]
        for yr, y_df in sec_df.groupby("year"):
            ces = y_df[y_df["is_cesena"]]
            if ces.empty:
                continue
            c_row = ces.iloc[0]

            core_peers = y_df[y_df["is_core_peer"]]
            all_peers = y_df[~y_df["is_cesena"]]

            # Rank among the 13 ER municipalities (1 = highest)
            va_add_rank = int((y_df["va_per_addetto_eur"].rank(ascending=False, method="min")).loc[c_row.name])
            retrib_dip_rank = int((y_df["retribuzione_per_dipendente_eur"].rank(ascending=False, method="min")).loc[c_row.name])

            records.append({
                "year": int(yr),
                "macro_sector": sector,
                # Cesena nominal & real levels
                "cesena_va_per_addetto_eur": float(c_row["va_per_addetto_eur"]),
                "cesena_va_per_addetto_real_eur": float(c_row["va_per_addetto_real_eur"]),
                "cesena_retrib_per_dipendente_eur": float(c_row["retribuzione_per_dipendente_eur"]),
                "cesena_retrib_per_dipendente_real_eur": float(c_row["retribuzione_per_dipendente_real_eur"]),
                "cesena_va_su_fatturato_pct": float(c_row["va_su_fatturato_pct"]) if pd.notna(c_row["va_su_fatturato_pct"]) else None,
                "cesena_retrib_su_va_pct": float(c_row["retribuzioni_su_va_pct"]) if pd.notna(c_row["retribuzioni_su_va_pct"]) else None,
                # Core Peers medians
                "core_peer_va_per_addetto_eur": float(core_peers["va_per_addetto_eur"].median()),
                "core_peer_va_per_addetto_real_eur": float(core_peers["va_per_addetto_real_eur"].median()),
                "core_peer_retrib_per_dipendente_eur": float(core_peers["retribuzione_per_dipendente_eur"].median()),
                "core_peer_retrib_per_dipendente_real_eur": float(core_peers["retribuzione_per_dipendente_real_eur"].median()),
                # All ER Peers medians
                "all_peer_va_per_addetto_eur": float(all_peers["va_per_addetto_eur"].median()),
                "all_peer_va_per_addetto_real_eur": float(all_peers["va_per_addetto_real_eur"].median()),
                "all_peer_retrib_per_dipendente_eur": float(all_peers["retribuzione_per_dipendente_eur"].median()),
                "all_peer_retrib_per_dipendente_real_eur": float(all_peers["retribuzione_per_dipendente_real_eur"].median()),
                # Gaps vs Core Peers (pp or %)
                "gap_va_per_addetto_core_pct": float((c_row["va_per_addetto_eur"] / core_peers["va_per_addetto_eur"].median() - 1.0) * 100.0),
                "gap_retrib_per_dipendente_core_pct": float((c_row["retribuzione_per_dipendente_eur"] / core_peers["retribuzione_per_dipendente_eur"].median() - 1.0) * 100.0),
                # Ranks
                "rank_va_per_addetto_er": va_add_rank,
                "rank_retrib_per_dipendente_er": retrib_dip_rank,
                "n_peers_total": len(y_df),
            })

    summary_df = pd.DataFrame(records).sort_values(["macro_sector", "year"]).reset_index(drop=True)
    summary_df.to_parquet(out_path, index=False)
    print(f"[istat_frame_sbs] Saved summary {len(summary_df)} rows to {out_path}")
    return summary_df


if __name__ == "__main__":
    force_run = "--force" in sys.argv
    p = build_frame_sbs_panel(force=force_run)
    s = build_frame_sbs_summary(p)
    print(p.head())
    print(s.head())
