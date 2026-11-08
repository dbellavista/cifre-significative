"""ISTAT 15th General Census: Municipal origin-destination commuting matrices (2011).

Downloads and processes the official ISTAT Origin-Destination commuting flow archive
('Matrice del pendolarismo 2011') covering habitual daily movements for work and study
across Italian municipalities.

Extracts three harmonised tables for the Cesena deep-dive:
1. `commuting_od_matrix.parquet`: Pairwise origin-destination flows for the 13 ER benchmark
   cities and 16 Comprensorio Cesenate municipalities.
2. `commuting_municipal_summary.parquet`: Municipal-level summary accounting for all 29 target
   municipalities (internal workers, outbound commuters, inbound commuters, total jobs,
   net commuter inflow, job self-containment rate, resident labor retention rate, modal split,
   and travel time distribution).
3. `commuting_belt_corridors.parquet`: Bilateral commuting exchanges between Cesena and each
   of the 16 surrounding belt municipalities and regional peer capitals.
"""
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import truststore

from cifresignificative.core.geo import (
    ALL_COMUNI_IDS,
    BELT_BY_ID,
    BELT_COMUNI_IDS,
    CESENA_BELT_COMUNI,
    COMUNI_BY_ID,
    ER_COMUNI,
)
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/istat_pendolarismo")
PROCESSED_DIR = Path("data/processed/cesena")

PENDOLARISMO_URL = "https://www.istat.it/storage/cartografia/matrici_pendolarismo/matrici_pendolarismo_2011.zip"
ZIP_NAME = "matrici_pendolarismo_2011.zip"
TXT_INNER_PATH = "MATRICE PENDOLARISMO 2011/matrix_pendo2011_10112014.txt"


def fetch(raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Download and cache the 2011 ISTAT commuting archive."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / ZIP_NAME
    if zip_path.exists() and not force:
        return zip_path

    print(f"[istat_pendolarismo] Downloading {PENDOLARISMO_URL}...")
    req = urllib.request.Request(
        PENDOLARISMO_URL,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        content = resp.read()
        zip_path.write_bytes(content)
        record(
            "istat_pendolarismo_2011",
            filename=ZIP_NAME,
            size_bytes=len(content),
            url=PENDOLARISMO_URL,
            note="ISTAT 15th Census commuting matrices for work and study",
        )
    return zip_path


def parse_commuting_records(
    zip_path: Path,
    target_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse raw fixed-width records and aggregate into municipal summaries and OD flows."""
    # Summary stats per municipality
    stats = defaultdict(lambda: {
        "work_internal": 0, "work_out": 0, "work_in": 0,
        "study_internal": 0, "study_out": 0, "study_in": 0,
        "mode_active": 0.0, "mode_transit": 0.0, "mode_car": 0.0, "mode_twowheel": 0.0, "mode_other": 0.0, "mode_total": 0.0,
        "time_le15": 0.0, "time_16_30": 0.0, "time_31_60": 0.0, "time_gt60": 0.0, "time_total": 0.0,
    })

    od_records: list[dict[str, object]] = []
    cesena_corridors: dict[str, dict[str, int]] = defaultdict(lambda: {
        "in_to_cesena": 0, "out_from_cesena": 0,
        "study_in_to_cesena": 0, "study_out_from_cesena": 0,
    })

    with zipfile.ZipFile(zip_path) as z, z.open(TXT_INNER_PATH) as f:
        for line in f:
                parts = line.split()
                if not parts:
                    continue
                rtype = parts[0]

                if rtype == b"S":
                    prov_o, com_o = parts[2].decode(), parts[3].decode()
                    orig_id = f"{int(prov_o):03d}{int(com_o):03d}"
                    prov_d, com_d = parts[7].decode(), parts[8].decode()
                    dest_id = f"{int(prov_d):03d}{int(com_d):03d}" if prov_d.isdigit() and com_d.isdigit() else "999999"

                    in_orig = orig_id in target_ids
                    in_dest = dest_id in target_ids
                    if not in_orig and not in_dest:
                        continue

                    motivo = int(parts[5])
                    sesso = int(parts[4])
                    n_ind = int(parts[14]) if parts[14].isdigit() else 0

                    if in_orig and in_dest:
                        od_records.append({
                            "origin_id": orig_id,
                            "dest_id": dest_id,
                            "purpose": "work" if motivo == 2 else "study",
                            "sex": "M" if sesso == 1 else "F",
                            "persons": n_ind,
                        })

                    if motivo == 2:  # Work
                        if orig_id == dest_id and in_orig:
                            stats[orig_id]["work_internal"] += n_ind
                        else:
                            if in_orig:
                                stats[orig_id]["work_out"] += n_ind
                            if in_dest:
                                stats[dest_id]["work_in"] += n_ind

                        # Bilateral flows with Cesena (040007)
                        if orig_id != "040007" and dest_id == "040007":
                            cesena_corridors[orig_id]["in_to_cesena"] += n_ind
                        elif orig_id == "040007" and dest_id != "040007":
                            cesena_corridors[dest_id]["out_from_cesena"] += n_ind

                    elif motivo == 1:  # Study
                        if orig_id == dest_id and in_orig:
                            stats[orig_id]["study_internal"] += n_ind
                        else:
                            if in_orig:
                                stats[orig_id]["study_out"] += n_ind
                            if in_dest:
                                stats[dest_id]["study_in"] += n_ind

                        if orig_id != "040007" and dest_id == "040007":
                            cesena_corridors[orig_id]["study_in_to_cesena"] += n_ind
                        elif orig_id == "040007" and dest_id != "040007":
                            cesena_corridors[dest_id]["study_out_from_cesena"] += n_ind

                elif rtype == b"L":
                    prov_o, com_o = parts[2].decode(), parts[3].decode()
                    orig_id = f"{int(prov_o):03d}{int(com_o):03d}"
                    if orig_id not in target_ids:
                        continue
                    motivo = int(parts[5])
                    if motivo != 2:  # mode and time metrics for workers
                        continue

                    mezzo = int(parts[10]) if parts[10].isdigit() else 0
                    tempo = int(parts[12]) if parts[12].isdigit() else 0
                    try:
                        stima = float(parts[13])
                    except ValueError:
                        stima = 0.0

                    # Transport modes:
                    # 10: bike, 12: walk
                    # 1: train, 2: tram, 3: metro, 4: bus urbano, 5: corriera extra-urbano, 6: bus aziendale/scolastico
                    # 7: car driver, 8: car passenger
                    # 9: moto/scooter
                    # 11: other
                    if mezzo in (10, 12):
                        stats[orig_id]["mode_active"] += stima
                    elif mezzo in (1, 2, 3, 4, 5, 6):
                        stats[orig_id]["mode_transit"] += stima
                    elif mezzo in (7, 8):
                        stats[orig_id]["mode_car"] += stima
                    elif mezzo == 9:
                        stats[orig_id]["mode_twowheel"] += stima
                    else:
                        stats[orig_id]["mode_other"] += stima
                    stats[orig_id]["mode_total"] += stima

                    # Travel time brackets: 1: <=15m, 2: 16-30m, 3: 31-60m, 4: >60m
                    if tempo == 1:
                        stats[orig_id]["time_le15"] += stima
                    elif tempo == 2:
                        stats[orig_id]["time_16_30"] += stima
                    elif tempo == 3:
                        stats[orig_id]["time_31_60"] += stima
                    elif tempo == 4:
                        stats[orig_id]["time_gt60"] += stima
                    stats[orig_id]["time_total"] += stima

    # Build municipal summary dataframe
    summary_rows: list[dict[str, object]] = []
    for cid in target_ids:
        s = stats[cid]
        name = COMUNI_BY_ID[cid].name if cid in COMUNI_BY_ID else BELT_BY_ID[cid].name
        is_peer = cid in COMUNI_BY_ID
        is_cesena = cid == "040007"
        group = "cesena_core" if is_cesena else (COMUNI_BY_ID[cid].nuts3 if is_peer else BELT_BY_ID[cid].belt_group)

        tot_jobs = s["work_internal"] + s["work_in"]
        tot_res = s["work_internal"] + s["work_out"]
        net_inflow = s["work_in"] - s["work_out"]

        tot_students_local = s["study_internal"] + s["study_in"]
        tot_students_res = s["study_internal"] + s["study_out"]
        net_study_inflow = s["study_in"] - s["study_out"]

        m_tot = s["mode_total"]
        pct_car = s["mode_car"] / m_tot if m_tot > 0 else 0.0
        pct_active = s["mode_active"] / m_tot if m_tot > 0 else 0.0
        pct_transit = s["mode_transit"] / m_tot if m_tot > 0 else 0.0
        pct_twowheel = s["mode_twowheel"] / m_tot if m_tot > 0 else 0.0

        t_tot = s["time_total"]
        pct_le15 = s["time_le15"] / t_tot if t_tot > 0 else 0.0
        pct_16_30 = s["time_16_30"] / t_tot if t_tot > 0 else 0.0
        pct_31_60 = s["time_31_60"] / t_tot if t_tot > 0 else 0.0
        pct_gt60 = s["time_gt60"] / t_tot if t_tot > 0 else 0.0

        summary_rows.append({
            "comune_id": cid,
            "name": name,
            "group": group,
            "is_peer": is_peer,
            "is_cesena": is_cesena,
            "jobs_local": tot_jobs,
            "resident_workers": tot_res,
            "internal_workers": s["work_internal"],
            "inbound_workers": s["work_in"],
            "outbound_workers": s["work_out"],
            "net_commuter_inflow": net_inflow,
            "job_self_containment_rate": s["work_internal"] / tot_jobs if tot_jobs > 0 else 0.0,
            "labor_retention_rate": s["work_internal"] / tot_res if tot_res > 0 else 0.0,
            "commuter_inflow_ratio": net_inflow / tot_jobs if tot_jobs > 0 else 0.0,
            "students_local": tot_students_local,
            "resident_students": tot_students_res,
            "internal_students": s["study_internal"],
            "inbound_students": s["study_in"],
            "outbound_students": s["study_out"],
            "net_student_inflow": net_study_inflow,
            "pct_car": pct_car,
            "pct_active": pct_active,
            "pct_transit": pct_transit,
            "pct_twowheel": pct_twowheel,
            "pct_commute_le15m": pct_le15,
            "pct_commute_16_30m": pct_16_30,
            "pct_commute_31_60m": pct_31_60,
            "pct_commute_gt60m": pct_gt60,
        })

    df_summary = pd.DataFrame(summary_rows).sort_values("jobs_local", ascending=False).reset_index(drop=True)

    # Build OD flow dataframe
    df_od = pd.DataFrame(od_records)
    if not df_od.empty:
        # Group by origin, dest, purpose and sum
        df_od = df_od.groupby(["origin_id", "dest_id", "purpose", "sex"]).sum().reset_index()
        # Add names
        id_to_name = {**{c.id: c.name for c in ER_COMUNI}, **{b.id: b.name for b in CESENA_BELT_COMUNI}}
        df_od["origin_name"] = df_od["origin_id"].map(id_to_name).fillna(df_od["origin_id"])
        df_od["dest_name"] = df_od["dest_id"].map(id_to_name).fillna(df_od["dest_id"])

    # Build corridor dataframe for Cesena
    corridor_rows: list[dict[str, object]] = []
    for cid, cf in cesena_corridors.items():
        name = COMUNI_BY_ID[cid].name if cid in COMUNI_BY_ID else (BELT_BY_ID[cid].name if cid in BELT_BY_ID else cid)
        group = "er_peer" if cid in COMUNI_BY_ID else (BELT_BY_ID[cid].belt_group if cid in BELT_BY_ID else "other")
        res_workers = stats[cid]["work_internal"] + stats[cid]["work_out"]
        in_c = cf["in_to_cesena"]
        out_c = cf["out_from_cesena"]
        net_c = in_c - out_c
        gross_c = in_c + out_c

        corridor_rows.append({
            "partner_id": cid,
            "partner_name": name,
            "group": group,
            "inbound_to_cesena": in_c,
            "outbound_from_cesena": out_c,
            "net_inflow_to_cesena": net_c,
            "gross_commuter_flow": gross_c,
            "share_partner_resident_workers": in_c / res_workers if res_workers > 0 else 0.0,
            "study_inbound_to_cesena": cf["study_in_to_cesena"],
            "study_outbound_from_cesena": cf["study_out_from_cesena"],
            "study_net_inflow_to_cesena": cf["study_in_to_cesena"] - cf["study_out_from_cesena"],
        })

    df_corridors = pd.DataFrame(corridor_rows).sort_values("inbound_to_cesena", ascending=False).reset_index(drop=True)

    return df_summary, df_od, df_corridors


def build(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
    force: bool = False,
) -> dict[str, Path]:
    """Execute complete ISTAT pendolarismo build and return generated paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = fetch(raw_dir=raw_dir, force=force)

    target_ids = set(ALL_COMUNI_IDS) | set(BELT_COMUNI_IDS)
    df_summary, df_od, df_corridors = parse_commuting_records(zip_path, target_ids)

    p_summary = out_dir / "commuting_municipal_summary.parquet"
    p_od = out_dir / "commuting_od_matrix.parquet"
    p_corridors = out_dir / "commuting_belt_corridors.parquet"

    df_summary.to_parquet(p_summary, index=False)
    df_od.to_parquet(p_od, index=False)
    df_corridors.to_parquet(p_corridors, index=False)

    print(f"[istat_pendolarismo] Saved {len(df_summary)} municipal summary rows to {p_summary}")
    print(f"[istat_pendolarismo] Saved {len(df_od)} OD flow rows to {p_od}")
    print(f"[istat_pendolarismo] Saved {len(df_corridors)} corridor rows to {p_corridors}")

    return {
        "commuting_municipal_summary": p_summary,
        "commuting_od_matrix": p_od,
        "commuting_belt_corridors": p_corridors,
    }


if __name__ == "__main__":
    build()
