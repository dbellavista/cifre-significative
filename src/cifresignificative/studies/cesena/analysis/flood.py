"""May 2023 Romagna Flood Shock & Resilience Analysis (`studies.cesena.analysis.flood`).

Quantifies the socio-economic, fiscal, and firm-level impact of the catastrophic May 2023 Emilia-Romagna flood
(Alluvione dell'Emilia-Romagna, D.L. 1 giugno 2023, n. 61) across affected vs unaffected municipalities in Emilia-Romagna.

Groups municipalities by how Allegato 1 of the decree lists them (checked on a reproduction of the annex, 2026-10-30; recheck against the Gazzetta before citing):
1. `whole_territory`: Cesena, Forlì, Faenza, Ravenna.
2. `part_territory`: Imola and Bologna (named frazioni only).
3. `not_listed`: Rimini, Reggio nell'Emilia, Parma, Modena, Piacenza, Carpi, Ferrara.
A listing is a legal perimeter for relief measures, not a measure of damage, and municipalities in the same group may have been hit very differently.

Evaluates multi-dimensional resilience across:
- **Household income shock (MEF IRPEF 2022 -> 2023)**: real mean income loss and tax base contraction.
- **Municipal capital reconstruction surge (MEF RGS BDAP SIOPE 2022 -> 2024)**: cash capital spending outlays.
- **Real economy & local unit survival (ISTAT ASIA 2022 -> 2024 & FRAME-SBS 2022 -> 2023)**: firm destruction vs scaling.
- **Demographic stability & relocation (ISTAT Demo POSAS 2022 -> 2025)**: net migration rates and resident population.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cifresignificative.core.geo import ALL_COMUNI_IDS, BELT_BY_ID, CESENA, COMUNI_BY_ID
from cifresignificative.studies.cesena.analysis.data import (
    PROCESSED_DIR,
    load_asia,
    load_demography,
    load_frame_sbs_panel,
    load_irpef_summary,
    load_siope_panel,
)

# How Allegato 1 of D.L. 61/2023 lists each benchmark municipality
FLOOD_TIERS: dict[str, str] = {
    "040007": "whole_territory",  # Cesena
    "040012": "whole_territory",  # Forlì
    "039010": "whole_territory",  # Faenza
    "039014": "whole_territory",  # Ravenna
    "037032": "part_territory",   # Imola (named frazioni)
    "037006": "part_territory",   # Bologna (frazione Paleotto)
    "099014": "not_listed",       # Rimini
    "035033": "not_listed",       # Reggio nell'Emilia
    "034027": "not_listed",       # Parma
    "036023": "not_listed",       # Modena
    "033032": "not_listed",       # Piacenza
    "036005": "not_listed",       # Carpi
    "038008": "not_listed",       # Ferrara
}

TIER_LABELS: dict[str, str] = {
    "whole_territory": "Listed in D.L. 61/2023: whole territory",
    "part_territory": "Listed in D.L. 61/2023: part of the territory",
    "not_listed": "Not listed in D.L. 61/2023",
    "not_checked": "Listing not checked",
}


def build_flood_shock_panel(
    out_dir: Path = PROCESSED_DIR,
    comuni_ids: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build the comprehensive municipal flood shock and recovery comparison dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "flood_shock_panel.parquet"

    target_ids = comuni_ids if comuni_ids is not None else ALL_COMUNI_IDS

    demo = load_demography()
    irpef = load_irpef_summary()
    asia = load_asia()
    siope = load_siope_panel()
    frame = load_frame_sbs_panel()

    records: list[dict[str, Any]] = []

    for cid in target_ids:
        if cid in COMUNI_BY_ID:
            c_name = COMUNI_BY_ID[cid].name
        elif cid in BELT_BY_ID:
            c_name = BELT_BY_ID[cid].name
        else:
            c_name = cid

        tier = FLOOD_TIERS.get(cid, "not_checked")

        # 1. Demographic metrics (2022 vs 2023 vs 2024 vs 2025)
        d_sub = demo[demo["comune_id"] == cid].set_index("year")
        pop_22 = float(d_sub.loc[2022, "pop_total"]) if 2022 in d_sub.index else np.nan
        pop_23 = float(d_sub.loc[2023, "pop_total"]) if 2023 in d_sub.index else np.nan
        pop_24 = float(d_sub.loc[2024, "pop_total"]) if 2024 in d_sub.index else np.nan
        pop_25 = float(d_sub.loc[2025, "pop_total"]) if 2025 in d_sub.index else np.nan

        migr_22 = float(d_sub.loc[2022, "net_migr_rate_per_1k"]) if 2022 in d_sub.index else np.nan
        migr_23 = float(d_sub.loc[2023, "net_migr_rate_per_1k"]) if 2023 in d_sub.index else np.nan
        migr_24 = float(d_sub.loc[2024, "net_migr_rate_per_1k"]) if 2024 in d_sub.index else np.nan
        pop_growth_22_25 = (pop_25 / pop_22 - 1.0) * 100.0 if (pop_22 and pop_25) else np.nan

        # 2. MEF IRPEF income shock (2022 vs 2023)
        i_sub = irpef[irpef["comune_id"] == cid].set_index("year")
        inc_22 = float(i_sub.loc[2022, "mean_income_real_eur"]) if 2022 in i_sub.index else np.nan
        inc_23 = float(i_sub.loc[2023, "mean_income_real_eur"]) if 2023 in i_sub.index else np.nan
        tax_22 = float(i_sub.loc[2022, "taxpayers_total"]) if 2022 in i_sub.index else np.nan
        tax_23 = float(i_sub.loc[2023, "taxpayers_total"]) if 2023 in i_sub.index else np.nan
        income_growth_22_23 = (inc_23 / inc_22 - 1.0) * 100.0 if (inc_22 and inc_23) else np.nan
        taxpayer_growth_22_23 = (tax_23 / tax_22 - 1.0) * 100.0 if (tax_22 and tax_23) else np.nan

        # 3. SIOPE municipal capital reconstruction cash spending (2022 vs 2024)
        s_muni = siope[(siope["comune_id"] == cid) & (siope["entity_type"] == "comune")].set_index("year")
        cap_22 = float(s_muni.loc[2022, "spese_capitale_cassa_eur"]) if 2022 in s_muni.index else np.nan
        cap_23 = float(s_muni.loc[2023, "spese_capitale_cassa_eur"]) if 2023 in s_muni.index else np.nan
        cap_24 = float(s_muni.loc[2024, "spese_capitale_cassa_eur"]) if 2024 in s_muni.index else np.nan
        cap_pc_22 = cap_22 / pop_22 if (cap_22 and pop_22) else np.nan
        cap_pc_23 = cap_23 / pop_23 if (cap_23 and pop_23) else np.nan
        cap_pc_24 = cap_24 / pop_24 if (cap_24 and pop_24) else np.nan
        cap_growth_22_24 = (cap_pc_24 / cap_pc_22 - 1.0) * 100.0 if (cap_pc_22 and cap_pc_24) else np.nan
        # A single base year exaggerates the jump where 2022 happened to be low (Faenza): also compare with the 2019-21 average.
        base_vals = [
            float(s_muni.loc[y, "spese_capitale_cassa_eur"]) / float(d_sub.loc[y, "pop_total"])
            for y in (2019, 2020, 2021) if y in s_muni.index and y in d_sub.index
        ]
        cap_pc_base = float(np.mean(base_vals)) if base_vals else np.nan
        cap_growth_base_24 = (cap_pc_24 / cap_pc_base - 1.0) * 100.0 if (cap_pc_base and cap_pc_24) else np.nan

        # 4. ASIA active local units and private employees (2022 vs 2023 vs 2024)
        a_tot = asia[(asia["comune_id"] == cid) & (asia["is_total"].astype(bool))].set_index("year")
        emp_22 = float(a_tot.loc[2022, "employees"]) if 2022 in a_tot.index else np.nan
        emp_23 = float(a_tot.loc[2023, "employees"]) if 2023 in a_tot.index else np.nan
        emp_24 = float(a_tot.loc[2024, "employees"]) if 2024 in a_tot.index else np.nan
        u_22 = float(a_tot.loc[2022, "local_units"]) if 2022 in a_tot.index else np.nan
        u_23 = float(a_tot.loc[2023, "local_units"]) if 2023 in a_tot.index else np.nan
        u_24 = float(a_tot.loc[2024, "local_units"]) if 2024 in a_tot.index else np.nan
        emp_growth_22_23 = (emp_23 / emp_22 - 1.0) * 100.0 if (emp_22 and emp_23) else np.nan
        emp_growth_22_24 = (emp_24 / emp_22 - 1.0) * 100.0 if (emp_22 and emp_24) else np.nan
        # Cesena's 2022 food-manufacturing record is anomalous (see `economy.employment_anomalies`), so also report the change without it.
        a_x10 = asia[(asia["comune_id"] == cid) & (~asia["is_total"].astype(bool)) & (asia["ateco_2d"] != "10")].groupby("year")["employees"].sum()
        emp_growth_x10_22_24 = (a_x10[2024] / a_x10[2022] - 1.0) * 100.0 if {2022, 2024} <= set(a_x10.index) else np.nan
        units_growth_22_24 = (u_24 / u_22 - 1.0) * 100.0 if (u_22 and u_24) else np.nan
        units_growth_23_24 = (u_24 / u_23 - 1.0) * 100.0 if (u_23 and u_24) else np.nan

        # 5. ISTAT FRAME-SBS real value added and turnover shock (2022 vs 2023)
        f_tot = frame[(frame["comune_id"] == cid) & (frame["macro_sector"] == "totale")].set_index("year")
        va_22 = float(f_tot.loc[2022, "valore_aggiunto_real_eur"]) if 2022 in f_tot.index else np.nan
        va_23 = float(f_tot.loc[2023, "valore_aggiunto_real_eur"]) if 2023 in f_tot.index else np.nan
        fatt_22 = float(f_tot.loc[2022, "fatturato_real_eur"]) if 2022 in f_tot.index else np.nan
        fatt_23 = float(f_tot.loc[2023, "fatturato_real_eur"]) if 2023 in f_tot.index else np.nan
        va_growth_22_23 = (va_23 / va_22 - 1.0) * 100.0 if (va_22 and va_23) else np.nan
        fatt_growth_22_23 = (fatt_23 / fatt_22 - 1.0) * 100.0 if (fatt_22 and fatt_23) else np.nan

        records.append({
            "comune_id": cid,
            "comune_name": c_name,
            "flood_tier": tier,
            "flood_tier_label": TIER_LABELS[tier],
            "is_cesena": cid == CESENA,
            # Demographic
            "pop_2022": pop_22,
            "pop_2025": pop_25,
            "pop_growth_22_25_pct": pop_growth_22_25,
            "net_migr_rate_2022": migr_22,
            "net_migr_rate_2023": migr_23,
            "net_migr_rate_2024": migr_24,
            # Income
            "real_mean_income_2022": inc_22,
            "real_mean_income_2023": inc_23,
            "income_growth_22_23_pct": income_growth_22_23,
            "taxpayer_growth_22_23_pct": taxpayer_growth_22_23,
            # Fiscal / Capital Spending
            "capitale_cassa_pc_2022": cap_pc_22,
            "capitale_cassa_pc_2023": cap_pc_23,
            "capitale_cassa_pc_2024": cap_pc_24,
            "capitale_cassa_growth_22_24_pct": cap_growth_22_24,
            "capitale_cassa_pc_base_19_21": cap_pc_base,
            "capitale_cassa_growth_base_24_pct": cap_growth_base_24,
            # Firm & Employment
            "employees_2022": emp_22,
            "employees_2024": emp_24,
            "emp_growth_22_23_pct": emp_growth_22_23,
            "emp_growth_22_24_pct": emp_growth_22_24,
            "emp_growth_x10_22_24_pct": emp_growth_x10_22_24,
            "local_units_2022": u_22,
            "local_units_2024": u_24,
            "units_growth_22_24_pct": units_growth_22_24,
            "units_growth_23_24_pct": units_growth_23_24,
            # FRAME-SBS Economic Accounts
            "real_va_growth_22_23_pct": va_growth_22_23,
            "real_turnover_growth_22_23_pct": fatt_growth_22_23,
        })

    df = pd.DataFrame(records).sort_values(["flood_tier", "comune_name"]).reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    print(f"[flood] Saved {len(df)} rows to {out_path}")
    return df


def compare_flood_tiers(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute aggregate median statistics across the three flood impact tiers."""
    if panel is None:
        panel = build_flood_shock_panel()

    metrics = [
        "income_growth_22_23_pct",
        "taxpayer_growth_22_23_pct",
        "capitale_cassa_pc_2022",
        "capitale_cassa_pc_2024",
        "capitale_cassa_growth_22_24_pct",
        "capitale_cassa_growth_base_24_pct",
        "emp_growth_22_24_pct",
        "emp_growth_x10_22_24_pct",
        "units_growth_23_24_pct",
        "real_va_growth_22_23_pct",
        "net_migr_rate_2023",
        "net_migr_rate_2024",
        "pop_growth_22_25_pct",
    ]

    tier_order = ["whole_territory", "part_territory", "not_listed"]
    summary_rows: list[dict[str, Any]] = []

    for t in tier_order:
        sub = panel[panel["flood_tier"] == t]
        row: dict[str, Any] = {
            "flood_tier": t,
            "tier_label": TIER_LABELS[t],
            "n_municipalities": len(sub),
            "municipalities": ", ".join(sub["comune_name"]),
        }
        for m in metrics:
            row[f"{m}_median"] = float(sub[m].median())
        summary_rows.append(row)

    # Add Cesena alone for direct focal benchmarking
    ces = panel[panel["is_cesena"]].iloc[0]
    ces_row: dict[str, Any] = {
        "flood_tier": "cesena_focal",
        "tier_label": "Cesena (Focal Municipality)",
        "n_municipalities": 1,
        "municipalities": "Cesena",
    }
    for m in metrics:
        ces_row[f"{m}_median"] = float(ces[m])
    summary_rows.insert(0, ces_row)

    return pd.DataFrame(summary_rows)
