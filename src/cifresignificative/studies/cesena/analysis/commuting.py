"""Analysis of ISTAT origin-destination commuting matrices for Cesena, ER peers, and the Comprensorio belt.

Links spatial mobility (habitual work and study commuting flows) to Pillar 3 (jobs & firms),
Pillar 4 (demography & housing affordability), and Track 4.1 (the suburbanization gradient).

Computes:
1. `peer_commuting_comparison`: Rankings of the 13 Emilia-Romagna benchmark cities on
   local jobs, resident labor force, net commuter balance, self-containment rate,
   labor retention rate, student net inflow, and modal split.
2. `comprensorio_commuter_network`: Bilateral work and study exchanges between Cesena
   and the 16 surrounding belt municipalities grouped into the 3 functional rings:
   - `rubicone_mare` (coastal/industrial belt)
   - `via_emilia_west` (Forlimpopoli & Bertinoro)
   - `unione_valle_savio` (mountain/hill belt)
   plus external regional capitals (Forlì, Ravenna, Rimini, Bologna).
3. `spatial_arbitrage_analysis`: Evaluates the joint spatial equilibrium between
   housing purchase savings (€/m² discount vs Cesena core) and resident commuter outflow
   into Cesena.
"""
from pathlib import Path

import pandas as pd

from cifresignificative.core.geo import (
    BELT_GROUP_LABELS,
    CESENA,
    ER_COMUNI,
)

PROCESSED_DIR = Path("data/processed/cesena")


def load_commuting_summary(data_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load municipal commuting summary and merge with housing & demographic indicators."""
    p_sum = data_dir / "commuting_municipal_summary.parquet"
    if not p_sum.exists():
        from cifresignificative.core.sources import istat_pendolarismo
        istat_pendolarismo.build(out_dir=data_dir)

    df_sum = pd.read_parquet(p_sum)

    # Optionally merge with housing affordability if available
    p_omi = data_dir / "omi_quotations_municipal.parquet"
    if p_omi.exists():
        df_omi = pd.read_parquet(p_omi)[["comune_id", "compr_mid_eur_m2", "loc_mid_eur_m2_mo"]]
        df_sum = df_sum.merge(df_omi, on="comune_id", how="left")

    return df_sum


def peer_commuting_comparison(data_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Compute comparative commuting metrics and ranks across the 13 Emilia-Romagna peer cities."""
    df_sum = load_commuting_summary(data_dir=data_dir)
    peers = df_sum[df_sum["is_peer"]].copy()

    # Core peer flag
    core_ids = {c.id for c in ER_COMUNI if c.core_peer}
    peers["is_core_peer"] = peers["comune_id"].isin(core_ids)

    # Compute ranks (1 = highest)
    peers["rank_jobs"] = peers["jobs_local"].rank(ascending=False).astype(int)
    peers["rank_net_inflow"] = peers["net_commuter_inflow"].rank(ascending=False).astype(int)
    peers["rank_self_containment"] = peers["job_self_containment_rate"].rank(ascending=False).astype(int)
    peers["rank_retention"] = peers["labor_retention_rate"].rank(ascending=False).astype(int)
    peers["rank_inflow_ratio"] = peers["commuter_inflow_ratio"].rank(ascending=False).astype(int)
    peers["rank_student_net_inflow"] = peers["net_student_inflow"].rank(ascending=False).astype(int)
    peers["rank_active_commute"] = peers["pct_active"].rank(ascending=False).astype(int)

    return peers.sort_values("jobs_local", ascending=False).reset_index(drop=True)


def comprensorio_commuter_network(data_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Aggregate bilateral commuting flows between Cesena and belt municipalities/peers."""
    p_corridors = data_dir / "commuting_belt_corridors.parquet"
    if not p_corridors.exists():
        from cifresignificative.core.sources import istat_pendolarismo
        istat_pendolarismo.build(out_dir=data_dir)

    df_corr = pd.read_parquet(p_corridors)

    # Filter for the 16 belt municipalities + 4 key regional neighbors
    key_peers = {"040012", "039014", "099014", "037006"}  # Forlì, Ravenna, Rimini, Bologna
    belt_mask = df_corr["group"].isin(["rubicone_mare", "via_emilia_west", "unione_valle_savio"])
    peer_mask = df_corr["partner_id"].isin(key_peers)

    df_focus = df_corr[belt_mask | peer_mask].copy()

    # Group-level summary
    belt_groups = ["rubicone_mare", "via_emilia_west", "unione_valle_savio"]
    df_rings = df_focus[df_focus["group"].isin(belt_groups)].groupby("group").agg({
        "inbound_to_cesena": "sum",
        "outbound_from_cesena": "sum",
        "net_inflow_to_cesena": "sum",
        "gross_commuter_flow": "sum",
        "study_inbound_to_cesena": "sum",
        "study_outbound_from_cesena": "sum",
        "study_net_inflow_to_cesena": "sum",
    }).reset_index()

    df_rings["group_label"] = df_rings["group"].map(BELT_GROUP_LABELS)

    # Regional neighbors summary
    df_neighbors = df_focus[df_focus["partner_id"].isin(key_peers)].copy()

    return {
        "municipal_flows": df_focus.sort_values("inbound_to_cesena", ascending=False).reset_index(drop=True),
        "ring_aggregates": df_rings.sort_values("inbound_to_cesena", ascending=False).reset_index(drop=True),
        "neighbor_capitals": df_neighbors.sort_values("inbound_to_cesena", ascending=False).reset_index(drop=True),
    }


def spatial_arbitrage_analysis(data_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Analyze the joint relationship between housing discounts and commuter outflows to Cesena."""
    df_sum = load_commuting_summary(data_dir=data_dir)
    p_corr = data_dir / "commuting_belt_corridors.parquet"
    df_corr = pd.read_parquet(p_corr)

    # Isolate belt municipalities
    belt_ids = df_sum[df_sum["group"].isin(["rubicone_mare", "via_emilia_west", "unione_valle_savio"])]["comune_id"]

    cesena_row = df_sum[df_sum["comune_id"] == CESENA].iloc[0]
    cesena_price = float(cesena_row.get("compr_mid_eur_m2", 1588.1))
    cesena_rent = float(cesena_row.get("loc_mid_eur_m2_mo", 5.56))

    sub = df_sum[df_sum["comune_id"].isin(belt_ids)].copy()
    sub = sub.merge(
        df_corr[["partner_id", "inbound_to_cesena", "outbound_from_cesena", "net_inflow_to_cesena", "share_partner_resident_workers"]],
        left_on="comune_id",
        right_on="partner_id",
        how="left",
    )

    if "compr_mid_eur_m2" in sub.columns:
        sub["price_discount_eur_m2"] = cesena_price - sub["compr_mid_eur_m2"]
        sub["price_discount_pct"] = sub["price_discount_eur_m2"] / cesena_price
        sub["purchase_savings_100m2"] = sub["price_discount_eur_m2"] * 100.0
    if "loc_mid_eur_m2_mo" in sub.columns:
        sub["rent_discount_eur_m2_mo"] = cesena_rent - sub["loc_mid_eur_m2_mo"]
        sub["rent_savings_80m2_annual"] = sub["rent_discount_eur_m2_mo"] * 80.0 * 12.0

    return sub.sort_values("share_partner_resident_workers", ascending=False).reset_index(drop=True)
