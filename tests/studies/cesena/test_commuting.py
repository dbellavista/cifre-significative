"""Unit tests for Cesena commuting and spatial equilibrium analysis (`studies.cesena.analysis.commuting`)."""
from pathlib import Path

import pytest

from cifresignificative.studies.cesena.analysis.commuting import (
    comprensorio_commuter_network,
    peer_commuting_comparison,
    spatial_arbitrage_analysis,
)

PROCESSED_DIR = Path("data/processed/cesena")


@pytest.fixture
def require_commuting():
    if not (PROCESSED_DIR / "commuting_municipal_summary.parquet").exists():
        pytest.skip("Commuting parquet tables not yet built")


def test_peer_commuting_comparison(require_commuting) -> None:
    """Verify peer comparison table contains all 13 ER peers and ranks."""
    peers = peer_commuting_comparison()
    assert len(peers) == 13
    assert set(peers["name"]).issuperset({"Cesena", "Forlì", "Rimini", "Ravenna", "Bologna"})

    ces = peers[peers["name"] == "Cesena"].iloc[0]
    assert ces["jobs_local"] == 42437
    assert ces["net_commuter_inflow"] == 4597
    assert ces["rank_net_inflow"] <= 8
    # Bologna has highest jobs
    assert peers.iloc[0]["name"] == "Bologna"


def test_comprensorio_commuter_network(require_commuting) -> None:
    """Verify belt network aggregates and net commuter flows to Cesena."""
    net = comprensorio_commuter_network()
    assert "municipal_flows" in net
    assert "ring_aggregates" in net
    assert "neighbor_capitals" in net

    rings = net["ring_aggregates"]
    assert len(rings) == 3
    # All 3 rings must have positive net inflows to Cesena
    assert (rings["net_inflow_to_cesena"] > 0).all()

    # Total belt inbound commuters must be over 8,000
    tot_belt_inbound = rings["inbound_to_cesena"].sum()
    assert tot_belt_inbound > 8000
    assert rings.loc[rings["group"] == "rubicone_mare", "inbound_to_cesena"].iloc[0] > 4000

    # Neighbor capitals include Forlì, Ravenna, Rimini, Bologna
    neigh = net["neighbor_capitals"]
    assert set(neigh["partner_name"]).issuperset({"Forlì", "Ravenna", "Rimini", "Bologna"})


def test_spatial_arbitrage_analysis(require_commuting) -> None:
    """Verify the spatial arbitrage table linking housing discounts and commuting shares."""
    arb = spatial_arbitrage_analysis()
    assert len(arb) == 16  # 16 belt municipalities
    assert "share_partner_resident_workers" in arb.columns
    # Top commuting shares should be > 20%
    top_share = arb.iloc[0]["share_partner_resident_workers"]
    assert top_share > 0.25
