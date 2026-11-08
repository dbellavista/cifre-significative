"""Tests for Agenzia delle Entrate OMI data source (`core.sources.agenzia_entrate_omi`)."""

import pandas as pd

from cifresignificative.core.geo import ALL_COMUNI_IDS, BELT_COMUNI_IDS, CESENA
from cifresignificative.core.sources import agenzia_entrate_omi


def test_omi_quotations_municipal_structure():
    df = agenzia_entrate_omi.build_omi_quotations()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(ALL_COMUNI_IDS) + len(BELT_COMUNI_IDS)

    required_cols = [
        "comune_id",
        "comune_name",
        "nuts3",
        "prov_sigla",
        "is_cesena",
        "is_core_peer",
        "belt_group",
        "compr_min_eur_m2",
        "compr_max_eur_m2",
        "compr_mid_eur_m2",
        "loc_min_eur_m2_mo",
        "loc_max_eur_m2_mo",
        "loc_mid_eur_m2_mo",
        "gross_rental_yield_pct",
        "diff_vs_cesena_eur_m2",
        "diff_vs_cesena_pct",
        "diff_rent_vs_cesena_eur_mo",
        "diff_rent_vs_cesena_pct",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col}"

    ces = df[df["is_cesena"]].iloc[0]
    assert ces["comune_id"] == CESENA
    assert 1400.0 < ces["compr_mid_eur_m2"] < 1800.0
    assert 4.5 < ces["loc_mid_eur_m2_mo"] < 7.0
    assert ces["diff_vs_cesena_pct"] == 0.0


def test_omi_zones_cesena_structure():
    df = agenzia_entrate_omi.build_omi_zones_cesena()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 10

    required_cols = [
        "comune_id",
        "comune_name",
        "zona",
        "zona_label",
        "tipo",
        "compr_mid_eur_m2",
        "loc_mid_eur_m2_mo",
    ]
    for col in required_cols:
        assert col in df.columns

    # Centro Storico should be among the highest priced zones
    cs = df[df["zona"].str.contains("CENTRO STORICO", case=False)]
    assert not cs.empty
    assert cs["compr_mid_eur_m2"].max() > 1900.0


def test_omi_transactions_panel():
    df = agenzia_entrate_omi.build_omi_transactions()
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert "regione" in df.columns
        assert "prov_sigla" in df.columns
        assert "year" in df.columns
        assert "ntn_residential" in df.columns
        assert 2011 in df["year"].values
        assert 2024 in df["year"].values


def test_zero_rent_quotes_are_missing_not_free():
    """OMI files a zone without a rental quote as 0, which must not pull the municipal mean down (Bologna fell to 5.10 EUR/m2)."""
    df = agenzia_entrate_omi.build_omi_quotations().set_index("comune_name")
    assert df.loc["Bologna", "loc_mid_eur_m2_mo"] > 7.0
    assert df.loc["Bologna", "rent_quote_share_pct"] < 100.0
    assert df.loc["Cesena", "rent_quote_share_pct"] == 100.0
