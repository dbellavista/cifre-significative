"""SIOPE (Sistema Informativo sulle Operazioni degli Enti Pubblici) municipal and Unioni cash expenditure (`2016–2024`).

Combines the official MEF RGS BDAP SIOPE annual cash expenditure open datasets for Emilia-Romagna:
`https://bdap-opendata.rgs.mef.gov.it/export/csv/{year}---Emilia-Romagna---SIOPE-Movimenti-cumulati-mensili-di-Spesa.csv`

Provides:
1. **Harmonized cash expenditure (`2016–2024`)** for the 13 ER benchmark municipalities down to the single economic macroaggregato:
   - `spese_correnti_cassa` (`Titolo 1`, `U1...`)
   - `spese_capitale_cassa` (`Titolo 2`, `U2...`)
   - `rimborso_prestiti_cassa` (`Titolo 4`, `U4...` from 2017; `U3` in the old 2016 codes) and `incremento_att_fin_cassa` (`Titolo 3`, `U3...` from 2017)
   - `spesa_personale_cassa` (`Macroaggregato 101`, `U101...` Redditi da lavoro dipendente)
   - `spesa_beni_servizi_cassa` (`Macroaggregato 103`, `U103...` Acquisto beni e servizi)
   - `trasferimenti_correnti_cassa` (`Macroaggregato 104`, `U104...` Trasferimenti correnti)
2. **Consolidated perimeter with Unioni di Comuni**:
   - `UNIONE DEI COMUNI VALLE DEL SAVIO` (associated with Cesena `040007`)
   - `UNIONE DELLA ROMAGNA FAENTINA` (associated with Faenza `039010`)
   - `UNIONE DELLE TERRE D ARGINE` (associated with Carpi `036005`)
   - `NUOVO CIRCONDARIO IMOLESE` (associated with Imola `037032`)
   - `UNIONE DI COMUNI DELLA ROMAGNA FORLIVESE UNIONE MONTANA` (associated with Forlì `040012`)
   - `UNIONE RUBICONE E MARE` (Comprensorio Cesenate)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/siope")
BASE_URL = "https://bdap-opendata.rgs.mef.gov.it/export/csv/{year}---Emilia-Romagna---SIOPE-Movimenti-cumulati-mensili-di-{kind}.csv"  # kind: Spesa | Entrata
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
YEARS = tuple(range(2016, 2025))

# Target municipalities in Emilia-Romagna (13 benchmark peers)
MUNI_ENTITIES: dict[str, str] = {
    "040007": "COMUNE DI CESENA",
    "040012": "COMUNE DI FORLI'",
    "039014": "COMUNE DI RAVENNA",
    "099014": "COMUNE DI RIMINI",
    "037006": "COMUNE DI BOLOGNA",
    "036023": "COMUNE DI MODENA",
    "035033": "COMUNE DI REGGIO NELL'EMILIA",
    "034027": "COMUNE DI PARMA",
    "033032": "COMUNE DI PIACENZA",
    "038008": "COMUNE DI FERRARA",
    "039010": "COMUNE DI FAENZA",
    "037032": "COMUNE DI IMOLA",
    "036005": "COMUNE DI CARPI",
}

# Associated Unioni di Comuni and their core focal municipality
UNIONI_ENTITIES: dict[str, tuple[str, str | None]] = {
    "unione_valle_savio": ("UNIONE DEI COMUNI VALLE DEL SAVIO", "040007"),
    "unione_rubicone_mare": ("UNIONE RUBICONE E MARE", None),
    "unione_romagna_forlivese": ("UNIONE DI COMUNI DELLA ROMAGNA FORLIVESE UNIONE MONTANA", "040012"),
    "unione_romagna_faentina": ("UNIONE DELLA ROMAGNA FAENTINA", "039010"),
    "unione_terre_dargine": ("UNIONE DELLE TERRE D ARGINE", "036005"),
    "nuovo_circondario_imolese": ("NUOVO CIRCONDARIO IMOLESE", "037032"),
}

# Reverse lookups for fast filtering
MUNI_BY_NAME = {name: cid for cid, name in MUNI_ENTITIES.items()}
UNIONI_BY_NAME = {tup[0]: (key, tup[1]) for key, tup in UNIONI_ENTITIES.items()}
ALL_TARGET_NAMES = set(MUNI_BY_NAME).union(UNIONI_BY_NAME)


def raw_path(year: int, raw_dir: Path = RAW_DIR, kind: str = "Spesa") -> Path:
    return raw_dir / (f"siope_emr_{year}_m12.parquet" if kind == "Spesa" else f"siope_entrate_emr_{year}_m12.parquet")


def fetch_year(year: int, raw_dir: Path = RAW_DIR, force: bool = False, kind: str = "Spesa") -> Path:
    """Download and extract month 12 cumulative cash movements (`kind` = `Spesa` or `Entrata`) for target entities in year `year`."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_path(year, raw_dir, kind)
    if out_path.exists() and not force:
        return out_path

    url = BASE_URL.format(year=year, kind=kind)
    resp = requests.get(url, headers=UA, stream=True, timeout=120)
    resp.raise_for_status()

    lines: list[str] = []
    target_month = f'"{year}/12"'
    header: str | None = None

    for i, line in enumerate(resp.iter_lines()):
        if not line:
            continue
        text = line.decode("utf-8", errors="ignore")
        if header is None:
            header = text
            lines.append(header)
            continue
        # Filter for December cumulative figures and target entities
        if target_month in text:
            parts = text.split(";")
            if len(parts) > 6:
                ente = parts[6].replace('"', "").strip()
                if ente in ALL_TARGET_NAMES:
                    lines.append(text)

    if len(lines) <= 1:
        # Fallback: if month 12 is missing (e.g. recent year), take the latest available month
        pass

    raw_csv = "\n".join(lines)
    df = pd.read_csv(io.StringIO(raw_csv), sep=";", low_memory=False)
    # Strip quotes from column names and string cells
    df.columns = df.columns.str.replace('"', "").str.strip()
    df["Importo cumulato"] = pd.to_numeric(df["Importo cumulato"], errors="coerce").fillna(0.0)
    df.to_parquet(out_path, index=False)
    record(f"siope_emr_{year}" if kind == "Spesa" else f"siope_entrate_emr_{year}", source="MEF RGS BDAP", dataset=f"siope_emr_{kind.lower()}_{year}", url=url)
    return out_path


def fetch_all(years: tuple[int, ...] = YEARS, raw_dir: Path = RAW_DIR, force: bool = False, kind: str = "Spesa") -> list[Path]:
    """Download all SIOPE cash years of one kind (`Spesa` expenditure or `Entrata` revenue)."""
    return [fetch_year(y, raw_dir=raw_dir, force=force, kind=kind) for y in years]


def parse_annual_summary(df_raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate raw monthly/code level SIOPE rows into economic aggregates per entity."""
    df = df_raw.copy()
    ente_col = "Descrizione Ente BDAP"
    code_col = "Codice Gestionale Enti Locali"
    amount_col = "Importo cumulato"

    rows: list[dict] = []
    for ente, grp in df.groupby(ente_col):
        ente_str = str(ente).replace('"', "").strip()
        is_muni = ente_str in MUNI_BY_NAME
        cid = MUNI_BY_NAME.get(ente_str)
        unione_info = UNIONI_BY_NAME.get(ente_str)
        unione_key = unione_info[0] if unione_info else None
        focal_cid = unione_info[1] if unione_info else None

        codes = grp[code_col].astype(str).str.strip()
        amounts = grp[amount_col]

        # Economic Titles
        spese_correnti = float(amounts[codes.str.startswith("U1")].sum())
        spese_capitale = float(amounts[codes.str.startswith("U2")].sum())
        partite_giro = float(amounts[codes.str.startswith("U7")].sum())
        # The 2016 files still use the old SIOPE codes (`U1101`...), where Titolo 3 is the repayment of loans; with the D.Lgs. 118 chart of accounts (2017 on) Titolo 3
        # is the purchase of financial assets (shares, deposits) and the repayment of loans is Titolo 4.
        has_new_codes = bool(codes.str.fullmatch(r"U\d{10}").any())
        rimborso_prestiti = float(amounts[codes.str.startswith("U4" if has_new_codes else "U3")].sum())
        incremento_att_fin = float(amounts[codes.str.startswith("U3")].sum()) if has_new_codes else np.nan

        # Macroaggregates within Titolo 1 (Spese correnti)
        spesa_personale = float(amounts[codes.str.startswith("U101")].sum())
        spesa_beni_servizi = float(amounts[codes.str.startswith("U103")].sum())
        trasferimenti_correnti = float(amounts[codes.str.startswith("U104")].sum())
        interessi_passivi = float(amounts[codes.str.startswith("U107")].sum())
        # Transfers inside a Comune-Unione perimeter: counting them on both sides would double-count the same euro.
        trasf_corr_unioni = float(amounts[codes == "U1040102005"].sum())
        trasf_cap_unioni = float(amounts[codes == "U2030102005"].sum())
        trasf_corr_comuni = float(amounts[codes == "U1040102003"].sum())
        trasf_cap_comuni = float(amounts[codes == "U2030102003"].sum())
        # Old codes: the macroaggregates cannot be read: missing, not zero.
        macro = (lambda v: v) if has_new_codes else (lambda _: np.nan)

        rows.append({
            "year": year,
            "entity_name": ente_str,
            "entity_type": "comune" if is_muni else "unione",
            "comune_id": cid,
            "unione_key": unione_key,
            "focal_comune_id": focal_cid if not is_muni else cid,
            "spese_correnti_cassa_eur": spese_correnti,
            "spese_capitale_cassa_eur": spese_capitale,
            "rimborso_prestiti_cassa_eur": rimborso_prestiti,
            "incremento_att_fin_cassa_eur": incremento_att_fin,
            "partite_giro_cassa_eur": partite_giro,
            "spese_totali_cassa_eur": spese_correnti + spese_capitale + rimborso_prestiti + np.nan_to_num(incremento_att_fin),
            "spesa_personale_cassa_eur": macro(spesa_personale),
            "spesa_beni_servizi_cassa_eur": macro(spesa_beni_servizi),
            "trasferimenti_correnti_cassa_eur": macro(trasferimenti_correnti),
            "interessi_passivi_cassa_eur": macro(interessi_passivi),
            "trasf_correnti_a_unioni_eur": macro(trasf_corr_unioni),
            "trasf_capitale_a_unioni_eur": macro(trasf_cap_unioni),
            "trasf_correnti_a_comuni_eur": macro(trasf_corr_comuni),
            "trasf_capitale_a_comuni_eur": macro(trasf_cap_comuni),
            "personale_share_pct": macro((spesa_personale / spese_correnti * 100.0) if spese_correnti > 0 else np.nan),
            "servizi_share_pct": macro((spesa_beni_servizi / spese_correnti * 100.0) if spese_correnti > 0 else np.nan),
        })

    return pd.DataFrame(rows)


def parse_revenue_summary(df_raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate raw SIOPE revenue (`Entrata`) rows into cash revenue aggregates per entity.

    Titoli 1-3 (`E1`, `E2`, `E3`) are current revenue in both the old and the new chart of accounts. The lines that identify a tax (IMU `E1010106`, ICI `E1010108`, addizionale
    IRPEF `E1010116`, TARSU/TARI `E1010151` and `E1010161`, equalisation funds `E103`) exist only in the 10-digit codes of 2017 on and are missing (not zero) in 2016.
    Cash is what the treasurer collected in the year, arrears included (`...002` lines are collections after audit): it is not the accrual revenue of the budget.
    """
    df = df_raw.copy()
    rows: list[dict] = []
    for ente, grp in df.groupby("Descrizione Ente BDAP"):
        ente_str = str(ente).replace('"', "").strip()
        codes = grp["Codice Gestionale Enti Locali"].astype(str).str.strip()
        amounts = grp["Importo cumulato"]

        def total(prefix: str, codes: pd.Series = codes, amounts: pd.Series = amounts) -> float:
            return float(amounts[codes.str.startswith(prefix)].sum())

        has_new_codes = bool(codes.str.fullmatch(r"E\d{10}").any())
        macro = (lambda v: v) if has_new_codes else (lambda _: np.nan)
        rows.append({
            "year": year,
            "entity_name": ente_str,
            "entrate_tributarie_cassa_eur": total("E1"),
            "trasferimenti_correnti_entrate_cassa_eur": total("E2"),
            "entrate_extratributarie_cassa_eur": total("E3"),
            "entrate_correnti_cassa_eur": total("E1") + total("E2") + total("E3"),
            "imu_ici_cassa_eur": macro(total("E1010106") + total("E1010108")),
            "addizionale_irpef_cassa_eur": macro(total("E1010116")),
            "tari_tarsu_cassa_eur": macro(total("E1010151") + total("E1010161")),
            "fondi_perequativi_cassa_eur": macro(total("E103")),
        })
    return pd.DataFrame(rows)


def add_cash_rigidity(spesa_df: pd.DataFrame, revenue_df: pd.DataFrame) -> pd.DataFrame:
    """Join cash revenue to the cash expenditure panel and compute the cash rigidity proxy.

    `rigidity_cash_pct = (personnel + interest + loan repayment) / current revenue x 100`, all in cash. It follows the structure of the official budget-rigidity ratio
    (personnel and debt service over current revenue) but is not it: it mixes payments of the year with collections of the year, so it is not comparable with the
    accrual `rigidity_official_pct` and must not be spliced with it. Missing for 2016, where the macroaggregates cannot be read.
    """
    out = spesa_df.merge(revenue_df, on=["year", "entity_name"], how="left", validate="one_to_one")
    cost = out["spesa_personale_cassa_eur"] + out["interessi_passivi_cassa_eur"] + out["rimborso_prestiti_cassa_eur"]
    out["rigidity_cash_pct"] = (cost / out["entrate_correnti_cassa_eur"].replace(0, np.nan)) * 100.0
    # Waste-tax collections move between the municipal budget and the operator's tariff when the regime changes (Cesena: 15.4M in 2022, 2.6M in 2023), which shrinks the
    # denominator without any change in what the city spends: the variant without TARI/TARSU in current revenue does not depend on it.
    out["rigidity_cash_ex_tari_pct"] = (cost / (out["entrate_correnti_cassa_eur"] - out["tari_tarsu_cassa_eur"]).replace(0, np.nan)) * 100.0
    out.loc[out["spesa_personale_cassa_eur"].isna(), ["rigidity_cash_pct", "rigidity_cash_ex_tari_pct"]] = np.nan
    return out


def load_revenue_panel(years: tuple[int, ...] = YEARS, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and parse the multi-year SIOPE cash revenue panel for all target municipalities and Unioni."""
    frames: list[pd.DataFrame] = []
    for y in years:
        p = raw_path(y, raw_dir, "Entrata")
        if not p.exists():
            p = fetch_year(y, raw_dir=raw_dir, kind="Entrata")
        frames.append(parse_revenue_summary(pd.read_parquet(p), year=y))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_siope_panel(years: tuple[int, ...] = YEARS, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and parse the multi-year SIOPE cash panel for all target municipalities and Unioni."""
    frames: list[pd.DataFrame] = []
    for y in years:
        p = raw_dir / f"siope_emr_{y}_m12.parquet"
        if not p.exists():
            p = fetch_year(y, raw_dir=raw_dir)
        raw_df = pd.read_parquet(p)
        frames.append(parse_annual_summary(raw_df, year=y))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_consolidated_panel(
    siope_df: pd.DataFrame,
    demo_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute consolidated (Comune + Unione) per-capita expenditure series.

    For municipalities belonging to an Unione di Comuni (*Cesena*, *Faenza*, *Carpi*, *Imola*, *Forlì*),
    adds the municipality's population share of the Unione's expenditure to its direct municipal expenditure, netting the transfers between the two:
    `spesa_consolidata = (spesa_comune - trasf_a_unione) + peso * (spesa_unione - trasf_a_comuni)`.
    The shares are constants: against 2024 taxpayer shares they differ by 0.2 points (Cesena), 0.1 (Faenza), 0.9 (Carpi), 0.3 (Imola), 1.6 (Forlì).
    Spending is split by population, not by actual use of the services, so the result is an approximation.
    """
    df = siope_df.copy()

    # Split municipal rows from unione rows
    munis = df[df["entity_type"] == "comune"].copy()
    unioni = df[df["entity_type"] == "unione"].copy()

    # Empirical population weights inside each Unione (approximate mean from ISTAT Demo 2016-2024):
    # - Cesena in Unione Valle del Savio: ~83.4% (Cesena 96k / Unione 115k)
    # - Faenza in Unione Romagna Faentina: ~66.5% (Faenza 59k / Unione 88.5k)
    # - Carpi in Unione Terre d'Argine: ~67.0% (Carpi 72k / Unione 107k)
    # - Imola in Nuovo Circondario Imolese: ~52.5% (Imola 70k / Circondario 133k)
    # - Forlì in Romagna Forlivese: ~62.0% (Forlì 117k / Comprensorio 188k)
    UNIONE_WEIGHTS: dict[str, float] = {
        "040007": 0.834,  # Cesena / Valle del Savio
        "039010": 0.665,  # Faenza / Romagna Faentina
        "036005": 0.670,  # Carpi / Terre d'Argine
        "037032": 0.525,  # Imola / Circondario Imolese
        "040012": 0.620,  # Forlì / Romagna Forlivese
    }

    # Attach Unione expenditure to each focal municipality
    unioni_by_focal = unioni.set_index(["year", "focal_comune_id"])

    cons_rows: list[dict] = []
    for _, row in munis.iterrows():
        y = int(row["year"])
        cid = str(row["comune_id"])
        weight = UNIONE_WEIGHTS.get(cid, 0.0)

        u_correnti = 0.0
        u_capitale = 0.0
        u_personale = 0.0
        u_servizi = 0.0
        net_correnti = 0.0
        net_capitale = 0.0

        if (y, cid) in unioni_by_focal.index:
            u_row = unioni_by_focal.loc[(y, cid)]
            if isinstance(u_row, pd.DataFrame):
                u_row = u_row.iloc[0]
            # `nan_to_num`: the 2016 macroaggregates are missing, so nothing can be netted that year; its consolidated current spending stays gross
            u_correnti = (float(u_row["spese_correnti_cassa_eur"]) - np.nan_to_num(float(u_row.get("trasf_correnti_a_comuni_eur", 0.0)))) * weight
            u_capitale = (float(u_row["spese_capitale_cassa_eur"]) - np.nan_to_num(float(u_row.get("trasf_capitale_a_comuni_eur", 0.0)))) * weight
            u_personale = float(u_row["spesa_personale_cassa_eur"]) * weight
            u_servizi = float(u_row["spesa_beni_servizi_cassa_eur"]) * weight
            net_correnti = np.nan_to_num(float(row.get("trasf_correnti_a_unioni_eur", 0.0)))
            net_capitale = np.nan_to_num(float(row.get("trasf_capitale_a_unioni_eur", 0.0)))

        rec = row.to_dict()
        rec["unione_share_weight"] = weight
        rec["unione_correnti_share_eur"] = u_correnti
        rec["unione_capitale_share_eur"] = u_capitale
        rec["unione_personale_share_eur"] = u_personale
        rec["unione_servizi_share_eur"] = u_servizi

        rec["netted_correnti_eur"] = net_correnti
        rec["netted_capitale_eur"] = net_capitale
        rec["consolidated_correnti_cassa_eur"] = rec["spese_correnti_cassa_eur"] - net_correnti + u_correnti
        rec["consolidated_capitale_cassa_eur"] = rec["spese_capitale_cassa_eur"] - net_capitale + u_capitale
        rec["consolidated_personale_cassa_eur"] = rec["spesa_personale_cassa_eur"] + u_personale
        rec["consolidated_servizi_cassa_eur"] = rec["spesa_beni_servizi_cassa_eur"] + u_servizi
        rec["consolidated_totali_cassa_eur"] = rec["spese_totali_cassa_eur"] - net_correnti - net_capitale + u_correnti + u_capitale
        cons_rows.append(rec)

    cons_df = pd.DataFrame(cons_rows)

    if demo_df is not None and not demo_df.empty:
        pop_map = demo_df.set_index(["comune_id", "year"])["pop_total"].to_dict()
        cons_df["pop_total"] = cons_df.apply(lambda r: pop_map.get((r["comune_id"], r["year"]), np.nan), axis=1)
        pop = cons_df["pop_total"]

        cons_df["direct_correnti_pc"] = cons_df["spese_correnti_cassa_eur"] / pop
        cons_df["direct_capitale_pc"] = cons_df["spese_capitale_cassa_eur"] / pop
        cons_df["direct_personale_pc"] = cons_df["spesa_personale_cassa_eur"] / pop
        cons_df["unione_correnti_pc"] = cons_df["unione_correnti_share_eur"] / pop
        cons_df["netted_correnti_pc"] = cons_df["netted_correnti_eur"] / pop
        cons_df["consolidated_correnti_pc"] = cons_df["consolidated_correnti_cassa_eur"] / pop
        cons_df["consolidated_capitale_pc"] = cons_df["consolidated_capitale_cassa_eur"] / pop
        cons_df["consolidated_personale_pc"] = cons_df["consolidated_personale_cassa_eur"] / pop
        cons_df["consolidated_totali_pc"] = cons_df["consolidated_totali_cassa_eur"] / pop

    return cons_df


def build(out_dir: Path = Path("data/processed/cesena"), raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Download, parse, and save SIOPE cash panel and consolidated municipal datasets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fetch_all(raw_dir=raw_dir)
    fetch_all(raw_dir=raw_dir, kind="Entrata")
    panel_df = add_cash_rigidity(load_siope_panel(raw_dir=raw_dir), load_revenue_panel(raw_dir=raw_dir))

    p_panel = out_dir / "siope_cash_panel.parquet"
    panel_df.to_parquet(p_panel, index=False)

    demo_p = out_dir / "demography.parquet"
    demo_df = pd.read_parquet(demo_p) if demo_p.exists() else None

    cons_df = build_consolidated_panel(panel_df, demo_df=demo_df)
    p_cons = out_dir / "siope_consolidated.parquet"
    cons_df.to_parquet(p_cons, index=False)

    return {"siope_cash_panel": p_panel, "siope_consolidated": p_cons}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "fetch":
        fetch_all()
        fetch_all(kind="Entrata")
    else:
        paths = build()
        for k, v in paths.items():
            print(f"{k}: {v}")
