"""PNRR ItaliaDomani / ReGiS: Public Procurement and Tender Execution by Municipality (2021–2026).

Extracts and parses project-level public procurement data (tenders, award procedures, award lags,
and contractor characteristics) from the open-data releases of the Italian National Recovery and Resilience
Plan (PNRR / Italia Domani, https://www.italiadomani.gov.it).
Enables micro-level procurement benchmarking of Cesena against the 12 Emilia-Romagna peer municipalities:
direct award shares, tender execution lags (days between CIG publication and award), contract sizes, and
contractor sectoral specialisation (ATECO).

Exploratory only: the file has no contracting authority or location, so a tender is matched to a municipality by finding its name anywhere in the row. Lots that merely mention a city are attributed to it (88% of "Cesena's" EUR 501M is three national utility lots), and a multi-city lot goes to the first name found. Do not rank municipalities with this; a sound comparison needs project-level ItaliaDomani data with the implementing body. The Cesena study retracted these results.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.core.geo import ALL_COMUNI_IDS, COMUNI_BY_ID
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/pnrr")
PROCESSED_DIR = Path("data/processed/cesena")

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}

FILES = {
    "PNRR_Gare.csv": "https://www.italiadomani.gov.it/content/dam/sogei-ng/opendata/PNRR_Gare.csv",
    "PNRR_Aggiudicatari_Gare.csv": "https://www.italiadomani.gov.it/content/dam/sogei-ng/opendata/PNRR_Aggiudicatari_Gare.csv",
}

# Regex to match target Emilia-Romagna municipalities in tender records
MUNI_TEXT_MAP: dict[str, str] = {
    "CESENA": "040007",
    "FORLÌ": "040012",
    "FORLI": "040012",
    "RIMINI": "099014",
    "RAVENNA": "039014",
    "IMOLA": "037032",
    "CARPI": "036005",
    "REGGIO NELL'EMILIA": "035033",
    "REGGIO EMILIA": "035033",
    "FERRARA": "038008",
    "MODENA": "036023",
    "PARMA": "034027",
    "PIACENZA": "033032",
    "FAENZA": "039010",
    "BOLOGNA": "037006",
}

_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(MUNI_TEXT_MAP.keys(), key=len, reverse=True)) + r")\b", re.IGNORECASE)


def fetch(force: bool = False) -> Path:
    """Download the PNRR Gare and Aggiudicatari open-data CSVs if missing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in FILES.items():
        dest = RAW_DIR / fname
        if dest.exists() and not force:
            continue
        resp = requests.get(url, headers=UA, timeout=300)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        record(f"pnrr_{fname}", source="ItaliaDomani_PNRR", dataset=fname, url=url)
    return RAW_DIR


def load_municipal_tenders(comuni_ids: set[str] | tuple[str, ...] | list[str] | None = None, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Stream and parse PNRR tenders for target municipalities from PNRR_Gare.csv."""
    target = set(ALL_COMUNI_IDS) if comuni_ids is None else set(comuni_ids)
    p = raw_dir / "PNRR_Gare.csv"
    if not p.exists():
        return pd.DataFrame()

    records = []
    with open(p, encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split(";")
        cig_idx = header.index("CIG")
        cup_idx = header.index("CUP")
        subm_idx = header.index("\ufeffCodice Univoco Submisura") if "\ufeffCodice Univoco Submisura" in header else 0
        proc_idx = header.index("Descrizione Procedura di Aggiudicazione")
        obj_idx = header.index("Oggetto Gara")
        pub_idx = header.index("Data Pubblicazione del CIG")
        award_date_idx = header.index("Data Aggiudicazione Definitiva")
        imp_gara_idx = header.index("Importo Complessivo Gara")
        imp_agg_idx = header.index("Importo Aggiudicazione")

        for line in f:
            m = _PATTERN.search(line)
            if not m:
                continue
            parts = line.strip().split(";")
            if len(parts) <= max(imp_agg_idx, award_date_idx):
                continue
            cid = MUNI_TEXT_MAP[m.group(0).upper()]
            if cid not in target:
                continue

            records.append({
                "comune_id": cid,
                "cig": parts[cig_idx].strip(),
                "cup": parts[cup_idx].strip(),
                "submisura_id": parts[subm_idx].strip(),
                "procedura": parts[proc_idx].strip(),
                "oggetto": parts[obj_idx].strip(),
                "data_pub": parts[pub_idx].strip(),
                "data_agg": parts[award_date_idx].strip(),
                "importo_gara": parts[imp_gara_idx].strip(),
                "importo_agg": parts[imp_agg_idx].strip(),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    for col in ("importo_gara", "importo_agg"):
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce").fillna(0.0)

    for dcol in ("data_pub", "data_agg"):
        df[dcol] = pd.to_datetime(df[dcol].astype(str).str.strip(), format="%d/%m/%Y", errors="coerce")

    df["award_lag_days"] = (df["data_agg"] - df["data_pub"]).dt.days
    df["is_direct"] = df["procedura"].str.contains("DIRETTO", case=False, na=False)
    df["rebate_pct"] = ((df["importo_gara"] - df["importo_agg"]) / df["importo_gara"].replace(0, float("nan"))).clip(lower=0.0, upper=1.0)
    df["comune_name"] = df["comune_id"].map(lambda c: COMUNI_BY_ID[c].name if c in COMUNI_BY_ID else c)

    return df


def load_municipal_contractors(cigs: set[str] | list[str] | None = None, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load contractor details and ATECO sectors for given CIGs from PNRR_Aggiudicatari_Gare.csv."""
    p = raw_dir / "PNRR_Aggiudicatari_Gare.csv"
    if not p.exists():
        return pd.DataFrame()

    target_cigs = set(cigs) if cigs is not None else None
    contractors = []
    with open(p, encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split(";")
        cig_idx = header.index("CIG")
        cup_idx = header.index("CUP")
        vat_idx = header.index("Codice Fiscale/P.IVA")
        name_idx = header.index("Denominazione Aggiudicatario")
        ateco_idx = header.index("Codice ATECO Aggiudicatario")
        form_idx = header.index("Descrizione Forma Giuridica Aggiudicatario") if "Descrizione Forma Giuridica Aggiudicatario" in header else -1

        for line in f:
            parts = line.strip().split(";")
            if len(parts) <= max(cig_idx, ateco_idx):
                continue
            cig_val = parts[cig_idx].strip()
            if target_cigs is not None and cig_val not in target_cigs:
                continue

            contractors.append({
                "cig": cig_val,
                "cup": parts[cup_idx].strip(),
                "contractor_vat": parts[vat_idx].strip(),
                "contractor_name": parts[name_idx].strip(),
                "legal_form": parts[form_idx].strip() if form_idx >= 0 and form_idx < len(parts) else "",
                "ateco": parts[ateco_idx].strip(),
            })

    return pd.DataFrame(contractors) if contractors else pd.DataFrame()


def summarize_municipal_procurement(comuni_ids: set[str] | tuple[str, ...] | list[str] | None = None, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Compute aggregate procurement execution benchmarks for the 13 ER municipalities."""
    df = load_municipal_tenders(comuni_ids, raw_dir=raw_dir)
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby(["comune_id", "comune_name"]).agg(
        n_tenders=("cig", "nunique"),
        total_tendered_eur=("importo_gara", "sum"),
        total_awarded_eur=("importo_agg", "sum"),
        direct_award_share=("is_direct", "mean"),
        mean_award_lag_days=("award_lag_days", "mean"),
        median_award_lag_days=("award_lag_days", "median"),
        mean_rebate_pct=("rebate_pct", "mean"),
    ).reset_index()

    summary["realisation_ratio"] = summary["total_awarded_eur"] / summary["total_tendered_eur"].replace(0, float("nan"))
    return summary.sort_values("total_awarded_eur", ascending=False, ignore_index=True)


def build(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build processed PNRR procurement parquet tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tenders = load_municipal_tenders()
    summary = summarize_municipal_procurement()

    tenders.to_parquet(out_dir / "pnrr_municipal_tenders.parquet", index=False)
    summary.to_parquet(out_dir / "pnrr_municipal_summary.parquet", index=False)
    return summary


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "fetch":
        fetch("--force" in sys.argv)
    else:
        df = build()
        print(f"Built PNRR procurement summary: {len(df)} municipalities.")
        print(df.to_string())
