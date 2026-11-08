"""ISTAT / Coeweb / Regione Emilia-Romagna: Foreign trade and provincial export statistics (1991–2025).

Downloads and processes official foreign trade datasets from the Emilia-Romagna regional statistical
office ('Esploradati Coeweb - Istat'), providing detailed trade balance, imports, exports, and
commodity/country breakdowns across the 9 Emilia-Romagna provinces with a focus on Forlì-Cesena (ITH58).

Produces 4 harmonised tables:
1. `coeweb_provinces_export.parquet`: Provincial cross-section (2025) of export, import, trade balance,
   per-capita metrics, coverage ratios, and regional export shares for all 9 ER provinces.
2. `coeweb_emr_time_series.parquet`: Annual time series (1991–2025) of Emilia-Romagna export, import,
   and trade balance.
3. `coeweb_commodities_export.parquet`: Top merchandise categories by export value and trade balance (2025),
   identifying the agrifood and agricultural machinery specialization.
4. `coeweb_countries_balance.parquet`: Regional trade balances with top destination countries (EU27 and global).
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import truststore

from cifresignificative.core.geo import ER_PROVINCES
from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/coeweb_trade")
PROCESSED_DIR = Path("data/processed/cesena")

COEWEB_ZIP_URL = "https://statistica.regione.emilia-romagna.it/servizi-online/esploradaticoe/import-export-emilia-romagna.zip/@@download/file"
ZIP_NAME = "import_export_emilia_romagna.zip"


def fetch(raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Download and cache the Emilia-Romagna Coeweb foreign trade ZIP archive."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / ZIP_NAME
    if zip_path.exists() and not force:
        return zip_path

    req = urllib.request.Request(
        COEWEB_ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    zip_path.write_bytes(data)
    record("coeweb_trade", path=str(zip_path), url=COEWEB_ZIP_URL, size_bytes=len(data))
    return zip_path


def build_coeweb_provinces(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build provincial cross-section table (2025) of export, import, and balance for all 9 ER provinces."""
    zip_path = raw_dir / ZIP_NAME
    if not zip_path.exists():
        fetch(raw_dir)

    with zipfile.ZipFile(zip_path) as z:
        df_exp = pd.read_excel(io.BytesIO(z.read("3A. Esportazioni.xlsx")))
        df_imp = pd.read_excel(io.BytesIO(z.read("3B. Importazioni.xlsx")))
        df_bal = pd.read_excel(io.BytesIO(z.read("3C. Saldo commerciale.xlsx")))

    rows = []
    for p_name, p_info in ER_PROVINCES.items():
        row_exp = df_exp[df_exp.iloc[:, 0] == p_name]
        row_imp = df_imp[df_imp.iloc[:, 0] == p_name]
        row_bal = df_bal[df_bal.iloc[:, 0] == p_name]

        if row_exp.empty or row_imp.empty or row_bal.empty:
            continue

        exp_val = float(row_exp.iloc[0, 2])
        imp_val = float(row_imp.iloc[0, 1])
        bal_val = float(row_bal.iloc[0, 1])

        rows.append({
            "provincia": p_name,
            "nuts3": p_info.nuts3,
            "prov_sigla": p_info.prov_sigla,
            "pop_2024": p_info.pop_2024,
            "export_eur": exp_val,
            "import_eur": imp_val,
            "trade_balance_eur": bal_val,
            "coverage_ratio_pct": round(exp_val / imp_val * 100, 2) if imp_val > 0 else None,
            "export_per_capita_eur": round(exp_val / p_info.pop_2024, 2),
            "balance_per_capita_eur": round(bal_val / p_info.pop_2024, 2),
        })

    df = pd.DataFrame(rows)
    total_exp = df["export_eur"].sum()
    df["export_share_pct"] = (df["export_eur"] / total_exp * 100).round(2)
    df["rank_export"] = df["export_eur"].rank(ascending=False).astype(int)
    df["rank_balance"] = df["trade_balance_eur"].rank(ascending=False).astype(int)
    df["rank_export_per_capita"] = df["export_per_capita_eur"].rank(ascending=False).astype(int)
    df = df.sort_values("rank_export").reset_index(drop=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "coeweb_provinces_export.parquet"
    df.to_parquet(out_path, index=False)
    return df


def build_coeweb_time_series(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build annual time series (1991–2025) of Emilia-Romagna foreign trade."""
    zip_path = raw_dir / ZIP_NAME
    if not zip_path.exists():
        fetch(raw_dir)

    with zipfile.ZipFile(zip_path) as z:
        df_ts = pd.read_excel(io.BytesIO(z.read("2. Andamento delle importazioni e delle esportazioni.xlsx")), skiprows=1)

    df_ts.columns = ["year", "import_eur", "export_eur", "trade_balance_eur"]
    df_ts["year"] = df_ts["year"].astype(int)
    for c in ["import_eur", "export_eur", "trade_balance_eur"]:
        df_ts[c] = df_ts[c].astype(float)
    df_ts["coverage_ratio_pct"] = (df_ts["export_eur"] / df_ts["import_eur"] * 100).round(2)
    df_ts = df_ts.sort_values("year").reset_index(drop=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "coeweb_emr_time_series.parquet"
    df_ts.to_parquet(out_path, index=False)
    return df_ts


def build_coeweb_commodities(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build top export and trade surplus commodity categories (2025)."""
    zip_path = raw_dir / ZIP_NAME
    if not zip_path.exists():
        fetch(raw_dir)

    with zipfile.ZipFile(zip_path) as z:
        df_6a = pd.read_excel(io.BytesIO(z.read("6A. Le 10 categorie merceologiche per valore delle Esportazioni.xlsx")), skiprows=1)
        df_6c = pd.read_excel(io.BytesIO(z.read("6C. Le 10 categorie merceologiche con maggiore Saldo commerciale.xlsx")), skiprows=1)

    df_6a.columns = ["commodity", "export_eur"]
    df_6a["export_eur"] = df_6a["export_eur"].astype(float)
    df_6a["code"] = df_6a["commodity"].str.split("-").str[0].str.strip()
    df_6a["name"] = df_6a["commodity"].str.split("-").str[1:].str.join("-").str.strip()

    df_6c.columns = ["commodity", "trade_balance_eur"]
    df_6c["trade_balance_eur"] = df_6c["trade_balance_eur"].astype(float)
    df_6c["code"] = df_6c["commodity"].str.split("-").str[0].str.strip()
    df_6c["name"] = df_6c["commodity"].str.split("-").str[1:].str.join("-").str.strip()

    merged = pd.merge(
        df_6a[["code", "name", "export_eur"]],
        df_6c[["code", "name", "trade_balance_eur"]],
        on="code",
        how="outer",
    )
    merged["name"] = merged["name_x"].combine_first(merged["name_y"])
    merged = merged.drop(columns=["name_x", "name_y"])
    merged["is_agrifood"] = merged["code"].isin(["CA101", "CA108", "CA120", "CK283"])

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "coeweb_commodities_export.parquet"
    merged.to_parquet(out_path, index=False)
    return merged


def build_coeweb_countries(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build destination country trade balances (EU27 and top global partners)."""
    zip_path = raw_dir / ZIP_NAME
    if not zip_path.exists():
        fetch(raw_dir)

    with zipfile.ZipFile(zip_path) as z:
        df_5 = pd.read_excel(io.BytesIO(z.read("5. Export, import, saldo per Aggregato geo-economico.xlsx")), skiprows=1)

    df_5.columns = ["country", "export_eur", "import_eur", "trade_balance_eur"]
    for c in ["export_eur", "import_eur", "trade_balance_eur"]:
        df_5[c] = df_5[c].astype(float)
    df_5["coverage_ratio_pct"] = (df_5["export_eur"] / df_5["import_eur"] * 100).round(2)
    df_5 = df_5.sort_values("trade_balance_eur", ascending=False).reset_index(drop=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "coeweb_countries_balance.parquet"
    df_5.to_parquet(out_path, index=False)
    return df_5


def build_all(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Build all Coeweb trade tables."""
    return {
        "provinces": build_coeweb_provinces(raw_dir, processed_dir),
        "time_series": build_coeweb_time_series(raw_dir, processed_dir),
        "commodities": build_coeweb_commodities(raw_dir, processed_dir),
        "countries": build_coeweb_countries(raw_dir, processed_dir),
    }


if __name__ == "__main__":
    dfs = build_all()
    print("Coeweb trade tables built successfully:")
    for k, v in dfs.items():
        print(f"  {k}: shape {v.shape}")
