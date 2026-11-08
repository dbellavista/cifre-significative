"""ISTAT 7th General Agricultural Census (2020): Land use (SAT/SAU) and farm size distribution.

Processes official Istat agricultural census tables compiled by Regione Emilia-Romagna,
providing provincial-level agricultural structures across all 9 ER provinces:
1. `agri_sat_land_use_provinces.parquet`: Total agricultural area (SAT) and utilised agricultural
   area (SAU) by land use category (arable land, cereals, fruit orchards, vineyards, pastures, woods).
2. `agri_sau_size_classes_provinces.parquet`: Farm size distribution by SAU class (<10 ha, 10–20 ha,
   20–30 ha, 30–50 ha, 50–100 ha, >=100 ha), capturing the smallholder fragmentation vs consolidation.
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/istat_agricoltura")
PROCESSED_DIR = Path("data/processed/cesena")

SAT_FILE = "sat_utilizzo_provincia_2020.csv"
SAU_CLASS_FILE = "sau_classe_provincia_2020.csv"
SAU_LEGAL_FILE = "sau_forma_zona_2020.csv"


def build_sat_land_use(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build land use and crop categories across all 9 ER provinces (7th Agricultural Census 2020)."""
    csv_path = raw_dir / SAT_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing agricultural land use file: {csv_path}")

    df = pd.read_csv(csv_path, sep=";", skiprows=1, index_col=0)
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    # Standardise index names and identify sub-items
    categories = []
    is_subitem = []
    parent_category = []
    clean_names = []

    current_parent = None
    for idx in df.index:
        raw_name = str(idx).strip()
        if raw_name.startswith(("–", "-")):
            clean_name = raw_name.lstrip("–- ").strip()
            categories.append(clean_name)
            is_subitem.append(True)
            parent_category.append(current_parent)
        else:
            clean_name = raw_name
            categories.append(clean_name)
            is_subitem.append(False)
            parent_category.append(None)
            current_parent = clean_name
        clean_names.append(clean_name)

    df["category"] = clean_names
    df["is_subitem"] = is_subitem
    df["parent_category"] = parent_category

    # Reorder columns: metadata first, then provinces
    meta_cols = ["category", "is_subitem", "parent_category"]
    prov_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + prov_cols].reset_index(drop=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "agri_sat_land_use_provinces.parquet"
    df.to_parquet(out_path, index=False)
    return df


def build_sau_size_classes(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build farm size class distribution across all 9 ER provinces (7th Agricultural Census 2020)."""
    csv_path = raw_dir / SAU_CLASS_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing farm size class file: {csv_path}")

    df = pd.read_csv(csv_path, sep=";", skiprows=1, skipfooter=1, engine="python", index_col=0)
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    clean_classes = []
    for idx in df.index:
        s = str(idx).strip()
        if "Fino a" in s:
            clean_classes.append("< 10 ha")
        elif "100 e oltre" in s:
            clean_classes.append(">= 100 ha")
        elif s == "Totale":
            clean_classes.append("Totale")
        else:
            clean_classes.append(s + " ha")

    df["size_class"] = clean_classes
    meta_cols = ["size_class"]
    prov_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + prov_cols].reset_index(drop=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "agri_sau_size_classes_provinces.parquet"
    df.to_parquet(out_path, index=False)
    return df


def build_all(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Build all agricultural census tables."""
    return {
        "land_use": build_sat_land_use(raw_dir, processed_dir),
        "size_classes": build_sau_size_classes(raw_dir, processed_dir),
    }


if __name__ == "__main__":
    dfs = build_all()
    print("Agricultural census tables built successfully:")
    for k, v in dfs.items():
        print(f"  {k}: shape {v.shape}")
