"""Prepare source-level municipal income components from the frozen MEF archives.

Run from the repository root with ``uv run python studies/cesena/prepare_income_evidence.py``.
Blanks are not interpreted as zero and may reflect suppression or missing reporting.
"""
from pathlib import Path

import pandas as pd

from cifresignificative.core.geo import COMUNI_BY_ID
from cifresignificative.core.sources.mef_comuni import (
    SUMMARY_PATTERNS,
    YEARS,
    _to_num,
    parse_year,
    read_year_frame,
    zip_path,
)
from cifresignificative.studies.cesena.analysis.outcomes import bracket_gini_and_median

BASELINE = Path("data/raw/cesena_income_baseline")
OUTPUT = Path("data/processed/cesena/income_followup")
COMPONENTS = ("reddito_dipendente", "reddito_pensione", "reddito_autonomo", "reddito_impresa_ordinaria", "reddito_impresa_semplificata", "reddito_partecipazione", "reddito_fabbricati", "reddito_complessivo")


def main() -> None:
    patterns = {key: pat for key, pat, _ in SUMMARY_PATTERNS}
    by_cat = {c.catastale: c for c in COMUNI_BY_ID.values()}
    rows, denominators = [], []
    for year in YEARS:
        path = zip_path(year, BASELINE)
        frame = read_year_frame(path, set(by_cat))
        brackets, summary = parse_year(path)
        columns = {key: {"freq": [], "eur": []} for key in COMPONENTS}
        for key in COMPONENTS:
            for col in frame.columns:
                match = patterns[key].match(col)
                if match:
                    columns[key]["freq" if match.group(1).lower().startswith("freq") else "eur"].append(col)
            if any(len(matches) > 1 for matches in columns[key].values()):
                raise ValueError(f"Ambiguous columns for {year} {key}: {columns[key]}")
        for _, source in frame.iterrows():
            municipality = by_cat[source["catastale"]]
            for key in COMPONENTS:
                freq_cols, eur_cols = columns[key]["freq"], columns[key]["eur"]
                freq_col, eur_col = (freq_cols[0] if freq_cols else None), (eur_cols[0] if eur_cols else None)
                freq = _to_num(source[freq_col]) if freq_col else float("nan")
                amount = _to_num(source[eur_col]) if eur_col else float("nan")
                rows.append({"comune_id": municipality.id, "catastale": municipality.catastale, "tax_year": year, "component": key, "amount_nominal_eur": amount, "recipient_count": freq, "amount_source_column": eur_col, "count_source_column": freq_col, "amount_status": "absent_schema" if eur_col is None else ("blank_or_unparsed" if pd.isna(amount) else "observed"), "count_status": "absent_schema" if freq_col is None else ("blank_or_unparsed" if pd.isna(freq) else "observed"), "source_file": path.name})
            b = brackets[brackets.comune_id.eq(municipality.id)]
            d = bracket_gini_and_median(b)
            s = summary[summary.comune_id.eq(municipality.id)].iloc[0]
            denominators.append({"comune_id": municipality.id, "catastale": municipality.catastale, "tax_year": year, "taxpayers_total": s.taxpayers_total, "total_income_frequency": s.get("reddito_complessivo_freq"), "total_income_eur": s.get("reddito_complessivo_eur"), "positive_bracket_taxpayers": d["taxpayers_pos"], "positive_bracket_income_eur": d["income_pos_eur"], "positive_brackets_missing": int((b.class_order.ge(1) & (b.taxpayers.isna() | b.income_eur.isna())).sum()), "imputed_brackets": int(b.imputed.sum()), "nonpositive_bracket_missing": bool(b.loc[b.class_order.eq(0), ["taxpayers", "income_eur"]].isna().any(axis=None))})
    components = pd.DataFrame(rows)
    if components.duplicated(["comune_id", "tax_year", "component"]).any():
        raise ValueError("Duplicate city-year-component")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    components.to_parquet(OUTPUT / "income_components.parquet", index=False)
    pd.DataFrame(denominators).to_csv(OUTPUT / "income_denominators.csv", index=False)
    print(f"{len(components)} component rows; {len(denominators)} city-years in {OUTPUT}")


if __name__ == "__main__":
    main()
