"""Build data/external/piaac.csv from the OECD Education GPS export of the Survey of Adult Skills 2023 (PIAAC).

The OECD GPS endpoint (gpseducation.oecd.org/getdata/EaGUnicode) is behind a Cloudflare challenge for scripts, so the table was captured by a
person in their own browser (HAR export, 2026-10-13) and the response body saved as data/external/source/piaac/gpseducation_piaac2023.tsv.
2012 levels are derived as the 2023 level minus the OECD's reported change 2012-2023 (columns P087/P088).
Usage: uv run python scripts/import_piaac.py
"""
import csv
import io
from operator import itemgetter
from pathlib import Path

import pandas as pd

from cifresignificative.countries import NAMES

SRC = Path("data/external/source/piaac/gpseducation_piaac2023.tsv")
OUT = Path("data/external/piaac.csv")
META_ROWS = {"ID", "DESCRIPTION", "UNIT", "PARSETYPE", "SHEET", "PRECISION", "--------"}
# panel indicator -> GPS column id (2023 levels)
SERIES = {
    "piaac_literacy": "P001", "piaac_numeracy": "P002", "piaac_adaptive_problem_solving": "P062",
    "piaac_low_literacy_pct": "P013", "piaac_low_numeracy_pct": "P015", "piaac_high_literacy_pct": "P012",
    "piaac_literacy_16_24": "Q001", "piaac_literacy_25_34": "Q002", "piaac_literacy_55_65": "Q005", "piaac_numeracy_25_34": "Q007",
    "piaac_literacy_women": "Q012", "piaac_literacy_men": "Q011",
    "piaac_literacy_below_upper_secondary": "Q019", "piaac_literacy_tertiary": "Q021",
    "piaac_literacy_parents_no_upper_secondary": "Q054", "piaac_literacy_parent_tertiary": "Q056",
}
DERIVED_2012 = {"piaac_literacy": "P087", "piaac_numeracy": "P088"}  # change 2012-2023 (points)


def main() -> None:
    raw = pd.read_csv(io.StringIO(SRC.read_text(encoding="utf-8")), sep="\t", dtype=str)
    ids = raw.iloc[0].to_dict()  # first row maps column name -> GPS indicator id
    col = {v: k for k, v in ids.items() if isinstance(v, str)}
    data = raw[~raw.META.isin(META_ROWS) & raw.CODE.isin(NAMES)].set_index("CODE")
    num = lambda gps: pd.to_numeric(data[col[gps]], errors="coerce")
    base = "OECD Education GPS, Survey of Adult Skills 2023 (PIAAC), indicator "
    rows: list[tuple[str, int, str, float, str, str]] = []
    for name, gps in SERIES.items():
        for iso, v in num(gps).dropna().items():
            rows.append((iso, 2023, name, round(float(v), 1), "education", f"{base}{gps} (browser export 2026-10-13)"))
    for name, chg in DERIVED_2012.items():
        lvl, ch = num(SERIES[name]), num(chg)
        for iso, v in (lvl - ch).dropna().items():
            rows.append((iso, 2012, name, round(float(v), 1), "education", f"derived: 2023 level ({SERIES[name]}) minus reported change 2012-2023 ({chg}), OECD Education GPS"))
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iso3", "year", "indicator", "value", "theme", "source"])
        w.writerows(sorted(rows, key=itemgetter(2, 1, 0)))
    got = pd.DataFrame(rows, columns=["iso3", "year", "indicator", "value", "theme", "source"])
    print(f"{len(got)} values; countries with data: {sorted(got.iso3.unique())}")
    it = got[got.iso3 == "ITA"].set_index(["indicator", "year"]).value
    print("Italy:", {f"{k[0]}_{k[1]}": v for k, v in it.items() if k[0] in ("piaac_literacy", "piaac_numeracy", "piaac_low_literacy_pct")})


if __name__ == "__main__":
    main()
