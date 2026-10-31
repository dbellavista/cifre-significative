"""Fill data/external/pisa_2022.csv from the OECD workbook (PISA 2022 Results Vol. I, Annex B1, chapter 2 tables).

The workbook is the StatLink https://stat.link/files/53f23881-en/xmrlsh.xlsx, saved as
data/external/source/pisa2022_ch2_tables.xlsx (oecd.org blocks scripted downloads; stat.link does not).
Usage: uv run python scripts/import_pisa_2022.py
"""
import re
from pathlib import Path

import pandas as pd

from cifresignificative.core.countries import NAMES

SRC = Path("data/external/source/pisa2022_ch2_tables.xlsx")
OUT = Path("data/external/pisa_2022.csv")
TABLES = {"wb_pisa_math": ("Table I.B1.2.1", "mathematics"), "wb_pisa_reading": ("Table I.B1.2.2", "reading"), "wb_pisa_science": ("Table I.B1.2.3", "science")}
ALIASES = {"Czech Republic": "CZE", "Slovak Republic": "SVK", "Türkiye": None, "Slovenia": "SVN"}
NAME_TO_ISO = {v: k for k, v in NAMES.items()} | {k: v for k, v in ALIASES.items() if v}


def clean_name(raw: str) -> str:
    return re.sub(r"[\*†‡§¹²³\d\s]+$", "", str(raw)).strip()


def read_means(sheet: str, subject: str) -> pd.Series:
    d = pd.read_excel(SRC, sheet_name=sheet, header=None)
    title = str(d.iloc[1, 0]).lower()
    assert subject in title, f"{sheet} is '{title}', expected {subject}"
    header_row = d.index[d[1].astype(str).str.strip().eq("Mean score")][-1]  # row with 'Mean score' / 'S.E.' labels
    rows = d.iloc[header_row + 1 :, [0, 1]].dropna()
    rows = rows[pd.to_numeric(rows[1], errors="coerce").notna()]
    out = {}
    for name, val in zip(rows[0], rows[1], strict=True):
        iso = NAME_TO_ISO.get(clean_name(name))
        if iso in NAMES:
            out[iso] = round(float(val), 1)
    return pd.Series(out)


def main() -> None:
    tpl = pd.read_csv(OUT, dtype={"source": str})
    report = {}
    for ind, (sheet, subject) in TABLES.items():
        means = read_means(sheet, subject)
        mask = tpl.indicator == ind
        tpl.loc[mask, "value"] = tpl.loc[mask, "iso3"].map(means)
        tpl.loc[mask & tpl.value.notna(), "source"] = f"PISA 2022 Results Vol I, {sheet} (mean score); stat.link/files/53f23881-en/xmrlsh.xlsx"
        missing = sorted(set(tpl.loc[mask, "iso3"]) - set(means.index))
        report[ind] = (int(tpl.loc[mask, "value"].notna().sum()), missing)
    tpl.to_csv(OUT, index=False)
    for ind, (n, missing) in report.items():
        print(f"{ind}: {n} countries filled; not in the table: {missing}")


if __name__ == "__main__":
    main()
