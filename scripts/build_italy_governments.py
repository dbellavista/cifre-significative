"""Build data/external/italy_governments.csv from the English Wikipedia list of Italian prime ministers, cross-checked against the Italian
Wikipedia list of governments (start and end dates must agree). Usage: uv run python scripts/build_italy_governments.py [en.html it.html]
"""
import io
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

EN = "https://en.wikipedia.org/wiki/List_of_prime_ministers_of_Italy"
IT = "https://it.wikipedia.org/wiki/Governi_italiani_per_durata"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
MONTHS_IT = {"gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5, "giugno": 6, "luglio": 7, "agosto": 8, "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12}
MONTHS_EN = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
OUT = Path("data/external/italy_governments.csv")


def html(url: str, path: str | None) -> str:
    return Path(path).read_text(encoding="utf-8") if path else requests.get(url, headers=UA, timeout=60).text


def parse_date(text: str, months: dict[str, int]) -> date | None:
    m = re.search(r"(\d{1,2})\s+([A-Za-zàù]+)\s+(\d{4})", str(text))
    return date(int(m.group(3)), months[m.group(2)], int(m.group(1))) if m and m.group(2) in months else None


def clean(s: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", str(s)).strip()


def main() -> None:
    en = pd.read_html(io.StringIO(html(EN, sys.argv[1] if len(sys.argv) > 1 else None)))[1]
    en.columns = [" ".join(dict.fromkeys(map(str, c))) for c in en.columns]
    rows = []
    for _, r in en.iterrows():
        start, end = parse_date(r["Term of office Took office"], MONTHS_EN), parse_date(r["Term of office Left office"], MONTHS_EN)
        if start is None:
            continue
        pm = clean(re.sub(r"\(.*?\)", "", r["Name (Birth–Death)"]))
        rows.append({"government": clean(r["Government"]), "pm": pm, "party": clean(r["Party Party.1"]), "start": start.isoformat(), "end": end.isoformat() if end else "",
                     "composition": clean(r["Composition"]), "legislature": clean(r["Legislature (Election)"]), "president": clean(r["President (Tenure)"])})
    gov = pd.DataFrame(rows)
    it = pd.read_html(io.StringIO(html(IT, sys.argv[2] if len(sys.argv) > 2 else None)))[0]
    it.columns = [clean(c) for c in it.columns]
    check = {}
    for _, r in it.iterrows():
        name, period = clean(r["Governo"]), str(r["Periodo in carica"])
        parts = re.split(r"\s+[–-]\s+", period)
        s = parse_date(parts[0], MONTHS_IT)
        e = parse_date(parts[1], MONTHS_IT) if len(parts) > 1 else None
        if s:
            check[name.lower()] = (s, e)
    agree, mismatch, missing = 0, [], []
    for r in gov.to_dict("records"):
        c = check.get(r["government"].lower())
        if c is None:
            missing.append(r["government"])
            continue
        ok = c[0].isoformat() == r["start"] and ((c[1].isoformat() if c[1] else "") == r["end"])
        agree += ok
        if not ok:
            mismatch.append((r["government"], r["start"], r["end"], c[0].isoformat(), c[1].isoformat() if c[1] else ""))
    gov["start_end_confirmed_by_it_wikipedia"] = gov.government.str.lower().map(lambda g: g in check and check[g][0].isoformat() == gov.loc[gov.government.str.lower() == g, "start"].iloc[0])
    gov["source"] = EN + " ; " + IT
    OUT.parent.mkdir(parents=True, exist_ok=True)
    gov.to_csv(OUT, index=False)
    print(f"{len(gov)} governments written; start/end agree with it.wikipedia for {agree}; not found there: {len(missing)}; disagree: {len(mismatch)}")
    for m in mismatch[:10]:
        print("  mismatch", m)
    print("  not found:", missing[:12])


if __name__ == "__main__":
    main()
