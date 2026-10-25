"""Crawler and downloader for the MEF Department of Finance open-data portal of tax-return statistics (licence CC BY 3.0 IT):
https://www1.finanze.gov.it/finanze/analisi_stat/public/index.php?opendata=yes

The listing page of a classification (`search_class[0]=cG_VENT` for income twentieths, ...) is paginated, ten table cards per page across all filing years; each card carries the title
(`2025 a.i. 2024 / IRPEF / Persone fisiche totali / ...`), the theme, the classification, an update date and the export link `index.php?tree=<year><table code>&export=3`. Empty responses mean
the table does not exist for that year. Requests are single-threaded with a pause (public data, be polite); files already present are not requested again. Parsing is separated from the network
so it can be tested on saved HTML. `sources/mef_irpef.py` parses the CSVs this module downloads.
"""
import html
import re
import time
from pathlib import Path

import pandas as pd
import requests
import truststore

truststore.inject_into_ssl()
BASE = "https://www1.finanze.gov.it/finanze/analisi_stat/public/index.php"
CLASSIFICATIONS = {"classi_reddito": "cCLRED740", "ventili": "cG_VENT", "regione": "cG_REGIONE", "attivita": "cCOD_SEZ", "eta": "cETA", "fatturato": "cG_VOLUME_AFFARI", "comune": "cCOMUNE"}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; cifresignificative-research; public open data, 1 request per second)"}
PERSONS = r"^(AAPF|ABPF|AE)"  # individuals: all, with a VAT number (employees, pensioners, entrepreneurs, professionals, farmers), and by prevalent income (AE)


def _text(fragment: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment))).strip()


def parse_cards(page: str) -> list[dict]:
    """Table cards of one listing page: code (year + table code), title, theme, classification, update date; cards without an export link are skipped."""
    out = []
    for card in page.split('class="card card-primary')[1:]:
        code = re.search(r"tree=(20\d\d[A-Z0-9]{9,12})", card)
        title = re.search(r"<h3[^>]*>(.*?)</h3>", card, flags=re.DOTALL)
        if not (code and title):
            continue
        field = {k: (re.search(rf"<h4[^>]*>\s*{k}:\s*</h4>.*?<span[^>]*>(.*?)</span>", card, flags=re.DOTALL)) for k in ("Data", "Tematica", "Classificazione")}
        out.append({"code": code.group(1), "year": int(code.group(1)[:4]), "table": code.group(1)[4:], "title": _text(title.group(1)),
                    **{k.lower(): _text(m.group(1)) if m else "" for k, m in field.items()}})
    return out


def list_tables(classification: str, max_pages: int = 100, pause: float = 1.0, session: requests.Session | None = None) -> pd.DataFrame:
    """Crawl every page of the listing for a classification key (`CLASSIFICATIONS` value) until a page brings no new table."""
    s = session or requests.Session()
    seen: dict[str, dict] = {}
    for p in range(1, max_pages + 1):
        r = s.get(BASE, params={"opendata": "yes", "export": "0", "media": "media", "personalizza": "no", "search_class[0]": classification, "page": p}, headers=HEADERS, timeout=90)
        r.raise_for_status()
        new = [c for c in parse_cards(r.text) if c["code"] not in seen]
        seen.update({c["code"]: c for c in new})
        if not new:
            break
        time.sleep(pause)
    return pd.DataFrame(list(seen.values())).sort_values(["table", "year"]).reset_index(drop=True) if seen else pd.DataFrame(columns=["code", "year", "table", "title", "data", "tematica", "classificazione"])


def download(code: str, out: Path, pause: float = 1.0, session: requests.Session | None = None) -> Path | None:
    """Export one table to `out/<year>/<code>.csv`; returns the path, or None when the portal returns an empty body (table absent for that year). Existing files are kept."""
    path = out / code[:4] / f"{code}.csv"
    if path.exists() and path.stat().st_size:
        return path
    r = (session or requests).get(BASE, params={"tree": code, "export": 3, "aggiornato": 1615465800}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    time.sleep(pause)
    if not r.content:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return path


def fetch(classification: str, out: Path, tables: str = PERSONS, years: tuple[int, int] = (2005, 2030), pause: float = 1.0) -> pd.DataFrame:
    """List the tables of a classification, keep those whose table code matches the regex `tables` and whose year is in `years`, download them, and write `catalog.csv` next to the files
    (with a `file` column: the saved path, or empty when the portal had no data). Returns the catalog."""
    s = requests.Session()
    cat = list_tables(classification, pause=pause, session=s)
    cat = cat[cat.table.str.match(tables) & cat.year.between(*years)].copy()
    cat["file"] = [str(p) if (p := download(c, out, pause, s)) else "" for c in cat.code]
    out.mkdir(parents=True, exist_ok=True)
    cat.to_csv(out / "catalog.csv", index=False)
    return cat
