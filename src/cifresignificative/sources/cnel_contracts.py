"""Renewal dates of the national collective agreements (CNEL archive, open-data API), for the agreements mapped to the Istat agreement groups.

CNEL's "Archivio nazionale dei contratti e degli accordi collettivi di lavoro" publishes every deposited agreement as JSON (https://www.cnel.it/Archivio-Contratti-Collettivi/Contratti-Open-Data, licence IODL 2.0, updated weekly): contract code (`codiceCcnl`,
the CNEL/INPS alphanumeric code), type (`tipologiaAccordo`), signing date (`dataStipula`), start (`dataDecorrenza`) and contractual expiry (`dataScadenzaContrattuale`) and the signatories. Employee counts (`dipendenti`) are empty in every row.
Two traps, both handled in `agreements` and `cycles`:
- A code collects the side-signings of minor unions on later dates (the Confcommercio commerce contract, H011, has rows of 2017 and 2019 signed by UGL only; the CGIL-CISL-UIL renewals are 2015-03 and 2024-03). Each row is flagged `confederal` (CGIL, CISL and UIL federations among the signatories) and a renewal cycle takes the earliest confederal signing.
- Private contracts record a renewal as `Accordo di rinnovo`; public ones (ARAN, codes `S...`) as an `Ipotesi` (also typed `Accordo di rinnovo` when present) followed by the `Testo definitivo`, which carries the three-year period: for the public codes the definitive text's date is used, because the pay rise
  takes effect after it. The consolidated `Testo definitivo` of a private renewal repeats the renewal's date and is ignored, except when a private cycle has no renewal row (C010, the AGIDAE schools): the definitive text then stands for it (`kind` says which). Wage-only agreements (`Economico secondo biennio`, `Accordo economico`) are not renewals and are ignored.
The API is queried by a developer step (`uv run python -m cifresignificative.sources.cnel_contracts fetch|build|check`); the raw JSON goes to `data/raw/cnel/` (gitignored) and the build writes `data/external/cnel_contracts.csv` (committed, one row per renewal cycle of a mapped code),
so tests and CI never fetch. The mapping of Istat groups to CNEL codes is hand-curated in `data/external/cnel_istat_map.csv` (evidence labels: see `data/external/source/cnel/README.md`).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.sources._common import record

truststore.inject_into_ssl()

API = "https://az-apim-cne-sa-0002-lgc-we.azure-api.net/ricerca-api/ricerca/pubblica/open-data"
ARCHIVES = {"current": "ARCHIVIO_CORRENTE", "historical": "ARCHIVIO_STORICO"}
RAW_DIR = Path("data/raw/cnel")
CSV = Path("data/external/cnel_contracts.csv")
MAP = Path("data/external/cnel_istat_map.csv")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
ASOF = pd.Timestamp("2026-10-17")  # the day the archive was queried: renewals after it are unknown (censored)
FIRST_YEAR = "2005"  # the Istat wage index starts in 2005; keep every cycle that can matter for the lag at December of a start year
CONFEDERAL = [re.compile(p) for p in (r"CGIL", r"CISL", r"UIL")]


def path(name: str) -> Path:
    return RAW_DIR / f"{ARCHIVES[name]}.json"


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, kind in ARCHIVES.items():
        if path(name).exists() and not force:
            continue
        url = f"{API}?type={kind}"
        resp = requests.get(url, headers=UA, timeout=600)
        resp.raise_for_status()
        path(name).write_bytes(resp.content)
        record(f"cnel_{name}", source="cnel-open-data", dataset=kind, url=url)
    return RAW_DIR


def agreements() -> pd.DataFrame:
    """Every agreement of both archives: code, id, kind, signed, start, expiry, confederal, public, title."""
    rows = []
    for name in ARCHIVES:
        for r in json.loads(path(name).read_text())["data"]:
            unions = " ".join(r["partiSindacaliFirmatarie"] or []).upper()
            rows.append({"code": r["codiceCcnl"], "id": r["idAccordo"], "kind": r["tipologiaAccordo"], "signed": r["dataStipula"], "start": r["dataDecorrenza"], "expiry": r["dataScadenzaContrattuale"], "confederal": all(p.search(unions) for p in CONFEDERAL), "title": r["titoloCcnl"]})
    d = pd.DataFrame(rows).drop_duplicates("id")
    for c in ("signed", "start", "expiry"):
        d[c] = pd.to_datetime(d[c])
    return d


def cycles(d: pd.DataFrame, codes: set[str] | None = None) -> pd.DataFrame:
    """One row per renewal cycle of a code (an expiry date; a row with no period is its own cycle with empty start and expiry): the earliest confederal signing (the earliest of any signing when no confederal row exists, `confederal` False), the period it covers and the number of agreement rows behind it."""
    public = d.code.str.startswith("S")
    keep = d.kind.isin(["Accordo di rinnovo", "Testo definitivo"]) & (d.signed >= pd.Timestamp(FIRST_YEAR))
    if codes is not None:
        keep &= d.code.isin(codes)
    r = d[keep].assign(rank=lambda x: ((x.kind == "Testo definitivo") != public[x.index]).astype(int))  # 0 = the row that stands for the renewal: the renewal for private codes, the definitive text for public ones
    out = []
    r = r.assign(cycle=lambda x: x.expiry.dt.strftime("%Y-%m-%d").where(x.expiry.notna(), "id" + x.id.astype(str)))  # a row without a period (11 mapped renewals, mostly 2021-24) stands alone
    for (code, _), g in r.groupby(["code", "cycle"]):
        g = g[g["rank"] == g["rank"].min()]  # a private cycle with no renewal row falls back to the definitive text
        gc = g[g.confederal]
        pick = gc if len(gc) else g
        first = pick.sort_values("signed").iloc[0]
        out.append({"code": code, "signed": first.signed, "start": first.start, "expiry": first.expiry, "kind": first.kind, "confederal": bool(len(gc)), "n_rows": len(g), "ids": ";".join(str(i) for i in sorted(g.id))})
    return pd.DataFrame(out).sort_values(["code", "signed"], ignore_index=True)


def primary_cycles(c: pd.DataFrame) -> pd.DataFrame:
    """The cycles that stand for a contract: the confederal ones when the code has any (the rest are minor-union side-signings with their own expiry dates), all of them otherwise (C010, the Stellantis group contract, FIOM did not sign; G031)."""
    return c[c.confederal | ~c.code.map(c.groupby("code").confederal.any())].reset_index(drop=True)


def load_map() -> pd.DataFrame:
    return pd.read_csv(MAP, keep_default_na=False)


def load(csv: Path = CSV) -> pd.DataFrame:
    return pd.read_csv(csv, parse_dates=["signed", "start", "expiry"]).assign(confederal=lambda d: d.confederal.astype(bool))


def build_csv(out: Path = CSV) -> pd.DataFrame:
    codes = set(load_map().cnel_code) - {""}
    c = cycles(agreements(), codes)
    missing = codes - set(c.code)
    if missing:
        raise ValueError(f"mapped CNEL codes without a renewal cycle since {FIRST_YEAR}: {sorted(missing)}")
    c.to_csv(out, index=False, date_format="%Y-%m-%d")
    return c


def check(c: pd.DataFrame | None = None) -> pd.DataFrame:
    """Problems in the committed cycles: a code of the map without cycles, a period that ends before it starts, a signing more than seven years away from the start, two cycles of a code that overlap in the wrong order. Empty when clean."""
    c = load() if c is None else c
    bad = []
    for code in sorted(set(load_map().cnel_code) - {""} - set(c.code)):
        bad.append({"code": code, "problem": "no cycle"})
    dated = c.dropna(subset=["start", "expiry"])
    for code, signed, start, expiry in zip(dated.code, dated.signed, dated.start, dated.expiry):
        if expiry < start:
            bad.append({"code": code, "problem": f"expiry {expiry:%Y-%m-%d} before start {start:%Y-%m-%d}"})
        if abs((signed - start).days) > 7 * 366:  # the 2019-21 schools and research contract (S305) was signed in January 2024
            bad.append({"code": code, "problem": f"signed {signed:%Y-%m-%d} far from start {start:%Y-%m-%d}"})
    for code, g in c.groupby("code"):
        g = primary_cycles(g).dropna(subset=["expiry"]).sort_values("signed")
        if not g.expiry.is_monotonic_increasing:
            bad.append({"code": code, "problem": "expiry dates not increasing with the signing date"})
    return pd.DataFrame(bad, columns=["code", "problem"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "fetch":
        print(fetch("--force" in sys.argv))
    elif cmd == "build":
        out = build_csv()
        print(f"wrote {CSV}: {len(out)} cycles, {out.code.nunique()} codes, confederal share {out.confederal.mean():.2f}")
    else:
        c = load()
        bad = check(c)
        print(f"{len(c)} cycles, {c.code.nunique()} codes; problems: {len(bad)}")
        print(bad.to_string())
