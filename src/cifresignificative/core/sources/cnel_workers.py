"""Employees and employers per national collective agreement (CNEL-INPS, 2025), to weight the Istat agreement groups of notebook 39.

CNEL publishes, for the private sector, the number of employers and employees that declare each agreement code (INPS UNIEMENS flow, December 2025 snapshot) as Excel files behind the "CCNL settore privato" link of the archive. Three downloaded files are used, in this order of priority when a code appears in more than one:
`ccnl-di-settore-vigenti-e-ultrattivi` (sheet `1 - CCNL di settore, elenco`, 159 sector agreements, 14.9 million employees), `ccnl-altri` (sheet `CCNL elenco`) and `contratti-aziendali-di-livello-nazionale` (sheet `1 - CCNL elenco`; company-level
national agreements such as Poste, K700). The other files of the download (managers, collaborators, self-employed AECs, new deposits) have no counts. Public-sector (ARAN, `S...`) agreements are not in the private-sector files: they have no count here.
The counts are a 2025 snapshot, used as relative weights for a 2021-24 outcome; a count of 0 or a tiny one (Stellantis C010, 37) is kept as published and weighs next to nothing.
The Excel files are committed under `data/external/source/cnel/workers/`; the build writes `data/external/cnel_workers.csv` (`code, title, employers, employees, file`).
`uv run python -m cifresignificative.core.sources.cnel_workers build`.
"""
from pathlib import Path

import pandas as pd

SRC = Path("data/external/source/cnel/workers")
CSV = Path("data/external/cnel_workers.csv")
FILES = [("ccnl-di-settore-vigenti-e-ultrattivi.xlsx", "1 - CCNL di settore, elenco"), ("ccnl-altri.xlsx", "CCNL elenco"), ("contratti-aziendali-di-livello-nazionale.xlsx", "1 - CCNL elenco")]


def build_csv(out: Path = CSV, src: Path = SRC) -> pd.DataFrame:
    parts = []
    for name, sheet in FILES:
        d = pd.read_excel(src / name, sheet_name=sheet)
        parts.append(pd.DataFrame({"code": d["CCNL id"].astype(str).str.strip(), "title": d["CCNL titolo"], "employers": d["n° datori di lavoro 2025"], "employees": d["n° dipendenti 2025"], "file": name}))
    d = pd.concat(parts, ignore_index=True).dropna(subset=["employees"])
    d = d.drop_duplicates("code", keep="first").sort_values("code", ignore_index=True)  # files are concatenated in priority order
    d[["employers", "employees"]] = d[["employers", "employees"]].astype(int)
    d.to_csv(out, index=False)
    return d


def load(csv: Path = CSV) -> pd.DataFrame:
    return pd.read_csv(csv)


if __name__ == "__main__":
    d = build_csv()
    print(len(d), "codes;", int(d.employees.sum()), "employees;", d.groupby("file").size().to_dict())
