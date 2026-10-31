"""Eurosystem public-sector purchases by country: PSPP (2015-03 on) and PEPP (2020-03 on), monthly, from the two ECB CSV exports saved by hand in `data/external/source/eurosystem_holdings/` (README there).
Each file has three blocks: monthly net purchases (EUR million, net of redemptions, amortised cost; the last column is the ECB's cumulative figure), the WAM of the portfolio and the WAM of the eligible universe (years).
`stock` = cumulative sum of net purchases, an approximation of the holdings at amortised cost that is checked against the ECB's own cumulative column (`cumulative_check`); it is not the market value."""
import csv
from pathlib import Path

import pandas as pd

DIR = Path("data/external/source/eurosystem_holdings")
FILES = {"PSPP": "PSPP_breakdown_history.csv", "PEPP": "PEPP_public_sector_securities_breakdown_history.csv"}


def _num(x: str) -> float:
    x = x.strip().replace(",", "")
    return float(x) if x not in ("", "-") else float("nan")


def _measure(title: str) -> str | None:
    t = title.lower()
    if t.startswith(("monthly net purchases", "net purchases")):
        return "net_purchases"
    if t.startswith("wam of eligible universe"):
        return "wam_universe"
    if t.startswith("wam of"):
        return "wam_portfolio"
    return None


def _dates(cells: list[str]) -> pd.Series:
    """Month ends from `dd/mm/yyyy` (PSPP) or `Mon-yy` (PEPP) headers."""
    ts = pd.Series([pd.to_datetime(c, format="%d/%m/%Y" if "/" in c else "%b-%y") for c in cells])
    return (ts + pd.offsets.MonthEnd(0)).dt.normalize()


def parse(path: Path, programme: str) -> tuple[pd.DataFrame, pd.Series]:
    """Long `programme, country, measure, period, value` and the ECB's cumulative net purchases per country (last column of the first block)."""
    rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
    out, cum, measure, dates, ncol = [], {}, None, None, 0
    for r in rows:
        head = r[0].strip()
        if head and (m := _measure(head)):
            measure, dates = m, None
        elif measure and not head and dates is None and any(c.strip() for c in r[1:]):
            cells = [c.strip() for c in r[1:]]
            ncol = max(i for i, c in enumerate(cells) if c and c.lower() not in ("cumulative", "cumulative net purchases")) + 1
            dates = _dates(cells[:ncol])
        elif measure and dates is not None and head and len(r) > ncol:
            if head.lower().startswith(("in eur", "figures", "remaining", "end-of", "disclaimer")):  # footnotes
                continue
            vals = [_num(c) for c in r[1:ncol + 1]]
            out += [(programme, head, measure, d, v) for d, v in zip(dates, vals) if pd.notna(v)]
            if measure == "net_purchases" and len(r) > ncol + 1:
                cum[head] = _num(r[ncol + 1])
    return pd.DataFrame(out, columns=["programme", "country", "measure", "period", "value"]), pd.Series(cum, name="cumulative")


def load(directory: Path = DIR) -> pd.DataFrame:
    """Long table for both programmes, plus the `stock` measure (cumulative net purchases, EUR million)."""
    d = pd.concat([parse(directory / f, p)[0] for p, f in FILES.items()], ignore_index=True)
    npur = d[d.measure == "net_purchases"].sort_values("period")
    stock = npur.assign(measure="stock", value=npur.groupby(["programme", "country"]).value.cumsum())
    return pd.concat([d, stock], ignore_index=True).assign(source="ecb:asset-purchase-programmes").sort_values(["programme", "country", "measure", "period"], ignore_index=True)


def cumulative_check(directory: Path = DIR) -> pd.DataFrame:
    """The sum of monthly net purchases against the ECB's cumulative column, per programme and country (should agree up to rounding)."""
    rows = []
    for p, f in FILES.items():
        d, cum = parse(directory / f, p)
        s = d[d.measure == "net_purchases"].groupby("country").value.sum()
        rows.append(pd.DataFrame({"programme": p, "sum_monthly": s, "ecb_cumulative": cum}).dropna().rename_axis("country").reset_index())
    return pd.concat(rows, ignore_index=True).assign(diff=lambda x: x.sum_monthly - x.ecb_cumulative)
