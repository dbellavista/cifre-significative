"""Consolidated accounts of the Comune di Cesena group ("Bilancio consolidato"), from the PDFs on the old transparency site (`core.sources.comune_archive`).

The report prints, for the year of the document, the income statement and the balance sheet in three columns (the Comune alone, the effect of consolidation, the group) and, from 2021, the same aggregates against the previous year. Files with a text layer: 2018 and 2021 (the 2021 one also gives 2020 in its comparison table). The 2019 and 2020 files are scans of the statutory schemes (income statement and balance sheet in full, two years side by side). 2019 is read from the PaddleOCR cache (`core.sources.ocr`, `data/raw/ocr/`; never run by the loader): the group aggregates only (`source` 'ocr'), with every identity of the statements checked (see `parse_ocr_statements`). The 2020 scan closes every identity once one unreadable token is fixed by hand (`data/external/ocr_review/`), and its 2020 column equals the 2021 report's prior-year column; the loader keeps the 2021 report for 2020.
The perimeter table lists each body with the share held and the method (full or proportional); it changes between years (Valore Città in liquidation and Cesena Fiera are in the 2018 group, not in 2021). The group figures are the accounting aggregates of the old scheme (extraordinary items still exist): the Comune's own operating result is negative every year by construction of that scheme, read the result after extraordinary items and financial adjustments.
"""
import re
from pathlib import Path

import pandas as pd

from cifresignificative.core.geo import CESENA
from cifresignificative.core.sources import ocr
from cifresignificative.studies.cesena.analysis.indicatori import pdf_text
from cifresignificative.studies.cesena.analysis.rendiconto import ARCHIVE_DIR

AMOUNT = re.compile(r"(-)?€\s*(\d{1,3}(?:\.\d{3})*)")
DOC_YEAR = re.compile(r"BILANCIO CONSOLIDATO\s+(20\d\d)")
COMPARE = re.compile(r"Bilancio consolidato (20\d\d)\s+Bilancio consolidato (20\d\d)\s+Differenza")
PERIMETER = re.compile(r"^\s*([A-Z][A-Za-z0-9 .'\-]+?)\s{2,}(\d{1,3},\d{1,3})%\s+(Integrale|Proporzionale)\s*$")
ITEMS = {
    "valore della produzione": "output", "costi della produzione": "costs", "risultato della gestione operativa": "operating_result", "proventi e oneri finanziari": "financial_result",
    "rettifiche di valore di attività finanziarie": "financial_adjustments", "proventi e oneri straordinari": "extraordinary_result", "imposte sul reddito": "taxes",
    "risultato di esercizio": "net_result", "risultato di esercizio comprensivo della quota di pertinenza di terzi": "net_result",
    "immobilizzazioni immateriali": "intangible_assets", "immobilizzazioni materiali": "tangible_assets", "immobilizzazioni finanziarie": "financial_assets", "totale immobilizzazioni": "fixed_assets",
    "rimanenze": "inventories", "crediti": "receivables", "attività finanziarie che non cost, imm.ni": "current_financial_assets", "disponibilità liquide": "cash", "totale attivo circolante": "current_assets",
    "totale ratei e risconti attivi": "prepayments", "totale dell'attivo": "total_assets", "patrimonio netto": "equity", "fondi per rischi ed oneri": "provisions", "tfr": "severance", "debiti": "debts",
    "ratei e risconti passivi": "deferred_income", "totale del passivo": "total_liabilities",
}


def amount(sign: str, digits: str) -> float:
    return -float(digits.replace(".", "")) if sign else float(digits.replace(".", ""))


def parse_statements(text: str) -> pd.DataFrame:
    """Rows (year, item, capogruppo_eur, impact_eur, consolidated_eur, source) from the group table (Comune alone, consolidation effect, group; `source` 'group table') and, when the report prints it, the previous year of the comparison table (consolidated only, `source` 'prior-year column'); empty when the layout is not found. A table is read from the line that names it ("Impatto dei valori" or "Bilancio consolidato 2021 Bilancio consolidato 2020 Differenza") until the next one."""
    doc = DOC_YEAR.search(text[:800])
    if doc is None:
        return pd.DataFrame()
    lines = text.splitlines()
    kind: tuple[str, int, int] | None = None  # ("group", year, year) or ("compare", year, previous)
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    for i, ln in enumerate(lines):
        if "Impatto dei valori" in ln or any("Impatto dei valori" in x for x in lines[max(0, i - 1):i]) and ln.strip().startswith("consolidati"):
            kind = ("group", int(doc[1]), int(doc[1]))
        elif (m := COMPARE.search(ln)):
            kind = ("compare", int(m[1]), int(m[2]))
        found = list(AMOUNT.finditer(ln))
        label = " ".join(ln[:found[0].start()].split()).lower() if found else ""
        item = ITEMS.get(label)
        if kind is None or item is None or len(found) != 3 or (kind[0], item) in seen:
            continue
        vals = [amount(m[1], m[2]) for m in found]
        seen.add((kind[0], item))
        if kind[0] == "group":
            rows.append({"year": kind[1], "item": item, "capogruppo_eur": vals[0], "impact_eur": vals[1], "consolidated_eur": vals[2], "source": "group table"})
        else:
            rows.append({"year": kind[2], "item": item, "capogruppo_eur": float("nan"), "impact_eur": float("nan"), "consolidated_eur": vals[1], "source": "prior-year column"})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.insert(0, "comune_id", CESENA)
    return out


def parse_perimeter(text: str) -> pd.DataFrame:
    """Rows (year, body, share_pct, method) of the first perimeter table, the lines `NAME  72,16%  Integrale`; the body names are as printed."""
    doc = DOC_YEAR.search(text[:800])
    rows = []
    for ln in text.splitlines():
        if (m := PERIMETER.match(ln)) and doc:
            if m[1].strip() in {r["body"] for r in rows}:
                break  # the second table (with the equity of each body) starts: the first one is complete
            rows.append({"comune_id": CESENA, "year": int(doc[1]), "body": m[1].strip(), "share_pct": float(m[2].replace(",", ".")), "method": m[3]})
    return pd.DataFrame(rows)


OCR_LABELS = {  # label (lower case, as read) -> item; `startswith` on the longest text cell of a row
    "totale componenti positivi della gestione": "output", "totale componenti negativi della gestione": "costs", "differenza fra comp. positivi e negativi": "operating_result", "totale (c)": "financial_result",
    "totale ( d)": "financial_adjustments", "totale (e)": "extraordinary_result", "imposte": "taxes", "terzi)": "net_result", "risultato dell'esercizio (comprensivo": "net_result_label", "totale immobilizzazioni imm": "intangible_assets",
    "totale immobilizzazioni mat": "tangible_assets", "totale immobilizzazioni fin": "financial_assets", "totale immobilizzazioni (b)": "fixed_assets", "rimanenze": "inventories",
    "totale attività finanziarie": "current_financial_assets", "totale disponibilità liquide": "cash", "totale attivo circolante": "current_assets", "totale ratei e risconti d": "prepayments", "totale dell'attivo": "total_assets",
    "totale patrimonio netto": "equity", "totale fondi rischi": "provisions", "totale t.f.r.": "severance", "totale debiti": "debts", "totale ratei e risconti (e)": "deferred_income", "totale del passivo": "total_liabilities",
}
OCR_ITEMS = tuple(dict.fromkeys(i for i in OCR_LABELS.values() if i != "net_result_label")) + ("receivables",)
IDENTITIES = {  # name -> {item: sign}, each sum must be 0 (euros): the identities of the statutory schemes
    "assets = liabilities": {"total_assets": 1, "total_liabilities": -1}, "assets = fixed + current + prepayments": {"total_assets": 1, "fixed_assets": -1, "current_assets": -1, "prepayments": -1},
    "fixed assets": {"fixed_assets": 1, "intangible_assets": -1, "tangible_assets": -1, "financial_assets": -1}, "current assets": {"current_assets": 1, "inventories": -1, "receivables": -1, "current_financial_assets": -1, "cash": -1},
    "liabilities": {"total_liabilities": 1, "equity": -1, "provisions": -1, "severance": -1, "debts": -1, "deferred_income": -1}, "operating result": {"operating_result": 1, "output": -1, "costs": 1},
    "net result": {"net_result": 1, "operating_result": -1, "financial_result": -1, "financial_adjustments": -1, "extraordinary_result": -1, "taxes": 1},
}
AMOUNT_CELL = re.compile(r"-?[\dBoO][\dBoO.,]*")  # a cell that looks like an amount, even when misread (`B.319.520`, `1.077,413.797`)
REVIEW_DIR = Path("data/external/ocr_review")


def ocr_row(cells: list[str]) -> tuple[str, list[str]]:
    """(label, amount-like cells after it) of one OCR row: the label is the longest cell that is not amount-like; row numbers before it and codes after the amounts (`A5c`) are dropped."""
    label = max((c for c in cells if not AMOUNT_CELL.fullmatch(c.strip())), key=len, default="")
    after = cells[cells.index(label) + 1:] if label else cells
    return label.strip().lower(), [c.strip() for c in after if AMOUNT_CELL.fullmatch(c.strip())]


class OcrReading:
    """What a scan gives for one year pair: `raw[item]` the two printed texts (this year, previous year), `values[item]` the two numbers (None when the text is not a number), and the identities that fail."""

    def __init__(self, raw: dict[str, list[str]], year: int, review: dict[str, list[float | None]] | None = None):
        self.year, self.raw = year, raw
        self.values: dict[str, list[float | None]] = {it: [ocr.parse_amount(t) for t in raw.get(it, ["", ""])] for it in OCR_ITEMS}
        self.reviewed = set(review or {})
        for it, vals in (review or {}).items():
            self.values[it] = vals
        self.failures: dict[tuple[str, int], float] = {}
        for k in (0, 1):
            for name, terms in IDENTITIES.items():
                vs = [self.values[it][k] for it in terms]
                if all(v is not None for v in vs) and abs(gap := sum(sg * (v or 0.0) for sg, v in zip(terms.values(), vs))) > 1:
                    self.failures[(name, k)] = gap
        self.missing = [(it, k) for it in OCR_ITEMS for k in (0, 1) if self.values[it][k] is None]

    @property
    def ok(self) -> bool:
        return not self.failures and not self.missing

    def suspects(self) -> set[tuple[str, int]]:
        """(item, column) involved in a failed identity or unreadable."""
        out = set(self.missing)
        for name, k in self.failures:
            out |= {(it, k) for it in IDENTITIES[name]}
        return out


def read_ocr_statements(pages: list[list[dict]], year: int, review: dict[str, list[float | None]] | None = None) -> OcrReading:
    """Group aggregates of a scanned consolidated account (income statement and balance sheet, two years side by side), read by label prefix.
    Receivables are the second bare `Totale` of the assets page; prepayments are accrued plus deferred assets (the printed total is often misread); an unlabeled pair of rows after the treasury lines is cash then current assets when those labels are missing. `review` replaces values with the ones checked by hand (`load_review`)."""
    flat = [ocr_row(r) for p in pages for r in ocr.row_texts(p)]
    raw: dict[str, list[str]] = {}
    totale: list[list[str]] = []
    for label, am in flat:
        if len(am) < 2:
            continue
        if label == "totale":
            totale.append(am[:2])
        item = next((it for lab, it in OCR_LABELS.items() if label.startswith(lab)), None)
        if item and item != "net_result_label" and item not in raw:
            raw[item] = am[:2]
    if len(totale) >= 2:
        raw["receivables"] = totale[1]
    if "cash" not in raw:
        for i, (label, _) in enumerate(flat):
            if label.startswith("altri conti presso la tesoreria") and len(nxt := [a for lab, a in flat[i + 1:i + 3] if not lab and len(a) >= 2]) == 2:
                raw["cash"], raw["current_assets"] = nxt[0][:2], nxt[1][:2]
    cells_rows = [r for p in pages for r in ocr.row_texts(p)]
    accrued = {key: [ocr_row(r)[1] for r in cells_rows if any(c.strip().lower() == key for c in r) and len(ocr_row(r)[1]) >= 2][:1] for key in ("ratei attivi", "risconti attivi")}
    if all(accrued.values()):
        pair = [[ocr.parse_amount(t) for t in a[0][:2]] for a in accrued.values()]
        if all(v is not None for p in pair for v in p):
            raw["prepayments"] = [f"{int(a + b):,}".replace(",", ".") for a, b in zip(*pair)]
    return OcrReading(raw, year, review)


def review_path(pdf: Path, review_dir: Path = REVIEW_DIR) -> Path:
    return review_dir / f"{pdf.stem}.txt"


def load_review(path: Path) -> dict[str, list[float | None]]:
    """Values from a review file: lines `item | value this year | value previous year | ...` (`#` lines ignored); a blank or `?` value stays unknown."""
    out: dict[str, list[float | None]] = {}
    for ln in path.read_text().splitlines() if path.exists() else []:
        cells = [c.strip() for c in ln.split("|")]
        if ln.strip().startswith("#") or len(cells) < 3 or cells[0] not in OCR_ITEMS:
            continue
        out[cells[0]] = [float(c) if re.fullmatch(r"-?\d+", c) else ocr.parse_amount(c) for c in (x.replace("−", "-") for x in cells[1:3])]
    return out


def write_review(pdf: Path, reading: OcrReading, path: Path | None = None) -> Path:
    """The editable text file for a scan whose numbers do not add up: one line per aggregate with what the OCR read (`?` when unreadable), the lines involved in a failed identity marked `CHECK`, and the failed identities at the top."""
    path = path or review_path(pdf)
    path.parent.mkdir(parents=True, exist_ok=True)
    y0, y1 = reading.year, reading.year - 1
    bad = reading.suspects()
    head = [f"# Review of {pdf.name}", f"# Open the PDF: {pdf.resolve()}", f"# Columns: {y0} (first) | {y1} (second). Edit the numbers: euros, no thousands separator, minus sign for negatives, `?` if you cannot read it. Lines starting with # are ignored.",
            "# Lines marked CHECK are involved in an identity that does not close (or could not be read). The other lines are what the OCR read and are accepted as they stand once the identities close.", "#"]
    head += [f"# FAILED {name} for {(y0, y1)[k]}: off by {d:,.0f}" for (name, k), d in reading.failures.items()]
    head += [f"# UNREADABLE {it} for {(y0, y1)[k]}" for it, k in reading.missing]
    rows = []
    for it in OCR_ITEMS:
        vals = [None if v is None else f"{v:.0f}" for v in reading.values[it]]
        flag = "CHECK" if any((it, k) in bad for k in (0, 1)) else ""
        rows.append(f"{it:<26}| {vals[0] or '?':>14} | {vals[1] or '?':>14} | {(reading.raw.get(it) or ['?', '?'])!s:<36} {flag}")
    path.write_text("\n".join(head + [f"# item{'':<21}| {y0:>14} | {y1:>14} | OCR text"] + rows) + "\n")
    return path


def ocr_frame(reading: OcrReading) -> pd.DataFrame:
    out = pd.DataFrame([{"year": reading.year - k, "item": it, "capogruppo_eur": float("nan"), "impact_eur": float("nan"), "consolidated_eur": v[k], "source": "ocr"} for it, v in reading.values.items() for k in (0, 1)])
    out.insert(0, "comune_id", CESENA)
    return out


def parse_ocr_statements(pages: list[list[dict]], year: int, review: dict[str, list[float | None]] | None = None) -> pd.DataFrame:
    """Rows (year, item, consolidated_eur, source 'ocr') for `year` and `year - 1` of a scan; raises ValueError (naming the failed identities and unreadable values) unless every identity closes to the euro, so a misread figure never enters a series. Use `write_review` and `load_review` for the hand check."""
    reading = read_ocr_statements(pages, year, review)
    if not reading.ok:
        raise ValueError(f"OCR statements {year} do not close: failed {sorted({n for n, _ in reading.failures})}, unreadable {reading.missing}")
    return ocr_frame(reading)


def consolidated_files(archive_dir: Path = ARCHIVE_DIR) -> list[Path]:
    return sorted((archive_dir / "files").glob("trasparenza-bilancioconsolidato*/Bilancio_consolidato_*.pdf"))


def load_consolidated(archive_dir: Path = ARCHIVE_DIR, review_dir: Path = REVIEW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(statements, perimeter) over every consolidated-account PDF with a text layer; the group table of a year wins over the prior-year column of a later report, which wins over a scan. Empty frames when none."""
    st, pe = [], []
    for f in consolidated_files(archive_dir):
        t = pdf_text(f)
        if len(s := parse_statements(t)):
            st.append(s)
        elif (y := re.search(r"(20\d\d)", f.stem)) and (ocr.CACHE_DIR / f"{f.stem}.json").exists():
            try:
                scan = parse_ocr_statements(ocr.ocr_file(f, "it"), int(y[1]), load_review(review_path(f, review_dir)))
                st.append(scan[scan["year"] == int(y[1])])
            except (RuntimeError, ValueError):
                pass  # no cache matching the file, or the numbers do not add up (run `python -m ...consolidato` to write the review file): leave the year out
        if len(p := parse_perimeter(t)):
            pe.append(p)
    statements = pd.concat(st, ignore_index=True) if st else pd.DataFrame()
    if len(statements):
        statements = statements.sort_values("source", key=lambda s: s.map({"group table": 0, "prior-year column": 1}).fillna(2), kind="stable").drop_duplicates(["year", "item"]).sort_values(["year", "item"]).reset_index(drop=True)
    return statements, pd.concat(pe, ignore_index=True) if pe else pd.DataFrame()


def main() -> None:
    """Read every scanned consolidated account from the OCR cache (running the OCR service when a file is not cached) and write a review file for each one whose numbers do not add up."""
    for f in consolidated_files():
        if len(parse_statements(pdf_text(f))):
            continue
        y = int(re.findall(r"20\d\d", f.stem)[0])
        reading = read_ocr_statements(ocr.ocr_file(f, "it"), y, load_review(review_path(f)))
        if reading.ok:
            print(f"{f.name}: identities close")
        else:
            print(f"{f.name}: REVIEW NEEDED -> {write_review(f, reading)}")


if __name__ == "__main__":
    main()
