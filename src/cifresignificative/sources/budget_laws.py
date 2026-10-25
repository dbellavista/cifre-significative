"""Composition of the Italian budget laws (manovre): resources and uses, from the dossiers "Effetti sui saldi e conto risorse e impieghi" of the Servizi Bilancio of the Camera and the Senate.

Each dossier reproduces the government's technical report as a table: *impieghi* (uses: higher spending and lower revenue) against *risorse* (resources: higher revenue and lower spending), the difference being the manovra's effect on the
balance, for three aggregates (saldo netto da finanziare, fabbisogno, indebitamento netto) and three years, in EUR million. A dossier holds several such tables (initial text, a continuation page, the decree-law that goes with the bill,
effects of changes in the first reading), so `parse_pdf` returns every one of them and `data/external/budget_laws.csv` records which table stands for which budget year. These are ex-ante estimates, not realised effects.

The PDFs are downloaded to `data/raw/budget_laws/` (gitignored); parsing needs poppler's `pdftotext` and is a developer step: the committed CSV is the source of truth, so the build and CI never parse. `uv run python -m cifresignificative.sources.budget_laws
fetch|parse|check` downloads, parses and checks the accounting identities.
"""
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import requests
import truststore

from cifresignificative.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/budget_laws")
CSV = Path("data/external/budget_laws.csv")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
BASE = "https://documenti.camera.it"
# document id -> (budget year, what the dossier covers, URL). senato.it answers scripts with an empty HTTP 202 page, so only the Camera copies are fetched; the two URL schemes (`Pdf/DFP11.pdf`, `pdf/DFP006.pdf`) both exist.
DOSSIERS = {
    "L17_DFP11": (2016, "Legge di stabilita 2016 (bill)", f"{BASE}/leg17/dossier/Pdf/DFP11.pdf"),
    "L17_DFP16": (2017, "Legge di bilancio 2017 (bill) and DL 193/2016", f"{BASE}/leg17/dossier/Pdf/DFP16.pdf"),
    "L17_DFP20": (2018, "Legge di bilancio 2018 (bill, changes in the first reading) and DL 148/2017", f"{BASE}/leg17/dossier/Pdf/DFP20.pdf"),
    "S18_AS2448_nota274": (2022, "Legge di bilancio 2022 (bill, A.S. 2448), net effects and measure lines (gross split rebuilt): summary prospetto at the end of the Senate budget service's Nota di lettura 274 (manual download from senato.it)", ""),
    "A17_AS2960": (2018, "Legge di bilancio 2018 (bill, A.S. 2960) and DL 148/2017; copy hosted by Astrid", "https://www.astrid-online.it/static/upload/effe/effetti-sui-saldi.pdf"),
    "N18_DFP006": (2019, "Legge di bilancio 2019 (bill) and DL fiscale", f"{BASE}/leg18/dossier/pdf/DFP006.pdf"),
    "L18_DFP09": (2020, "Legge di bilancio 2020 (bill) and DL 124/2019", f"{BASE}/leg18/dossier/Pdf/DFP09.pdf"),
    "L18_DFP21": (2021, "Legge di bilancio 2021 (bill)", f"{BASE}/leg18/dossier/Pdf/DFP21.pdf"),
    "L18_DFP19": (2020, "Pandemic decree-laws 18, 23, 34 and 104 of 2020, cumulative", f"{BASE}/leg18/dossier/Pdf/DFP19.pdf"),
    "L18_DFP20": (2020, "Decree-laws 137 and 149 of 2020 (ristori), initial texts", f"{BASE}/leg18/dossier/Pdf/DFP20.pdf"),
    "L18_DFP26": (2021, "Decree-law 41/2021 (sostegni)", f"{BASE}/leg18/dossier/Pdf/DFP26.pdf"),
    "N19_DFP006": (2023, "Legge di bilancio 2023 (bill)", f"{BASE}/leg19/dossier/pdf/DFP006.pdf"),
    "L19_DFP11": (2024, "Legge di bilancio 2024 (bill) and DL 145/2023", f"{BASE}/leg19/dossier/Pdf/DFP11.pdf"),
    "L19_DFP14": (2025, "Legge di bilancio 2025 (bill) and DL 155/2024", f"{BASE}/leg19/dossier/Pdf/DFP14.pdf"),
    "L19_DFP17": (2026, "Legge di bilancio 2026 (bill)", f"{BASE}/leg19/dossier/Pdf/DFP17.pdf"),
}
AGGREGATES = ["saldo_netto_da_finanziare", "fabbisogno", "indebitamento"]
TOTALS = ["totale_maggiori_spese", "totale_minori_entrate", "totale_impieghi", "totale_maggiori_entrate", "totale_minori_spese", "totale_risorse", "manovra"]
NUMBER = re.compile(r"(?<![\w,.])[-−–]{0,2}\d{1,3}(?:\.\d{3})*(?:,\d+)?(?![\w%,])|(?<=\s)-(?=\s|$)")
YEAR = re.compile(r"(?<!\d)(20[12]\d)(?!\d)")


def path(doc: str) -> Path:
    return RAW_DIR / f"{doc}.pdf"


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for doc, (year, title, url) in DOSSIERS.items():
        if not url or (path(doc).exists() and not force):  # no URL: a manual download
            continue
        resp = requests.get(url, headers=UA, timeout=180)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            raise RuntimeError(f"{doc}: not a PDF: {resp.content[:60]!r}")
        path(doc).write_bytes(resp.content)
        record(f"budget_laws_{doc}", source="documenti.camera.it", dataset=title, url=url)
    return RAW_DIR


def to_number(token: str) -> float:
    """Italian number (`1.234,5`, a lone `-` is zero); the sources contain typos such as a doubled minus sign."""
    t = token.replace("−", "-").replace("–", "-")
    if t == "-":
        return 0.0
    neg = t.startswith("-")
    return (-1 if neg else 1) * float(t.lstrip("-").replace(".", "").replace(",", "."))


def slug(label: str) -> str:
    """Row label -> item name; totals are recognised by their start (`Totale impieghi (Manovra lorda)` is `totale_impieghi`), the balance row (`Manovra lorda` or `Manovra netta`, resources minus uses) is `manovra`."""
    s = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode().lower()
    if "al netto" in s:  # "Totale risorse (al netto rimodulazione PNRR)" is a subtotal, not the total
        return re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    s = re.sub(r"\([^)]*\)|[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"^[a-z] ", "", s)  # "A) Impieghi", "B) totale impieghi"
    s = {"totale mezzi di finanziamento": "totale risorse", "totale interventi": "totale impieghi", "saldo manovra": "manovra", "saldo": "manovra", "totale maggiore spesa primaria": "totale maggiori spese"}.get(s, s)
    for total in TOTALS[:-1]:
        if s.startswith(total.replace("_", " ")):
            return total
    return "manovra" if s in ("manovra lorda", "manovra netta") else s.replace(" ", "_")


def pages(pdf: Path) -> list[str]:
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    return out.split("\f")


def _columns(header: str) -> list[tuple[int, int]]:
    return [(m.end(), int(m.group(1))) for m in YEAR.finditer(header)]


def parse_pdf(pdf: Path, doc: str) -> pd.DataFrame:
    """Every composition table of a dossier (tables with a `Totale impieghi` and a `Totale risorse` row) as long data: doc, page, table (sequence number in the dossier), title, label, item, aggregate, year, value_meur.
    Numbers are assigned to columns by the position of their last digit under the year labels, because empty cells would shift them; a number that fits no column is reported with aggregate `unassigned`."""
    rows, seq = [], 0
    for page_no, text in enumerate(pages(pdf), 1):
        lines = text.split("\n")
        header, cols, title, table, pending, consumed, n_agg = None, [], "", None, "", set(), 3
        for i, line in enumerate(lines):
            if i in consumed:
                continue
            if re.match(r"\s*(Tabella|Tavola)\b", line):
                title = re.sub(r"\s+", " ", line.strip())[:200]
            ys = _columns(line)
            above = lines[max(0, i - 3):i + 1]
            only_year_labels = not NUMBER.search(YEAR.sub("", line).replace("/", ""))
            if len(ys) >= 6 and len(ys) % 3 == 0 and only_year_labels and any("Saldo netto" in x or "Fabbisogno" in x or "Indebitamento" in x for x in above):
                header, cols, table, pending, n_agg = i, ys, None, "", 3
                continue
            if len(ys) == 3 and only_year_labels and any("Indebitamento netto" in x for x in above) and not any("Fabbisogno" in x for x in above):  # decree dossiers that show the net borrowing only
                header, cols, table, pending, n_agg = i, ys, None, "", 1
                continue
            if header is None:
                continue
            if re.match(r"\s*(Fonte|Note?:|\*)", line):
                header = None
                continue
            line = re.sub(r"^\s*\d(?=\s{2,}[A-Za-z])", "", line)  # footnote or line number in the margin
            nums = [(m.end(), m.group()) for m in NUMBER.finditer(line)]
            label = re.split(NUMBER, line, maxsplit=1)[0].strip() if nums else line.strip()
            if not nums:
                pending = (pending + " " + label).strip() if label and (label.startswith("(") or label[:1].islower() or pending) else label
                continue
            first = NUMBER.search(line)
            label = (line[:first.start()].strip() if first else label)
            if pending and (not label or label.startswith("(") or label[:1].islower()):
                label = (pending + " " + label).strip()
            pending = ""
            nxt_i = next((k for k in range(i + 1, min(i + 4, len(lines))) if lines[k].strip()), None)  # next non-empty line
            nxt = lines[nxt_i] if nxt_i is not None else ""
            if label and re.fullmatch(r"\s*\((al netto|di cui)[A-Za-z ]{2,50}\)", nxt) or label and re.fullmatch(r"\s*[a-z][a-z ]{2,30}", nxt) and (len(label) < 14 or label.lower().endswith(" di")):  # qualifier on the line after the numbers
                label = f"{label} {nxt.strip()}"
                consumed.add(nxt_i)
            elif not label and re.fullmatch(r"\s*[a-z][a-z ]{2,30}", nxt):
                label = nxt.strip()
                consumed.add(nxt_i)
            if not label:
                continue
            if table is None:
                seq += 1
                table = seq
            n_years = len(cols) // n_agg
            slots, last = [], -1
            for k, (end, _) in enumerate(nums):  # in column order: all cells filled -> in order; otherwise the nearest later column (empty cells are skipped), never more tokens than columns left
                if len(nums) == len(cols):
                    j = k
                else:
                    room = range(last + 1, len(cols) - (len(nums) - k - 1))
                    j = min(room, key=lambda c: abs(cols[c][0] - end), default=-1)
                    j = j if j >= 0 and abs(cols[j][0] - end) <= 14 else -1
                slots.append(j)
                last = j if j >= 0 else last
            for (end, tok), j in zip(nums, slots, strict=True):
                rows.append({"doc": doc, "page": page_no, "table": table, "title": title, "label": label, "item": slug(label),
                             "aggregate": (AGGREGATES[j // n_years] if n_agg == 3 else "indebitamento") if j >= 0 else "unassigned", "year": cols[j][1] if j >= 0 else 0, "value_meur": to_number(tok), "end": end})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    has = d.groupby(["doc", "table"]).item.agg(lambda s: {"totale_impieghi", "totale_risorse"} <= set(s))
    d = d[d.set_index(["doc", "table"]).index.map(has)].drop(columns="end")
    d = d.assign(table=d.groupby("doc").table.transform(lambda t: t.rank(method="dense").astype(int)))  # n-th composition table of the dossier: stable when the parser starts seeing other tables
    return d.drop_duplicates(["doc", "table", "item", "aggregate", "year"], keep="first").reset_index(drop=True)


def identities(d: pd.DataFrame, tol: float = 0.15) -> pd.DataFrame:
    """Accounting identities per (doc, table, aggregate, year): total impieghi = total higher spending + total lower revenue, total risorse = total higher revenue + total lower spending, balance = risorse - impieghi
    (missing totals are skipped, missing cells count as zero). Returns the three differences and `ok` (all within `tol` EUR million, the figures are rounded to 0.1)."""
    keep = [*TOTALS, "spesa_per_interessi", "rimodulazione_pnrr"]  # the 2026 bill adds interest spending to the uses and the PNRR reshuffle to the resources
    w = d[d.item.isin(keep)].pivot_table(index=["doc", "table", "aggregate", "year"], columns="item", values="value_meur", aggfunc="first")
    for c in keep:
        if c not in w:
            w[c] = float("nan")
    out = pd.DataFrame(index=w.index)
    out["uses_diff"] = w.totale_impieghi - w.totale_maggiori_spese.fillna(0) - w.totale_minori_entrate.fillna(0) - w.spesa_per_interessi.fillna(0)
    out["resources_diff"] = w.totale_risorse - w.totale_maggiori_entrate.fillna(0) - w.totale_minori_spese.fillna(0) - w.rimodulazione_pnrr.fillna(0)
    out["balance_diff"] = w.manovra - (w.totale_risorse - w.totale_impieghi)
    out.loc[w.totale_maggiori_spese.isna() & w.totale_minori_entrate.isna(), "uses_diff"] = float("nan")
    out.loc[w.totale_maggiori_entrate.isna() & w.totale_minori_spese.isna(), "resources_diff"] = float("nan")
    out["ok"] = out.abs().max(axis=1, skipna=True).fillna(0) <= tol
    return out.reset_index()


NOTA_ROWS = {"TOTALE ENTRATE": "entrate_nette_sezione_i", "TOTALE SPESE": "spese_nette_sezione_i", "SALDO": "saldo_sezione_i", "TOTALE RIFINAZIAMENTI": "rifinanziamenti", "TOTALE RIPROGRAMMAZIONI": "riprogrammazioni",
             "TOTALE DEFINAZIAMENTI": "definanziamenti", "DL FISCALE": "saldo_dl_fiscale", "ENTRATE": "dl_fiscale_entrate", "SPESE": "dl_fiscale_spese", "TOTALE MANOVRA": "manovra"}  # (sic) the source spells "RIFINAZIAMENTI" and "DEFINAZIAMENTI"


def parse_nota_summary(pdf: Path, doc: str, budget_year: int) -> pd.DataFrame:
    """Net effects of the 2022 budget law from the summary prospetto in the last pages of the Servizio del Bilancio's Nota di lettura 274: sezione I entrate, spese and saldo, the sezione II lines (spending sign: positive = more spending),
    the DL fiscale saldo and the total manovra, for the three aggregates and 2022-24. The identity manovra = saldo - rifinanziamenti - riprogrammazioni - definanziamenti + saldo DL fiscale must hold. There is no gross split in this source."""
    rows = []
    for page_no, text in enumerate(pages(pdf), 1):
        if "TOTALE MANOVRA" not in text:
            continue
        for line in text.split("\n"):
            label = line.strip()
            for key, item in NOTA_ROWS.items():
                if re.match(rf"{re.escape(key)}\b", label):
                    nums = [m.group() for m in NUMBER.finditer(label[len(key):])]
                    if len(nums) == 9:  # all three aggregates and years filled
                        for j, tok in enumerate(nums):
                            rows.append({"budget_year": budget_year, "doc": doc, "table": 0, "page": page_no, "item": item, "label": key.title(), "aggregate": AGGREGATES[j // 3], "year": 2022 + j % 3, "value_meur": to_number(tok)})
    d = pd.DataFrame(rows).drop_duplicates(["item", "aggregate", "year"], keep="first")
    w = d.pivot_table(index=["aggregate", "year"], columns="item", values="value_meur")
    gap = w.manovra - (w.saldo_sezione_i - w.rifinanziamenti - w.riprogrammazioni - w.definanziamenti + w.saldo_dl_fiscale)
    if gap.abs().max() > 0.15 or len(w) != 9:
        raise ValueError(f"{doc}: the summary does not add up:\n{gap}")
    return d


ARTICLE_NATURE = {"C", "K", "T", "CO", "EXT", "T/C"}  # spesa corrente, spesa in conto capitale, entrata tributaria, contributiva, extra-tributaria, tributaria/contributiva
NUMBER_FULL = re.compile(NUMBER.pattern.split("|")[0])


def parse_nota_articles(pdf: Path) -> pd.DataFrame:
    """The article-by-article lines of the summary prospetto of the Nota di lettura 274 (Sezione I): one row per measure with its type (`S` spending, `E` revenue), nature and, for each aggregate and year 2022-24, the effect with the sign convention of the
    government's table (spending: positive = more spending; revenue: positive = more revenue). The nine numeric columns are positional and blank cells are not written, so each number is assigned to the column whose header year (`pdftotext -bbox`
    word positions) ends nearest to its own right edge; the text layout of `pdftotext -layout` is not reliable for this (the header and the data drift apart on some pages). Lines from `TOTALE ENTRATE` on are the net summary (`parse_nota_summary`)."""
    pg = pages(pdf)
    first = next(i for i, p in enumerate(pg) if "Prospetto riepilogativo degli effetti finanziari" in p and "Art." in p) + 1
    last = next(i for i, p in enumerate(pg) if "TOTALE MANOVRA" in p) + 1
    xml = subprocess.run(["pdftotext", "-bbox", "-f", str(first), "-l", str(last), str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    word = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="[\d.]+">([^<]*)</word>')
    rows = []
    for k, page in enumerate(xml.split("<page ")[1:]):
        lines: dict[int, list] = {}
        for x0, y0, x1, t in word.findall(page):
            lines.setdefault(round(float(y0)), []).append((float(x0), float(x1), t))
        hdr = [ln for ln in lines.values() if sum(bool(re.fullmatch(r"202[234]", t)) for _, _, t in ln) == 9]
        if not hdr:
            continue
        cols = sorted(x1 for x0, x1, t in hdr[0] if re.fullmatch(r"202[234]", t))
        for _, ln in sorted(lines.items()):
            ln = sorted(ln)
            if any(t.startswith("TOTALE") or t == "SALDO" for _, _, t in ln):
                break
            for idx in range(len(ln) - 1):
                if ln[idx][2] in ("S", "E") and ln[idx + 1][2] in ARTICLE_NATURE:
                    for _, x1, tok in ln[idx + 2:]:
                        if NUMBER_FULL.fullmatch(tok) or tok == "-":
                            j = min(range(9), key=lambda c: abs(cols[c] - x1))
                            rows.append({"page": first + k, "kind": ln[idx][2], "nature": ln[idx + 1][2], "aggregate": AGGREGATES[j // 3], "year": 2022 + j % 3, "value_meur": to_number(tok)})
                    break
    return pd.DataFrame(rows)


def reconstruct_gross(articles: pd.DataFrame, summary: pd.DataFrame, budget_year: int, doc: str, tol: float = 0.4) -> pd.DataFrame:
    """Gross uses and resources of a law with a net summary only, from its measure lines: in Sezione I each line is a use if it raises spending or lowers revenue and a resource otherwise; the Sezione II totals (refinancing, reprogramming,
    definancing, with the spending sign) and the DL fiscale entrate and spese lines are split by sign in the same way (the sub-lines of the Sezione II totals are not read: a small revenue line inside the refinancing total of 2023-24 is netted).
    Output rows use the item names of the dossier tables (`totale_maggiori_spese`, ..., `totale_impieghi`, `totale_risorse`). Checks: the Sezione I lines add up to the summary's net entrate and spese (within `tol`, EUR million, the lines are rounded)
    and resources minus uses equal the summary's total manovra."""
    a = articles.groupby(["aggregate", "year", "kind"]).value_meur.sum().unstack()
    s = summary.pivot_table(index=["aggregate", "year"], columns="item", values="value_meur", aggfunc="first")
    gap = pd.concat([a["E"] - s.entrate_nette_sezione_i, a["S"] - s.spese_nette_sezione_i], axis=1).abs().max().max()
    if gap > tol:
        raise ValueError(f"{doc}: the article lines do not add up to the Sezione I net totals (max gap {gap:.2f}):\n{a.join(s[['entrate_nette_sezione_i', 'spese_nette_sezione_i']])}")
    more_sp = articles[(articles.kind == "S") & (articles.value_meur > 0)].groupby(["aggregate", "year"]).value_meur.sum()
    less_sp = -articles[(articles.kind == "S") & (articles.value_meur < 0)].groupby(["aggregate", "year"]).value_meur.sum()
    more_rev = articles[(articles.kind == "E") & (articles.value_meur > 0)].groupby(["aggregate", "year"]).value_meur.sum()
    less_rev = -articles[(articles.kind == "E") & (articles.value_meur < 0)].groupby(["aggregate", "year"]).value_meur.sum()
    sec2 = s[["rifinanziamenti", "riprogrammazioni", "definanziamenti"]]  # spending sign, each section split by its own sign (refinancing is a use, definancing a resource, reprogramming either)
    g = pd.DataFrame({"totale_maggiori_spese": more_sp + sec2.clip(lower=0).sum(axis=1) + s.dl_fiscale_spese.clip(lower=0), "totale_minori_spese": less_sp + (-sec2).clip(lower=0).sum(axis=1) + (-s.dl_fiscale_spese).clip(lower=0),
                      "totale_maggiori_entrate": more_rev + s.dl_fiscale_entrate.clip(lower=0), "totale_minori_entrate": less_rev + (-s.dl_fiscale_entrate).clip(lower=0)}).fillna(0.0)
    g["totale_impieghi"] = g.totale_maggiori_spese + g.totale_minori_entrate
    g["totale_risorse"] = g.totale_maggiori_entrate + g.totale_minori_spese
    bal = (g.totale_risorse - g.totale_impieghi - s.manovra).abs().max()
    if bal > 0.5:
        raise ValueError(f"{doc}: reconstructed resources minus uses differ from the net manovra by {bal:.2f}")
    out = g.round(1).stack().rename("value_meur").reset_index().rename(columns={"level_2": "item", "year": "horizon_year"})
    return out.assign(budget_year=budget_year, doc=doc, table=0, page=0, label=out.item.str.replace("_", " ").str.capitalize())


ITEM = re.compile(r"^(maggiori_|minori_|totale_|manovra$|spesa_per_interessi$|rimodulazione_pnrr$)")
TABLES_CSV = Path("data/external/budget_laws_tables.csv")


def build_csv(out: Path = CSV) -> pd.DataFrame:
    """The tables chosen in `budget_laws_tables.csv` (one row per budget year and part: bill, decree or total) as `budget_laws.csv`: budget_year, role, doc, table, page, stage, item, label, aggregate, horizon_year, value_meur, evidence.
    Narrative lines that look like rows are dropped; the accounting identities must hold for every chosen table before anything is written."""
    reg = pd.read_csv(TABLES_CSV)
    nota = reg[reg.table == 0]  # net-only summaries (no gross split)
    reg = reg[reg.table != 0]
    parsed = parse_all(sorted(set(reg.doc)))
    idn = identities(parsed)
    chosen = parsed.merge(reg, on=["doc", "table"], how="inner").rename(columns={"year": "horizon_year", "budget_year_y": "budget_year"})
    titles = parsed.drop_duplicates(["doc", "table"]).set_index(["doc", "table"]).title
    for r in reg.itertuples():  # the chosen table must still be the one meant: the title is a guard against renumbering
        title = titles.get((r.doc, r.table), None)
        if title is None or str(r.title_contains) not in title:
            raise ValueError(f"{r.doc} table {r.table}: expected a title containing {r.title_contains!r}, found {title!r}")
    bad = idn.merge(reg, on=["doc", "table"])
    if not bad.ok.all():
        raise ValueError(f"accounting identities fail:\n{bad[~bad.ok]}")
    chosen = chosen[chosen.item.str.match(ITEM) & (chosen.aggregate != "unassigned")].copy()
    chosen["evidence"] = "identities_ok"
    extra = []
    for r in nota.itertuples():
        summ = parse_nota_summary(path(r.doc), r.doc, r.budget_year)
        extra.append(summ.rename(columns={"year": "horizon_year"}).assign(role=r.role, group=r.group, ref_date=r.ref_date, stage=r.stage, evidence="net_only_identity_ok"))
        gross = reconstruct_gross(parse_nota_articles(path(r.doc)), summ, r.budget_year, r.doc)  # a net-only law gets its gross split from its measure lines (identities checked inside)
        extra.append(gross.assign(role=r.role, group=r.group, ref_date=r.ref_date, stage=r.stage, evidence="reconstructed_from_measure_lines"))
    chosen = pd.concat([chosen, *extra], ignore_index=True)
    cols = ["budget_year", "role", "group", "ref_date", "doc", "table", "page", "stage", "item", "label", "aggregate", "horizon_year", "value_meur", "evidence"]
    chosen = chosen[cols].sort_values(["budget_year", "role", "table", "aggregate", "horizon_year"], ignore_index=True)
    chosen.to_csv(out, index=False)
    return chosen


def parse_all(docs: list[str] | None = None) -> pd.DataFrame:
    frames = [parse_pdf(path(doc), doc).assign(budget_year=DOSSIERS[doc][0]) for doc in (docs or DOSSIERS) if path(doc).exists()]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "fetch":
        print(fetch("--force" in sys.argv))
    elif cmd == "build":
        out = build_csv()
        print(f"wrote {CSV}: {len(out)} rows, budget years {sorted(out.budget_year.unique())}")
    else:
        parsed = parse_all()
        idn = identities(parsed)
        print(parsed.groupby(["doc", "table", "title"]).size().to_string())
        print("identity failures:", int((~idn.ok).sum()), "of", len(idn))
        print(idn[~idn.ok].round(2).to_string())
