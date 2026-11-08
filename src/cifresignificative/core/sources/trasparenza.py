"""Public documents of a municipality's "Amministrazione trasparente" portal built on the Liferay `jcitygov-albi` platform (`<comune>.trasparenza-valutazione-merito.it`).

The site is slow (15-25 s a page) and drops connections, so every request is throttled, retried with backoff and cached on disk; a run can be stopped and restarted.
Layouts handled:
- `papca` leaves (acts and publications, e.g. Bilanci): a list table (paged by a POST form, 20/50 rows) whose rows link to a detail page; the detail page lists the attachments, each with
  a download URL base64-encoded in an `onclick="window.open(atob('...'))"`.
- `pas` leaves (Albo Soggetti): blocks with a title and plain download links.
Raw files go to `data/raw/trasparenza/<site>/`: `catalog.json` (section -> leaf url), `lists/<leaf>.json`, `details/<doc>.json`, `files/<doc>/<attachment>_<name>` and `files.csv`.
Public data, one request at a time, `robots.txt` allows crawling. Usage (the base URL is a parameter, nothing is site-specific here):
`uv run python -m cifresignificative.core.sources.trasparenza --base-url https://<comune>.trasparenza-valutazione-merito.it catalog|list|details|files|run [options]`
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
import truststore

from cifresignificative.core.sources._common import record

truststore.inject_into_ssl()

RAW_DIR = Path("data/raw/trasparenza")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# Section presets (prefixes of the menu's `data-resource` path): what the budget and personnel questions need, and what may be of interest besides.
PRESETS: dict[str, tuple[str, ...]] = {
    "fiscal": ("Bilanci", "Controlli e rilievi sull'amministrazione - Organi di revisione", "Controlli e rilievi sull'amministrazione - Corte dei conti", "Enti controllati"),
    "extended": ("Bilanci", "Controlli e rilievi sull'amministrazione", "Enti controllati", "Personale - Dotazione organica", "Personale - Personale non a tempo indeterminato",
                 "Personale - Contrattazione integrativa", "Personale - Tassi di assenza", "Opere pubbliche", "Beni immobili e gestione patrimonio", "Pianificazione e governo del territorio",
                 "Sovvenzioni, contributi, sussidi, vantaggi economici", "Servizi erogati"),
}
# Subjects worth downloading from the acts lists (the Bilanci list also holds budget variations and unrelated resolutions).
DEFAULT_MATCH = r"rendiconto|conto consuntivo|consuntiv|indicatori|bilancio consolidato|bilancio di previsione|nota integrativa|documento unico|\bdup\b|equilibri|salvaguardia|revisor|relazione|partecipate|personale|fabbisogni"


@dataclass
class Row:
    doc_id: str
    leaf: str
    category: str
    number: str
    subject: str
    published: str
    n_attachments: int
    url: str
    year: int | None = None


@dataclass
class Attachment:
    att_id: str
    title: str
    description: str
    mimetype: str
    url: str


@dataclass
class Detail:
    doc_id: str
    leaf: str
    meta: dict[str, str] = field(default_factory=dict)
    attachments: list[Attachment] = field(default_factory=list)


def slug(text: str, maxlen: int = 80) -> str:
    """File-name safe ASCII slug."""
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-")
    return s[:maxlen] or "file"


def clean(text: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text)).split())


def decode_download_url(b64: str) -> str:
    """The detail page hides each download URL in `atob('...')`."""
    return base64.b64decode(b64).decode("utf-8")


def parse_menu(page: str) -> dict[str, str]:
    """Menu leaves: `data-resource` path (`Bilanci - Bilancio preventivo e consuntivo`) -> `data-mainurl` (`/web/trasparenza/papca-g/-/papca/igrid/41683`)."""
    u = html.unescape(page)
    out: dict[str, str] = {}
    for m in re.finditer(r'data-resource="([^"]+)"[^>]*?data-parentid="\d+"[^>]*?data-mainurl="([^"]*)"', u, re.DOTALL):
        if m.group(2):
            out[m.group(1)] = m.group(2)
    return out


def year_of(number: str, published: str) -> int | None:
    """Year of an act: from `2026/71` when it has one, else from the publication date `dd/mm/yyyy`."""
    m = re.match(r"(\d{4})/", number.strip())
    if m and m.group(1) != "0000":
        return int(m.group(1))
    m = re.search(r"\d{2}/\d{2}/(\d{4})", published)
    return int(m.group(1)) if m else None


def parse_list(page: str, leaf: str) -> tuple[list[Row], tuple[int, int, int] | None, str | None]:
    """Rows of a `papca` list page, its (page, pages, total), and the action URL of its paging form."""
    u = html.unescape(page)
    pos = re.search(r"Pagina (\d+) di (\d+) \((\d+) risultati\)", u)
    form = re.search(r'<form id="form_page_size" action="([^"]+)"', u)
    tbl = re.search(r'<table class="master-detail-list-table.*?</table>', u, re.DOTALL)
    rows: list[Row] = []
    for tr in re.findall(r"<tr.*?</tr>", tbl.group(0), re.DOTALL) if tbl else []:
        href = re.search(r'href="([^"]*/display/(\d+)[^"]*)"', tr)
        cells = [clean(c) for c in re.findall(r"<td.*?</td>", tr, re.DOTALL)]
        if not href or len(cells) < 4:
            continue
        n_att = int(cells[4]) if len(cells) > 4 and cells[4].isdigit() else 0
        rows.append(Row(href.group(2), leaf, cells[0], cells[1], cells[2], cells[3], n_att, href.group(1), year_of(cells[1], cells[3])))
    return rows, ((int(pos.group(1)), int(pos.group(2)), int(pos.group(3))) if pos else None), (form.group(1) if form else None)


def parse_detail(page: str, doc_id: str, leaf: str) -> Detail:
    """Metadata (`Oggetto`, `Anno atto`...) and attachments of a detail page."""
    u = html.unescape(page)
    det = Detail(doc_id, leaf)
    text = clean(re.sub(r"<script.*?</script>|<style.*?</style>", " ", u, flags=re.DOTALL))
    for key in ("Categoria", "Sottocategoria", "Anno atto", "Numero atto", "Data documento", "Proponente", "Oggetto", "Data atto", "Classifica", "Periodo Pubblicazione"):
        m = re.search(rf"{key} (.+?)(?= (?:Categoria|Sottocategoria|Anno atto|Numero atto|Data documento|Proponente|Oggetto|Data atto|Data esecutività|Classifica|Periodo Pubblicazione|Provenienza|Stato|Numero allegati|Titolo) |$)", text)
        if m:
            det.meta[key] = m.group(1).strip()
    m = re.search(r"Numero allegati (\d+)", text)
    if m:
        det.meta["Numero allegati"] = m.group(1)
    tbl = re.search(r'<table class="table table-striped table-hover allegati-table">.*?</table>', u, re.DOTALL)
    for tr in re.findall(r'<tr data-chiave-allegato="(\d+)" data-mimetype="([^"]*)">(.*?)</tr>', tbl.group(0), re.DOTALL) if tbl else []:
        cells = re.findall(r"<td.*?</td>", tr[2], re.DOTALL)
        enc = re.search(r"atob\('([^']+)'\)", tr[2])
        plain = re.search(r'href="(https?://[^"]*downloadAllegato[^"]*)"', tr[2])
        url = decode_download_url(enc.group(1)) if enc else (plain.group(1) if plain else "")
        if url:
            det.attachments.append(Attachment(tr[0], clean(cells[0]) if cells else "", clean(cells[1]) if len(cells) > 1 else "", tr[1], url))
    return det


def parse_soggetti(page: str, leaf: str) -> list[Detail]:
    """`pas` leaves: one block per item (title in `table-albi-soggetto-header`), with plain `downloadAllegato` links."""
    u = html.unescape(page)
    parts = re.split(r'<th[^>]*class="table-albi-soggetto-header"[^>]*>', u)
    out: list[Detail] = []
    for k, part in enumerate(parts[1:]):
        title = clean(part.split("</th>")[0])
        det = Detail(f"{slug(leaf, 30)}_{k:03d}", leaf, {"Oggetto": title})
        block = part.split("table-albi-soggetto-header")[0]
        m = re.search(r"Data di pubblicazione (\d{2}/\d{2}/\d{4})", clean(block))
        if m:
            det.meta["Data documento"] = m.group(1)
        for h in re.findall(r'href="(https?://[^"]*downloadAllegato[^"]*)"', block):
            aid = re.search(r"_id=(\d+)", h)
            det.attachments.append(Attachment(aid.group(1) if aid else slug(h, 20), title, "", "", h))
        out.append(det)
    return out


class Site:
    """HTTP session with throttling, retries and exponential backoff."""

    def __init__(self, base_url: str, raw_dir: Path = RAW_DIR, delay: float = 2.0, timeout: int = 240, retries: int = 5):
        self.base = base_url.rstrip("/")
        self.name = urlparse(self.base).netloc.split(".")[0]
        self.dir = raw_dir / self.name
        self.delay, self.timeout, self.retries = delay, timeout, retries
        self.session = requests.Session()
        self.session.headers.update(UA)
        for sub in ("lists", "details", "files"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)

    def request(self, method: str, url: str, **kw) -> requests.Response:
        url = url if url.startswith("http") else self.base + url
        err: Exception | None = None
        for k in range(self.retries):
            try:
                r = self.session.request(method, url, timeout=self.timeout, **kw)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                time.sleep(self.delay)
                return r
            except (requests.RequestException, OSError) as e:
                err = e
                wait = min(120, 10 * 2**k)
                print(f"  retry {k + 1}/{self.retries} after {type(e).__name__} ({wait}s): {url[:100]}", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"giving up on {url}: {err}")

    def get(self, url: str, **kw) -> requests.Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, **kw) -> requests.Response:
        return self.request("POST", url, **kw)


def leaf_id(url: str) -> str:
    m = re.search(r"/(?:i?grid)/(\d+)", url)
    return m.group(1) if m else slug(url, 30)


def fetch_catalog(site: Site, force: bool = False) -> dict[str, str]:
    p = site.dir / "catalog.json"
    if p.exists() and not force:
        return json.loads(p.read_text(encoding="utf-8"))
    cat = parse_menu(site.get("/").text)
    if not cat:
        raise RuntimeError("no menu leaves found: the portal layout changed")
    p.write_text(json.dumps(cat, ensure_ascii=False, indent=1), encoding="utf-8")
    record(f"trasparenza_{site.name}_catalog", source="municipal transparency portal", dataset="catalog", url=site.base)
    return cat


def select_leaves(cat: dict[str, str], prefixes: tuple[str, ...]) -> dict[str, str]:
    return {k: v for k, v in cat.items() if any(k.startswith(p) for p in prefixes)}


def list_leaf(site: Site, resource: str, url: str, page_size: int = 50, force: bool = False) -> list[Row]:
    """All rows of a `papca` leaf (cached in `lists/<leaf>.json`). `pas` leaves are returned as pseudo-rows by `list_soggetti`."""
    p = site.dir / "lists" / f"{leaf_id(url)}.json"
    if p.exists() and not force:
        return [Row(**r) for r in json.loads(p.read_text(encoding="utf-8"))["rows"]]
    page = site.get(url).text
    rows, pos, form = parse_list(page, resource)
    if pos and pos[1] > 1 and form:
        page = site.post(form, data={"hidden_page_size": str(page_size), "hidden_page_to": ""}).text
        rows, pos, form = parse_list(page, resource)
        for n in range(2, (pos[1] if pos else 1) + 1):
            if not form:
                break
            page = site.post(form, data={"hidden_page_size": "", "hidden_page_to": str(n)}).text
            more, pos, form = parse_list(page, resource)
            rows += more
    if pos and len(rows) != pos[2]:
        print(f"  WARNING {resource}: read {len(rows)} rows, the page says {pos[2]}", flush=True)
    p.write_text(json.dumps({"resource": resource, "url": url, "rows": [asdict(r) for r in rows]}, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def fetch_detail(site: Site, row: Row, force: bool = False) -> Detail:
    p = site.dir / "details" / f"{row.doc_id}.json"
    if p.exists() and not force:
        d = json.loads(p.read_text(encoding="utf-8"))
        return Detail(d["doc_id"], d["leaf"], d["meta"], [Attachment(**a) for a in d["attachments"]])
    det = parse_detail(site.get(row.url).text, row.doc_id, row.leaf)
    det.meta.setdefault("Oggetto", row.subject)
    p.write_text(json.dumps(asdict(det), ensure_ascii=False, indent=1), encoding="utf-8")
    return det


def list_soggetti(site: Site, resource: str, url: str, force: bool = False) -> list[Detail]:
    p = site.dir / "details" / f"{slug(leaf_id(url), 30)}_soggetti.json"
    if p.exists() and not force:
        return [Detail(d["doc_id"], d["leaf"], d["meta"], [Attachment(**a) for a in d["attachments"]]) for d in json.loads(p.read_text(encoding="utf-8"))]
    items = parse_soggetti(site.get(url).text, resource)
    p.write_text(json.dumps([asdict(i) for i in items], ensure_ascii=False, indent=1), encoding="utf-8")
    return items


def filename_from(resp: requests.Response, att: Attachment) -> str:
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", cd)
    name = unquote(m.group(1)) if m else att.title
    ext = Path(name).suffix
    if not ext:
        ext = {"application/pdf": ".pdf", "application/zip": ".zip", "application/xml": ".xml", "text/xml": ".xml", "application/vnd.ms-excel": ".xls",
               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx"}.get(resp.headers.get("Content-Type", "").split(";")[0].strip(), "")
    return f"{att.att_id}_{slug(Path(name).stem)}{ext}"


def download(site: Site, doc: Detail, att: Attachment) -> dict[str, str | int]:
    """Download one attachment (skips a file already on disk); return its manifest row."""
    folder = site.dir / "files" / doc.doc_id
    folder.mkdir(parents=True, exist_ok=True)
    existing = sorted(folder.glob(f"{att.att_id}_*"))
    if existing:
        path = existing[0]
    else:
        r = site.get(att.url)
        path = folder / filename_from(r, att)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.rename(path)
    data = path.read_bytes()
    return {"doc_id": doc.doc_id, "leaf": doc.leaf, "subject": doc.meta.get("Oggetto", ""), "att_id": att.att_id, "title": att.title, "description": att.description,
            "file": str(path.relative_to(site.dir)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "pdf": data[:5] == b"%PDF-"}


SIGNED_MIMETYPES = ("application/x-pkcs7-mime", "application/pkcs7-mime")


def wanted(att: Attachment, with_signed: bool = False) -> bool:
    """Skip the `.p7m` signature envelopes (a signed copy of a PDF that is listed beside it) unless asked."""
    return with_signed or att.mimetype not in SIGNED_MIMETYPES


def write_manifest(site: Site, rows: list[dict]) -> Path:
    p = site.dir / "files.csv"
    old = {r["att_id"]: r for r in csv.DictReader(p.open(encoding="utf-8"))} if p.exists() else {}
    for r in rows:
        old[str(r["att_id"])] = {k: str(v) for k, v in r.items()}
    if old:
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(next(iter(old.values())).keys()))
            w.writeheader()
            w.writerows(old.values())
    return p


def run(site: Site, prefixes: tuple[str, ...], match: str, from_year: int | None, listing_only: bool = False, limit: int | None = None, with_signed: bool = False) -> None:
    cat = fetch_catalog(site)
    leaves = select_leaves(cat, prefixes)
    print(f"{len(leaves)} leaves selected of {len(cat)}", flush=True)
    rx = re.compile(match, re.IGNORECASE)
    docs: list[Detail] = []
    for res, url in leaves.items():
        print(f"[{res}]", flush=True)
        if "/papca" in url:
            rows = list_leaf(site, res, url)
            keep = [r for r in rows if rx.search(r.subject) and (from_year is None or (r.year or 0) >= from_year)]
            print(f"  {len(rows)} rows, {len(keep)} match", flush=True)
            if listing_only:
                continue
            for r in keep[:limit]:
                docs.append(fetch_detail(site, r))
        elif "/pas" in url:
            items = list_soggetti(site, res, url)
            print(f"  {len(items)} items", flush=True)
            docs += [] if listing_only else items
        else:
            print("  layout not handled, skipped", flush=True)
    if listing_only:
        return
    manifest: list[dict] = []
    docs = [Detail(d.doc_id, d.leaf, d.meta, [a for a in d.attachments if wanted(a, with_signed)]) for d in docs]
    total = sum(len(d.attachments) for d in docs)
    print(f"{len(docs)} documents, {total} attachments to download", flush=True)
    n = 0
    for d in docs:
        for a in d.attachments:
            n += 1
            try:
                manifest.append(download(site, d, a))
            except RuntimeError as e:
                print(f"  FAILED {a.att_id}: {e}", flush=True)
            if n % 10 == 0:
                write_manifest(site, manifest)
                print(f"  {n}/{total}", flush=True)
    path = write_manifest(site, manifest)
    record(f"trasparenza_{site.name}_files", source="municipal transparency portal", dataset="documents", url=site.base)
    print(f"manifest: {path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("cmd", choices=["catalog", "list", "run"], help="catalog: menu leaves; list: all rows of the selected leaves; run: list, details and downloads")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="fiscal")
    ap.add_argument("--section", action="append", help="menu path prefix (repeatable); replaces the preset")
    ap.add_argument("--match", default=DEFAULT_MATCH, help="regex on the subject of an act")
    ap.add_argument("--from-year", type=int, default=2019)
    ap.add_argument("--limit", type=int, help="documents per leaf (for a trial)")
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--with-signed", action="store_true", help="also download the .p7m signature envelopes")
    a = ap.parse_args(argv)
    site = Site(a.base_url, delay=a.delay)
    if a.cmd == "catalog":
        for k, v in sorted(fetch_catalog(site, force=True).items()):
            print(k, "->", v)
        return
    run(site, tuple(a.section) if a.section else PRESETS[a.preset], a.match, a.from_year, listing_only=a.cmd == "list", limit=a.limit, with_signed=a.with_signed)


if __name__ == "__main__":
    main()
