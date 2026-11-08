"""Pre-2023 documents of a municipality's old "Amministrazione trasparente" site (Flex CM pages, `sititematici.<comune>.fc.it`).

The Liferay portal read by `trasparenza` starts in January 2023; the older acts stay on the previous site, as plain HTML pages whose attachments are links `.../ServeAttachment.php/L/IT/D/<hash>/P/BLOB:ID=<n>/E/<ext>?mode=download` with the title and size as link text.
`robots.txt` asks for a 2 s crawl delay (the `Site` throttle default). Files go to `data/raw/trasparenza/<site>/files/<page>/<title>.<ext>` with `files.csv`; a run can be stopped and restarted (existing files are kept).
Usage: `uv run python -m cifresignificative.core.sources.comune_archive --base-url https://sititematici.comune.cesena.fc.it [--page PATH ...]` (default: the pages in `CESENA_PAGES`).
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from cifresignificative.core.sources._common import record
from cifresignificative.core.sources.trasparenza import RAW_DIR, Site, clean, slug

# Rendiconto 2018-2021 (Entrate/Spese CSV + PDFs), budget indicators 2020-2022, consolidated accounts 2018-2021.
CESENA_PAGES = tuple(f"/trasparenza/bilancioconsuntivo{y}" for y in (2018, 2019, 2020, 2021)) + ("/flex/cm/pages/ServeBLOB.php/L/IT/IDPagina/13304", "/trasparenza-bilancioconsolidato", "/trasparenza-bilancioconsolidato2019", "/trasparenza-bilancioconsolidato2018")
LINK = re.compile(r"<a [^>]*href=['\"]([^'\"]*ServeAttachment\.php/L/IT/D/[^'\"]*/E/(\w+)\?mode=download)['\"][^>]*>(.*?)</a>", re.DOTALL)
SIZE = re.compile(r"\s*\(\d+(?:\.\d+)?\s*[KM]B\)\s*$")


def parse_attachments(page: str) -> list[tuple[str, str, str]]:
    """(url, extension, title) of every downloadable attachment on a page, titles without the trailing `(1.24 MB)`; repeated links of a page are kept once."""
    seen, out = set(), []
    for href, ext, text in LINK.findall(page):
        url = href.replace("&amp;", "&")
        if url not in seen:
            seen.add(url)
            out.append((url, ext.lower(), SIZE.sub("", clean(text))))
    return out


def page_name(path: str) -> str:
    return slug(path.rstrip("/").split("/")[-1] if "IDPagina" not in path else "IDPagina" + path.rstrip("/").split("/")[-1])


def run(site: Site, pages: tuple[str, ...] = CESENA_PAGES) -> Path:
    rows = []
    for path in pages:
        for url, ext, title in parse_attachments(site.get(path).text):
            dest = site.dir / "files" / page_name(path) / f"{slug(title)}.{ext}"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(site.get(url).content)
            rows.append({"page": path, "title": title, "ext": ext, "file": str(dest.relative_to(site.dir)), "bytes": dest.stat().st_size})
            print(f"  {dest.relative_to(site.dir)} ({dest.stat().st_size} bytes)", flush=True)
    out = site.dir / "files.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["page", "title", "ext", "file", "bytes"])
        w.writeheader()
        w.writerows(rows)
    record(f"comune_archive_{site.name}", files=len(rows), pages=len(pages))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--page", action="append", help="page path (repeatable); default: the Cesena preset")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    a = ap.parse_args(argv)
    run(Site(a.base_url, a.raw_dir), tuple(a.page) if a.page else CESENA_PAGES)


if __name__ == "__main__":
    main()
