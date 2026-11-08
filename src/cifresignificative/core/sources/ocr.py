"""OCR of scanned PDFs and images with the `pensieri-ocr` shell command (PaddleOCR server of the sister project; Tesseract is not used).

`pensieri-ocr` is installed once as a uv tool (`uv tool install <pensieri checkout>/packages/ocr`) and talks gRPC to a PaddleOCR server on `127.0.0.1:50051` that runs natively on the Mac (`apps/ocr-server`, Python 3.13, its own venv; `ensure_service` starts it, models load for ~30 s). Languages: `en`, `it` (keeps accents), `ch`. No job queue is involved.
The command renders each PDF page at ~150 dpi, applies the page's `/Rotate` flag (the 2019 consolidated accounts of the Comune are stored sideways) and prints one line per recognised text with its page, confidence and box (`--json`); about 25 s a page on CPU, so the timeout is 600 s.
Results are cached as JSON next to the other raw files, `data/raw/ocr/<file stem>.json` (`sha256` of the file and the language inside: a changed file is read again); `rows` joins the tokens of a printed line and `parse_amount` reads Italian number formats.
Usage: `uv run python -m cifresignificative.core.sources.ocr FILE [--lang it] [--start] [--force]` prints the rows of every page.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from statistics import mean

from cifresignificative.core.sources._common import record

CACHE_DIR = Path("data/raw/ocr")
SERVER_DIR = Path(os.environ.get("OCR_SERVER_DIR", "~/pensieri/apps/ocr-server")).expanduser()
HOST, PORT = os.environ.get("OCR_HOST", "127.0.0.1"), os.environ.get("OCR_PORT", "50051")
TIMEOUT_SECS = 600  # per request: a full page takes ~25 s on CPU, a long PDF is one request
START_TIMEOUT_SECS = 180.0
ROW_TOLERANCE = 0.6  # a token joins a row when its centre is within this share of its height of the row's mean centre
AMOUNT = re.compile(r"-?\d{1,3}(?:\.\d{3})+(?:,\d+)?|-?\d+(?:,\d+)?")


def command(*args: str) -> list[str]:
    exe = shutil.which("pensieri-ocr")
    if exe is None:
        raise RuntimeError("`pensieri-ocr` not found: `uv tool install <pensieri checkout>/packages/ocr`")
    return [exe, "--host", HOST, "--port", PORT, "--timeout", str(TIMEOUT_SECS), *args]


def service_up() -> bool:
    return subprocess.run(command("health"), capture_output=True, check=False).returncode == 0


def ensure_service() -> subprocess.Popen | None:
    """Start the server natively (`uv run python -m pensieri_ocr_server` in `SERVER_DIR`) unless a healthy one answers; returns the process to stop, or None when one was already running."""
    if service_up():
        return None
    proc = subprocess.Popen(["uv", "run", "python", "-m", "pensieri_ocr_server"], cwd=SERVER_DIR, env={**os.environ, "OCR_GRPC_PORT": PORT}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + START_TIMEOUT_SECS
    while not service_up():
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.terminate()
            raise RuntimeError(f"OCR server did not become healthy at {HOST}:{PORT} (server dir {SERVER_DIR}; stubs and models: see its docs/ocr.md)")
        time.sleep(2)
    return proc


def run_cli(path: Path, lang: str) -> list[list[dict]]:
    """Tokens per page from `pensieri-ocr run FILE --lang LANG --json` (`text`, `confidence`, `x1`, `y1`, `x2`, `y2` in the upright page image)."""
    done = subprocess.run(command("run", str(path), "--lang", lang, "--json"), capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"pensieri-ocr failed on {path.name}: {done.stderr.strip()}")
    result = json.loads(done.stdout)
    pages: list[list[dict]] = [[] for _ in range(result["pages"])]
    for line in result["lines"]:
        x1, y1, x2, y2 = line["bbox"]
        pages[line["page"] - 1].append({"text": line["text"], "confidence": round(line["confidence"], 4), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return pages


def rows(tokens: list[dict], tolerance: float = ROW_TOLERANCE) -> list[list[dict]]:
    """Tokens grouped into printed lines, top to bottom, each line left to right; a token joins a line when its vertical centre is within `tolerance` x its own height of the line's mean centre."""
    lines: list[list[dict]] = []
    for t in sorted(tokens, key=lambda t: t["y1"] + t["y2"]):
        centre, height = (t["y1"] + t["y2"]) / 2, max(t["y2"] - t["y1"], 1)
        if lines and abs(mean((u["y1"] + u["y2"]) / 2 for u in lines[-1]) - centre) < tolerance * height:
            lines[-1].append(t)
        else:
            lines.append([t])
    return [sorted(line, key=lambda t: t["x1"]) for line in lines]


def row_texts(tokens: list[dict], tolerance: float = ROW_TOLERANCE) -> list[list[str]]:
    return [[t["text"] for t in line] for line in rows(tokens, tolerance)]


def parse_amount(text: str) -> float | None:
    """Italian number (`1.077.413.797`, `-8.174.355`, `(1.234)`, `12,5`) as a float; None for anything else, so a misread token is never turned into a number."""
    s = text.strip().replace("€", "").replace(" ", "")
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if not AMOUNT.fullmatch(s):
        return None
    value = float(s.replace(".", "").replace(",", "."))
    return -value if negative else value


def ocr_file(path: str | Path, lang: str = "it", force: bool = False, cache_dir: Path = CACHE_DIR) -> list[list[dict]]:
    """Tokens of every page of an image or PDF, from the cache when the file and language are unchanged."""
    path = Path(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = cache_dir / f"{path.stem}.json"
    if cache.exists() and not force:
        cached = json.loads(cache.read_text())
        if (cached["sha256"], cached["lang"]) == (sha, lang):
            return cached["pages"]
    pages = run_cli(path, lang)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"file": path.name, "sha256": sha, "lang": lang, "pages": pages}, ensure_ascii=False))
    record(f"ocr_{path.stem}", pages=len(pages), tokens=sum(map(len, pages)), lang=lang)
    return pages


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", type=Path)
    ap.add_argument("--lang", default="it", choices=["en", "it", "ch"])
    ap.add_argument("--start", action="store_true", help="start the OCR server natively when none answers (stopped at the end)")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args(argv)
    proc = ensure_service() if args.start else None
    try:
        pages = ocr_file(args.file, args.lang, args.force)
    finally:
        if proc:
            proc.terminate()
    for n, tokens in enumerate(pages, 1):
        print(f"--- page {n}: {len(tokens)} tokens")
        for line in row_texts(tokens):
            print(" | ".join(line))


if __name__ == "__main__":
    main()
