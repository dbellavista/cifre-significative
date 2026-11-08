"""OCR connector (`core.sources.ocr`): row grouping, Italian amounts, the `pensieri-ocr` call and the cache, with no service; the cached 2019 consolidated accounts (skipped without data/raw) are checked against their own accounting identity."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cifresignificative.core.sources import ocr


def tok(text, x1, y1, x2, y2):
    return {"text": text, "confidence": 0.9, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_rows_group_by_line_and_order_left_to_right():
    tokens = [tok("2019", 500, 52, 560, 78), tok("Terreni", 10, 50, 120, 80), tok("Fabbricati", 10, 120, 150, 150), tok("7.928.721", 500, 118, 600, 152)]
    assert ocr.row_texts(tokens) == [["Terreni", "2019"], ["Fabbricati", "7.928.721"]]


def test_rows_keep_a_tall_token_with_its_line_but_split_distinct_lines():
    tokens = [tok("a", 0, 0, 10, 40), tok("b", 20, 8, 30, 32), tok("c", 0, 45, 10, 85)]
    assert ocr.row_texts(tokens) == [["a", "b"], ["c"]]


@pytest.mark.parametrize("text,value", [("1.077.413.797", 1077413797.0), ("-8.174.355", -8174355.0), ("(1.234)", -1234.0), ("12,5", 12.5), ("1.234,56", 1234.56), ("0", 0.0), (" 251.682 ", 251682.0)])
def test_parse_amount_reads_italian_formats(text, value):
    assert ocr.parse_amount(text) == value


@pytest.mark.parametrize("text", ["", "BI1", "1.07.413", "1.2345", "S99'890", "TOTALE", "1,2,3"])
def test_parse_amount_rejects_text_and_malformed_thousands(text):
    assert ocr.parse_amount(text) is None


def fake_cli(calls, text="1.000"):
    def run(path, lang):
        calls.append((path.name, lang))
        return [[tok(text, 1, 2, 3, 4)]]

    return run


def test_run_cli_splits_the_json_lines_into_pages(monkeypatch):
    out = json.dumps({"pages": 2, "lines": [{"page": 2, "text": "b", "confidence": 0.91234567, "bbox": [5, 6, 7, 8]}, {"page": 1, "text": "a", "confidence": 1.0, "bbox": [1, 2, 3, 4]}]})
    monkeypatch.setattr(ocr, "command", lambda *a: ["x", *a])
    monkeypatch.setattr(ocr.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=out, stderr=""))
    pages = ocr.run_cli(Path("f.pdf"), "it")
    assert [[t["text"] for t in p] for p in pages] == [["a"], ["b"]] and pages[1][0] == {"text": "b", "confidence": 0.9123, "x1": 5, "y1": 6, "x2": 7, "y2": 8}


def test_run_cli_reports_the_command_error(monkeypatch):
    monkeypatch.setattr(ocr, "command", lambda *a: ["x", *a])
    monkeypatch.setattr(ocr.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="unavailable"))
    with pytest.raises(RuntimeError, match="unavailable"):
        ocr.run_cli(Path("f.pdf"), "it")


def test_command_needs_the_installed_tool(monkeypatch):
    monkeypatch.setattr(ocr.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="uv tool install"):
        ocr.command("health")


def test_ocr_file_reads_the_cache_and_rereads_when_the_file_or_language_changes(tmp_path, monkeypatch):
    f = tmp_path / "s.pdf"
    f.write_bytes(b"one")
    calls = []
    monkeypatch.setattr(ocr, "record", lambda *a, **k: None)
    monkeypatch.setattr(ocr, "run_cli", fake_cli(calls))
    cache = tmp_path / "cache"
    first = ocr.ocr_file(f, cache_dir=cache)
    assert first == [[tok("1.000", 1, 2, 3, 4)]] and ocr.ocr_file(f, cache_dir=cache) == first and len(calls) == 1
    ocr.ocr_file(f, "en", cache_dir=cache)
    f.write_bytes(b"two")
    ocr.ocr_file(f, "en", cache_dir=cache)
    assert calls == [("s.pdf", "it"), ("s.pdf", "en"), ("s.pdf", "en")]


CACHE = Path("data/raw/ocr/Bilancio_consolidato_2019.json")


@pytest.mark.skipif(not CACHE.exists(), reason="OCR cache of the 2019 consolidated accounts missing (uv run python -m cifresignificative.core.sources.ocr <pdf> --start)")
def test_consolidated_2019_balance_sheet_closes():
    pages = json.loads(CACHE.read_text())["pages"]
    assert len(pages) == 3

    def line(page, label):
        (row,) = [r for r in ocr.row_texts(pages[page]) if r[0].upper().startswith(label)]
        return [ocr.parse_amount(x) for x in row[1:3]]

    assets, liabilities = line(0, "TOTALE DELL'ATTIVO"), line(1, "TOTALE DEL PASSIVO")
    assert assets == liabilities == [1077413797.0, 1082068665.0]
    parts = [line(1, k)[0] for k in ("TOTALE PATRIMONIO NETTO", "TOTALE FONDI RISCHI", "TOTALE T.F.R.", "TOTALE DEBIT", "TOTALE RATEI E RISCONTI (E")]
    assert sum(parts) == pytest.approx(liabilities[0], abs=1)
