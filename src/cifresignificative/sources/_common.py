"""Shared helpers for source modules: manifest bookkeeping."""
import json
from datetime import UTC, datetime
from pathlib import Path

MANIFEST = Path("data/raw/manifest.json")


def record(name: str, **info) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[name] = {**info, "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds")}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
