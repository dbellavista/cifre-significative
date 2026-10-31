"""Registry of studies: every `studies/<name>/study.toml` describes one study built on the shared platform."""
import tomllib
from dataclasses import dataclass
from pathlib import Path

STUDIES_DIR = Path(__file__).resolve().parents[2] / "studies"


@dataclass(frozen=True)
class Study:
    name: str
    title: str
    description: str
    status: str
    root: Path
    notebooks: Path
    reports: Path
    publications: Path | None
    cells: Path | None
    dashboard: str | None


def load_study(folder: Path) -> Study:
    cfg = tomllib.loads((folder / "study.toml").read_text())
    if cfg["name"] != folder.name and folder.name != "_template":
        raise ValueError(f"study.toml name {cfg['name']!r} differs from its folder {folder.name!r}")
    pub = cfg.get("publications", "")
    return Study(cfg["name"], cfg["title"], cfg["description"], cfg["status"], folder,
                 (folder / cfg["notebooks"]).resolve(), (folder / cfg["reports"]).resolve(), (folder / pub).resolve() if pub else None,
                 (folder / cfg["cells"]).resolve() if cfg.get("cells") else None,
                 cfg.get("dashboard") or None)


def load_studies(base: Path = STUDIES_DIR) -> dict[str, Study]:
    """All studies under `base` by name (the `_template` skeleton is skipped)."""
    return {s.name: s for s in (load_study(p.parent) for p in sorted(base.glob("*/study.toml")) if p.parent.name != "_template")}
