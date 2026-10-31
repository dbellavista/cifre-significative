"""Scaffold a new study: `uv run python scripts/new_study.py NAME "Title" "One-sentence question"` (or `uv run inv study.new ...`).

Creates studies/NAME (copy of studies/_template with study.toml filled in), src/cifresignificative/studies/NAME/{__init__,analysis/__init__}.py
and tests/studies/NAME/test_registry.py. Nothing outside those three folders is touched; add the files with `git add` before `inv publish.check`."""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"[a-z][a-z0-9_]*")


def scaffold(name: str, title: str, description: str, root: Path = ROOT) -> list[Path]:
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"study name must be a lowercase python identifier, got {name!r}")
    content, code, tests = root / "studies" / name, root / "src/cifresignificative/studies" / name, root / "tests/studies" / name
    for p in (content, code, tests):
        if p.exists():
            raise FileExistsError(f"{p} exists")
    shutil.copytree(root / "studies/_template", content, ignore=shutil.ignore_patterns("__pycache__"))
    toml = (content / "study.toml").read_text()
    toml = re.sub(r'^name = "template"', f'name = "{name}"', toml, flags=re.MULTILINE)
    toml = re.sub(r'^title = ".*?"', lambda _: f'title = {_q(title)}', toml, flags=re.MULTILINE)
    toml = re.sub(r'^description = ".*?"', lambda _: f'description = {_q(description)}', toml, flags=re.MULTILINE)
    (content / "study.toml").write_text(toml)
    readme = (content / "README.md").read_text().replace("<Study title>", title).replace("<name>", name)
    (content / "README.md").write_text(readme)
    (code / "analysis").mkdir(parents=True)
    (code / "__init__.py").write_text(f'"""{title}: study code (may import cifresignificative.core, never another study)."""\n')
    (code / "analysis" / "__init__.py").write_text('"""Pure statistics and loaders of this study; runners that write to data/processed live in experiments."""\n')
    tests.mkdir(parents=True)
    (tests / "test_registry.py").write_text(
        f'"""The study is registered and its code package imports."""\nfrom cifresignificative.study import load_studies\n\n\n'
        f'def test_{name}_is_registered():\n    s = load_studies()["{name}"]\n    assert s.title and s.notebooks.exists()\n\n\n'
        f'def test_{name}_package_imports():\n    import cifresignificative.studies.{name}.analysis  # noqa: F401\n'
    )
    return [content, code, tests]


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    for p in scaffold(sys.argv[1], sys.argv[2], sys.argv[3]):
        print("created", p.relative_to(ROOT))
