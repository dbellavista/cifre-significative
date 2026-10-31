# Studies

Each folder is one self-contained study built on the shared platform in `src/cifresignificative/core/` (sources, cleaning, catalog, glossary, viz). A study is registered by its `study.toml`; `cifresignificative.study.load_studies()` reads them all (`uv run inv study.list`).

| study | question | status |
|---|---|---|
| [`italy_productivity/`](italy_productivity/) | Italy against the other EU27 members: productivity, wages, the South, public finance | active (Substack series) |
| [`_template/`](_template/) | skeleton copied by `inv study.new` | |

A study is three folders with the same name:

| what | where |
|---|---|
| content: notebooks, cells, findings, publications, README | `studies/<name>/` |
| code: analysis modules, optional `dashboard.py` | `src/cifresignificative/studies/<name>/` |
| tests | `tests/studies/<name>/` |

Rule of thumb: if a piece of code or a table would still make sense for a different topic, it belongs to the platform (`core`, tests in `tests/core`); otherwise to the study. Studies may use the platform, the platform never imports a study, and studies do not import each other: promote the shared piece to `core` first.

New study: `uv run inv study.new <name> "<Title>" "<question>"`, then follow the `add-study` agent skill.
