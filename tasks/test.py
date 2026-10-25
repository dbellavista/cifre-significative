"""Tests: focused files or keywords, optional slow notebook execution."""
from invoke import Context, task

from tasks.common import runner


@task(
    iterable=["f"],
    help={"f": "test file or node id (repeat -f for several)", "k": "pytest -k expression", "x": "stop at the first failure", "slow": "RUN_SLOW=1: also execute every notebook (2-4 minutes)",
          "rtk": "force/disable RTK output filtering (default: auto)"},
)
def run(c: Context, f: list[str] | None = None, k: str | None = None, x: bool = False, slow: bool = False, rtk: bool | None = None) -> None:
    """Run pytest quietly; without -f the whole suite runs. Data-dependent tests skip when data/processed is missing."""
    cmd = f"{runner(rtk)}pytest -q {' '.join(f or [])} {f'-k {k!r} ' if k else ''}{'-x ' if x else ''}".strip()
    c.run(cmd, pty=True, env={"RUN_SLOW": "1"} if slow else {})
