"""Static checks: ruff and ty, both blocking."""
from invoke import Context, task

from tasks.common import PATHS, runner

HELP = {"fix": "apply ruff --fix", "ty": "also run the ty type checker (--no-ty to skip)", "rtk": "force/disable RTK output filtering (default: auto)"}


def _checks(c: Context, scope: str, fix: bool, ty: bool, rtk: bool | None) -> None:
    run = runner(rtk)
    c.run(f"{run}ruff check {'--fix ' if fix else ''}{scope}", pty=True)  # failure stops here
    if ty:
        c.run(f"{run}ty check {scope}", pty=True)  # failure stops here


@task(help=HELP)
def verify(c: Context, fix: bool = False, ty: bool = True, rtk: bool | None = None) -> None:
    """Whole-repo checks: ruff on src, tests, scripts, tasks; then ty."""
    _checks(c, PATHS, fix, ty, rtk)


@task(help={**HELP, "scope": "space-separated files or directories (default: the whole repo)"})
def scope_checks(c: Context, scope: str = PATHS, fix: bool = False, ty: bool = True, rtk: bool | None = None) -> None:
    """Checks on a touched scope, e.g. --scope "src/cifresignificative/analysis tests/test_hf.py"."""
    _checks(c, scope.strip() or PATHS, fix, ty, rtk)
