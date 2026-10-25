"""Generated documentation: the data catalog."""
from invoke import Context, task

CMD = "uv run python scripts/make_data_catalog.py"


@task
def catalog(c: Context) -> None:
    """Regenerate docs/data_catalog.md (needs the built panel); never edit that file by hand."""
    c.run(CMD, pty=True)


@task
def check(c: Context) -> None:
    """Fail when docs/data_catalog.md is stale (full check with data, static sections without)."""
    c.run(f"{CMD} --check", pty=True)
