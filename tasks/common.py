"""Shared helpers for the invoke tasks."""
import shutil

PATHS = "src tests scripts tasks"


def runner(rtk: bool | None) -> str:
    """'uv run rtk ' when RTK is requested (default: when `rtk` is on the PATH), else 'uv run '."""
    use = shutil.which("rtk") is not None if rtk is None else rtk
    return "uv run rtk " if use else "uv run "
