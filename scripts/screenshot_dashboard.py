"""Start the dashboard, open each view in headless Chromium and save a full-page screenshot (visual check / regression aid).

Usage: uv run python scripts/screenshot_dashboard.py [--lang it] [--views overview explorer ...] [--theme dark]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import requests

from cifresignificative.core.sources import browser as br
from cifresignificative.studies.italy_productivity.dashboard import VIEWS

OUT = Path("studies/italy_productivity/reports/dashboard_screenshots")
PORT = 8791


def wait_ready(url: str, timeout: int = 60) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url + "/_stcore/health", timeout=2).text.strip() == "ok":
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("dashboard did not start")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--views", nargs="*")
    args = ap.parse_args()
    views = args.views or list(VIEWS)
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "streamlit", "run", "src/cifresignificative/dashboard/app.py", "--server.headless", "true", "--server.port", str(PORT), f"--theme.base={args.theme}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://localhost:{PORT}"
    try:
        wait_ready(base)
        with br.browser(viewport=(1500, 1000)) as ctx:
            page = ctx.new_page()
            for v in views:
                page.goto(f"{base}/?view={v}&lang={args.lang}", wait_until="networkidle", timeout=90000)
                try:
                    page.wait_for_selector(".js-plotly-plot, [data-testid='stMetric'], [data-testid='stMarkdownContainer']", timeout=45000)
                except Exception:  # noqa: BLE001
                    print(f"  (no chart found for {v})")
                page.wait_for_timeout(2500)
                errors = page.locator("[data-testid='stException']").count()
                path = OUT / f"{v}_{args.lang}_{args.theme}.png"
                page.screenshot(path=str(path), full_page=True)
                print(f"{v:14s} -> {path}  exceptions on page: {errors}")
    finally:
        proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
