"""Scriptable-browser helpers for sources that need a real browser (Playwright, Chromium).

Download protocol (see docs/download_protocol.md): official API/file -> plain HTTP -> headless browser that reads the page's own
data requests -> headed browser where a human solves any anti-bot check, then the script continues with the same session.
We never try to defeat or hide from anti-bot systems automatically.
"""
import re
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import BrowserContext, Page, sync_playwright

PROFILE_DIR = Path("data/.browser_profile")  # persistent cookies for headed sessions (gitignored, never commit)
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
INTERACTIVE = False  # set by `cifresignificative.core.fetch --interactive`: allow falling back to a visible browser with human help


@contextmanager
def browser(headed: bool = False, viewport: tuple[int, int] = (1600, 1000)) -> Generator[BrowserContext]:
    """Yield a browser context. Headed sessions use a persistent profile so a solved check is remembered."""
    with sync_playwright() as p:
        if headed:
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            ctx = p.chromium.launch_persistent_context(str(PROFILE_DIR), headless=False, viewport={"width": viewport[0], "height": viewport[1]}, user_agent=UA)
        else:
            b = p.chromium.launch(headless=True)
            ctx = b.new_context(viewport={"width": viewport[0], "height": viewport[1]}, user_agent=UA)
        try:
            yield ctx
        finally:
            ctx.close()


def handoff(page: Page, message: str, ready: Callable[[Page], bool] | None = None, timeout: int = 600) -> None:
    """Ask the human to act in the visible browser (e.g. pass a check); continue when `ready(page)` is true or after Enter."""
    print(f"\n[browser needs you] {message}\nSolve it in the open window; the script continues automatically (or press Enter here).")
    start = time.time()
    while time.time() - start < timeout:
        if ready and ready(page):
            return
        page.wait_for_timeout(1500)
        if not ready:
            input("Press Enter when done... ")
            return
    raise TimeoutError("timed out waiting for the human step")


def is_blocked(page: Page) -> bool:
    """Heuristic for an anti-bot / access-denied page."""
    try:
        text = page.inner_text("body", timeout=3000)[:1500].lower()
    except Exception:  # noqa: BLE001
        return False
    return bool(re.search(r"access denied|403 forbidden|verify you are human|captcha|are you a robot|unusual traffic", text))
