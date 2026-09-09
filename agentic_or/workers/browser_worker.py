from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Dict, Optional, Set

from agentic_or.models import TaskNode, DispatchAction, WorkloadType
from agentic_or.workers.base_worker import (
    BaseWorker,
    TaskExecutionResult,
    CaptchaBlockedError,
    RateLimitedError,
)

if TYPE_CHECKING:
    from playwright.async_api import Browser, Page, Playwright

logger = logging.getLogger(__name__)

# playwright is an OPTIONAL, heavy dependency (~200MB browser binary) - see
# pyproject.toml's [project.optional-dependencies] "browser" extra. Import
# it lazily/defensively so agentic_or works for LOCAL/API-only usage without
# ever requiring it. `_PLAYWRIGHT_AVAILABLE` gates every real-automation
# code path below; when False, BrowserWorker behaves EXACTLY like the
# original pure simulation (existing tests/demo.py never pass
# task.metadata["url"], so they never touch playwright at all either way).
try:
    from playwright.async_api import async_playwright, Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

# Heuristic only - detects that a page is SHOWING a block, so the incident
# can be escalated to CaptchaSolver (self-healing), same as any other
# worker. This does not attempt to solve/bypass anything.
_CAPTCHA_MARKERS = (
    "captcha", "are you a robot", "are you human", "verify you are human",
    "unusual traffic", "access denied", "checking your browser",
)


def _strip_html_text(text: str, max_chars: int = 3000) -> str:
    """Same cap as web_fetch_agent.py/main_agent.py - a raw page's text can
    still be large; an uncapped fetch once blew a real LLM token limit."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"... [truncated, {len(text)} chars total]"
    return text


class BrowserWorker(BaseWorker):
    """
    Browser automation worker. Two modes, chosen per-task, no config needed:

    - `task.metadata["url"]` given AND `playwright` installed
      (`pip install desktop-agent-or[browser] && playwright install chromium`):
      REAL headless Chromium navigation - launches a real browser (lazily,
      on first use), reuses the same page/tab for repeat tasks sharing
      `task.affinity_key` (true warm-start: 0 navigation overhead, matching
      the Context-Affinity concept the C++ scheduler already optimizes
      for), and returns the real page title + stripped visible text.
    - Otherwise (no `url`, or playwright not installed): the ORIGINAL
      simulated behavior (asyncio.sleep standing in for cold-start/setup/
      exec time) - so every existing caller that doesn't pass a real URL
      (demo.py, the test suite) is completely unaffected.

    Raises `CaptchaBlockedError` when a page's content matches a common
    "you're blocked" pattern, and `RateLimitedError` on an HTTP 429/403
    response - both route through the same Self-Healing Daemons as any
    other worker; this never attempts to solve/bypass an actual challenge.
    """

    _warned_missing_playwright = False  # class-level: warn once per process, not once per instance

    def __init__(self, worker_id: str):
        super().__init__(worker_id, WorkloadType.BROWSER)
        # --- simulated-mode state (used whenever real automation doesn't apply) ---
        self.is_browser_launched: bool = False
        self.open_domains: Set[str] = set()

        # --- real Playwright state (only ever touched if _PLAYWRIGHT_AVAILABLE) ---
        self._playwright: Optional["Playwright"] = None
        self._browser: Optional["Browser"] = None
        # Warm page pool keyed by affinity_key - reusing the SAME Page for a
        # repeat key is the real equivalent of the simulated "0ms warm-start".
        self._pages: Dict[str, "Page"] = {}
        # Which running event loop the above are bound to. A worker instance
        # can outlive one asyncio.run() call (e.g. the CLI's persistent
        # session Orchestrator across multiple `run` commands) - Playwright's
        # objects, like an httpx.AsyncClient, break ("Event loop is closed")
        # if reused from a different one than they were created on. Same
        # pattern as ApiWorker/FetchAgent's _get_client().
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def execute_task(self, task: TaskNode, action: DispatchAction) -> TaskExecutionResult:
        url = task.metadata.get("url")
        if url and _PLAYWRIGHT_AVAILABLE:
            return await self._execute_real(task, action, url)
        if url and not _PLAYWRIGHT_AVAILABLE:
            if not BrowserWorker._warned_missing_playwright:
                logger.warning(
                    "[BrowserWorker] task.metadata['url'] given but playwright is not installed - "
                    "falling back to simulated browsing. Install with: "
                    "pip install desktop-agent-or[browser] && playwright install chromium"
                )
                BrowserWorker._warned_missing_playwright = True
        return await self._execute_simulated(task, action)

    # ------------------------------------------------------------------
    # Simulated mode - unchanged from the original implementation, kept
    # verbatim so every existing test/demo.py call behaves identically.
    # ------------------------------------------------------------------
    async def _execute_simulated(self, task: TaskNode, action: DispatchAction) -> TaskExecutionResult:
        try:
            if not self.is_browser_launched:
                logger.info(f"[{self.worker_id}] Launching browser instance (Cold Start)...")
                await asyncio.sleep(min(0.3, action.setup_cost_ms / 1000.0))
                self.is_browser_launched = True
            elif task.affinity_key and task.affinity_key in self.open_domains:
                logger.info(f"[{self.worker_id}] Warm-start reuse for '{task.affinity_key}' (0ms setup)!")
            else:
                await asyncio.sleep(min(0.1, action.setup_cost_ms / 1000.0))

            if task.affinity_key:
                self.open_domains.add(task.affinity_key)

            exec_time = min(0.3, task.estimated_duration_ms / 1000.0)
            await asyncio.sleep(exec_time)

            return TaskExecutionResult(
                task.task_id,
                success=True,
                output=f"Browser scraped: {task.name or task.task_id} @ {task.affinity_key}"
            )
        except Exception as e:
            return TaskExecutionResult(task.task_id, success=False, error=str(e))

    # ------------------------------------------------------------------
    # Real Playwright mode
    # ------------------------------------------------------------------
    async def _ensure_browser(self) -> "Browser":
        loop = asyncio.get_running_loop()
        if self._browser is None or self._loop is not loop:
            # Loop changed (or first use): any previous playwright/browser
            # objects are bound to a dead loop - just drop the references,
            # do NOT try to await .close() on them (that would itself
            # require the dead loop). Start completely fresh.
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            self._pages = {}
            self._loop = loop
            logger.info(f"[{self.worker_id}] Real Chromium launched (Cold Start).")
        return self._browser

    async def _get_page(self, affinity_key: str) -> tuple["Page", bool]:
        """Returns (page, was_warm_reused)."""
        browser = await self._ensure_browser()
        if affinity_key and affinity_key in self._pages:
            page = self._pages[affinity_key]
            if not page.is_closed():
                logger.info(f"[{self.worker_id}] Warm-start reuse for '{affinity_key}' (same tab).")
                return page, True
            del self._pages[affinity_key]  # was closed elsewhere - fall through to a fresh one

        page = await browser.new_page()
        if affinity_key:
            self._pages[affinity_key] = page
        return page, False

    async def _execute_real(self, task: TaskNode, action: DispatchAction, url: str) -> TaskExecutionResult:
        try:
            page, warm = await self._get_page(task.affinity_key)
        except (PlaywrightError, RuntimeError) as e:
            return TaskExecutionResult(task.task_id, success=False, error=f"Browser launch failed: {e}")

        try:
            resp = await page.goto(url, timeout=15000, wait_until="domcontentloaded")
        except PlaywrightTimeoutError:
            return TaskExecutionResult(task.task_id, success=False, error=f"Timed out loading {url}")
        except PlaywrightError as e:
            return TaskExecutionResult(task.task_id, success=False, error=f"Navigation failed: {e}")

        if resp is not None and resp.status in (429, 403):
            raise RateLimitedError(f"{url} returned HTTP {resp.status}")

        try:
            title = await page.title()
            body_text = await page.inner_text("body")
        except PlaywrightError as e:
            return TaskExecutionResult(task.task_id, success=False, error=f"Reading page content failed: {e}")

        haystack = f"{title} {body_text[:500]}".lower()
        if any(marker in haystack for marker in _CAPTCHA_MARKERS):
            raise CaptchaBlockedError(f"{url} appears to show a verification/block page")

        if task.affinity_key:
            self.open_domains.add(task.affinity_key)

        output = f"[{url}] {title}\n{_strip_html_text(body_text)}"
        logger.info(f"[{self.worker_id}] {'Warm' if warm else 'Cold'} navigation to {url} OK ({len(body_text)} chars raw).")
        return TaskExecutionResult(task.task_id, success=True, output=output)

    # ------------------------------------------------------------------
    def release_resources(self) -> None:
        """
        Close tabs and free browser memory (called by MemoryJanitor under
        RAM pressure). Contract-mandated to be sync (see BaseWorker), but
        real Playwright cleanup is async - so this only RESETS the
        simulated-mode state synchronously, and schedules the real
        browser's async close as a best-effort background task if one is
        currently running (this is an emergency cleanup path: freeing RAM
        "soon" is the goal, not blocking the caller until it's done).
        """
        self.open_domains.clear()
        self.is_browser_launched = False

        if self._browser is not None:
            browser, playwright = self._browser, self._playwright
            self._browser, self._playwright, self._pages, self._loop = None, None, {}, None
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_close_browser(browser, playwright, self.worker_id))
            except RuntimeError:
                pass  # no running loop right now - nothing to schedule against
        logger.info(f"[{self.worker_id}] Browser context released and memory freed.")

    async def close(self) -> None:
        """
        Explicit, awaited real cleanup - call this when you're done with a
        BrowserWorker for good (end of a script, shutdown), the same way
        ApiWorker.close() works. Without it, an open real Chromium's driver
        subprocess is only torn down by process exit, which can print a
        harmless-but-noisy "Event loop is closed" warning from asyncio's own
        GC-time cleanup. release_resources() (above) is for the OOM-pressure
        case instead - fire-and-forget, not awaited by its caller.
        """
        if self._browser is not None:
            await _close_browser(self._browser, self._playwright, self.worker_id)
            self._browser, self._playwright, self._pages, self._loop = None, None, {}, None


async def _close_browser(browser: "Browser", playwright: "Playwright", worker_id: str) -> None:
    try:
        await browser.close()
        await playwright.stop()
        logger.info(f"[{worker_id}] Real Chromium instance closed.")
    except Exception as e:
        logger.warning(f"[{worker_id}] Error closing browser during cleanup: {e}")
