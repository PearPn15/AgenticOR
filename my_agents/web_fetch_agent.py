"""
Sample custom agent: fetches a real webpage over HTTP, then asks an LLM
(Groq or Gemini) to summarize it in a couple of sentences.

Setup (pick one):
    export GROQ_API_KEY="gsk_..."
    export GEMINI_API_KEY="..."      # GOOGLE_API_KEY also works

Optional:
    export AGENT_LLM_PROVIDER="groq"   # or "gemini" - default: whichever key is set (groq wins if both)
    export AGENT_LLM_MODEL="qwen/qwen3.8-27b"   # or e.g. "gemini-3.5-flash-lite"
    # Google deprecates dated Gemini model names over time (e.g.
    # gemini-2.0-flash / gemini-2.5-flash / gemini-2.5-flash-lite all 404
    # "no longer available" as of this writing) - if the default below ever
    # 404s, list what's live for your key: GET /v1beta/models?key=...

No key set at all? The agent still runs - it just returns the raw fetched
text instead of an LLM summary, so you can test the fetch/dispatch/monitor
flow before wiring up a key.

Each TaskNode this agent runs needs a URL in its metadata:
    TaskNode(..., metadata={"url": "https://example.com"})

See docs/custom-agents.md in the repo for the full BaseWorker contract this
follows, and README at the bottom of this file for how to run it via the CLI.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

import httpx

from agentic_or.models import WorkloadType
from agentic_or.workers.base_worker import (
    BaseWorker,
    TaskExecutionResult,
    RateLimitedError,
)


logger = logging.getLogger("my_agents.web_fetch_agent")


def _strip_html(html: str, max_chars: int = 6000) -> str:
    """Good-enough text extraction without adding a bs4 dependency."""
    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&quot;|&#39;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


class WebFetchSummarizerAgent(BaseWorker):
    """
    Real custom agent (WorkloadType.API - network-bound, not memory-heavy):
    1. GETs `task.metadata["url"]` for real over HTTP.
    2. Sends the extracted text to Groq or Gemini for a short summary.
    Raises RateLimitedError on a real 429/403 from either the target site or
    the LLM provider, so the Orchestrator's DomainCircuitBreaker/ProxyRotator
    handle it exactly like a built-in worker would.
    """

    def __init__(self, worker_id: str):
        super().__init__(worker_id, WorkloadType.API)
        self._client: httpx.AsyncClient | None = None
        # Which running event loop `self._client` was created for. This
        # worker instance can outlive a single `asyncio.run()` call (e.g.
        # the CLI's persistent session Orchestrator reused across multiple
        # `run` commands) - each such call gets its own fresh event loop,
        # and an httpx client silently breaks ("Event loop is closed") if
        # reused from a different one. `_get_client()` recreates it
        # whenever the loop has changed instead of failing on the 2nd `run`.
        self._client_loop: asyncio.AbstractEventLoop | None = None

        groq_key = os.environ.get("GROQ_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.provider = os.environ.get("AGENT_LLM_PROVIDER", "").lower() or ("groq" if groq_key else "gemini")
        self.api_key = groq_key if self.provider == "groq" else gemini_key
        self.model = os.environ.get("AGENT_LLM_MODEL") or (
            "qwen/qwen3.8-27b" if self.provider == "groq" else "gemini-3.5-flash-lite"
        )

        # Reported immediately at construction time (i.e. right when 'agents
        # add' runs) rather than only discovered later when a task fails to
        # summarize - this is the process's own os.environ, inherited at
        # `agentic-or` startup, NOT re-read afterward. If you `export` a key
        # in a different terminal, or after this process already started,
        # it will NOT show up here - restart `agentic-or` in the terminal
        # where the key is actually set.
        if self.api_key:
            masked = self.api_key[:6] + "…" + self.api_key[-4:] if len(self.api_key) > 12 else "…"
            print(f"🔑 [{worker_id}] Detected {self.provider} API key ({masked}), model={self.model}")
        else:
            print(
                f"⚠️  [{worker_id}] No GROQ_API_KEY/GEMINI_API_KEY visible to this process - "
                f"will fetch pages but skip LLM summaries. If you did 'export' one, make sure it "
                f"was in THIS terminal, BEFORE running 'agentic-or' (env vars aren't re-read after startup)."
            )

    async def _get_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            # A real User-Agent avoids a plain "generic HTTP client" being
            # blocked outright (e.g. Wikipedia returns 403 without one).
            self._client = httpx.AsyncClient(
                timeout=20.0,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"},
            )
            self._client_loop = loop
        return self._client

    async def execute_task(self, task, action) -> TaskExecutionResult:
        url = task.metadata.get("url")
        if not url:
            return TaskExecutionResult(task.task_id, success=False, error="task.metadata['url'] is required")

        client = await self._get_client()
        try:
            resp = await client.get(url)
        except httpx.HTTPError as e:
            return TaskExecutionResult(task.task_id, success=False, error=f"Fetch failed: {e}")

        if resp.status_code in (429, 403):
            raise RateLimitedError(f"{url} returned HTTP {resp.status_code}")
        if resp.status_code >= 400:
            return TaskExecutionResult(task.task_id, success=False, error=f"HTTP {resp.status_code} fetching {url}")

        page_text = _strip_html(resp.text)

        if not self.api_key:
            output = (
                f"[No GROQ_API_KEY/GEMINI_API_KEY set - skipped LLM summary] "
                f"Fetched {len(resp.text)} bytes from {url}. First 300 chars:\n{page_text[:300]}"
            )
            logger.info(f"[{self.worker_id}] {output}")
            return TaskExecutionResult(task.task_id, success=True, output=output)

        try:
            summary = await self._summarize(page_text, url)
        except RateLimitedError:
            raise
        except httpx.HTTPStatusError as e:
            return TaskExecutionResult(task.task_id, success=False, error=f"LLM HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            return TaskExecutionResult(task.task_id, success=False, error=f"LLM summarize failed: {e}")

        # Printed directly (not just logged) so the summary is visible right
        # in the CLI/REPL output while 'agents'/--monitor shows this worker
        # as busy, without needing to inspect the Orchestrator in Python.
        print(f"\n📄 [{self.worker_id}] Summary of {url}:\n{summary}\n")
        return TaskExecutionResult(task.task_id, success=True, output=f"[{url}]\n{summary}")

    async def _summarize(self, text: str, url: str) -> str:
        prompt = f"Summarize this webpage ({url}) in 2-3 sentences:\n\n{text}"
        client = await self._get_client()

        if self.provider == "gemini":
            api_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model}:generateContent?key={self.api_key}"
            )
            resp = await client.post(api_url, json={"contents": [{"parts": [{"text": prompt}]}]})
            if resp.status_code in (429, 403):
                raise RateLimitedError(f"Gemini HTTP {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

        # Groq: OpenAI-compatible chat completions
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}]},
        )
        if resp.status_code in (429, 403):
            raise RateLimitedError(f"Groq HTTP {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
