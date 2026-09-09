from __future__ import annotations

import asyncio
import logging
import httpx
from typing import Optional
from agentic_or.models import TaskNode, DispatchAction, WorkloadType
from agentic_or.workers.base_worker import BaseWorker, TaskExecutionResult

logger = logging.getLogger(__name__)


class ApiWorker(BaseWorker):
    """
    Worker for network requests, async API calls, and real LLM inference.
    Supports:
    - Gemini API (default model: gemini-3.5-flash-lite - Google deprecates
      dated model names over time; pass task.metadata["model"] to override,
      or check `GET /v1beta/models?key=...` for what's currently live)
    - OpenAI-compatible endpoints (OpenAI, Groq, DeepSeek, OpenRouter, Ollama)
    - Generic REST HTTP APIs with rate limit tracking.
    """

    def __init__(
        self,
        worker_id: str,
        client: Optional[httpx.AsyncClient] = None,
        default_llm_provider: Optional[str] = None,
        default_api_key: Optional[str] = None,
    ):
        super().__init__(worker_id, WorkloadType.API)
        self._client = client
        # Which running event loop `self._client` was created for. A worker
        # instance can outlive a single `asyncio.run()` call (e.g. the CLI's
        # persistent session Orchestrator across multiple `run` commands) -
        # each such call gets its own fresh event loop, and an httpx client
        # (like any asyncio-bound resource) silently breaks ("Event loop is
        # closed") if reused from a different one. `_get_client()` below
        # transparently recreates it whenever the loop has changed instead
        # of failing on the second `run`.
        self._client_loop: Optional[asyncio.AbstractEventLoop] = None
        self.default_llm_provider = default_llm_provider
        self.default_api_key = default_api_key

    async def _get_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = httpx.AsyncClient(timeout=30.0)
            self._client_loop = loop
        return self._client

    async def _call_gemini_api(self, prompt: str, api_key: str, model: str = "gemini-3.5-flash-lite") -> TaskExecutionResult:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        max_retries = 3
        backoff = 2.0
        client = await self._get_client()
        for attempt in range(max_retries):
            resp = await client.post(url, json=payload)
            if resp.status_code in (429, 403):
                if attempt < max_retries - 1:
                    logger.warning(f"[ApiWorker] Gemini HTTP {resp.status_code}. Backing off {backoff:.1f}s (Attempt {attempt+1}/{max_retries})...")
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
                return TaskExecutionResult(
                    "", success=False, error=f"Rate Limit 429: {resp.status_code}"
                )
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return TaskExecutionResult("", success=True, output=text)

    async def _call_openai_compatible(
        self, prompt: str, api_key: str, base_url: str, model: str
    ) -> TaskExecutionResult:
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        max_retries = 5
        backoff = 3.0
        client = await self._get_client()
        for attempt in range(max_retries):
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code in (429, 403):
                retry_after_str = resp.headers.get("retry-after")
                wait_time = backoff
                if retry_after_str:
                    try:
                        wait_time = max(float(retry_after_str), backoff)
                    except ValueError:
                        pass
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[ApiWorker] Groq/OpenAI HTTP {resp.status_code}. "
                        f"Backing off {wait_time:.1f}s (Attempt {attempt+1}/{max_retries})... "
                        f"Detail: {resp.text[:80]}"
                    )
                    await asyncio.sleep(wait_time)
                    backoff = max(backoff * 1.5, 5.0)
                    continue
                return TaskExecutionResult(
                    "", success=False, error=f"Rate Limit 429: {resp.text}"
                )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return TaskExecutionResult("", success=True, output=text)

    async def execute_task(self, task: TaskNode, action: DispatchAction) -> TaskExecutionResult:
        # 1. Check if this is an LLM inference task
        llm_prompt = task.metadata.get("llm_prompt") or task.metadata.get("prompt")
        if llm_prompt:
            import os
            provider = task.metadata.get("llm_provider") or self.default_llm_provider or "gemini"

            # Support mock/offline provider for immediate testing without API key
            if provider.lower() in ("mock", "test", "simulated"):
                await asyncio.sleep(0.2)
                return TaskExecutionResult(
                    task.task_id,
                    success=True,
                    output=(
                        f"🤖 [Mock AI Response for '{task.name}']:\n"
                        f"• Insight: Schedule optimized by the C++ RCPSP-ALNS engine (~150k iters/50ms).\n"
                        f"• Resource safety: RAM={task.ram_mb}MB and CPU={task.cpu_percent}% both within limits.\n"
                        f"• Status: Multi-Agent LLM DAG context successfully linked."
                    )
                )

            api_key = (
                task.metadata.get("api_key")
                or self.default_api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("GROQ_API_KEY")
            )

            try:
                if provider.lower() in ("gemini", "google"):
                    if not api_key:
                        return TaskExecutionResult(
                            task.task_id,
                            success=False,
                            error="Missing GEMINI_API_KEY or GOOGLE_API_KEY"
                        )
                    model = task.metadata.get("model", "gemini-3.5-flash-lite")
                    res = await self._call_gemini_api(llm_prompt, api_key, model=model)
                    res.task_id = task.task_id
                    return res

                elif provider.lower() in ("openai", "groq", "openrouter", "deepseek", "ollama"):
                    base_url = task.metadata.get("base_url")
                    if not base_url:
                        if provider.lower() == "groq":
                            base_url = "https://api.groq.com/openai/v1"
                        elif provider.lower() == "openrouter":
                            base_url = "https://openrouter.ai/api/v1"
                        elif provider.lower() == "deepseek":
                            base_url = "https://api.deepseek.com/v1"
                        elif provider.lower() == "ollama":
                            base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434/v1")
                        else:
                            base_url = "https://api.openai.com/v1"

                    model = task.metadata.get("model")
                    if not model:
                        if provider.lower() == "groq":
                            model = "qwen/qwen3.8-27b"
                        elif provider.lower() == "deepseek":
                            model = "deepseek-chat"
                        elif provider.lower() == "ollama":
                            model = "llama3"
                        else:
                            model = "gpt-4o-mini"

                    res = await self._call_openai_compatible(llm_prompt, api_key or "", base_url, model)
                    res.task_id = task.task_id
                    return res

            except httpx.HTTPStatusError as e:
                return TaskExecutionResult(task.task_id, success=False, error=f"HTTP {e.response.status_code}: {e.response.text}")
            except Exception as e:
                return TaskExecutionResult(task.task_id, success=False, error=str(e))

        # 2. Check if generic REST URL
        url = task.metadata.get("url")
        method = task.metadata.get("method", "GET").upper()

        try:
            if url:
                headers = task.metadata.get("headers", {})
                json_data = task.metadata.get("json")
                client = await self._get_client()
                resp = await client.request(method, url, headers=headers, json=json_data)
                
                if resp.status_code in (429, 403):
                    return TaskExecutionResult(
                        task.task_id,
                        success=False,
                        error=f"HTTP Rate Limit Error: {resp.status_code}"
                    )
                resp.raise_for_status()
                return TaskExecutionResult(task.task_id, success=True, output=resp.text)

            # Simulated network / API inference latency
            delay = task.estimated_duration_ms / 1000.0
            await asyncio.sleep(min(0.2, delay))
            return TaskExecutionResult(
                task.task_id,
                success=True,
                output=f"API Response: {task.name or task.task_id}"
            )

        except httpx.HTTPStatusError as e:
            return TaskExecutionResult(task.task_id, success=False, error=f"HTTP {e.response.status_code}")
        except Exception as e:
            return TaskExecutionResult(task.task_id, success=False, error=str(e))

    async def close(self):
        if self._client is not None:
            await self._client.aclose()

