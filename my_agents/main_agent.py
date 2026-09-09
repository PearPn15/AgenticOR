"""
MAIN AGENT: an autonomous, chat-driven agent (like Claude Code deciding to
call tools/subagents on its own) - except every real action it decides to
take is dispatched through `Orchestrator.spawn_agent()`, the single unified
entry point every agent (Main Agent, or any sub-agent spawning a further
sub-agent) uses identically - no per-agent-type registration logic needed
anywhere. spawn_agent() internally: validates the agent_type, allocates an
id, instantiates + registers the agent (attaching it under its parent in the
agent tree), builds a TaskNode for the work, submits it to the C++-scheduled
Orchestrator, and returns a handle to await the result.

    User -> Main Agent -> spawn_agent(agent_type=...) -> AgenticOR
                              (validate/register/attach-parent/submit/schedule)
                                          |
                    Agent A ---spawn_agent()---> Agent C (auto-discovered)

AgenticOR does NOT restrict WHAT the LLM decides to do - that's the whole
point (per the design discussion this file came out of: "it should behave
like Claude Code, fully autonomous; AgenticOR's job is just to keep it
resource-safe"). It only governs HOW/WHEN actions actually run: OOMGuard can
hold a heavy action back if RAM is low, ThermalBatteryGuard can force
ECO_SILENT and serialize everything if the machine is hot/low battery, and
the bandit caps concurrency dynamically.

IMPORTANT - read before running: "safe" here means resource-safe, NOT
content-safe. A `shell` action runs a REAL subprocess on THIS machine with
YOUR permissions - AgenticOR does not vet what the command does, only how
much RAM/CPU it's allowed to consume while doing it. Don't hand this a goal
you wouldn't hand a shell prompt to.

Setup:
    export GROQ_API_KEY="..."   # or GEMINI_API_KEY
    uv run python my_agents/main_agent.py

While it's running, watch the whole agent TREE live from ANOTHER terminal:
    uv run agentic-or watch
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import urllib.parse
from typing import Optional

import httpx

from agentic_or.models import WorkloadType
from agentic_or.orchestrator import Orchestrator
from agentic_or.telemetry.live_monitor import write_session_status, write_status_fields
from agentic_or.workers.base_worker import BaseWorker, TaskExecutionResult
from agentic_or.workers.local_worker import LocalWorker

SYSTEM_PROMPT = """You are the Main Agent of a local automation system called AgentOR.
You decide what to do; the system underneath you enforces RAM/CPU safety - you don't
need to worry about resource limits, just decide the right action.

On EVERY turn, respond with ONLY one raw JSON object (no markdown fences, no prose),
one of these five shapes:

1. Search the web for something (no URL needed - use this when you don't already
   know the exact page to look at):
   {"action": "search", "query": "<search query>", "why": "<short reason>"}

2. Run a real shell command on this machine:
   {"action": "shell", "command": "<a real shell command>", "why": "<short reason>"}

3. Fetch a URL over real HTTP (use this once you have a specific URL, e.g. from a
   prior "search" result):
   {"action": "fetch", "url": "<https url>", "why": "<short reason>"}

4. Research a URL (fetches it AND summarizes word count/length - a sub-agent that
   itself spawns a further fetch sub-agent to do this):
   {"action": "research", "url": "<https url>", "why": "<short reason>"}

5. Reply to the user in chat (use this when you have your final answer, or to ask
   a clarifying question, or after an action's result already answers the goal):
   {"action": "reply", "text": "<your message to the user>"}

After a "search"/"shell"/"fetch"/"research" action, you will be given the real result
as an "observation" and asked to decide the NEXT action - keep deciding actions until
you have enough information, then use "reply". A typical flow for an open-ended
question is: "search" for it first, THEN "fetch" or "research" the most relevant
result URL from the search observation. Don't repeat an action that already gave you
what you need.
"""


class SearchAgent(BaseWorker):
    """
    Real web search (WorkloadType.API) via DuckDuckGo's HTML endpoint - no
    API key needed. Returns the top results as "N. <title> - <url>" lines,
    which the LLM reads as an observation and can then "fetch"/"research"
    from directly.
    """

    def __init__(self, worker_id: str):
        super().__init__(worker_id, WorkloadType.API)
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        # Same loop-aware recreation pattern as WebFetchSummarizerAgent /
        # ApiWorker - see docs/custom-agents.md's "Don't stash a long-lived
        # async resource" gotcha.
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = httpx.AsyncClient(
                timeout=15.0, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
            )
            self._client_loop = loop
        return self._client

    async def execute_task(self, task, action) -> TaskExecutionResult:
        query = task.metadata.get("query")
        if not query:
            return TaskExecutionResult(task.task_id, success=False, error="metadata.query is required")

        client = await self._get_client()
        try:
            resp = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
        except httpx.HTTPError as e:
            return TaskExecutionResult(task.task_id, success=False, error=f"Search failed: {e}")
        if resp.status_code != 200:
            return TaskExecutionResult(task.task_id, success=False, error=f"Search HTTP {resp.status_code}")

        matches = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', resp.text, re.S)
        results = []
        for href, raw_title in matches[:5]:
            title = re.sub(r"<[^>]+>", "", raw_title).strip()
            # DuckDuckGo's HTML results link through a redirect wrapping the
            # real URL in ?uddg=<url-encoded> - decode it so a downstream
            # "fetch"/"research" action gets a directly usable URL.
            parsed = urllib.parse.urlparse(href if href.startswith("http") else f"https:{href}")
            real_url = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
            results.append(f"{title} - {real_url}")

        if not results:
            return TaskExecutionResult(task.task_id, success=True, output=f"No results for '{query}'.")

        formatted = "\n".join(f"{i+1}. {r}" for i, r in enumerate(results))
        return TaskExecutionResult(task.task_id, success=True, output=f"Search results for '{query}':\n{formatted}")


def _strip_html(html: str, max_chars: int = 3000) -> str:
    """Good-enough text extraction, same approach as web_fetch_agent.py -
    no bs4 dependency. Hard-capped: a raw page can be tens of KB, and this
    text ends up verbatim in the next LLM turn's context - an uncapped
    fetch caused a real 413 "Request too large" (61839 tokens vs. a 7000
    limit) once, so this cap is load-bearing, not cosmetic."""
    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&quot;|&#39;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"... [truncated, {len(text)} chars total]"
    return text


class FetchAgent(BaseWorker):
    """
    Real HTTP GET (WorkloadType.API) for a page an LLM is going to read
    directly - unlike the generic built-in ApiWorker (meant for arbitrary
    REST/JSON calls, returns raw response text uncapped), this strips HTML
    to plain text and hard-truncates it (see _strip_html) before it can
    reach the chat context. Also sends a real User-Agent - Wikipedia (and
    many other sites) return 403 to a client that doesn't send one, which
    is exactly what using the generic ApiWorker for this hit in practice.
    """

    def __init__(self, worker_id: str):
        super().__init__(worker_id, WorkloadType.API)
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = httpx.AsyncClient(
                timeout=20.0, follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"},
            )
            self._client_loop = loop
        return self._client

    async def execute_task(self, task, action) -> TaskExecutionResult:
        url = task.metadata.get("url")
        if not url:
            return TaskExecutionResult(task.task_id, success=False, error="metadata.url is required")

        client = await self._get_client()
        try:
            resp = await client.get(url)
        except httpx.HTTPError as e:
            return TaskExecutionResult(task.task_id, success=False, error=f"Fetch failed: {e}")

        if resp.status_code in (429, 403):
            from agentic_or.workers.base_worker import RateLimitedError
            raise RateLimitedError(f"{url} returned HTTP {resp.status_code}")
        if resp.status_code >= 400:
            return TaskExecutionResult(task.task_id, success=False, error=f"HTTP {resp.status_code} fetching {url}")

        return TaskExecutionResult(task.task_id, success=True, output=_strip_html(resp.text))


class ResearchAgent(BaseWorker):
    """
    Demonstrates REAL recursion: this agent's own execute_task calls
    self.orchestrator.spawn_agent(agent_type="fetch", ...) to dynamically
    create and delegate to a further sub-agent that was never declared
    anywhere upfront - the exact "Agent A spawns Agent C" shape from the
    design this came out of (docs/custom-agents.md#the-main-agent), driven
    by real main_agent.py code, not just a test.
    """

    def __init__(self, worker_id: str):
        super().__init__(worker_id, WorkloadType.LOCAL)

    async def execute_task(self, task, action) -> TaskExecutionResult:
        url = task.metadata.get("url")
        if not url:
            return TaskExecutionResult(task.task_id, success=False, error="metadata.url is required")

        handle = await self.orchestrator.spawn_agent(
            parent_id=self.worker_id,
            agent_type="fetch",
            task={"name": f"fetch for research: {url[:30]}", "metadata": {"url": url}},
        )
        try:
            page_text = str(await handle.result())
        except RuntimeError as e:
            return TaskExecutionResult(task.task_id, success=False, error=f"fetch sub-agent failed: {e}")

        word_count = len(re.findall(r"\w+", page_text))
        summary = f"{url}: fetched {len(page_text)} chars, ~{word_count} words."
        return TaskExecutionResult(task.task_id, success=True, output=summary)


def _get_llm_config():
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    provider = os.environ.get("AGENT_LLM_PROVIDER", "").lower() or ("groq" if groq_key else "gemini")
    api_key = groq_key if provider == "groq" else gemini_key
    model = os.environ.get("AGENT_LLM_MODEL") or (
        "qwen/qwen3.8-27b" if provider == "groq" else "gemini-3.5-flash-lite"
    )
    return provider, api_key, model


class LLMCallError(RuntimeError):
    """Raised by _call_llm after retries are exhausted - carries the real
    response body (not just the status code) so a failure is diagnosable
    from the chat transcript alone, without needing to reproduce it."""


async def _call_llm(
    client: httpx.AsyncClient, provider: str, api_key: str, model: str, messages: list[dict],
    max_retries: int = 3,
) -> str:
    """
    A single LLM provider hiccup must not kill the whole chat session (it
    used to: an uncaught HTTPStatusError from here crashed handle_goal, and
    with it the entire main_agent.py process - caught by hitting a real,
    apparently-transient 400 from Groq mid-session that a same-request
    retry moments later returned 200 OK for). Retries transient-looking
    errors (429, 5xx) with backoff; a non-retryable error (4xx other than
    429) or exhausted retries raises LLMCallError with the real response
    body included, instead of a bare status code.
    """
    backoff = 1.5
    last_error = ""

    for attempt in range(max_retries):
        try:
            if provider == "gemini":
                # Gemini has no separate "system" role - fold it into the first user turn.
                contents = []
                for m in messages:
                    role = "model" if m["role"] == "assistant" else "user"
                    contents.append({"role": role, "parts": [{"text": m["content"]}]})
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                resp = await client.post(url, json={"contents": contents})
            else:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model, "messages": messages},
                )
        except httpx.HTTPError as e:
            last_error = f"network error: {e}"
        else:
            if resp.status_code == 200:
                data = resp.json()
                if provider == "gemini":
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                return data["choices"][0]["message"]["content"]

            last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
            if resp.status_code not in (429, 500, 502, 503, 504):
                # Not the kind of error a retry is likely to fix (bad
                # request, bad auth, ...) - fail now with the real body.
                raise LLMCallError(last_error)

        if attempt < max_retries - 1:
            print(f"⚠️  LLM call failed ({last_error[:120]}) - retrying in {backoff:.1f}s "
                  f"({attempt + 1}/{max_retries})...", file=sys.stderr)
            await asyncio.sleep(backoff)
            backoff *= 2

    raise LLMCallError(f"Gave up after {max_retries} attempts. Last error: {last_error}")


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


# Hard cap on any action's result before it's fed back into the LLM's own
# context as an observation - see _dispatch_action. A real run once sent a
# whole raw HTML page through and got a 413 "Request too large" (61839
# tokens vs. a 7000 limit) from the LLM provider.
_MAX_OBSERVATION_CHARS = 4000

# Maps the LLM's "action" field to a spawn_agent() agent_type + how to build
# that type's task metadata from the decided action dict. Adding a new kind
# of action is exactly: one new BaseWorker subclass + one line here - no
# other code needs to change (that's the point of the unified spawn_agent API).
_ACTION_TO_AGENT_TYPE = {
    "search": ("search", lambda a: {"query": a["query"]}),
    "shell": ("shell", lambda a: {"command": a["command"]}),
    "fetch": ("fetch", lambda a: {"url": a["url"]}),
    "research": ("research", lambda a: {"url": a["url"]}),
}


class MainAgent:
    def __init__(self):
        self.provider, self.api_key, self.model = _get_llm_config()
        if not self.api_key:
            print("⚠️  No GROQ_API_KEY/GEMINI_API_KEY set - the Main Agent needs one to decide anything.", file=sys.stderr)
            sys.exit(1)

        # One persistent Orchestrator for the whole chat session: every
        # decided action goes through it (real RAM/CPU governance carries
        # across turns), and monitor_render_to_terminal=False keeps its
        # dashboard OUT of this terminal (this terminal is the chat) while
        # still mirroring the whole agent tree to the status file for
        # `agentic-or watch`.
        # num_*_workers=0: every LOCAL/API task here goes through an
        # explicitly spawn_agent()'d, named agent_type - NOT a generic
        # default worker. Leaving the built-in default pools non-empty
        # would let the C++ scheduler route a task to one of THEM instead
        # of the specific agent spawn_agent() just registered for it (they
        # share the same WorkloadType pool, and slot assignment isn't
        # guaranteed to prefer the newest one) - caught by testing this
        # file directly: a "research" action was silently executed by a
        # plain default LocalWorker instead of ResearchAgent.
        self.orchestrator = Orchestrator(
            num_local_workers=0, num_api_workers=0, num_browser_workers=0,
            enable_live_monitor=True,
            monitor_refresh_seconds=0.5,
            monitor_render_to_terminal=False,
            alns_time_budget_ms=30,
        )

        # The capability catalog spawn_agent() validates against - reuses
        # the ALREADY-REAL LocalWorker (metadata.command does real
        # subprocess work, see its own docstring) for "shell"; SearchAgent/
        # FetchAgent/ResearchAgent above are purpose-built for this file
        # (real User-Agent, HTML stripped + truncated for the chat context -
        # the generic built-in ApiWorker returns raw, uncapped response
        # text, which is right for a REST/JSON caller but caused a real 413
        # "Request too large" when a whole raw HTML page reached the LLM).
        self.orchestrator.register_agent_type("search", SearchAgent)
        self.orchestrator.register_agent_type("shell", LocalWorker)
        self.orchestrator.register_agent_type("fetch", FetchAgent)
        self.orchestrator.register_agent_type("research", ResearchAgent)

        # A purely symbolic tree root ("Main" in the design's diagram) -
        # schedulable=False so it's a valid parent_id for every action this
        # session spawns, WITHOUT ever being eligible to receive a real
        # dispatched task itself.
        root = LocalWorker("main_agent")
        self.root_id = self.orchestrator.register_sub_agent(root, parent_id=None, schedulable=False)

        self._spawn_seq = 0
        self._http = httpx.AsyncClient(timeout=30.0)

    async def _dispatch_action(self, action: dict) -> str:
        """Spawn the agent this decided action needs (via the SAME
        spawn_agent() API every agent uses), run it through the Orchestrator
        (RAM/CPU-governed, tree-tracked), and return a plain-text observation."""
        kind = action.get("action")
        mapping = _ACTION_TO_AGENT_TYPE.get(kind)
        if mapping is None:
            return f"ERROR: unknown action type '{kind}'"
        agent_type, build_metadata = mapping

        self._spawn_seq += 1
        label = f"{kind} -> {action.get('command') or action.get('url') or action.get('query')}"
        why = action.get("why", "")
        print(f"⚙️  spawn_agent(agent_type='{agent_type}')  {label}" + (f"  ({why})" if why else ""))
        # write_status_fields (not orchestrator.set_monitor_extra_status)
        # writes to the status file immediately - a `watch`-ing terminal
        # sees this the instant the action starts, not only once some
        # monitor cycle happens to render during the (possibly very short)
        # run() call below.
        write_status_fields(current_action=label)
        self.orchestrator.set_monitor_extra_status(current_action=label)

        handle = await self.orchestrator.spawn_agent(
            parent_id=self.root_id,
            agent_type=agent_type,
            task={
                "task_id": f"main_action_{self._spawn_seq}",
                "name": label[:60],
                "estimated_duration_ms": 500,
                "metadata": build_metadata(action),
            },
        )
        await self.orchestrator.run()

        try:
            result = str(await handle.result())
            print(f"✅ Result: {result[:300]}")
            # Defense in depth on top of FetchAgent's own truncation: NO
            # agent's result reaches the LLM's context uncapped, regardless
            # of type - this is what should have stopped the real 413
            # "Request too large" (61839 tokens vs. a 7000 limit) even if
            # some other agent later forgets to cap its own output.
            if len(result) > _MAX_OBSERVATION_CHARS:
                result = result[:_MAX_OBSERVATION_CHARS] + f"... [truncated, {len(result)} chars total]"
            return result
        except RuntimeError as e:
            print(f"❌ Failed: {e}")
            return f"FAILED: {e}"
        finally:
            # Clear the stale "currently: X" label now that this action has
            # actually settled - otherwise a `watch`-ing terminal keeps
            # showing "is currently: <last action>" forever, indistinguishable
            # from it genuinely still running (caught by watching a real run
            # after it had already finished). Written immediately (see
            # write_status_fields's docstring) - no monitor is active in
            # this exact gap (between one run() ending and the next
            # starting), so relying on set_monitor_extra_status alone would
            # leave this update sitting unwritten until some future render.
            write_status_fields(current_action=f"idle (last: {label})")
            self.orchestrator.set_monitor_extra_status(current_action=f"idle (last: {label})")

    async def handle_goal(self, goal: str) -> None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ]
        # Gemini has no "system" role - the caller folds it into the first
        # user turn (see _call_llm), so keep both providers working from
        # the same `messages` list regardless.

        for _ in range(8):  # hard cap: never loop forever on a confused LLM
            try:
                raw = await _call_llm(self._http, self.provider, self.api_key, self.model, messages)
            except LLMCallError as e:
                # A provider failure ends THIS goal, not the whole chat
                # session - the REPL loop in main() keeps running so the
                # user can just try again.
                print(f"❌ LLM call failed: {e}", file=sys.stderr)
                return
            cleaned = _strip_markdown_fences(raw)
            try:
                action = json.loads(cleaned)
            except json.JSONDecodeError:
                print(f"🤖 Main Agent (unstructured): {raw}")
                return

            if action.get("action") == "reply":
                print(f"🤖 Main Agent: {action.get('text', '')}")
                return

            messages.append({"role": "assistant", "content": cleaned})
            observation = await self._dispatch_action(action)
            messages.append({"role": "user", "content": f"Observation: {observation}\n\nDecide the next action."})

        print("🤖 Main Agent: (stopped after 8 actions without a final reply - ask a narrower question)")


async def main() -> None:
    agent = MainAgent()
    # session_active=True the whole time this process is up - a `watch`ing
    # terminal can now tell "briefly idle between actions" (session_active
    # stays True; only the per-run monitor_active flickers) apart from
    # "this process crashed or exited" (session_active=False, with the
    # real error attached if it wasn't a clean exit/Ctrl+C).
    write_session_status(active=True)
    print("🧠 Main Agent ready. Every decided action spawns a real agent through")
    print("   Orchestrator.spawn_agent() - RAM/CPU-governed, C++-scheduled, tree-tracked.")
    print("   Watch the whole agent tree live: `agentic-or watch` in another terminal.")
    print("   Type your goal, or 'exit' to quit.\n")

    crash_reason: Optional[str] = None
    try:
        while True:
            goal = input("You › ").strip()
            if goal.lower() in ("exit", "quit"):
                break
            if not goal:
                continue
            await agent.handle_goal(goal)
            print()
    except (EOFError, KeyboardInterrupt):
        print()
    except Exception as e:
        crash_reason = f"{type(e).__name__}: {e}"
        print(f"❌ Main Agent crashed: {crash_reason}", file=sys.stderr)
    finally:
        write_session_status(active=False, error=crash_reason)
        await agent._http.aclose()
        print("👋 Main Agent stopped.")


if __name__ == "__main__":
    asyncio.run(main())
