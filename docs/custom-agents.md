# Custom Agents Guide

The built-in workers (`LocalWorker`, `ApiWorker`, `BrowserWorker`) each do
real work when `task.metadata` asks for it — a real shell command, a real
HTTP call, or a real Playwright navigation (see [demo.py](../demo.py) vs.
[test_llm.py](../test_llm.py)). With no such metadata they simulate
instead, `asyncio.sleep()`ing for the task's estimated duration, which is
what makes it possible to exercise the scheduler without touching the
network or launching a browser.

This doc covers how to plug your **own** real agent implementation into the
Orchestrator so it actually gets scheduled, resource-aware-dispatched, and
self-healed like the built-in workers — two ways: programmatically
(`register_worker_pool`) and from the CLI (`agents add`).

**Learning to write one?** Start from
[my_agents/agent_template.py](../my_agents/agent_template.py) — a minimal,
heavily-commented skeleton with no API key/network dependency, runnable
as-is:
```
uv run agentic-or
AgentOR › agents add my_agents/agent_template.py
AgentOR › run my_agents/agent_template_tasks.json
```
For a fuller real-world example (real HTTP fetch + real LLM summarization,
typed-incident self-healing), see
[my_agents/web_fetch_agent.py](../my_agents/web_fetch_agent.py).

## The contract

A custom agent is a subclass of `BaseWorker`
([agentic_or/workers/base_worker.py](../agentic_or/workers/base_worker.py)).
Its full docstring is the source of truth; summarized:

1. `__init__` must call `super().__init__(worker_id, workload_type)` with a
   `worker_id` unique across the whole Orchestrator, and `workload_type`
   one of `WorkloadType.LOCAL` / `.API` / `.BROWSER`. Only these 3 exist —
   they're what the C++ RCPSP-ALNS engine's resource/setup-cost model
   understands; a genuinely new resource category needs the C++ engine
   itself extended (out of scope here).
2. `execute_task(self, task, action)` must be `async def` and must never
   block the event loop — no synchronous network/disk calls; wrap blocking
   work in `asyncio.to_thread(...)`.
3. It must return a `TaskExecutionResult`. On failure, either set `error` (a
   message) or raise a typed incident (below) so the Orchestrator can route
   it to the right Self-Healing Daemon.
4. The `TaskNode`s it executes should carry accurate `ram_mb` /
   `cpu_percent` / `estimated_duration_ms` — the C++ scheduler uses these to
   build the dispatch plan; bad estimates produce a meaningless schedule.
5. Optionally override `release_resources()` to free real resources (close a
   real browser context, drop a big buffer, ...) when `MemoryJanitor` calls
   it on an idle worker under RAM pressure. Default is a no-op.

```python
from agentic_or.workers.base_worker import BaseWorker, TaskExecutionResult
from agentic_or.models import WorkloadType

class MyRealAgent(BaseWorker):
    def __init__(self, worker_id: str):
        super().__init__(worker_id, WorkloadType.API)

    async def execute_task(self, task, action) -> TaskExecutionResult:
        result = await call_my_real_thing(task)
        return TaskExecutionResult(task.task_id, success=True, output=result)
```

A full runnable example (writes real files to disk, wires up a self-healing
round trip) lives at [examples/custom_agent.py](../examples/custom_agent.py) —
run it with `uv run python examples/custom_agent.py`.

## Signaling incidents (self-healing)

Don't rely on stuffing a magic word into `error` — raise one of the typed
exceptions instead:

| Raise this | When | Handled by |
|---|---|---|
| `CaptchaBlockedError` | a page/API is behind a CAPTCHA wall | `CaptchaSolver` — requeues on success |
| `TwoFactorRequiredError` | a login flow needs 2FA | `CaptchaSolver` — requeues on success |
| `AuthExpiredError` | 401 / expired session / expired token | `SessionAuthManager` — requeues on success |
| `RateLimitedError` | 429/403 | trips the domain circuit breaker + rotates proxy — **not auto-requeued**, the task fails for real this attempt |

```python
from agentic_or.workers.base_worker import CaptchaBlockedError

async def execute_task(self, task, action):
    if blocked:
        raise CaptchaBlockedError("verification wall shown")
    ...
```

`run_action()` catches these automatically and sets
`TaskExecutionResult.error_code`, which the Orchestrator checks first; it
falls back to matching on the error message only when no typed code is
present. For `AuthExpiredError`, register how to refresh the
session first:

```python
orchestrator.auth_manager.register_session("session:my_service", refresh_callback=my_refresh_fn)
```

For `CaptchaBlockedError`/`TwoFactorRequiredError`, register an auto-solver
or the daemon falls back to a human-in-the-loop wait (bounded by
`human_timeout_seconds`, default 120s):

```python
orchestrator.captcha_solver.auto_solve_callback = my_auto_solver
```

## Registering it: `register_worker_pool`

```python
orchestrator = Orchestrator(num_api_workers=2)
orchestrator.register_worker_pool(WorkloadType.API, [
    MyRealAgent("my_agent_0"),
    MyRealAgent("my_agent_1"),
])
orchestrator.submit_tasks(tasks)
await orchestrator.run()
```

Call it **before** `submit_tasks()`/`run()`. It validates every instance
(right base class, `workload_type` matches the pool, `worker_id` unique
across every pool) and *replaces* that pool outright — pass the existing
pool plus your new instances yourself if you want to keep some built-in
workers alongside custom ones:

```python
orchestrator.register_worker_pool(WorkloadType.API, orchestrator.api_workers + [MyRealAgent("extra_0")])
```

## CLI: `agents add` / `agents reset` / `agents persisted`

Instead of writing a script, point the CLI/REPL at a `.py` file defining
your agent class(es):

```
AgentOR › agents add my_agent.py --count 2
✅ Registered 2x MyRealAgent as API agent(s): ['myrealagent_0', 'myrealagent_1']
AgentOR › agents
AgentOR › run tasks.json --monitor
```

`agents add <file.py>`:
- `--class X` — which class to use, if the file defines more than one
  (interactive numbered prompt in a real terminal; required flag otherwise)
- `--count N` — how many instances (default 1)
- `--prefix P` — `worker_id` prefix (default: the class name, lowercased)

Two modes:

| | `agents add file.py` | `agents add file.py --persist` |
|---|---|---|
| Stored | in-memory only | `~/.agentic_or/agents.json` |
| Survives | until the process exits | every future `agentic-or` startup |
| Works across separate `bash` invocations | **No** — each is a new process, state is gone | **Yes** — auto-loads before `run`/`agents` does anything |

`agents persisted` lists what's saved. `agents reset` drops the *session's*
custom agents (persisted config untouched — it'll auto-load again next
time); `agents reset --persist` also deletes the saved config.

## Multi-agent pipelines (A before B, B uses A's output)

`predecessors` in a task's JSON controls **ordering only** — task B won't be
considered ready until every task in `predecessors` has status `COMPLETED`.
It does **not** hand A's `TaskExecutionResult.output` to B automatically.
Ordering and data flow are two separate things you wire up yourself:

```python
_RESULTS: dict[str, str] = {}   # module-level "blackboard", shared by both classes below

class CrawlerAgent(BaseWorker):          # Stage 1 - e.g. WorkloadType.BROWSER
    async def execute_task(self, task, action):
        data = await fetch(...)
        _RESULTS[task.task_id] = data     # <- write, keyed by this task's own id
        return TaskExecutionResult(task.task_id, success=True, output=data)

class ProcessorAgent(BaseWorker):        # Stage 2 - e.g. WorkloadType.LOCAL
    async def execute_task(self, task, action):
        upstream_id = task.metadata["read_from"]   # <- which upstream task to read
        data = _RESULTS[upstream_id]                # <- read what Stage 1 wrote
        ...
```

```json
[
  { "task_id": "crawl_1",   "workload_type": "BROWSER", "metadata": {"topic": "..."} },
  { "task_id": "process_1", "workload_type": "LOCAL",
    "predecessors": ["crawl_1"], "metadata": {"read_from": "crawl_1"} }
]
```

`predecessors` guarantees `process_1` doesn't start before `crawl_1`
finishes; `metadata.read_from` is what actually lets `process_1` find
`crawl_1`'s result. A full 2-stage, 2-agent-class working example (tested
end to end) is [my_agents/pipeline_agents.py](../my_agents/pipeline_agents.py)
+ [my_agents/pipeline_tasks.json](../my_agents/pipeline_tasks.json):

```
AgentOR › agents add my_agents/pipeline_agents.py --class CrawlerAgent
AgentOR › agents add my_agents/pipeline_agents.py --class ProcessorAgent
AgentOR › run my_agents/pipeline_tasks.json
```

Registering two different classes **from the same file** like this only
shares module-level state (like `_RESULTS` above) correctly because
`agents add` caches an imported file by path — the same module object is
reused across both calls. A plain dict only survives within one process,
though; if the two stages need to run as **separate `agentic-or`
processes** (e.g. one persisted agent added days apart), swap it for
something that outlives the process - a file, SQLite, or a real key-value
store.

## Generating a pipeline with a "planner" agent instead of hand-writing it

[my_agents/planner_agent.py](../my_agents/planner_agent.py) turns a
natural-language goal into a task-JSON file, using an LLM that's told about
`AGENT_CATALOG` (what agents exist and the `metadata` shape each expects),
then validates the result (schema, duplicate ids, dangling/cyclic
`predecessors`) before writing it:

```
export GROQ_API_KEY="..."   # or GEMINI_API_KEY
uv run python my_agents/planner_agent.py "Research topic X and summarize a reference page about it"
# review the printed summary + my_agents/generated_pipeline.json, THEN:
uv run agentic-or
AgentOR › agents add my_agents/pipeline_agents.py --class CrawlerAgent
AgentOR › agents add my_agents/pipeline_agents.py --class ProcessorAgent
AgentOR › run my_agents/generated_pipeline.json
```

By design it **only writes the file** - it never calls `submit_tasks()`/
`run()` itself. Nothing the LLM invented (a paid API call, a real web fetch,
...) executes without a human reading the plan first. Edit `AGENT_CATALOG`
in the file to describe your own agents once you've written more than the
examples here.

## The Main Agent (fully autonomous, resource-governed)

The planner above is deliberately review-first: it writes a plan, a human
reads it, then runs it. [my_agents/main_agent.py](../my_agents/main_agent.py)
is the other end of the spectrum - **no fixed catalog, no plan file to
review**. An LLM decides real actions turn by turn in a chat loop (run a
shell command, fetch a URL, or reply), like Claude Code deciding to call a
tool - and AgenticOR's role is *not* to restrict what it decides, only to
make sure whatever it decides runs through the same RAM/CPU-governed C++
scheduler as everything else in this framework, instead of as an
unsupervised raw subprocess.

```
export GROQ_API_KEY="..."   # or GEMINI_API_KEY
uv run python my_agents/main_agent.py
You › check how much disk space is free
⚙️  Dispatching: shell `df -h .`  (check free disk space)
✅ Result: ...
🤖 Main Agent: You have ... free on this disk.
```

Watch it live from a **different terminal** while it's running:
```
uv run agentic-or watch
```
This works because every `agentic-or --monitor`/`main_agent.py` process
mirrors its state to `~/.agentic_or/status.json` (write-then-rename, safe to
read concurrently) on every refresh cycle; `watch` is a separate, read-only
process that just polls that file - it needs no Orchestrator of its own, so
it works from any terminal, even one that never ran `agentic-or` before.

**Read this before pointing it at anything real:** "resource-governed" means
AgenticOR's OOMGuard/ThermalBatteryGuard/bandit decide *when* and *how many*
of its actions run concurrently - it does **not** vet *what* a `shell`
action's command actually does. It runs with your real permissions on your
real machine. Don't hand it a goal you wouldn't hand a raw shell prompt.

### Dynamic agent registration: `spawn_agent()` (the agent TREE)

The Main Agent doesn't need a fixed catalog of sub-agents declared upfront.
**Don't call `register_sub_agent()` directly** - it's the low-level
primitive `spawn_agent()` uses internally. `spawn_agent()` is the one
abstraction every caller (Main Agent, or any sub-agent spawning a further
sub-agent) uses identically, regardless of `agent_type`:

```python
orchestrator.register_agent_type("research", ResearchAgent)  # once, at setup

class SomeAgent(BaseWorker):
    async def execute_task(self, task, action):
        # self.orchestrator is set automatically once THIS agent was itself
        # spawned - it's how a sub-agent reaches back into AgenticOR to
        # spawn its OWN children, discovered purely from execution.
        handle = await self.orchestrator.spawn_agent(
            parent_id=self.worker_id, agent_type="research",
            task={"name": "look into X", "metadata": {"url": "https://..."}},
        )
        result = await handle.result()
        ...
```

`spawn_agent(parent_id, agent_type, task)` does, in order: validate
`agent_type` is known (`register_agent_type` first) → allocate an id →
instantiate + register the agent (`register_sub_agent` internally, which
attaches `parent_id` in `orchestrator.agent_registry` - a parent → child
tree, independent of the flat LOCAL/API/BROWSER pools that actually get
scheduled) → build a `TaskNode` for `task` → submit it → return an
`AgentHandle` to `await .result()` on. Works identically called at the top
level or recursively - no nested `run()` needed for the recursive case, the
already-active dispatch loop just picks up the new task.

`agentic-or watch` renders the resulting tree live (indented, parent above
child) from another terminal - see
[my_agents/main_agent.py](../my_agents/main_agent.py)'s `ResearchAgent` for
a real 3-level example (`main_agent -> research_N -> fetch_M`, the fetch
sub-agent spawned entirely from `ResearchAgent`'s own code) and
[tests/test_spawn_agent.py](../tests/test_spawn_agent.py) for the
`Main -> {Research -> {Search, Browser}, Coding -> {Test, Debug}}` shape
tested directly.

**Agents of the same `WorkloadType` share one real scheduling pool** - a
`spawn_agent()`-created agent is a **one-shot**: once its task settles, it's
automatically retired from that pool (it stays in `agent_registry` for the
tree/history) so it can never later catch some unrelated task just because
it's idle and the right type. Without this, any idle worker of the same
`WorkloadType` - including a plain default `LocalWorker` - could be handed
a task meant for a specific spawned agent, since the scheduler matches on
workload type, not on agent identity.

**Resource tracking caveat:** every agent here is an `asyncio` coroutine in
ONE Python process, not a separate OS process - so per-agent RAM/CPU in the
tree is whatever the agent's current task *declares* (`ram_mb`/`cpu_percent`
on the `TaskNode`). Don't read that number as a live measurement.

The exception is an agent wrapping a real OS subprocess - currently
`LocalWorker` running a `metadata["command"]` - which has a real PID and so
*is* measured for real: `psutil` samples RSS and CPU across the whole
process tree while it runs, surfacing as
`TaskExecutionResult.measured_resources` and shown separately in
`agentic-or watch` as `[measured: ...]`, so a measured number is never
confused with a declared one.

### Constructor limitation

Agents added via `agents add` are instantiated as `worker_cls(worker_id)` —
**only** `worker_id`, nothing else. If your class needs an API key, a file
path, etc., read it from an environment variable or a config file inside
`__init__`/`execute_task`, not from a constructor argument:

```python
class MyRealAgent(BaseWorker):
    def __init__(self, worker_id: str):
        super().__init__(worker_id, WorkloadType.API)
        self.api_key = os.environ["MY_SERVICE_API_KEY"]  # not a ctor param
```

### Gotchas

- **Session mode only shares state within one REPL session.** Two separate
  `agentic-or agents add ...` / `agentic-or run ...` calls from bash are two
  separate processes and share nothing — use `--persist`, or do both inside
  one `agentic-or` REPL session.
- **A shared broker across multiple `run` calls in one session accumulates
  tasks.** `history`/task counts become cumulative across runs sharing the
  session Orchestrator; re-submitting the same `task_id` twice raises a
  clear error rather than silently corrupting its checkpoint. Run
  `run <file.json> --fresh` to reset just the task queue (keeps your
  registered agents), or `agents reset` for a full clean slate.
- **A broken persisted entry (moved file, renamed class, new required ctor
  arg) is skipped with a warning at startup, not fatal** — the rest of the
  CLI still works.
- **Don't stash a long-lived async resource (`httpx.AsyncClient`,
  `aiohttp.ClientSession`, a Playwright browser handle, ...) on `self` and
  assume it lives forever.** A worker instance can outlive a single
  `asyncio.run()` call once session/persist mode is in play - each `run` is
  its own fresh event loop, and such resources silently break ("Event loop
  is closed") if reused from a different one than they were created on.
  Recreate it lazily, keyed by the currently running loop:
  ```python
  async def _get_client(self) -> httpx.AsyncClient:
      loop = asyncio.get_running_loop()
      if self._client is None or self._client_loop is not loop:
          self._client = httpx.AsyncClient(...)
          self._client_loop = loop
      return self._client
  ```
  This is exactly what `ApiWorker` and
  [my_agents/web_fetch_agent.py](../my_agents/web_fetch_agent.py) do — copy
  the pattern for any similar resource.
