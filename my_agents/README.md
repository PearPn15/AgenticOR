# my_agents/ — build, run, and watch your own AgentOR agent

This folder is where **your** agents live — nothing in here is part of the
AgenticOR framework itself (that's `agentic_or/`). Everything below is a
worked example you can copy, or run as-is to see it work first.

Full reference: [docs/custom-agents.md](../docs/custom-agents.md). This
file is the fast path to "I want to build one right now."

## What's already here

| File | Level | Teaches | Try it |
|---|---|---|---|
| `agent_template.py` | Start here | The bare `BaseWorker` contract - no API key, no network | `agents add my_agents/agent_template.py` |
| `web_fetch_agent.py` | Real agent | A complete real agent: HTTP fetch + LLM summary | `agents add my_agents/web_fetch_agent.py` |
| `pipeline_agents.py` | Multi-agent | Two *different* agent classes, one feeding the other's output | `agents add my_agents/pipeline_agents.py --class CrawlerAgent` (+ `ProcessorAgent`) |
| `planner_agent.py` | Advanced | An LLM plans a task DAG; a human reviews it before anything runs | `python my_agents/planner_agent.py "your goal"` |
| `main_agent.py` | Advanced | Fully autonomous chat agent - decides actions and spawns real sub-agents live | `python my_agents/main_agent.py` |

Each `*_tasks.json` next to an agent file is a ready-to-run task list for
it (`run my_agents/<name>_tasks.json` in the REPL).

## Build your own agent in 6 steps

Copy [agent_template.py](agent_template.py) and fill in the TODOs, or start
from scratch following this contract (full version: `BaseWorker`'s own
docstring in [agentic_or/workers/base_worker.py](../agentic_or/workers/base_worker.py)):

```python
from agentic_or.models import WorkloadType
from agentic_or.workers.base_worker import BaseWorker, TaskExecutionResult

class MyAgent(BaseWorker):
    # 1. Pick ONE type: LOCAL (light/local work), API (network calls),
    #    or BROWSER (heavy, session-based).
    def __init__(self, worker_id: str):
        super().__init__(worker_id, WorkloadType.LOCAL)
        # 2. (optional) one-time setup - config from an env var, NOT a ctor
        #    arg (agents added via the CLI only ever get worker_id).

    # 3. The actual work. `task.metadata` is a plain dict YOU define.
    async def execute_task(self, task, action) -> TaskExecutionResult:
        value = task.metadata.get("some_key")
        # 4. MUST be real async work - never a blocking call.
        result = await do_something_real(value)
        # 5. Return success or failure (or raise a typed WorkerIncident -
        #    CaptchaBlockedError/AuthExpiredError/RateLimitedError - to
        #    trigger self-healing instead of just failing).
        return TaskExecutionResult(task.task_id, success=True, output=result)

    # 6. (optional) override release_resources() to free real resources
    #    under RAM pressure. Default is a safe no-op - delete if unused.
```

## Run it — two modes

### Mode A: you write the task list, review it, then run

```
uv run agentic-or
AgentOR › agents add my_agents/my_agent.py
AgentOR › run my_agents/my_tasks.json
```
Add `--persist` to `agents add` to make it auto-load on every future
`agentic-or` startup, not just this session.

### Mode B: fully autonomous, agent decides + spawns sub-agents at runtime

The `main_agent.py` pattern - see it for the full working example. The gist:

```python
orchestrator.register_agent_type("my_type", MyAgent)   # once, at setup

handle = await orchestrator.spawn_agent(
    parent_id=some_parent_id, agent_type="my_type",
    task={"name": "...", "metadata": {"some_key": "..."}},
)
result = await handle.result()
```
`spawn_agent()` validates the type, creates + registers the agent (attaching
it under `parent_id` in the agent tree), submits its task, and returns a
handle - usable from the top level OR recursively from inside another
agent's own `execute_task` (an agent spawning a further agent).

## Watch it running — from ANOTHER terminal

```bash
uv run agentic-or watch
```
Shows the live agent tree (parent → child, who spawned whom), RAM/CPU,
task counts, and whatever the running agent is currently doing - refreshes
every second, reads a shared status file so it works from a **completely
separate terminal/process**, no setup needed beyond the command itself.

If nothing shows: the agent process needs to actually be running with
`--monitor` (a `run --monitor` command) or be `main_agent.py` (which always
mirrors). A `❌ SESSION CRASHED: ...` line reports the actual exception that
ended the watched agent process, together with its message.

## Gotchas

- **Constructor**: agents added via `agents add`/`spawn_agent` are built as
  `YourClass(worker_id)` - no other required arguments.
- **Long-lived async resources** (an `httpx.AsyncClient` held in `__init__`)
  break across separate `run()`/process-lifetime boundaries with "Event
  loop is closed" unless recreated per-loop - see `FetchAgent` in
  `main_agent.py` for the pattern.
- **Don't feed a whole raw web page to an LLM** - a single page easily runs
  to tens of thousands of tokens and will blow a model's context limit.
  Strip the HTML and cap the length (see `_strip_html` in
  `main_agent.py`/`web_fetch_agent.py`).
- **`predecessors` in a task JSON controls ORDER only** - it does not pass
  data between tasks. See "Multi-agent pipelines" in
  [docs/custom-agents.md](../docs/custom-agents.md) for how `pipeline_agents.py`
  wires actual data flow.
