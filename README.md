# Desktop-Agent-OR (`HexaDispatch`)

**OS-Native Multi-Agent Orchestrator** — a desktop automation framework that pairs a high-performance **Operations Research** scheduler written in C++20 with an **asynchronous (asyncio)** agent execution layer written in Python.

The core idea: pure LLM multi-agent frameworks (CrewAI, AutoGen, ...) reason well but cannot solve combinatorial scheduling — they blow up RAM, hit rate limits, and have no real control over machine resources. Pure OR solvers (OR-Tools, VROOM, ...) schedule extremely fast but model the world statically, and cannot adapt to asynchronous events (captchas, expired sessions, changing DOM). Desktop-Agent-OR is the hybrid: a **C++ engine** owns scheduling and multi-dimensional resource allocation in real time, a **Python orchestrator** owns real agent execution and incident self-healing, with PyBind11 between them.

Every feature described below is **implemented and running** in this repository.

---

## Quick start

**Requirements:** Python ≥ 3.11 and a C++20-capable compiler (`g++` ≥ 11 or `clang` ≥ 14) — the scheduling core is a C++ extension compiled at install time.

### Install

```bash
git clone <repo-url> && cd AgentOR
uv sync                      # installs dependencies + builds the C++ extension
```

Without `uv`, use pip inside a virtualenv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Configure an API key

The framework supports several LLM providers. Export a key for **one** of them:

```bash
export GROQ_API_KEY="..."        # or:
export GEMINI_API_KEY="..."      # (GOOGLE_API_KEY is accepted too)
```

If `GROQ_API_KEY` is set, Groq is selected by default; otherwise Gemini is used. To pick a provider explicitly, or change the model:

```bash
export AGENT_LLM_PROVIDER="gemini"          # groq | gemini
export AGENT_LLM_MODEL="gemini-3.5-flash-lite"
```

### Run the autonomous agent

```bash
uv run python my_agents/main_agent.py
```

A chat loop in which **the LLM decides its own action every turn** — search the web, run a shell command, fetch a page, or delegate to a sub-agent — and every action goes through the C++ scheduler and the resource safety guards:

```
You › check how much disk space is free
⚙️  Dispatching: shell `df -h .`  (check free disk space)
✅ Result: ...
🤖 Main Agent: You have ... free on this disk.
```

> ⚠️ This agent runs real commands **with your own permissions on your real machine**. The framework governs *when* and *how many* actions run concurrently, but it does **not** vet *what* a shell command actually does. Don't hand it a goal you wouldn't type into a terminal yourself.

### Run the multi-agent LLM pipeline

```bash
uv run agentic-or llm --provider groq        # gemini | openai | groq | deepseek | openrouter | ollama
```

It reads the key from the `<PROVIDER>_API_KEY` environment variable, or takes one directly via `--key`; change the model with `--model`.

### Watch it run

While an agent is running in your first terminal, open a **second terminal**:

```bash
uv run agentic-or watch      # live view in the terminal
uv run agentic-or ui         # or a web dashboard at http://127.0.0.1:8420
```

Both show the same data: RAM/CPU/battery/temperature, task counts by status, and the full live agent tree (who spawned whom) — refreshed every second.

Running `uv run agentic-or` with no subcommand opens an interactive REPL; type `help` for the full command list.

### Optional: enable real browser automation

```bash
uv sync --extra browser && uv run playwright install chromium
```

Skipping this is fine — browser tasks then run in simulation mode.

To write your own agent, see [docs/custom-agents.md](docs/custom-agents.md) and the [my_agents/](my_agents/) folder.

---

## Features

### 🧠 C++ scheduling core (RCPSP-ALNS + Contextual Bandit)
- Solves the resource-constrained project scheduling problem (**RCPSP**) with **ALNS** (Adaptive Large Neighborhood Search): 4 destroy operators + 4 repair operators, running inside a fixed real-time budget (30–100ms per cycle by default), so it can never stall the Python loop above it.
- Computes the **critical path (CPM)** over the task DAG, so the branch that actually determines total completion time gets prioritized correctly.
- A **contextual bandit** (linear LinUCB over 6 telemetry features) picks the concurrency level (2/4/6/8 workers at once) from the machine's real state at that moment, learning its reward per individual task rather than per batch (so tasks rescued by self-healing don't poison the signal).
- Models **setup time** (the context-switch cost between two consecutive tasks on the same worker — e.g. launching a fresh browser is far more expensive than reusing an open tab) and **affinity keys** (grouping same-domain tasks onto one worker to reuse a "hot" context). Both are honored by the dispatch loop when it makes real placement decisions.
- Scheduling posture is switchable by profile (`TURBO_SPEED` favors raw speed, `ECO_SILENT` heavily penalizes anything that stresses the machine) without touching code.

### ⚙️ Asynchronous rolling dispatch loop
- Not a "wait for the whole batch, then release the next batch" model — a genuine **rolling pipeline**: whichever task finishes first and whichever worker frees up first gets new work immediately, without waiting for the rest of the batch.
- Each cycle: reap finished tasks → sample real machine telemetry → filter tasks through the circuit breaker and OOM guard → let the bandit choose a concurrency cap → have the C++ solver plan for exactly the free slots that actually exist → reserve the worker synchronously (so two tasks can never claim the same one) → dispatch independently, without waiting for the batch to settle.

### 🛡️ Resource safety invariants
- **OOM guard**: when available RAM drops below the warning threshold, new heavy tasks are deferred; below the critical threshold, memory reclamation is forced immediately.
- **Thermal & battery guard**: an overheating machine, or a low battery while unplugged, forces the eco profile (lower concurrency, heavy resource-stress penalty); a healthy, plugged-in machine unlocks maximum speed.
- **Domain circuit breaker**: consecutive 429/403 errors against a domain temporarily blocks it, with exponentially increasing cooldown, so the system never keeps hammering a service that is already refusing it.

### 🩹 Self-healing daemons
- Four shared daemons, invoked automatically when a worker reports a matching incident: `ProxyRotator` (rotate proxies when blocked), `SessionAuthManager` (refresh expired logins), `CaptchaSolver` (handle captcha walls), `MemoryJanitor` (reclaim memory under pressure).
- Four explicitly typed incidents (`CaptchaBlockedError`, `TwoFactorRequiredError`, `AuthExpiredError`, `RateLimitedError`) route each failure to the daemon that actually knows how to handle it, with automatic requeue (up to 3 attempts) for recoverable incidents — and deliberately no blind requeue for rate limits.

### 🧩 Plugin system — write your own agents
- Write real agents of your own by subclassing `BaseWorker`, following a clear 5-point contract (constructor shape, async execution method, result type, honest resource declarations, optional resource release) — without touching the framework's core.
- Custom agents inherit the **entire** scheduling, self-healing and safety infrastructure, exactly like the built-in workers.
- The CLI can plug an agent in two ways: for the current session only, or **persisted**, so it auto-loads on every future startup — including separate, one-off CLI invocations from another terminal.

### 🌳 Dynamic agent tree & runtime spawning (`spawn_agent`)
- No need to declare the full roster of agents upfront. An agent can decide, mid-execution, to spawn a brand-new child agent the system never knew about — and the parent–child relationship is recorded from the real execution flow, not from a static declaration.
- Works **recursively**: a child agent can spawn grandchildren of its own, with no nested orchestration loop needed — the already-running dispatch loop simply picks up the new work.
- A spawned agent is one-shot and retires from the scheduling pool once its task settles, so different agent types sharing a workload category can never be scheduled into each other's work.
- The entire agent tree — including agents created milliseconds ago — is visible live through the monitoring tools.

### 🌐 Real browser automation (Playwright)
- When a task specifies a URL and browser support is enabled, the system drives a real headless Chromium: real navigation, real page content.
- **Reuses an already-open tab** for tasks sharing a hot context (same domain) instead of launching a fresh browser each time — the exact saving the scheduler's setup-time/affinity model accounts for, realized at the execution layer.
- Detects captcha walls and rate-limited responses, feeding them straight into the self-healing flow above.
- Real browsing is **optional**: with no URL in the task, or without Playwright installed, the system runs in simulation mode — useful for exercising the scheduler without a browser environment.

### 📊 Real per-agent resource measurement
- For any agent that wraps a real OS process (running a shell command, for instance), RAM and CPU are **measured for real** by sampling the running process — including the entire tree of child processes it spawns, not just the top-level one.
- Measured numbers are displayed **distinctly** from the figures an agent declared before running, so an estimate is never mistaken for a measurement.

### 👀 Cross-process monitoring (`agentic-or watch`)
- Observe a running agent session **from a completely separate terminal** — no server or socket to start, based on a continuously written shared status file.
- Clearly separates three states that all look like "silence" from the outside: session running, session idle between two actions, and session **crashed, with the specific error**.
- Displays real RAM/CPU/battery/temperature, task counts by status, the full live agent tree, and self-healing state (which domains are blocked, how many captchas are pending, ...).

### 🖥️ Lightweight graphical dashboard (`agentic-or ui`)
- A local web dashboard showing exactly what `watch` shows, in a more readable browser view — with no extra libraries to install and no separate build toolchain (no Node, no Tauri/Electron).
- Refreshes every second: RAM/CPU/battery/temperature cards, a session status banner (running / idle / crashed), and the live agent tree including real measured resources where available.
- Read-only — no buttons or forms that send commands back into the system, consistent with the "observe, don't intervene" philosophy across the whole monitoring layer. Served locally only, never exposed to the network.

### 🤖 Fully autonomous main agent (bundled example)
- A complete, genuinely runnable example: a chat loop against a real LLM in which the model **decides its own action each turn** (search, run a command, fetch a page, delegate to a research sub-agent, ...) — not confined to a fixed, pre-programmed action catalog, much like a coding assistant deciding which tool to call.
- Every action it decides on still passes through the full C++ scheduler and the safety guards above before it executes for real — the framework does not police *what* the agent decides, only guarantees that *how* it executes stays safe for the machine.

---

## Architecture overview

```
CLI / REPL
    │
    ▼
Orchestrator  (Python, asyncio)
    │                              │
    ▼                              ▼
C++ OR Engine                 Agent Registry (tree)
(RCPSP-ALNS + Bandit)         (built at runtime)
    │
    ▼
Worker Pools + Self-Healing Daemons
(LocalWorker · ApiWorker · BrowserWorker · custom agents)
    │
    ▼
Cross-process monitoring (watch)  +  Lightweight dashboard (ui)
```

For a layer-by-layer description, the full diagram and the design decisions behind them, see [Pipeline.md](Pipeline.md).

---

## Implementation status

| Component | Status |
| :--- | :--- |
| C++ RCPSP-ALNS scheduler + contextual bandit | ✅ Implemented, built, tested |
| Asynchronous dispatch loop + safety invariants | ✅ Implemented |
| `LocalWorker`, `ApiWorker` | ✅ Implemented, execute real work |
| Browser automation (Playwright) | ✅ Implemented — optional; simulation mode when not installed |
| Self-healing daemons | ✅ Implemented |
| Custom agent plugin system | ✅ Implemented, tested |
| Dynamic agent tree / `spawn_agent()` | ✅ Implemented, recursion tested |
| Cross-process monitoring (`watch`) | ✅ Implemented, with real measurements where available |
| Lightweight dashboard (`ui`) | ✅ Implemented |
| Autonomous main agent (bundled example) | ✅ Implemented, runs against real LLMs |
| Real resource measurement for subprocess-backed agents | ✅ Implemented |
| Durable checkpoint store | ❌ Not yet — state lives in memory only |
| External message broker (NATS JetStream, ...) | ❌ Not yet — single-process in-memory queue |

A fuller table with explanations is in [Pipeline.md](Pipeline.md).

---

## Documentation

- **Writing your own agents (plugin contract, `agents add`, session vs. persisted modes)**: [docs/custom-agents.md](docs/custom-agents.md)
- **Runnable example agents**: [my_agents/](my_agents/) (see [my_agents/README.md](my_agents/README.md))
- **Minimal example**: [examples/custom_agent.py](examples/custom_agent.py)
