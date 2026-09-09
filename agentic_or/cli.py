from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Type

from agentic_or.banner import print_banner
from agentic_or.broker.local_queue import AsyncLocalBroker
from agentic_or.models import TaskNode, WorkloadType
from agentic_or.orchestrator import Orchestrator
from agentic_or.telemetry.live_monitor import format_agents_table, STATUS_FILE_PATH
from agentic_or.telemetry.system_monitor import SystemTelemetryMonitor
from agentic_or.workers.base_worker import BaseWorker

# `_SESSION_ORCHESTRATOR` (below) is the opt-in exception to the note above:
# once a user registers a custom agent via `agents add`, that Orchestrator
# instance - and its worker pools/broker - is kept alive and reused by every
# subsequent `run` in the same CLI/REPL process, instead of each `run`
# building a disposable one. Nothing changes for anyone who never touches
# `agents add`: `run` still builds a fresh Orchestrator exactly as before.
#
# The Orchestrator only exists for the lifetime of one blocking `run`/`demo`/
# `llm` call - there is no persistent background daemon. So `agents` (below)
# is intentionally real-time only: once a pipeline finishes and control comes
# back to the prompt, there are, by definition, 0 Agents running right now.
# `_LAST_ORCHESTRATOR` exists solely so `agents` can tell a *finished* run
# (0 running, but N configured) apart from *no run at all* (nothing to show).
# Past-run summaries live separately in `_RUN_HISTORY`, surfaced by `history`.
_LAST_ORCHESTRATOR: Optional[Orchestrator] = None
_RUN_HISTORY: List[dict] = []

# Set only once the user registers at least one custom agent via `agents
# add`. From then on it IS `_LAST_ORCHESTRATOR` too (same object) and every
# `run` reuses it instead of building a disposable Orchestrator. `agents
# reset` drops it back to None.
_SESSION_ORCHESTRATOR: Optional[Orchestrator] = None

# Which WorkloadType pools `agents add` has already taken over this session.
# The FIRST `agents add` for a given type replaces that type's built-in
# simulated pool outright (otherwise the custom agent would just sit
# alongside 4 fake ApiWorkers and might never actually be the one picked for
# a task); subsequent adds for the same type accumulate on top of that.
_CUSTOMIZED_WORKLOAD_TYPES: set = set()

_POOL_ATTR = {
    WorkloadType.LOCAL: "local_workers",
    WorkloadType.API: "api_workers",
    WorkloadType.BROWSER: "browser_workers",
}

# Mode 2 ("hard add"): `agents add ... --persist` writes an entry here
# instead of (well, in addition to) only living in this process's memory.
# Every future `agentic-or` invocation - REPL OR a one-shot call from bash,
# a fresh process each time - auto-loads this file the first time it needs
# an Orchestrator, so a persisted agent is active from the moment the tool
# starts, not just within the REPL session that added it (see
# `_apply_persisted_agent_specs` / `_get_or_create_session_orchestrator`).
_PERSIST_CONFIG_PATH = Path.home() / ".agentic_or" / "agents.json"


def _remember_orchestrator(orchestrator: Optional[Orchestrator]) -> None:
    global _LAST_ORCHESTRATOR
    if orchestrator is not None:
        _LAST_ORCHESTRATOR = orchestrator


def _load_persisted_agent_specs() -> List[dict]:
    if not _PERSIST_CONFIG_PATH.exists():
        return []
    try:
        data = json.loads(_PERSIST_CONFIG_PATH.read_text())
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"⚠️  Ignoring corrupt {_PERSIST_CONFIG_PATH}: {e}", file=sys.stderr)
        return []


def _has_persisted_agents() -> bool:
    return bool(_load_persisted_agent_specs())


def _persist_agent_spec(file: str, class_name: str, count: int, prefix: str) -> None:
    """Upsert one entry (keyed by file+class) into the persisted config."""
    specs = _load_persisted_agent_specs()
    specs = [s for s in specs if not (s.get("file") == file and s.get("class") == class_name)]
    specs.append({"file": file, "class": class_name, "count": count, "prefix": prefix})
    _PERSIST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PERSIST_CONFIG_PATH.write_text(json.dumps(specs, indent=2))


def _apply_persisted_agent_specs(orchestrator: Orchestrator) -> None:
    """
    Auto-register every persisted agent spec into `orchestrator`. Called once
    when the session Orchestrator is first created. A broken entry (file
    moved/deleted, class renamed, constructor now needs new args, ...) is
    skipped with a warning rather than crashing the whole CLI on startup.
    """
    for spec in _load_persisted_agent_specs():
        file = spec.get("file", "")
        try:
            worker_cls = _resolve_worker_class(Path(file), spec.get("class"), interactive=False)
            instances = _instantiate_workers(worker_cls, spec.get("count", 1), spec.get("prefix") or worker_cls.__name__.lower())
        except SystemExit:
            print(f"⚠️  Skipped persisted agent from {file} (see error above).", file=sys.stderr)
            continue
        except Exception as e:
            print(f"⚠️  Skipped persisted agent from {file}: {e}", file=sys.stderr)
            continue

        workload_type = instances[0].workload_type
        prefix = spec.get("prefix") or worker_cls.__name__.lower()
        if workload_type in _CUSTOMIZED_WORKLOAD_TYPES:
            existing = [
                w for w in getattr(orchestrator, _POOL_ATTR[workload_type])
                if not w.worker_id.startswith(f"{prefix}_")
            ]
        else:
            existing = []
        orchestrator.register_worker_pool(workload_type, existing + instances)
        _CUSTOMIZED_WORKLOAD_TYPES.add(workload_type)
        print(f"🔌 Auto-loaded {len(instances)}x {worker_cls.__name__} ({workload_type.value}) from persisted config.")


def _get_or_create_session_orchestrator() -> Orchestrator:
    """The Orchestrator instance `agents add`/`agents reset` operate on, and
    that `run` reuses once it exists. Created lazily on the first `agents
    add` - or the first thing that needs an Orchestrator at all, if there
    are persisted agents to auto-load (see `_apply_persisted_agent_specs`)."""
    global _SESSION_ORCHESTRATOR
    if _SESSION_ORCHESTRATOR is None:
        _SESSION_ORCHESTRATOR = Orchestrator()
        _CUSTOMIZED_WORKLOAD_TYPES.clear()
        _remember_orchestrator(_SESSION_ORCHESTRATOR)
        _apply_persisted_agent_specs(_SESSION_ORCHESTRATOR)
    return _SESSION_ORCHESTRATOR


# Cache of already-imported agent files, keyed by resolved absolute path.
# Without this, `agents add file.py --class A` followed by
# `agents add file.py --class B` (a real pipeline pattern: several
# specialized agent classes in one file, e.g. my_agents/pipeline_agents.py)
# would exec the file's top-level code TWICE, producing two independent
# module objects - any state those classes share at module level (a "shared
# results" dict between pipeline stages, a shared httpx client, ...) would
# silently NOT actually be shared. Reusing the same module object for the
# same file fixes that, and avoids re-running top-level side effects twice.
_LOADED_MODULES: Dict[str, object] = {}


def _load_module_from_path(path: Path):
    key = str(path.resolve())
    if key in _LOADED_MODULES:
        return _LOADED_MODULES[key]

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import a module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # let the file's own imports resolve normally
    spec.loader.exec_module(module)
    _LOADED_MODULES[key] = module
    return module


def _discover_worker_classes(module) -> Dict[str, Type[BaseWorker]]:
    """BaseWorker subclasses *defined in* `module` (not merely imported into
    it, e.g. `from agentic_or.workers.local_worker import LocalWorker`), and
    concrete (no leftover unimplemented abstract methods)."""
    found: Dict[str, Type[BaseWorker]] = {}
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, BaseWorker)
            and obj is not BaseWorker
            and obj.__module__ == module.__name__
            and not inspect.isabstract(obj)
        ):
            found[name] = obj
    return found


def _resolve_worker_class(path: Path, class_name: Optional[str], interactive: bool = True) -> Type[BaseWorker]:
    """
    Import `path`, find its BaseWorker subclasses, and resolve to exactly
    one: via `class_name` if given, the sole match if there's only one, or -
    only when `interactive` and stdin is a real TTY - a numbered prompt.
    Raises SystemExit (with a message already printed) on any failure, so
    callers that must not let a bad entry kill the whole process (persisted
    auto-load) catch SystemExit around this call.
    """
    if not path.exists():
        print(f"❌ File not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        module = _load_module_from_path(path)
    except Exception as e:
        print(f"❌ Failed to import {path}: {e}", file=sys.stderr)
        sys.exit(1)

    classes = _discover_worker_classes(module)
    if not classes:
        print(f"❌ No BaseWorker subclass found in {path}.", file=sys.stderr)
        print("    Your file needs: class MyAgent(BaseWorker): ...", file=sys.stderr)
        sys.exit(1)

    if class_name is None:
        if len(classes) == 1:
            class_name = next(iter(classes))
        elif interactive and sys.stdin.isatty():
            names = list(classes)
            print(f"Found {len(names)} agent classes in {path.name}:")
            for i, n in enumerate(names, 1):
                print(f"  [{i}] {n}")
            choice = input(f"Choose [1-{len(names)}]: ").strip()
            try:
                class_name = names[int(choice) - 1]
            except (ValueError, IndexError):
                print("❌ Invalid choice.", file=sys.stderr)
                sys.exit(1)
        else:
            print(
                f"❌ {path.name} defines multiple agent classes ({', '.join(classes)}); "
                f"pass --class to pick one.",
                file=sys.stderr,
            )
            sys.exit(1)

    worker_cls = classes.get(class_name)
    if worker_cls is None:
        print(f"❌ Class '{class_name}' not found in {path.name}. Available: {', '.join(classes)}", file=sys.stderr)
        sys.exit(1)
    return worker_cls


def _instantiate_workers(worker_cls: Type[BaseWorker], count: int, prefix: str) -> List[BaseWorker]:
    """Create `count` instances named `{prefix}_0`, `{prefix}_1`, ... Raises
    SystemExit (message already printed) if the constructor needs more than
    just `worker_id` - see the contract in BaseWorker's docstring."""
    instances: List[BaseWorker] = []
    for i in range(max(1, count)):
        worker_id = f"{prefix}_{i}"
        try:
            instances.append(worker_cls(worker_id))
        except TypeError as e:
            print(
                f"❌ Could not instantiate {worker_cls.__name__}('{worker_id}') - its __init__ needs extra "
                f"required arguments ({e}).\n"
                f"    Agents added via 'agents add' must be constructible from worker_id alone - "
                f"read any extra config (API keys, paths, ...) from an env var or a config file "
                f"inside your class instead.",
                file=sys.stderr,
            )
            sys.exit(1)
    return instances


def _record_history(command: str, orchestrator: Optional[Orchestrator], elapsed_seconds: float) -> None:
    """Append one completed-run summary. This is the ONLY place run history is written."""
    if orchestrator is None:
        return
    broker = orchestrator.broker
    _RUN_HISTORY.append({
        "time": time.strftime("%H:%M:%S"),
        "command": command,
        "total_tasks": len(broker._tasks),
        "completed": len(broker._completed_tasks),
        "failed": len(broker._failed_tasks),
        "agents_breakdown": (
            f"{len(orchestrator.local_workers)}L/"
            f"{len(orchestrator.api_workers)}A/"
            f"{len(orchestrator.browser_workers)}B"
        ),
        "elapsed_seconds": elapsed_seconds,
    })

# demo.py / test_llm.py live at the repo root, not inside the agentic_or package.
# When invoked through the installed `agentic-or` console script (rather than
# `python demo.py` or `python -m ...`), the repo root is not on sys.path, so
# `import demo` fails. Make it resolvable regardless of how the CLI was launched.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _print_status_snapshot(monitor: SystemTelemetryMonitor, clear: bool = False) -> None:
    snap = monitor.capture_snapshot()
    if clear:
        sys.stdout.write("\x1b[H\x1b[2J")
    print("=" * 60)
    print("       🖥️  DESKTOP-AGENT-OR: SYSTEM TELEMETRY" + ("   (watching, Ctrl+C to stop)" if clear else ""))
    print("=" * 60)
    print(f"  • RAM Available  : {snap.ram_free_mb:.1f} MB / {snap.ram_total_mb:.1f} MB ({snap.ram_free_ratio * 100:.1f}%)")
    print(f"  • CPU Load       : {snap.cpu_load_1m * 100:.1f}%")
    print(f"  • CPU Usage      : {snap.cpu_percent:.1f}%")
    print(f"  • Battery Level  : {snap.battery_percent:.1f}% (Charging: {snap.is_charging})")
    print(f"  • SoC Temperature: {snap.cpu_temperature_c:.1f}°C")
    print(f"  • Bandit Feature : {snap.to_bandit_feature_vector()}")
    print("=" * 60)


def cmd_status(args) -> None:
    """Check and display instantaneous OS hardware telemetry (optionally --watch continuously)."""
    monitor = SystemTelemetryMonitor()
    if not getattr(args, "watch", False):
        _print_status_snapshot(monitor)
        return

    interval = max(0.2, getattr(args, "interval", 1.0))
    try:
        while True:
            _print_status_snapshot(monitor, clear=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def _render_status_file(data: dict) -> str:
    sep = "=" * 70
    dash = "-" * 70
    tasks = data.get("tasks", {})
    healing = data.get("self_healing", {})
    workers = data.get("workers", [])

    # session_active/session_error (set by a long-lived caller like
    # main_agent.py via write_session_status - see live_monitor.py) track
    # the WHOLE process's lifecycle. monitor_active only reflects one
    # Orchestrator.run() call's dispatch loop and flips back to True on the
    # very next call - on its own it can't tell "briefly idle between
    # actions" apart from "the process crashed or was killed". Prefer the
    # session-level signal whenever a caller actually reports one; fall back
    # to monitor_active alone for a caller that doesn't (e.g. a plain
    # `agentic-or run --monitor`, which is one Orchestrator.run() call and
    # nothing more).
    session_active = data.get("session_active")
    session_error = data.get("session_error")
    active = data.get("monitor_active", False)

    if session_active is False and session_error:
        header = f"AGENTOR WATCH — ❌ SESSION CRASHED: {session_error[:40]}"
    elif session_active is False:
        header = "AGENTOR WATCH — session ended (exited normally)"
    elif active:
        header = "AGENTOR WATCH (cross-terminal)"
    else:
        header = "AGENTOR WATCH — idle between actions, showing last known state"

    lines = [
        sep,
        f"  {header:<48}{data.get('time', '')}   Profile: {data.get('profile', '?')}",
    ]
    if session_active is False and session_error:
        lines.append(f"  Full error: {session_error}")
    lines += [
        sep,
        f"  RAM  {data.get('ram_free_mb', 0):.0f}/{data.get('ram_total_mb', 0):.0f} MB free   "
        f"CPU {data.get('cpu_percent', 0):.1f}%   "
        f"Battery {data.get('battery_percent', 0):.1f}% "
        f"({'charging' if data.get('is_charging') else 'on battery'})   "
        f"Temp {data.get('cpu_temperature_c', 0):.1f}°C",
        dash,
        f"  Tasks  total={tasks.get('total', 0)}  pending={tasks.get('pending', 0)}  "
        f"running={tasks.get('running', 0)}  completed={tasks.get('completed', 0)}  "
        f"failed={tasks.get('failed', 0)}",
        dash,
        f"  Self-Healing  circuit-broken={healing.get('circuit_broken_domains', [])}  "
        f"captcha_pending={healing.get('captcha_pending', 0)}  "
        f"locked_sessions={healing.get('locked_sessions', 0)}",
        sep,
    ]

    current_action = data.get("current_action")
    if current_action:
        lines += [f"  Main agent: {current_action}", sep]

    agent_tree = data.get("agent_tree")
    if agent_tree:
        lines.append("  AGENT TREE  (who registered whom - built at runtime, not declared upfront)")
        lines.append(dash)
        for node in agent_tree:
            lines.extend(_render_tree_node(node, depth=0))
        lines.append(sep)
    elif workers:
        # Fallback for a status file written before agent_tree existed.
        id_w = max([len(w["worker_id"]) for w in workers] + [len("WORKER ID")])
        lines.append(f"  {'WORKER ID':<{id_w}}  TYPE      STATUS   CURRENT TASK")
        for w in workers:
            status = "● busy" if w.get("busy") else "○ idle"
            lines.append(f"  {w['worker_id']:<{id_w}}  {w['type']:<9} {status:<7}  {w.get('current_task') or '-'}")
        lines.append(sep)

    return "\n".join(lines)


def _render_tree_node(node: dict, depth: int) -> List[str]:
    indent = "  " + "  " * depth + ("└─ " if depth else "")
    status = "● RUNNING" if node.get("status") == "RUNNING" else "○ idle"
    task = node.get("current_task") or "-"
    line = f"{indent}{node['agent_id']}  [{node.get('type', '?')}]  {status}  task={task}"
    measured = node.get("measured_resources")
    if measured:
        # REAL psutil measurement (only ever set for an agent that spawned
        # a real OS subprocess) - marked "measured" so it's never confused
        # with a TaskNode's declared ram_mb/cpu_percent estimate.
        line += f"  [measured: {measured.get('peak_ram_mb', 0):.1f}MB, {measured.get('avg_cpu_percent', 0):.1f}% CPU]"
    out = [line]
    for child in node.get("children", []):
        out.extend(_render_tree_node(child, depth + 1))
    return out


def cmd_watch(args) -> None:
    """
    Watch another `agentic-or` process's live run from THIS terminal - reads
    the status file that process's LiveMonitor mirrors to disk (see
    agentic_or/telemetry/live_monitor.py's STATUS_FILE_PATH). Needs that
    other process to be running with `--monitor` (or be main_agent.py, which
    always mirrors). Does not need - and does not create - an Orchestrator
    of its own; this is read-only from a completely separate process.
    """
    interval = max(0.2, getattr(args, "interval", 1.0))
    try:
        while True:
            sys.stdout.write("\x1b[H\x1b[2J")
            if not STATUS_FILE_PATH.exists():
                print(f"⏳ No status file yet at {STATUS_FILE_PATH}.")
                print("   Run something with --monitor (or main_agent.py) in another terminal first.")
            else:
                try:
                    data = json.loads(STATUS_FILE_PATH.read_text())
                    print(_render_status_file(data))
                except (json.JSONDecodeError, OSError):
                    print("⏳ Status file is mid-write, retrying...")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def cmd_ui(args) -> None:
    """
    Lightweight LOCAL web dashboard - a browser-based `watch`. Same data
    source (~/.agentic_or/status.json), same read-only guarantee, just a
    nicer view - see agentic_or/webui.py. Pure Python stdlib, no new
    dependency, nothing to build/bundle.
    """
    from agentic_or.webui import run_ui
    run_ui(port=getattr(args, "port", 8420), open_browser=not getattr(args, "no_browser", False))


async def _run_tasks(tasks: List[TaskNode], alns_ms: int = 50, enable_monitor: bool = False, fresh: bool = False):
    # Reuse the session Orchestrator (custom agents from `agents add`, or
    # auto-loaded from a persisted config even in a brand-new process) if
    # either already applies; otherwise build a disposable one exactly as
    # before this feature existed - zero behavior change for anyone who
    # never touches `agents add`.
    if _SESSION_ORCHESTRATOR is not None or _has_persisted_agents():
        orchestrator = _get_or_create_session_orchestrator()
        if fresh:
            # Reset just the task queue/checkpoints so re-running the same
            # file (same task_ids) in this session doesn't hit the
            # duplicate-task_id guard in AsyncLocalBroker.register_tasks -
            # registered custom agents, the bandit's learned weights, and
            # daemon state (circuit breaker cooldowns, auth sessions, ...)
            # are untouched.
            orchestrator.broker = AsyncLocalBroker()
    else:
        orchestrator = Orchestrator()
    orchestrator.alns_time_budget_ms = alns_ms
    orchestrator.enable_live_monitor = enable_monitor
    orchestrator.submit_tasks(tasks)
    res = await orchestrator.run()
    return res, orchestrator


def cmd_run(args) -> None:
    """Execute a task pipeline defined in a JSON file."""
    path = Path(args.file)
    if not path.exists():
        print(f"❌ Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print("❌ Error: JSON file must contain a list of task objects.", file=sys.stderr)
        sys.exit(1)

    tasks: List[TaskNode] = []
    for item in data:
        tasks.append(TaskNode(
            task_id=item["task_id"],
            name=item.get("name", item["task_id"]),
            workload_type=WorkloadType(item.get("workload_type", "LOCAL")),
            predecessors=item.get("predecessors", []),
            estimated_duration_ms=item.get("estimated_duration_ms", 1000),
            ram_mb=item.get("ram_mb", 100),
            cpu_percent=item.get("cpu_percent", 10.0),
            affinity_key=item.get("affinity_key", ""),
            metadata=item.get("metadata", {})
        ))

    print(f"🚀 Loaded {len(tasks)} tasks from {path.name}. Starting orchestration...")
    t0 = time.time()
    try:
        res, orchestrator = asyncio.run(
            _run_tasks(tasks, alns_ms=args.budget, enable_monitor=args.monitor, fresh=getattr(args, "fresh", False))
        )
    except ValueError as e:
        # e.g. a duplicate task_id when reusing the session Orchestrator's
        # broker across multiple 'run' calls (see AsyncLocalBroker.register_tasks).
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    _remember_orchestrator(orchestrator)
    _record_history("run", orchestrator, time.time() - t0)
    print(f"✅ Workflow completed in {res['elapsed_seconds']:.2f}s (Total: {res['total_tasks']} tasks)")


def cmd_demo(args) -> None:
    """Run the built-in 11-task demonstration."""
    try:
        import demo
    except ModuleNotFoundError:
        # demo.py is development scaffolding kept out of the published
        # package - every other command works without it.
        print("❌ 'demo' needs demo.py at the repo root, which isn't part of this "
              "installation.", file=sys.stderr)
        print("   Try 'run <file.json>' with your own tasks, or 'llm --provider <name>'.",
              file=sys.stderr)
        return
    t0 = time.time()
    orchestrator = asyncio.run(demo.main(enable_monitor=args.monitor))
    _remember_orchestrator(orchestrator)
    _record_history("demo", orchestrator, time.time() - t0)


def cmd_agents(args) -> None:
    """Dispatch 'agents' [status] / 'agents add' / 'agents reset' / 'agents persisted'."""
    action = getattr(args, "agents_action", None)
    if action == "add":
        cmd_agents_add(args)
    elif action == "reset":
        cmd_agents_reset(args)
    elif action == "persisted":
        cmd_agents_persisted(args)
    else:
        cmd_agents_status(args)


def cmd_agents_status(args) -> None:
    """
    Show Agents/Workers running RIGHT NOW - real-time only, never history.
    Once a pipeline finishes there are, by definition, 0 Agents running - use
    'history' to see summaries of runs that already completed. If any custom
    agents were registered via 'agents add', the roster is shown even while
    idle, so you can confirm what's registered before running anything.
    """
    if _SESSION_ORCHESTRATOR is not None or _has_persisted_agents():
        print(format_agents_table(_get_or_create_session_orchestrator()))
        return

    busy_workers = []
    if _LAST_ORCHESTRATOR is not None:
        all_workers = (
            _LAST_ORCHESTRATOR.local_workers
            + _LAST_ORCHESTRATOR.api_workers
            + _LAST_ORCHESTRATOR.browser_workers
        )
        busy_workers = [w for w in all_workers if w.is_busy]

    if not busy_workers:
        print("🟢 0 Agents running right now.")
        print("    Pass --monitor to 'demo'/'run'/'llm' to watch Agents live while they run.")
        print("    Type 'agents add <file.py>' to plug in your own agent.")
        if _RUN_HISTORY:
            print("    Type 'history' to see runs already completed in this session.")
        return

    print(format_agents_table(_LAST_ORCHESTRATOR))


def cmd_agents_add(args) -> None:
    """
    Load a .py file, find the BaseWorker subclass(es) it defines, instantiate
    `--count` of them, and register them into the session Orchestrator (kept
    alive so the next 'run' actually uses them). See BaseWorker's docstring
    (agentic_or/workers/base_worker.py) for the full contract a custom
    class must satisfy.

    Two modes:
    - (default) session-only: registered agents live only in this process's
      memory - gone once you exit the REPL / the one-shot command returns.
    - `--persist`: ALSO written to ~/.agentic_or/agents.json, so every future
      `agentic-or` invocation (REPL or a fresh one-shot call from bash) picks
      it up automatically before doing anything else - no need to stay in
      the same session. Use 'agents persisted' to see what's stored, and
      'agents reset --persist' to remove it again.
    """
    path = Path(args.file)
    worker_cls = _resolve_worker_class(path, args.class_name)

    prefix = args.prefix or worker_cls.__name__.lower()
    count = max(1, args.count)
    instances = _instantiate_workers(worker_cls, count, prefix)

    workload_type = instances[0].workload_type
    orchestrator = _get_or_create_session_orchestrator()

    # First time this workload_type is customized this session: replace the
    # built-in simulated pool outright rather than appending alongside it
    # (see _CUSTOMIZED_WORKLOAD_TYPES above). Later adds of the same type
    # accumulate on top of what was already registered - EXCEPT any earlier
    # instances using this exact prefix, which this add replaces outright
    # (so re-running 'agents add' for the same agent with a new --count
    # doesn't collide with its own previous worker_ids; this also covers
    # re-persisting an already-persisted class, since _get_or_create_session_
    # orchestrator() auto-loads the OLD persisted entry moments before this
    # runs).
    if workload_type in _CUSTOMIZED_WORKLOAD_TYPES:
        existing = [
            w for w in getattr(orchestrator, _POOL_ATTR[workload_type])
            if not w.worker_id.startswith(f"{prefix}_")
        ]
    else:
        existing = []

    try:
        orchestrator.register_worker_pool(workload_type, existing + instances)
    except (TypeError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    _CUSTOMIZED_WORKLOAD_TYPES.add(workload_type)

    print(
        f"✅ Registered {count}x {worker_cls.__name__} as {workload_type.value} agent(s): "
        f"{[w.worker_id for w in instances]}"
    )
    print(format_agents_table(orchestrator))

    if getattr(args, "persist", False):
        _persist_agent_spec(str(path.resolve()), worker_cls.__name__, count, prefix)
        print(f"💾 Persisted to {_PERSIST_CONFIG_PATH} - will auto-load on every 'agentic-or' startup from now on.")
    else:
        print("    Session-only: gone when this process exits. Pass --persist to make it permanent.")


def cmd_agents_reset(args) -> None:
    """
    Drop the session's custom agents - the next 'run' goes back to the
    built-in simulated workers. Session-only by default; pass --persist to
    ALSO remove the ~/.agentic_or/agents.json config, so nothing auto-loads
    on future startups either.
    """
    global _SESSION_ORCHESTRATOR
    wipe_persisted = getattr(args, "persist", False)
    had_session = _SESSION_ORCHESTRATOR is not None

    _SESSION_ORCHESTRATOR = None
    _CUSTOMIZED_WORKLOAD_TYPES.clear()
    # Also drop the cached agent-file imports, so if you edited a .py file
    # between resets, the next 'agents add' actually re-reads it from disk
    # instead of reusing the stale in-memory module.
    _LOADED_MODULES.clear()

    if wipe_persisted:
        if _PERSIST_CONFIG_PATH.exists():
            _PERSIST_CONFIG_PATH.unlink()
            print("✅ Cleared session agents AND the persisted config - nothing auto-loads on next startup.")
        else:
            print("✅ Cleared session agents. (No persisted config existed.)")
        return

    if not had_session:
        print("ℹ️  No custom agents registered this session - nothing to reset.")
    else:
        print("✅ Session agents cleared. The next 'run'/'demo'/'llm' uses the default built-in workers again.")
    if _has_persisted_agents():
        print("    Note: persisted agents still exist and will auto-load again - 'agents reset --persist' to drop those too.")


def cmd_agents_persisted(args) -> None:
    """List what's in ~/.agentic_or/agents.json - the Mode-2 'hard add' registry."""
    specs = _load_persisted_agent_specs()
    if not specs:
        print(f"ℹ️  No persisted agents ({_PERSIST_CONFIG_PATH} is empty/missing).")
        print("    Use 'agents add <file.py> --persist' to add one.")
        return

    print(f"Persisted agents (auto-loaded on every 'agentic-or' startup, from {_PERSIST_CONFIG_PATH}):")
    for s in specs:
        print(f"  - {s.get('count', 1)}x {s.get('class')}  ({s.get('file')})")


def cmd_history(args) -> None:
    """Show summaries of pipelines that already finished in this session - NOT real-time."""
    if not _RUN_HISTORY:
        print("ℹ️  No run history yet in this session. Try 'demo', 'run <file.json>', or 'llm' first.")
        return

    sep = "─" * 72
    print(sep)
    print(f"  {'TIME':<10}{'CMD':<7}{'TASKS':<10}{'AGENTS':<10}{'ELAPSED':<10}STATUS")
    print(sep)
    for e in _RUN_HISTORY:
        status = "✅ OK" if e["failed"] == 0 else f"⚠️  {e['failed']} failed"
        tasks_str = f"{e['completed']}/{e['total_tasks']}"
        elapsed_str = f"{e['elapsed_seconds']:.2f}s"
        print(f"  {e['time']:<10}{e['command']:<7}{tasks_str:<10}{e['agents_breakdown']:<10}{elapsed_str:<10}{status}")
    print(sep)


def cmd_llm(args) -> None:
    """Run the real multi-agent LLM pipeline against a chosen provider."""
    import test_llm

    default_models = {
        "gemini": "gemini-3.5-flash-lite",
        "openai": "gpt-4o-mini",
        "groq": "qwen/qwen3.8-27b",
        "deepseek": "deepseek-chat",
        "openrouter": "google/gemini-2.0-flash-exp:free",
        "ollama": "llama3",
    }
    model = args.model or default_models.get(args.provider, "gemini-3.5-flash-lite")
    api_key = args.key or os.environ.get(f"{args.provider.upper()}_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key and args.provider not in ("ollama", "mock"):
        print(f"❌ Please provide an API key via --key or export {args.provider.upper()}_API_KEY='...'", file=sys.stderr)
        sys.exit(1)
    t0 = time.time()
    orchestrator = asyncio.run(test_llm.run_llm_pipeline(args.provider, api_key or "", model))
    _remember_orchestrator(orchestrator)
    _record_history("llm", orchestrator, time.time() - t0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-or",
        description="OS-Native Multi-Agent Orchestrator CLI powered by C++ Operations Research"
    )
    # Not required: running `agentic-or` with no subcommand drops into the
    # interactive REPL instead of erroring out (see main()/run_repl()).
    subparsers = parser.add_subparsers(dest="command")

    # status command
    status_parser = subparsers.add_parser("status", help="Inspect real-time OS hardware telemetry and health")
    status_parser.add_argument("--watch", action="store_true", help="Continuously refresh instead of a single snapshot")
    status_parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds for --watch (default: 1.0)")

    # run command
    run_parser = subparsers.add_parser("run", help="Execute a DAG workflow from a JSON file")
    run_parser.add_argument("file", help="Path to JSON task definition file")
    run_parser.add_argument("--budget", type=int, default=50, help="C++ ALNS time budget in ms (default: 50)")
    run_parser.add_argument("--monitor", action="store_true", help="Show a live read-only dashboard while the pipeline runs")
    run_parser.add_argument(
        "--fresh", action="store_true",
        help="Reset the task queue before this run (keeps agents registered via 'agents add') - "
             "use to re-run the same task file/task_ids in one session",
    )

    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run the built-in heterogeneous DAG workflow demo")
    demo_parser.add_argument("--monitor", action="store_true", help="Show a live read-only dashboard while the pipeline runs")

    # llm command
    llm_parser = subparsers.add_parser("llm", help="Run real LLM multi-agent workflow")
    llm_parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "groq", "deepseek", "openrouter", "ollama", "mock"],
        default="mock",
    )
    llm_parser.add_argument("--key", help="API Key")
    llm_parser.add_argument("--model", help="Model name")

    # agents command - real-time status (default/'status'), plus 'add'/'reset'
    # to plug your own BaseWorker subclass into the session Orchestrator.
    agents_parser = subparsers.add_parser(
        "agents", help="Show Agents running RIGHT NOW (real-time), or add/reset your own"
    )
    agents_sub = agents_parser.add_subparsers(dest="agents_action")

    agents_sub.add_parser("status", help="(default) Show Agents running right now")

    agents_add_parser = agents_sub.add_parser(
        "add", help="Load a BaseWorker subclass from a .py file and register it for the next 'run'"
    )
    agents_add_parser.add_argument("file", help="Path to a .py file defining one or more BaseWorker subclasses")
    agents_add_parser.add_argument("--class", dest="class_name", help="Which class to use, if the file defines more than one")
    agents_add_parser.add_argument("--count", type=int, default=1, help="How many instances to create (default: 1)")
    agents_add_parser.add_argument("--prefix", help="worker_id prefix (default: the class name, lowercased)")
    agents_add_parser.add_argument(
        "--persist", action="store_true",
        help="Also save to ~/.agentic_or/agents.json so this agent auto-loads on every future 'agentic-or' startup "
             "(REPL or a fresh one-shot call), not just this session",
    )

    agents_reset_parser = agents_sub.add_parser(
        "reset", help="Drop the session's custom agents - 'run' goes back to the built-in workers"
    )
    agents_reset_parser.add_argument(
        "--persist", action="store_true",
        help="Also remove ~/.agentic_or/agents.json, so nothing auto-loads on future startups either",
    )

    agents_sub.add_parser("persisted", help="List agents saved with 'agents add --persist'")

    # history command - past runs in this session (the opposite of `agents`)
    subparsers.add_parser("history", help="Show summaries of pipelines already finished in this session")

    # watch command - cross-process/cross-terminal live view (reads the
    # status file another `agentic-or --monitor` run or main_agent.py writes)
    watch_parser = subparsers.add_parser(
        "watch", help="Watch another agentic-or process's live run from this terminal"
    )
    watch_parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds (default: 1.0)")

    # ui command - lightweight local web dashboard (browser-based `watch`)
    ui_parser = subparsers.add_parser(
        "ui", help="Open a lightweight local web dashboard (browser-based `watch`)"
    )
    ui_parser.add_argument("--port", type=int, default=8420, help="Local port to serve on (default: 8420)")
    ui_parser.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab")

    # repl command - explicit way to (re-)enter the interactive shell
    subparsers.add_parser("repl", help="Enter the interactive AgentOR shell")

    return parser


def dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Route a parsed Namespace to its handler. Shared by one-shot CLI calls and the REPL."""
    if args.command == "status":
        cmd_status(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "llm":
        cmd_llm(args)
    elif args.command == "agents":
        cmd_agents(args)
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "ui":
        cmd_ui(args)
    elif args.command == "repl" or args.command is None:
        from agentic_or.repl import run_repl
        run_repl(parser)
    else:
        parser.print_help()


def main() -> None:
    print_banner()

    parser = build_parser()
    argv = sys.argv[1:]

    if not argv:
        # Bare `agentic-or` -> interactive shell, Claude-Code style.
        from agentic_or.repl import run_repl
        run_repl(parser)
        return

    args = parser.parse_args(argv)
    dispatch(args, parser)


if __name__ == "__main__":
    main()
