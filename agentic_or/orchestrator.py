from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Dict, Optional, Any, Type, Union

import agentic_or._cxx_engine as cxx
from agentic_or.models import (
    TaskNode,
    WorkloadType,
    ExecutionProfile,
    DispatchAction,
    DispatchActionType,
    DispatchPlan,
)
from agentic_or.telemetry.system_monitor import SystemTelemetryMonitor
from agentic_or.safety.guards import OOMGuard, ThermalBatteryGuard, DomainCircuitBreaker
from agentic_or.broker.local_queue import AsyncLocalBroker
from agentic_or.workers.base_worker import BaseWorker, TaskExecutionResult
from agentic_or.workers.local_worker import LocalWorker
from agentic_or.workers.api_worker import ApiWorker
from agentic_or.workers.browser_worker import BrowserWorker
from agentic_or.daemons.proxy_rotator import ProxyRotator
from agentic_or.daemons.memory_janitor import MemoryJanitor
from agentic_or.daemons.auth_manager import SessionAuthManager
from agentic_or.daemons.captcha_handler import CaptchaSolver, ChallengeType
from agentic_or.telemetry.live_monitor import LiveMonitor
from agentic_or.registry import AgentRegistry, AgentHandle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("agentic_or.orchestrator")


class Orchestrator:
    """
    Main OS-Native Agentic-OR Orchestrator.
    Bridges Live OS Telemetry -> Safety Invariants -> C++ ALNS/Bandit Engine -> Async Workers.
    """

    def __init__(
        self,
        num_local_workers: int = 2,
        num_api_workers: int = 4,
        num_browser_workers: int = 2,
        min_ram_free_mb: float = 1200.0,
        alns_time_budget_ms: int = 100,
        enable_live_monitor: bool = False,
        monitor_refresh_seconds: float = 1.0,
        monitor_render_to_terminal: bool = True,
    ):
        self.alns_time_budget_ms = alns_time_budget_ms
        self.enable_live_monitor = enable_live_monitor
        self.monitor_refresh_seconds = monitor_refresh_seconds
        # False for a caller that owns this process's stdout for something
        # else (e.g. main_agent.py's own chat prompt) - the dashboard still
        # mirrors to LiveMonitor's status file for `agentic-or watch` in
        # another terminal, it just doesn't clear/redraw THIS terminal.
        self.monitor_render_to_terminal = monitor_render_to_terminal
        # Merged into the status file LiveMonitor writes each cycle (see
        # set_monitor_extra_status()) - e.g. main_agent.py sets
        # {"current_action": "..."} before a run() call so a `watch`-ing
        # terminal can show what it's currently doing.
        self._monitor_extra_status: Dict[str, Any] = {}

        # 1. Telemetry & Safety
        self.telemetry = SystemTelemetryMonitor(window_seconds=60)
        self.oom_guard = OOMGuard(min_ram_free_mb=min_ram_free_mb)
        self.thermal_guard = ThermalBatteryGuard()
        self.circuit_breaker = DomainCircuitBreaker()

        # 2. Broker
        self.broker = AsyncLocalBroker()

        # 3. Workers
        self.local_workers: List[LocalWorker] = [
            LocalWorker(f"local_worker_{i}") for i in range(num_local_workers)
        ]
        self.api_workers: List[ApiWorker] = [
            ApiWorker(f"api_worker_{i}") for i in range(num_api_workers)
        ]
        self.browser_workers: List[BrowserWorker] = [
            BrowserWorker(f"browser_worker_{i}") for i in range(num_browser_workers)
        ]
        self.all_workers: Dict[str, BaseWorker] = {}
        for w in self.local_workers + self.api_workers + self.browser_workers:
            self.all_workers[w.worker_id] = w

        # 3b. Agent tree (who spawned whom) - see agentic_or/registry.py.
        # The built-in workers above start as roots (no parent); anything
        # registered later via register_sub_agent() gets a real parent_id.
        self.agent_registry = AgentRegistry()
        for w in self.all_workers.values():
            w.orchestrator = self
            self.agent_registry.register(w, parent_id=None)

        # 3c. Agent-type catalog for spawn_agent() - see register_agent_type()/
        # spawn_agent() below. Empty until a caller registers types; this is
        # what lets spawn_agent("research") work without the caller needing
        # to know how to construct/register a ResearchAgent itself.
        self._agent_types: Dict[str, Type[BaseWorker]] = {}
        self._spawn_counter = 0

        # 4. Autonomous Daemons
        self.proxy_rotator = ProxyRotator()
        self.memory_janitor = MemoryJanitor(self.browser_workers)
        self.auth_manager = SessionAuthManager()
        self.captcha_solver = CaptchaSolver()
        # Cap how many times a single task may be requeued by self-healing
        # incident handling (Captcha/Auth) before it is allowed to fail for real.
        self.max_incident_retries = 3

        # 5. Core C++ Bandit
        self.bandit = cxx.LinearContextualBandit(0.2)

    def submit_tasks(self, tasks: List[TaskNode]) -> None:
        """Register batch of tasks with the broker."""
        self.broker.register_tasks(tasks)
        logger.info(f"Registered {len(tasks)} tasks into Orchestrator queue.")

    def set_monitor_extra_status(self, **fields: Any) -> None:
        """
        Free-form fields merged into LiveMonitor's status file on the next
        `run()` (see `agentic_or/telemetry/live_monitor.py`'s
        `STATUS_FILE_PATH` and `agentic-or watch`). Call before `run()` -
        e.g. `orchestrator.set_monitor_extra_status(current_action="fetching X")`
        so a `watch`-ing terminal can show what's currently being decided/done.
        """
        self._monitor_extra_status = dict(fields)

    def _pool_for(self, workload_type: WorkloadType) -> List[BaseWorker]:
        if workload_type == WorkloadType.BROWSER:
            return self.browser_workers
        elif workload_type == WorkloadType.API:
            return self.api_workers
        return self.local_workers

    def _retire_worker(self, worker: BaseWorker) -> None:
        """
        Remove a one-shot `spawn_agent()` worker from its schedulable pool
        now that its task has settled (called from `_execute_and_record`).
        It stays in `all_workers`/`agent_registry` - the tree and its final
        status/history are preserved - but the C++ scheduler can never pick
        it for some later, unrelated task of the same WorkloadType.
        """
        pool = self._pool_for(worker.workload_type)
        if worker in pool:
            pool.remove(worker)
            logger.info(f"[Orchestrator] Retired one-shot agent '{worker.worker_id}' from its pool (task settled).")

    def register_worker_pool(self, workload_type: WorkloadType, workers: List[BaseWorker]) -> None:
        """
        Replace the built-in simulated worker pool for `workload_type` with
        user-supplied `BaseWorker` subclass instances - e.g. a real
        Playwright-backed BrowserWorker, or an agent wrapping a proprietary
        SDK. See the contract documented on `BaseWorker` for what a custom
        worker must satisfy.

        Call this after constructing the Orchestrator and before
        `submit_tasks()` / `run()`. Only the 3 existing workload categories
        (LOCAL/API/BROWSER) are supported - they are what the C++ RCPSP-ALNS
        engine's resource/setup-cost model understands; adding a genuinely
        new resource category requires extending the C++ engine itself and
        is out of scope here.
        """
        other_ids = set()
        for wt in (WorkloadType.LOCAL, WorkloadType.API, WorkloadType.BROWSER):
            if wt == workload_type:
                continue
            other_ids.update(w.worker_id for w in self._pool_for(wt))

        seen_ids = set()
        for w in workers:
            if not isinstance(w, BaseWorker):
                raise TypeError(
                    f"register_worker_pool({workload_type.value}, ...): "
                    f"'{w!r}' must be an instance of a BaseWorker subclass."
                )
            if w.workload_type != workload_type:
                raise ValueError(
                    f"Worker '{w.worker_id}' declares workload_type={w.workload_type.value}, "
                    f"but is being registered into the {workload_type.value} pool."
                )
            if not w.worker_id or w.worker_id in other_ids or w.worker_id in seen_ids:
                raise ValueError(
                    f"Worker id '{w.worker_id}' is empty or already in use by another pool."
                )
            seen_ids.add(w.worker_id)

        pool = self._pool_for(workload_type)
        pool[:] = workers  # mutate in place: MemoryJanitor holds this same list object

        self.all_workers = {
            w.worker_id: w for w in self.local_workers + self.api_workers + self.browser_workers
        }
        logger.info(
            f"[Orchestrator] Registered {len(workers)} custom '{workload_type.value}' worker(s): "
            f"{[w.worker_id for w in workers]}"
        )

    def register_sub_agent(
        self, worker: BaseWorker, parent_id: Optional[str] = None, schedulable: bool = True,
    ) -> str:
        """
        Dynamically register ONE new agent at runtime - the entry point for
        "Main Agent calls a sub-agent" / "a sub-agent spawns a further
        agent" (see docs/custom-agents.md#the-main-agent). Unlike
        `register_worker_pool` (replaces a whole pool, called once before
        `run()`), this ADDS one worker to its pool and can be called at any
        time - including from inside another agent's OWN `execute_task`, via
        `self.orchestrator.register_sub_agent(child, parent_id=self.worker_id)`,
        which is how a sub-agent spawning a further sub-agent gets
        automatically discovered without AgenticOR knowing about it upfront.
        (`spawn_agent()` below is the higher-level, recommended way to do
        this - it calls register_sub_agent() internally.)

        `parent_id` is the calling agent's own `worker_id` - `None` makes
        this a new root of the agent tree, alongside the built-in workers.
        `schedulable=False` registers the agent in the tree (and in
        `all_workers`, so it's a valid `parent_id` for further registrations)
        WITHOUT adding it to its `WorkloadType` pool - use this for a purely
        symbolic tree node (e.g. a "main_agent" root that coordinates but
        never itself executes a dispatched task) so the C++ scheduler can
        never accidentally route a real task to it.
        Returns the new agent's id (== `worker.worker_id`).
        """
        if not isinstance(worker, BaseWorker):
            raise TypeError(f"register_sub_agent(...): '{worker!r}' must be an instance of a BaseWorker subclass.")
        if not worker.worker_id or worker.worker_id in self.all_workers:
            raise ValueError(f"Worker id '{worker.worker_id}' is empty or already registered.")
        if parent_id is not None and parent_id not in self.all_workers:
            raise ValueError(f"Parent agent id '{parent_id}' is not a registered agent.")

        if schedulable:
            self._pool_for(worker.workload_type).append(worker)
        self.all_workers[worker.worker_id] = worker
        worker.orchestrator = self
        self.agent_registry.register(worker, parent_id=parent_id)

        logger.info(
            f"[Orchestrator] Dynamically registered sub-agent '{worker.worker_id}' "
            f"({worker.workload_type.value})"
            + (f" under parent '{parent_id}'" if parent_id else " as a new agent-tree root")
        )
        return worker.worker_id

    def register_agent_type(self, agent_type: str, agent_class: Type[BaseWorker]) -> None:
        """
        Make `agent_type` a name `spawn_agent()` can instantiate - the
        "capability catalog" spawn_agent() validates against. `agent_class`
        must be constructible as `agent_class(worker_id)` (same convention
        as agents added via the CLI's `agents add`): read any other config
        from an env var/file inside its own `__init__`, not a ctor arg.
        """
        self._agent_types[agent_type] = agent_class

    async def spawn_agent(
        self,
        parent_id: Optional[str],
        agent_type: str,
        task: Union[TaskNode, dict],
        agent_id: Optional[str] = None,
    ) -> AgentHandle:
        """
        THE single entry point for "create a sub-agent and give it work" -
        the abstraction every caller (Main Agent, or any sub-agent spawning
        a further sub-agent from inside its own `execute_task`) uses
        identically, regardless of `agent_type`, instead of each needing its
        own registration/dispatch logic:

            validate agent_type -> allocate agent_id -> instantiate +
            register (register_sub_agent, which attaches parent_id and sets
            up the tree) -> build a TaskNode for `task` -> submit it to the
            scheduler -> return an AgentHandle to await the result.

        Works whether called at the top level (before any `run()` has
        started - the caller is expected to `await orchestrator.run()`
        itself afterward, same as `submit_tasks()` always required) or
        recursively from inside another agent's own `execute_task` (an
        already-active dispatch loop picks up the newly-submitted task on
        its own - no nested `run()` call needed or wanted).

        `task` is either a ready-made `TaskNode`, or a dict of TaskNode
        kwargs - `task_id` and `workload_type` are filled in automatically
        if omitted (workload_type is forced to match the spawned agent's
        own type; a mismatched explicit one is rejected, not silently
        overridden, so a copy-pasted task dict can't quietly target the
        wrong scheduling category).
        """
        agent_class = self._agent_types.get(agent_type)
        if agent_class is None:
            raise ValueError(
                f"Unknown agent_type '{agent_type}'. Register it first with "
                f"orchestrator.register_agent_type('{agent_type}', YourAgentClass)."
            )

        self._spawn_counter += 1
        new_agent_id = agent_id or f"{agent_type}_{self._spawn_counter}"
        worker = agent_class(new_agent_id)
        self.register_sub_agent(worker, parent_id=parent_id)
        # One-shot: this agent exists for exactly the one task submitted
        # below - retire it from its pool once that task settles (see
        # _retire_worker), so it can never later catch some unrelated task
        # of the same WorkloadType just because it happens to be idle. This
        # matters because ALL spawned agents of the same WorkloadType share
        # one flat pool for real scheduling - two distinct agent_types (e.g.
        # a "shell" and a "research" agent, both LOCAL) would otherwise be
        # interchangeable to the scheduler, which is never what a caller
        # asking for a SPECIFIC agent_type wants.
        worker._retire_after_task = True

        if isinstance(task, TaskNode):
            node = task
            if node.workload_type != worker.workload_type:
                raise ValueError(
                    f"Task '{node.task_id}' declares workload_type={node.workload_type.value}, "
                    f"but spawned agent_type='{agent_type}' is {worker.workload_type.value}."
                )
        else:
            task_kwargs = dict(task)
            task_kwargs.setdefault("task_id", f"{new_agent_id}_task_{self._spawn_counter}")
            declared_type = task_kwargs.get("workload_type")
            if declared_type is not None and WorkloadType(declared_type) != worker.workload_type:
                raise ValueError(
                    f"task dict declares workload_type={declared_type}, "
                    f"but spawned agent_type='{agent_type}' is {worker.workload_type.value}."
                )
            task_kwargs["workload_type"] = worker.workload_type
            node = TaskNode(**task_kwargs)

        self.submit_tasks([node])
        return AgentHandle(agent_id=new_agent_id, task_id=node.task_id, orchestrator=self)

    def _build_cxx_dag(self, tasks: List[TaskNode]) -> cxx.TaskDAG:
        dag = cxx.TaskDAG()
        id_to_idx: Dict[str, int] = {}

        for t in tasks:
            node = cxx.TaskData()
            node.task_id = t.task_id
            node.name = t.name
            node.duration_ms = t.estimated_duration_ms
            node.ram_mb = t.ram_mb
            node.cpu_percent = t.cpu_percent
            node.token_cost = t.token_cost
            node.affinity_key = t.affinity_key
            node.deadline_ms = t.deadline_ms if t.deadline_ms is not None else -1

            if t.workload_type == WorkloadType.BROWSER:
                node.workload_type = cxx.WorkloadType.BROWSER
            elif t.workload_type == WorkloadType.API:
                node.workload_type = cxx.WorkloadType.API
            else:
                node.workload_type = cxx.WorkloadType.LOCAL

            idx = dag.add_task(node)
            id_to_idx[t.task_id] = idx

        for t in tasks:
            for pred in t.predecessors:
                if pred in id_to_idx:
                    dag.add_dependency(id_to_idx[pred], id_to_idx[t.task_id])

        return dag

    async def run(self) -> Dict[str, Any]:
        """
        Main Event-Driven Scheduling and Dispatching Loop.
        """
        logger.info("Starting Orchestrator Dispatch Loop...")
        start_time = time.time()
        total_dispatched = 0

        # Optional read-only dashboard: runs as its own asyncio Task, polling
        # telemetry/broker/worker state on its own timer. It never calls any
        # scheduling/dispatch method, so it cannot affect task execution -
        # only start()/stop() bracket the loop below.
        monitor: Optional[LiveMonitor] = None
        if self.enable_live_monitor:
            monitor = LiveMonitor(
                self, refresh_seconds=self.monitor_refresh_seconds,
                render_to_terminal=self.monitor_render_to_terminal,
            )
            monitor.extra_status = self._monitor_extra_status
            monitor.start()

        try:
            total_dispatched = await self._dispatch_loop()
        finally:
            if monitor is not None:
                await monitor.stop()

        total_elapsed = time.time() - start_time
        logger.info(f"All tasks completed in {total_elapsed:.2f}s (Total dispatched: {total_dispatched})")

        return {
            "total_tasks": total_dispatched,
            "elapsed_seconds": total_elapsed,
            "status": "COMPLETED",
        }

    async def _dispatch_loop(self) -> int:
        total_dispatched = 0
        # Dispatched tasks run as independent asyncio.Tasks (task -> task_id)
        # instead of being awaited as one batch via asyncio.gather. This makes
        # dispatch a rolling pipeline: a slow task in one cycle never blocks
        # newly-ready work (or a worker it just freed up) from being picked up
        # on the very next cycle.
        inflight: Dict[asyncio.Task, str] = {}

        while not self.broker.is_all_done() or inflight:
            # Non-blocking reap of anything that already finished.
            for t in [t for t in inflight if t.done()]:
                await t  # propagate unexpected exceptions (should never raise)
                inflight.pop(t, None)

            backlog = self.broker.get_backlog_count()
            snapshot = self.telemetry.capture_snapshot(queue_backlog=backlog)

            # Safety Invariant 1: Hardware OOM-Guard
            can_run_heavy, need_emergency_gc = self.oom_guard.evaluate(snapshot)
            if need_emergency_gc:
                self.memory_janitor.perform_cleanup()
                await asyncio.sleep(0.5)
                continue

            # Safety Invariant 2: Thermal & Battery Profile
            profile = self.thermal_guard.determine_profile(snapshot)

            # Retrieve ready tasks from broker
            ready_tasks = self.broker.get_ready_tasks()
            if not ready_tasks:
                await self._wait_for_inflight_or_pause(inflight, default_sleep=0.05)
                continue

            # Filter tasks by circuit breaker and OOM guard
            candidate_tasks: List[TaskNode] = []
            for t in ready_tasks:
                if t.workload_type == WorkloadType.BROWSER and not can_run_heavy:
                    continue  # Hold heavy browser task until RAM clears
                if t.affinity_key.startswith("domain:"):
                    domain = t.affinity_key.split(":", 1)[1]
                    allowed, wait_sec = self.circuit_breaker.is_domain_allowed(domain)
                    if not allowed:
                        continue  # In cooldown
                candidate_tasks.append(t)

            if not candidate_tasks:
                await self._wait_for_inflight_or_pause(inflight, default_sleep=0.1)
                continue

            # Contextual Bandit: Determine concurrency cap
            context_vec = snapshot.to_bandit_feature_vector()
            action_arm = self.bandit.select_action(context_vec)
            concurrency_cap = self.bandit.action_to_concurrency(action_arm)

            if profile == ExecutionProfile.ECO_SILENT:
                concurrency_cap = min(concurrency_cap, 2)
            elif profile == ExecutionProfile.TURBO_SPEED:
                concurrency_cap = max(concurrency_cap, 4)

            remaining_capacity = concurrency_cap - len(inflight)
            if remaining_capacity <= 0:
                await self._wait_for_inflight_or_pause(inflight, default_sleep=0.05)
                continue

            # Core C++ Engine: Build DAG and solve via ALNS for the currently
            # free slots only (limits.max_concurrency = what's actually open).
            cxx_dag = self._build_cxx_dag(candidate_tasks)
            limits = cxx.ResourceLimits()
            limits.max_concurrency = remaining_capacity
            limits.max_ram_mb = int(snapshot.ram_free_mb * 0.8)

            solver = cxx.RCPSPSolver(cxx_dag, limits)
            if profile == ExecutionProfile.ECO_SILENT:
                solver.set_weights(0.5, 0.5, 2.0)  # Heavily penalize resource stress
            elif profile == ExecutionProfile.TURBO_SPEED:
                solver.set_weights(2.0, 0.2, 0.1)  # Prioritize fastest makespan

            cxx_result = solver.solve(time_budget_ms=self.alns_time_budget_ms)

            # Launch tasks against the C++ schedule: honor the ALNS's
            # per-slot worker assignment (spreads load across the pool
            # instead of piling everything onto one worker) and its computed
            # start-time stagger (spreads out resource-heavy launches instead
            # of firing the whole wave at once).
            now_epoch_ms = int(time.time() * 1000)
            launched_this_cycle = 0

            for assign in cxx_result.assignments:
                if launched_this_cycle >= remaining_capacity:
                    break

                task = candidate_tasks[assign.task_index]
                target_worker, used_preferred_slot = self._acquire_worker(task.workload_type, assign.worker_id)
                if target_worker is None:
                    continue  # every worker of this type is currently busy

                self.broker.mark_task_started(task.task_id)

                # The solver's start_time_ms bakes in this task's own
                # setup_cost_ms (e.g. a 2000ms browser cold-start counted as
                # dead time before the task "starts"). The worker's own
                # execute_task already simulates that same setup as leading
                # latency, so honoring the raw start_time_ms would wait for
                # it twice. Net it out, and only when the task landed on the
                # solver's preferred slot (used_preferred_slot) - if it got
                # reassigned to a different, already-idle worker via the
                # fallback in _acquire_worker, the solver's same-slot
                # sequencing assumption no longer applies and there is
                # nothing left to wait for.
                stagger_ms = max(0, assign.start_time_ms - assign.setup_cost_ms) if used_preferred_slot else 0

                action = DispatchAction(
                    task_id=task.task_id,
                    worker_id=target_worker.worker_id,
                    action_type=DispatchActionType.EXECUTE,
                    scheduled_start_ms=now_epoch_ms + stagger_ms,
                    estimated_duration_ms=task.estimated_duration_ms,
                    setup_cost_ms=assign.setup_cost_ms,
                )

                coro = self._execute_and_record(
                    target_worker, task, action, context_vec, action_arm, profile, snapshot.cpu_percent
                )
                inflight[asyncio.create_task(coro)] = task.task_id
                total_dispatched += 1
                launched_this_cycle += 1

            if launched_this_cycle == 0:
                # Every candidate's worker pool is saturated right now - wait
                # for a slot to free up rather than busy-spinning the solver.
                await self._wait_for_inflight_or_pause(inflight, default_sleep=0.05)

        return total_dispatched

    async def _wait_for_inflight_or_pause(
        self, inflight: Dict[asyncio.Task, str], default_sleep: float = 0.05
    ) -> None:
        """
        Block until at least one in-flight task finishes - so a freed worker
        or a newly-completed predecessor is reacted to immediately - or fall
        back to a short sleep when nothing is in flight yet.
        """
        if inflight:
            done, _ = await asyncio.wait(inflight.keys(), timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                await t
                inflight.pop(t, None)
        else:
            await asyncio.sleep(default_sleep)

    def _acquire_worker(
        self, workload_type: WorkloadType, preferred_slot: int
    ) -> "tuple[Optional[BaseWorker], bool]":
        """
        Reserve one idle worker of the given type for immediate dispatch.
        Reservation (`is_busy = True`) happens synchronously here, before the
        task's coroutine is even created, so two assignments picked in the
        same dispatch cycle can never both grab the same idle worker. The
        C++ solver's per-slot `worker_id` is tried first (to honor its
        load-balancing / context-affinity placement); any other idle worker
        of the same type is used as a fallback. Returns `(None, False)` if
        the whole pool is currently busy, or `(worker, used_preferred_slot)`
        - callers use `used_preferred_slot` to know whether the solver's
        computed schedule (e.g. same-slot sequencing) still applies to this
        worker.
        """
        if workload_type == WorkloadType.BROWSER:
            pool: List[BaseWorker] = self.browser_workers
        elif workload_type == WorkloadType.API:
            pool = self.api_workers
        else:
            pool = self.local_workers

        if not pool:
            return None, False

        preferred = pool[preferred_slot % len(pool)]
        if not preferred.is_busy:
            preferred.is_busy = True
            return preferred, True

        for w in pool:
            if not w.is_busy:
                w.is_busy = True
                return w, False

        return None, False

    async def _execute_and_record(
        self,
        worker: BaseWorker,
        task: TaskNode,
        action: DispatchAction,
        context_vec: List[float],
        action_arm: int,
        profile: ExecutionProfile,
        cpu_percent_at_dispatch: float,
    ) -> None:
        """
        Run one dispatched task to completion and feed the Contextual Bandit
        a reward for this single dispatch decision. Runs as an independent
        asyncio.Task (see `_dispatch_loop`) so a slow task never blocks
        newly-ready work from being picked up on the next cycle.
        """
        # Honor the C++ solver's computed stagger (Resource-Peak Smoother):
        # hold this worker reserved-but-idle until its planned start time
        # instead of firing every dispatched task at once.
        delay_s = (action.scheduled_start_ms - int(time.time() * 1000)) / 1000.0
        if delay_s > 0:
            await asyncio.sleep(delay_s)

        t0 = time.time()
        result = await self._execute_worker_task(worker, task, action)
        elapsed_s = max(0.05, time.time() - t0)

        # A task cleared by a Self-Healing Daemon (Captcha/2FA/Auth) is
        # requeued back to PENDING to run again later - it is neither a real
        # success nor a real failure yet, so it must not be scored as a
        # bandit failure (it would otherwise wrongly punish the concurrency
        # decision for a transient incident the daemons already fixed).
        if self.broker.get_task_status(task.task_id) == "PENDING":
            return

        if getattr(worker, "_retire_after_task", False):
            self._retire_worker(worker)

        reward = (1.0 / elapsed_s) if result.success else -2.0
        if profile == ExecutionProfile.ECO_SILENT:
            reward -= (cpu_percent_at_dispatch / 100.0) * 1.5
        self.bandit.update(context_vec, action_arm, reward)

    async def _execute_worker_task(
        self, worker: BaseWorker, task: TaskNode, action: DispatchAction
    ) -> TaskExecutionResult:
        result = await worker.run_action(task, action)
        domain = task.affinity_key.split(":", 1)[1] if task.affinity_key.startswith("domain:") else ""

        if result.success:
            self.broker.mark_task_completed(task.task_id, result.output)
            if domain:
                self.circuit_breaker.record_success(domain)
            self.telemetry.record_request_result(is_rate_limited=False)
            return result

        error = result.error or "Unknown error"
        code = result.error_code

        # Xử lý sự cố tự động (Step 5 of the pipeline): delegate blocked tasks
        # to the Self-Healing Daemons before giving up on them for real.
        # `error_code` (set when a Worker raises a typed WorkerIncident) is
        # checked first - it is what custom/plugged-in agents should use.
        # The substring checks on `error` are legacy fallbacks kept for the
        # built-in simulated workers and for plain exceptions raised by a
        # `task.metadata["handler"]` callable.
        is_captcha = code in ("CAPTCHA", "TWO_FACTOR") or "Captcha" in error or "2FA" in error
        is_auth = code == "AUTH_EXPIRED" or "401" in error or "Auth" in error or "Session Expired" in error or "Token Expired" in error
        is_rate_limited = code == "RATE_LIMIT" or "Rate Limit" in error

        if is_captcha:
            if await self._handle_captcha_incident(task, domain, error, code):
                return result
        elif is_auth:
            if await self._handle_auth_incident(task, error):
                return result

        self.broker.mark_task_failed(task.task_id, error)
        if is_rate_limited:
            self.telemetry.record_request_result(is_rate_limited=True)
            if domain:
                self.circuit_breaker.record_failure(domain)
                proxy = self.proxy_rotator.lease_proxy()
                self.proxy_rotator.report_error(proxy)

        return result

    async def _handle_captcha_incident(
        self, task: TaskNode, domain: str, error: str, error_code: Optional[str] = None
    ) -> bool:
        """Delegate a Captcha/2FA-blocked task to the CaptchaSolver daemon and requeue on success."""
        is_2fa = error_code == "TWO_FACTOR" if error_code else "2FA" in error
        challenge_type = ChallengeType.TWO_FACTOR if is_2fa else ChallengeType.CAPTCHA
        solution = await self.captcha_solver.resolve_challenge(
            domain=domain or task.affinity_key or task.task_id,
            challenge_type=challenge_type,
            context={"task_id": task.task_id, "error": error},
        )
        if solution is None:
            logger.error(f"[Orchestrator] Captcha/2FA incident on task '{task.task_id}' could not be resolved.")
            return False

        requeued = self.broker.requeue_task(task.task_id, max_incident_retries=self.max_incident_retries)
        if requeued:
            logger.info(f"[Orchestrator] Captcha/2FA cleared for '{task.task_id}'; task requeued.")
        return requeued

    async def _handle_auth_incident(self, task: TaskNode, error: str) -> bool:
        """Delegate an expired-session/token failure to the SessionAuthManager daemon and requeue on success."""
        session_key = task.metadata.get("session_key") or task.affinity_key or task.task_id
        fresh = await self.auth_manager.force_refresh(session_key)
        if fresh is None:
            logger.error(f"[Orchestrator] Auth/session incident on task '{task.task_id}' could not be resolved.")
            return False

        requeued = self.broker.requeue_task(task.task_id, max_incident_retries=self.max_incident_retries)
        if requeued:
            logger.info(f"[Orchestrator] Session refreshed for '{task.task_id}'; task requeued.")
        return requeued

