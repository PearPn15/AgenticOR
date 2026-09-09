from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Any
from agentic_or.models import TaskNode, DispatchAction, WorkloadType

if TYPE_CHECKING:
    from agentic_or.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class TaskExecutionResult:
    def __init__(
        self,
        task_id: str,
        success: bool,
        output: Any = None,
        error: Optional[str] = None,
        error_code: Optional[str] = None,
        measured_resources: Optional[dict] = None,
    ):
        self.task_id = task_id
        self.success = success
        self.output = output
        self.error = error
        # Typed incident signal (see WorkerIncident below) a Worker can set
        # so the Orchestrator can route the failure to the right Self-Healing
        # Daemon without relying on substring-matching `error`.
        self.error_code = error_code
        # REAL psutil measurement ({"peak_ram_mb", "avg_cpu_percent",
        # "samples"}), set only by a worker that spawned an actual OS
        # subprocess for this task (see agentic_or/telemetry/process_sampler.py
        # and LocalWorker's task.metadata["command"] path). None for every
        # other worker - which runs as an asyncio coroutine sharing this
        # process, with no OS-level boundary to measure separately. Never
        # confuse this with TaskNode.ram_mb/cpu_percent, which is only a
        # declared ESTIMATE the C++ scheduler plans against, not a measurement.
        self.measured_resources = measured_resources


class WorkerIncident(Exception):
    """
    Base class for the typed incidents a custom Worker's `execute_task` can
    raise to plug into the Orchestrator's Self-Healing Daemons (ProxyRotator /
    SessionAuthManager / CaptchaSolver) without relying on substring-matching
    an error message (which is fragile - e.g. a message merely mentioning
    "Authorization" would accidentally match a naive `"Auth" in error` check).
    `run_action` below catches these automatically and turns them into a
    failed TaskExecutionResult with `error_code` set; the Orchestrator checks
    `error_code` first and falls back to legacy substring matching only for
    the built-in simulated workers and for plain exceptions raised by a
    `task.metadata["handler"]` callable.
    """
    error_code: str = "UNKNOWN"


class CaptchaBlockedError(WorkerIncident):
    """Raise when a page/API is blocked behind a CAPTCHA wall."""
    error_code = "CAPTCHA"


class TwoFactorRequiredError(WorkerIncident):
    """Raise when a login flow is blocked behind a 2FA prompt."""
    error_code = "TWO_FACTOR"


class AuthExpiredError(WorkerIncident):
    """Raise on a 401 response / expired session / expired token."""
    error_code = "AUTH_EXPIRED"


class RateLimitedError(WorkerIncident):
    """Raise on a 429/403 rate-limit response."""
    error_code = "RATE_LIMIT"


class BaseWorker(ABC):
    """
    Abstract Worker interface. Subclass this to plug a real agent (a real
    Playwright session, a real API SDK, a real subprocess tool, ...) into the
    Orchestrator via `Orchestrator.register_worker_pool()`.

    Contract a subclass must satisfy:
    1. `__init__` must call `super().__init__(worker_id, workload_type)` with
       a `worker_id` unique across the whole Orchestrator and the same
       `workload_type` (LOCAL / API / BROWSER) as the pool it will be
       registered into.
    2. `execute_task` must be `async` and must never block the event loop
       (no synchronous network/disk calls - use async libraries, or run
       blocking work in a thread/process executor).
    3. `execute_task` must return a `TaskExecutionResult`. On failure, either
       set `error` (a human-readable message) or raise one of the
       `WorkerIncident` subclasses above so the Orchestrator routes the
       failure to the right Self-Healing Daemon instead of just failing the
       task.
    4. The `TaskNode` this worker executes should carry an accurate
       `ram_mb` / `cpu_percent` / `estimated_duration_ms` - the C++
       RCPSP-ALNS scheduler uses these to build the dispatch plan, so bad
       estimates produce a meaningless schedule.
    5. Optionally override `release_resources()` to free real resources
       (close a real browser context, drop a large in-memory buffer, ...)
       when `MemoryJanitor` calls it on an idle worker under RAM pressure.
       The default is a no-op, so this is safe to skip for light workers.

    Dynamic sub-agent registration: once registered via
    `Orchestrator.register_sub_agent()`, `self.orchestrator` is set - use
    `self.orchestrator.register_sub_agent(SomeChildAgent(...), parent_id=self.worker_id)`
    from inside `execute_task` if this agent itself needs to spawn/call
    another agent at runtime. See docs/custom-agents.md#the-main-agent.
    """

    def __init__(self, worker_id: str, workload_type: WorkloadType):
        self.worker_id = worker_id
        self.workload_type = workload_type
        self.is_busy = False
        self.current_task_id: Optional[str] = None
        self.current_affinity_key: str = ""
        # Set by Orchestrator.register_sub_agent() - None until then (e.g.
        # for a worker only ever added via the older register_worker_pool()
        # / the CLI's `agents add`, which don't track tree position).
        self.orchestrator: Optional["Orchestrator"] = None
        # Set by Orchestrator.spawn_agent() - a one-shot agent created for a
        # single task is removed from its schedulable pool once that task
        # settles (see Orchestrator._retire_worker), instead of lingering to
        # possibly catch some LATER, unrelated task of the same
        # WorkloadType. False for a persistent pool worker (the built-in
        # defaults, or anything added via register_worker_pool/
        # register_sub_agent directly) that's meant to be reused.
        self._retire_after_task: bool = False
        # Last real, measured resource usage this worker's execute_task
        # reported via TaskExecutionResult.measured_resources (see its
        # docstring) - None if this worker doesn't wrap a real OS
        # subprocess, or hasn't run a task yet. Read by AgentRegistry to
        # expose real (not declared-estimate) numbers in the agent tree
        # for the specific agents that can actually provide them.
        self.last_measured_resources: Optional[dict] = None

    @abstractmethod
    async def execute_task(self, task: TaskNode, action: DispatchAction) -> TaskExecutionResult:
        """Run the actual task logic."""
        pass

    def release_resources(self) -> None:
        """
        Optional hook: called by MemoryJanitor on an idle worker during an
        emergency RAM cleanup (see agentic_or.daemons.memory_janitor).
        Override to actually free real resources; default is a no-op.
        """
        pass

    async def run_action(self, task: TaskNode, action: DispatchAction) -> TaskExecutionResult:
        self.is_busy = True
        self.current_task_id = task.task_id
        try:
            logger.info(f"[{self.worker_id}] Starting task '{task.task_id}' ({task.workload_type.value})")
            try:
                res = await self.execute_task(task, action)
            except WorkerIncident as e:
                res = TaskExecutionResult(task.task_id, success=False, error=str(e), error_code=e.error_code)

            if res.measured_resources is not None:
                self.last_measured_resources = res.measured_resources
            if res.success:
                self.current_affinity_key = task.affinity_key
                logger.info(f"[{self.worker_id}] Finished task '{task.task_id}' successfully")
            else:
                logger.warning(f"[{self.worker_id}] Task '{task.task_id}' failed: {res.error}")
            return res
        finally:
            self.is_busy = False
            self.current_task_id = None

