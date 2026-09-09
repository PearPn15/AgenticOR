from __future__ import annotations

import asyncio
import time
import logging
from typing import Dict, Optional, List, Set, Any
from agentic_or.models import TaskNode, DispatchAction, DispatchPlan

logger = logging.getLogger(__name__)


class TaskCheckpoint:
    def __init__(self, task_id: str, status: str = "PENDING"):
        self.task_id = task_id
        self.status = status  # PENDING, RUNNING, COMPLETED, FAILED
        self.start_time: Optional[float] = None
        self.finish_time: Optional[float] = None
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        # Count of incident-driven requeues (Captcha/Auth self-healing), not
        # a generic retry counter - used to cap how many times a Self-Healing
        # Daemon may send a task back to PENDING before it is failed for real.
        self.incident_retry_count: int = 0


class AsyncLocalBroker:
    """
    In-memory Async Event Broker with checkpoint tracking.
    Facilitates non-blocking communication between Orchestrator and Worker Pools.
    """

    def __init__(self):
        # Queues for distributing work
        self._action_queue: asyncio.Queue[DispatchAction] = asyncio.Queue()
        # Storage for tasks definitions
        self._tasks: Dict[str, TaskNode] = {}
        # Checkpoints for progress tracking & crash resilience
        self._checkpoints: Dict[str, TaskCheckpoint] = {}
        # Precedence tracking
        self._completed_tasks: Set[str] = set()
        self._failed_tasks: Set[str] = set()

    def register_tasks(self, tasks: List[TaskNode]) -> None:
        for t in tasks:
            if t.task_id in self._tasks:
                raise ValueError(
                    f"Task id '{t.task_id}' is already registered. A broker accumulates tasks "
                    f"across multiple submit_tasks() calls (e.g. a persistent CLI session "
                    f"orchestrator) - re-submitting the same task_id would silently corrupt its "
                    f"checkpoint. Use a fresh, unique task_id per submission."
                )
            self._tasks[t.task_id] = t
            self._checkpoints[t.task_id] = TaskCheckpoint(t.task_id, status="PENDING")

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        return self._tasks.get(task_id)

    def get_task_status(self, task_id: str) -> Optional[str]:
        cp = self._checkpoints.get(task_id)
        return cp.status if cp else None

    def get_backlog_count(self) -> int:
        return sum(1 for cp in self._checkpoints.values() if cp.status == "PENDING")

    def get_ready_tasks(self) -> List[TaskNode]:
        """Return all tasks whose predecessors are completely finished."""
        ready = []
        for task_id, cp in self._checkpoints.items():
            if cp.status == "PENDING":
                task = self._tasks[task_id]
                if all(pred in self._completed_tasks for pred in task.predecessors):
                    ready.append(task)
        return ready

    async def publish_action(self, action: DispatchAction) -> None:
        await self._action_queue.put(action)

    async def get_next_action(self) -> DispatchAction:
        return await self._action_queue.get()

    def mark_task_started(self, task_id: str) -> None:
        if task_id in self._checkpoints:
            cp = self._checkpoints[task_id]
            cp.status = "RUNNING"
            cp.start_time = time.time()

    def mark_task_completed(self, task_id: str, result: Any = None) -> None:
        if task_id in self._checkpoints:
            cp = self._checkpoints[task_id]
            cp.status = "COMPLETED"
            cp.finish_time = time.time()
            cp.result = result
        self._completed_tasks.add(task_id)

    def requeue_task(self, task_id: str, max_incident_retries: int = 3) -> bool:
        """
        Send a RUNNING/FAILED task back to PENDING so it is picked up again by
        `get_ready_tasks()`, used by Self-Healing Daemons (SessionAuthManager,
        CaptchaSolver) after they successfully clear the underlying incident.
        Returns False (and leaves the task alone) once `max_incident_retries`
        has been exhausted, so a persistently blocked task still fails for real
        instead of looping forever.
        """
        cp = self._checkpoints.get(task_id)
        if cp is None:
            return False
        if cp.incident_retry_count >= max_incident_retries:
            return False

        cp.incident_retry_count += 1
        cp.status = "PENDING"
        cp.start_time = None
        cp.finish_time = None
        cp.error = None
        self._failed_tasks.discard(task_id)
        self._completed_tasks.discard(task_id)
        logger.info(
            f"[Broker] Task '{task_id}' requeued after self-healing "
            f"(attempt {cp.incident_retry_count}/{max_incident_retries})."
        )
        return True

    def mark_task_failed(self, task_id: str, error: str) -> None:
        if task_id in self._checkpoints:
            cp = self._checkpoints[task_id]
            cp.status = "FAILED"
            cp.finish_time = time.time()
            cp.error = error
        self._failed_tasks.add(task_id)

        # Cascade failure to dependent tasks that require this task
        for tid, task in self._tasks.items():
            if task_id in task.predecessors and tid not in self._completed_tasks and tid not in self._failed_tasks:
                self.mark_task_failed(tid, f"Predecessor '{task_id}' failed: {error}")

    def is_all_done(self) -> bool:
        total = len(self._tasks)
        finished = len(self._completed_tasks) + len(self._failed_tasks)
        return finished >= total

