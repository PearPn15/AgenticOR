"""
Multi-agent PIPELINE example: two DIFFERENT specialized agent classes,
one runs strictly after the other, and the second one actually CONSUMES
the first one's output.

Key thing this demonstrates that isn't obvious from a single-agent example:
`predecessors` in the task JSON only controls ORDER (A must finish before B
starts) - it does NOT automatically hand A's result to B. You have to wire
that yourself. The pattern here: a module-level dict both agent instances
share, keyed by task_id, written by the producer and read by the consumer
via a task_id the task JSON tells it to read from.

Try it:
    uv run agentic-or
    AgentOR › agents add my_agents/pipeline_agents.py --class CrawlerAgent
    AgentOR › agents add my_agents/pipeline_agents.py --class ProcessorAgent
    AgentOR › run my_agents/pipeline_tasks.json
"""

from __future__ import annotations

import asyncio

from agentic_or.models import WorkloadType
from agentic_or.workers.base_worker import BaseWorker, TaskExecutionResult

# Shared "blackboard" both agent classes below read/write to, keyed by
# task_id. This is the simplest way to pass real data between stages of a
# DAG in this framework - swap a plain dict for a file/SQLite/Redis if you
# need it to survive across separate `agentic-or` processes.
_RESULTS: dict[str, str] = {}


class CrawlerAgent(BaseWorker):
    """Stage 1 (WorkloadType.BROWSER - stands in for "goes and fetches
    something"). Produces text and stores it in `_RESULTS[task.task_id]`."""

    def __init__(self, worker_id: str):
        super().__init__(worker_id, WorkloadType.BROWSER)

    async def execute_task(self, task, action) -> TaskExecutionResult:
        topic = task.metadata.get("topic", "unknown")

        # TODO: replace with a real fetch (see web_fetch_agent.py for a
        # working httpx-based example). Stubbed here to keep this example
        # dependency-free.
        await asyncio.sleep(0.1)
        raw = f"[raw data about '{topic}' fetched by {self.worker_id}]"

        _RESULTS[task.task_id] = raw
        return TaskExecutionResult(task.task_id, success=True, output=raw)


class ProcessorAgent(BaseWorker):
    """
    Stage 2 (WorkloadType.LOCAL). Only makes sense to run AFTER a specific
    CrawlerAgent task - the task JSON must set:
      - "predecessors": ["<crawler_task_id>"]   -> enforces the ORDER
      - "metadata": {"read_from": "<crawler_task_id>"}  -> enforces the DATA FLOW
    Both are needed; neither implies the other.
    """

    def __init__(self, worker_id: str):
        super().__init__(worker_id, WorkloadType.LOCAL)

    async def execute_task(self, task, action) -> TaskExecutionResult:
        upstream_id = task.metadata.get("read_from")
        if not upstream_id:
            return TaskExecutionResult(task.task_id, success=False, error="metadata.read_from is required")

        raw = _RESULTS.get(upstream_id)
        if raw is None:
            # Should not normally happen if `predecessors` was set correctly
            # (the broker won't dispatch this task until upstream_id is
            # COMPLETED) - this only fires if read_from points at the wrong
            # task_id, or predecessors was forgotten in the task JSON.
            return TaskExecutionResult(
                task.task_id, success=False,
                error=f"No result found for '{upstream_id}' - did you set predecessors correctly?",
            )

        await asyncio.sleep(0.1)
        processed = raw.upper().replace("RAW DATA", "PROCESSED DATA")

        _RESULTS[task.task_id] = processed
        return TaskExecutionResult(task.task_id, success=True, output=processed)
