from __future__ import annotations

import asyncio
import logging
from agentic_or.models import TaskNode, DispatchAction, WorkloadType
from agentic_or.telemetry.process_sampler import sample_subprocess_resources
from agentic_or.workers.base_worker import BaseWorker, TaskExecutionResult

logger = logging.getLogger(__name__)


class LocalWorker(BaseWorker):
    """
    Worker for local computation, file operations, JSON manipulation, shell scripts.
    """

    def __init__(self, worker_id: str):
        super().__init__(worker_id, WorkloadType.LOCAL)

    async def execute_task(self, task: TaskNode, action: DispatchAction) -> TaskExecutionResult:
        try:
            # Check for custom callable in metadata
            handler = task.metadata.get("handler")
            if callable(handler):
                if asyncio.iscoroutinefunction(handler):
                    output = await handler(task)
                else:
                    output = handler(task)
                return TaskExecutionResult(task.task_id, success=True, output=output)

            # Check for shell command
            cmd = task.metadata.get("command")
            if cmd:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                # REAL resource measurement, not the declared TaskNode
                # estimate: this is a genuine OS subprocess with a real PID,
                # so psutil can sample its actual RSS/CPU while it runs -
                # see agentic_or/telemetry/process_sampler.py's docstring for
                # why this is the one case in the framework where that's
                # possible at all (every other worker shares this same
                # Python process as an asyncio coroutine).
                stop_sampling = asyncio.Event()
                sampler = asyncio.create_task(sample_subprocess_resources(proc.pid, stop_sampling))
                stdout, stderr = await proc.communicate()
                stop_sampling.set()
                measured = await sampler

                if proc.returncode == 0:
                    return TaskExecutionResult(
                        task.task_id, success=True, output=stdout.decode().strip(),
                        measured_resources=measured,
                    )
                else:
                    return TaskExecutionResult(
                        task.task_id, success=False, error=stderr.decode().strip(),
                        measured_resources=measured,
                    )

            # Default: simulated processing for estimated duration
            delay = task.estimated_duration_ms / 1000.0
            await asyncio.sleep(min(0.2, delay)) # cap sleep for fast responsive runs
            return TaskExecutionResult(task.task_id, success=True, output=f"Local execution completed: {task.name}")

        except Exception as e:
            return TaskExecutionResult(task.task_id, success=False, error=str(e))

