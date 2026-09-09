"""
Runnable example: how a user plugs their OWN real agent into the
Orchestrator, instead of the built-in simulated Local/Api/BrowserWorker.

Run it with:
    uv run python examples/custom_agent.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agentic_or.daemons.auth_manager import SessionCredentials
from agentic_or.models import TaskNode, WorkloadType, DispatchAction
from agentic_or.orchestrator import Orchestrator
from agentic_or.workers.base_worker import (
    BaseWorker,
    TaskExecutionResult,
    AuthExpiredError,
)

OUTPUT_DIR = Path("/tmp/agentic_or_reports")
SESSION_KEY = "session:demo_agent"


# Step 1 & 2: subclass BaseWorker, pick the workload_type this agent
# belongs to (LOCAL/API/BROWSER - see Orchestrator.register_worker_pool's
# docstring for why it's limited to these 3), implement execute_task().
class ReportWriterAgent(BaseWorker):
    """
    A real custom agent: it actually writes a file to disk (not a fake
    asyncio.sleep()). Subclasses BaseWorker directly - this is the "full
    control" plugin path, as opposed to the lighter-weight shortcut of
    passing task.metadata["handler"] to the built-in LocalWorker.

    `shared_state` is a plain dict shared across every agent instance in the
    pool, only so this demo can deterministically fail exactly one task's
    *first* attempt regardless of which worker instance ends up handling the
    retry - not something a real agent needs.
    """

    def __init__(self, worker_id: str, output_dir: Path, shared_state: dict):
        super().__init__(worker_id, WorkloadType.LOCAL)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._shared = shared_state

    async def execute_task(self, task: TaskNode, action: DispatchAction) -> TaskExecutionResult:
        # Step 4 (optional): if your agent can hit a real, recoverable
        # incident (captcha, expired auth, rate limit...), raise the typed
        # WorkerIncident instead of stuffing a magic substring into `error`
        # - the Orchestrator's Self-Healing Daemons key off this reliably.
        # Demonstrated here on one task's first attempt only, so you can see
        # SessionAuthManager kick in and the task get requeued + retried.
        if task.task_id == "report_2" and self._shared["report_2_attempts"] == 0:
            self._shared["report_2_attempts"] += 1
            raise AuthExpiredError("session token expired mid-run")

        content = task.metadata.get("content", f"Report for {task.name}")
        path = self.output_dir / f"{task.task_id}.txt"

        # Real (blocking) file I/O - offloaded to a thread so it never
        # blocks the Orchestrator's single asyncio event loop (contract #2
        # in BaseWorker's docstring: execute_task must never block).
        await asyncio.to_thread(path.write_text, content)

        return TaskExecutionResult(task.task_id, success=True, output=str(path))


async def refresh_demo_session(session_key: str) -> SessionCredentials:
    """Stand-in for a real re-login / refresh-token exchange."""
    return SessionCredentials(token="fresh-demo-token")


async def main() -> None:
    # Step 3: construct the Orchestrator as usual. num_local_workers here
    # just sizes the *default* pool - it gets thrown away by step 5 below.
    orchestrator = Orchestrator(num_local_workers=2, alns_time_budget_ms=30)

    # Register how SessionAuthManager should recover the session this agent
    # uses, so the AuthExpiredError raised above actually gets healed instead
    # of failing for real (see BaseWorker's docstring, requirement #3).
    orchestrator.auth_manager.register_session(SESSION_KEY, refresh_callback=refresh_demo_session)

    # Step 5: build your own agent instances and hand them to
    # register_worker_pool() BEFORE submit_tasks()/run(). This *replaces*
    # the simulated LocalWorker pool - validated against the contract
    # (right BaseWorker subclass, right workload_type, unique worker_id).
    shared_state = {"report_2_attempts": 0}
    my_agents = [
        ReportWriterAgent("my_report_agent_0", OUTPUT_DIR, shared_state),
        ReportWriterAgent("my_report_agent_1", OUTPUT_DIR, shared_state),
    ]
    orchestrator.register_worker_pool(WorkloadType.LOCAL, my_agents)

    tasks = [
        TaskNode(
            task_id=f"report_{i}",
            name=f"Report {i}",
            workload_type=WorkloadType.LOCAL,
            estimated_duration_ms=20,
            metadata={"content": f"Hello from a real custom agent, task {i}", "session_key": SESSION_KEY},
        )
        for i in range(6)
    ]
    orchestrator.submit_tasks(tasks)

    # Step 6: run it. Dispatch, resource-aware scheduling, and self-healing
    # (report_2's simulated AuthExpiredError gets auto-requeued and retried)
    # all work exactly as they would with the built-in workers.
    result = await orchestrator.run()

    print(f"\nDone: {result}")
    print(f"report_2 attempts: {shared_state['report_2_attempts']} (1 = healed on retry)")
    for path in sorted(OUTPUT_DIR.glob("*.txt")):
        print(f"  {path.name}: {path.read_text()!r}")


if __name__ == "__main__":
    asyncio.run(main())
