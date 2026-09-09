"""
=====================================================================
 AGENTOR CUSTOM AGENT TEMPLATE - copy this file to write your own
=====================================================================
No API key, no network call, no external dependency - this template runs
as-is, so you can see it work first, then replace STEP 3/4 with real logic.

Try it right now:
    uv run agentic-or
    AgentOR › agents add my_agents/agent_template.py
    AgentOR › run my_agents/agent_template_tasks.json

Full contract reference: docs/custom-agents.md and the BaseWorker docstring
in agentic_or/workers/base_worker.py.
"""

from __future__ import annotations

import asyncio

from agentic_or.models import WorkloadType
from agentic_or.workers.base_worker import BaseWorker, TaskExecutionResult

# Uncomment whichever of these this agent can realistically hit, to route a
# failure to the right Self-Healing Daemon instead of just failing the task.
# See "Signaling incidents" in docs/custom-agents.md.
# from agentic_or.workers.base_worker import (
#     CaptchaBlockedError,     # -> CaptchaSolver daemon, then auto-retried
#     TwoFactorRequiredError,  # -> CaptchaSolver daemon, then auto-retried
#     AuthExpiredError,        # -> SessionAuthManager daemon, then auto-retried
#     RateLimitedError,        # -> trips the domain circuit breaker (NOT auto-retried)
# )


class MyCustomAgent(BaseWorker):
    """
    TODO: one-line description of what this agent actually does.

    The 5-point contract every custom agent must satisfy (full version is
    BaseWorker's own docstring):
      1. __init__(self, worker_id) - the CLI's `agents add` only ever passes
         worker_id, nothing else. Need an API key/path/config? Read it from
         an env var or a file inside __init__ or execute_task, not a ctor arg.
      2. execute_task is `async def` and must NEVER block the event loop -
         wrap any blocking call in asyncio.to_thread(...).
      3. It must return a TaskExecutionResult (success or failure).
      4. The TaskNodes it runs should carry accurate ram_mb / cpu_percent /
         estimated_duration_ms - the C++ scheduler uses these for real.
      5. (optional) override release_resources() to free real resources
         under RAM pressure. Default (inherited) is a safe no-op.
    """

    # ------------------------------------------------------------------
    # STEP 1 - pick the ONE workload type that best matches this agent.
    # This is what the C++ scheduler uses to model cost/setup/RAM, and
    # which worker pool `register_worker_pool`/`agents add` puts it in.
    #   WorkloadType.LOCAL   - light I/O: files, text, subprocess, CPU work
    #   WorkloadType.API     - network calls: HTTP APIs, LLM inference
    #   WorkloadType.BROWSER - heavy, session-based (browser automation)
    # ------------------------------------------------------------------
    def __init__(self, worker_id: str):
        super().__init__(worker_id, WorkloadType.LOCAL)  # <-- change me

        # STEP 2 (optional) - one-time setup, kept fast and synchronous (no
        # real I/O here - that belongs in execute_task). Example:
        #   import os
        #   self.some_setting = os.environ.get("MY_AGENT_SETTING", "default")

    # ------------------------------------------------------------------
    # STEP 3 - the actual work. Called once per TaskNode dispatched here.
    # `task`   - the TaskNode: task.task_id, task.name, task.metadata (a
    #            plain dict YOU define - put whatever input this agent
    #            needs there when you write the task JSON/TaskNode).
    # `action` - the DispatchAction the C++ scheduler computed for this
    #            call (worker_id, scheduled_start_ms, setup_cost_ms, ...).
    #            Most agents never need to read this directly.
    # ------------------------------------------------------------------
    async def execute_task(self, task, action) -> TaskExecutionResult:
        message = task.metadata.get("message", "Hello from AgentOR!")

        try:
            # STEP 4 - replace the two lines below with real async work:
            # an httpx call, asyncio.create_subprocess_exec, a real SDK
            # call, or asyncio.to_thread(blocking_fn) around blocking code.
            # The sleep here is only a stand-in so this template runs as-is.
            await asyncio.sleep(0.1)
            result = message.upper()

        # except SomeRecoverableThing as e:
        #     raise RateLimitedError(str(e))  # let a Self-Healing Daemon try
        except Exception as e:
            # STEP 5 - a real failure: return it, don't let the exception
            # escape uncaught (the Orchestrator catches it either way, but
            # returning it yourself keeps the error message meaningful).
            return TaskExecutionResult(task.task_id, success=False, error=str(e))

        return TaskExecutionResult(task.task_id, success=True, output=result)

    # STEP 6 (optional) - only override this if the agent holds something
    # worth freeing under RAM pressure (a cache, a real browser context...).
    # Delete this whole method if you don't need it - the inherited default
    # is already a safe no-op.
    def release_resources(self) -> None:
        pass
