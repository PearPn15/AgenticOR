"""
AgentRegistry - the agent TREE (who spawned whom), separate from the flat
per-WorkloadType worker pools (local_workers/api_workers/browser_workers)
that actually get scheduled by the C++ engine.

A worker still only ever runs through one of those flat pools - the tree
here is purely a parent -> child bookkeeping layer on top, built dynamically
at runtime as agents register themselves (or get registered by whoever
decided to use them), NOT declared upfront. See
Orchestrator.register_sub_agent() for the entry point, and
docs/custom-agents.md#the-main-agent for the design this came out of.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agentic_or.orchestrator import Orchestrator
    from agentic_or.workers.base_worker import BaseWorker


@dataclass
class AgentHandle:
    """
    Returned by `Orchestrator.spawn_agent()` - the ONLY thing a caller needs
    to find out how its spawned agent's task went, regardless of whether
    spawn_agent() was called at the top level (before any run() started) or
    recursively from inside another agent's own execute_task (an
    already-active dispatch loop picks up the new task on its own; nothing
    special has to happen for the nested case to work).
    """
    agent_id: str
    task_id: str
    orchestrator: "Orchestrator"

    async def result(self, poll_interval: float = 0.05) -> Any:
        """Wait for this specific task to settle and return its output, or
        raise RuntimeError with its error message if it failed. Pure polling
        of shared broker state - safe to await from inside a task that is
        itself running under the same (or an ancestor's) dispatch loop."""
        broker = self.orchestrator.broker
        while True:
            status = broker.get_task_status(self.task_id)
            if status == "COMPLETED":
                return broker._checkpoints[self.task_id].result
            if status == "FAILED":
                raise RuntimeError(broker._checkpoints[self.task_id].error)
            await asyncio.sleep(poll_interval)


@dataclass
class AgentNode:
    agent_id: str
    worker: "BaseWorker"
    parent_id: Optional[str]
    children: List[str] = field(default_factory=list)
    registered_at: float = field(default_factory=time.time)

    def status(self) -> str:
        """RUNNING / IDLE - derived live from the worker, never cached
        (so it can never drift out of sync with the real dispatch state)."""
        return "RUNNING" if self.worker.is_busy else "IDLE"


class AgentRegistry:
    """
    Tracks parent -> child relationships between registered agents. This is
    NOT a scheduling structure (the C++ engine and the flat worker pools own
    that) - purely an observability layer: "who registered whom, and who's
    doing what right now."
    """

    def __init__(self):
        self.nodes: Dict[str, AgentNode] = {}

    def register(self, worker: "BaseWorker", parent_id: Optional[str] = None) -> str:
        agent_id = worker.worker_id
        if agent_id in self.nodes:
            raise ValueError(f"Agent id '{agent_id}' is already in the registry.")
        if parent_id is not None and parent_id not in self.nodes:
            raise ValueError(f"Parent agent id '{parent_id}' is not registered.")

        self.nodes[agent_id] = AgentNode(agent_id=agent_id, worker=worker, parent_id=parent_id)
        if parent_id is not None:
            self.nodes[parent_id].children.append(agent_id)
        return agent_id

    def children_of(self, agent_id: str) -> List[str]:
        node = self.nodes.get(agent_id)
        return list(node.children) if node else []

    def ancestors_of(self, agent_id: str) -> List[str]:
        """Root-first chain of parents above `agent_id` (not including it)."""
        chain: List[str] = []
        node = self.nodes.get(agent_id)
        while node and node.parent_id is not None:
            chain.append(node.parent_id)
            node = self.nodes.get(node.parent_id)
        return list(reversed(chain))

    def roots(self) -> List[str]:
        return [aid for aid, n in self.nodes.items() if n.parent_id is None]

    def to_tree_dict(self, agent_id: Optional[str] = None) -> Dict:
        """
        Nested {agent_id, type, status, current_task, measured_resources,
        children: [...]} structure, JSON-serializable - what LiveMonitor
        mirrors to the status file for `agentic-or watch` to render.
        `measured_resources` is `None` for almost every agent - everything
        here runs as an asyncio coroutine in ONE Python process, not a
        separate OS process, so there is usually no OS-level boundary to
        measure separately (see docs/custom-agents.md#the-main-agent). The
        one exception: an agent that spawned a real subprocess for its last
        task (currently: LocalWorker's task.metadata["command"] path) has a
        real PID psutil measured for real - see
        agentic_or/telemetry/process_sampler.py and
        BaseWorker.last_measured_resources. Never confuse this with a
        TaskNode's declared ram_mb/cpu_percent, which is only an estimate
        the C++ scheduler plans against.
        """
        roots = [agent_id] if agent_id is not None else self.roots()
        return {"agents": [self._node_dict(aid) for aid in roots if aid in self.nodes]}

    def _node_dict(self, agent_id: str) -> Dict:
        node = self.nodes[agent_id]
        w = node.worker
        return {
            "agent_id": agent_id,
            "type": w.workload_type.value,
            "status": node.status(),
            "current_task": w.current_task_id,
            "measured_resources": w.last_measured_resources,
            "children": [self._node_dict(c) for c in node.children],
        }
