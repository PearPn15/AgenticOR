from __future__ import annotations

import gc
import logging
import psutil
from typing import List
from agentic_or.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class MemoryJanitor:
    """
    Autonomous Memory Guard & Garbage Collection Daemon.
    Performs emergency cleanup when OOM-Guard detects high system pressure.
    """

    def __init__(self, browser_workers: List[BaseWorker]):
        # Kept as a reference to the Orchestrator's own list object (not a
        # copy), so Orchestrator.register_worker_pool() mutating that list
        # in place (swapping in custom BROWSER workers) is picked up here
        # automatically without needing to re-wire this daemon.
        self.browser_workers = browser_workers

    def perform_cleanup(self) -> float:
        """
        Execute GC and evict idle tabs across workers.
        Returns amount of RAM freed in MB.
        """
        mem_before = psutil.virtual_memory().available / (1024 * 1024)

        # 1. Release idle workers' resources (any BaseWorker subclass -
        # release_resources() defaults to a no-op, so this is safe even for
        # custom workers that don't manage heavy memory).
        for worker in self.browser_workers:
            if not worker.is_busy:
                worker.release_resources()

        # 2. Force Python Garbage Collection
        collected = gc.collect()

        mem_after = psutil.virtual_memory().available / (1024 * 1024)
        freed = max(0.0, mem_after - mem_before)

        logger.info(
            f"[MemoryJanitor] Cleanup completed: {collected} Python objects collected. "
            f"RAM available: {mem_before:.1f}MB -> {mem_after:.1f}MB (Freed ~{freed:.1f}MB)"
        )
        return freed

