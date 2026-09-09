"""
REAL per-process resource sampling via psutil, for a worker that spawns an
actual OS subprocess (currently: LocalWorker's task.metadata["command"]
path). This is the one case in the framework where "resource this agent is
using" can be a genuine measurement instead of the declared
ram_mb/cpu_percent estimate on its TaskNode - see the caveat in
docs/custom-agents.md's "Dynamic agent registration" section and
Pipeline.md's Mục 8: every OTHER agent runs as an asyncio coroutine in the
same Python process, with no OS-level boundary psutil could measure
separately.
"""

from __future__ import annotations

import asyncio
from typing import Dict

import psutil


def _tree_rss_and_cpu(proc: psutil.Process) -> tuple[int, float]:
    """
    Sum RSS and CPU% across `proc` AND all its live descendants. This
    matters a lot in practice: `asyncio.create_subprocess_shell(cmd)`
    returns the PID of `/bin/sh -c cmd`, and `sh` typically FORKS a real
    child for `cmd` rather than exec-replacing itself (confirmed: its own
    `cmdline()` stays `['/bin/sh', '-c', cmd]` for the process's whole
    life) - so measuring `proc` alone sees only the near-empty shell
    wrapper and misses essentially all of the command's actual RAM/CPU.
    """
    procs = [proc] + proc.children(recursive=True)
    rss = 0
    cpu = 0.0
    for p in procs:
        try:
            rss += p.memory_info().rss
            cpu += p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return rss, cpu


async def sample_subprocess_resources(
    pid: int, stop_event: asyncio.Event, interval: float = 0.05,
) -> Dict[str, float]:
    """
    Poll a real PID's RSS/CPU (plus any child processes it spawns - see
    `_tree_rss_and_cpu`) while `stop_event` is unset. Returns
    {"peak_ram_mb": ..., "avg_cpu_percent": ..., "samples": N}. A
    process that exits before the first poll (very fast commands, e.g.
    `echo hi`) yields `samples: 0` and zeroed values - genuinely too short
    to sample at this polling granularity, not a measurement of "0 usage".
    """
    try:
        proc = psutil.Process(pid)
        _tree_rss_and_cpu(proc)  # first call only primes each process's internal CPU baseline
    except psutil.NoSuchProcess:
        return {"peak_ram_mb": 0.0, "avg_cpu_percent": 0.0, "samples": 0}

    peak_rss_bytes = 0
    cpu_samples: list[float] = []

    while not stop_event.is_set():
        if not proc.is_running():
            break
        rss, cpu = _tree_rss_and_cpu(proc)
        peak_rss_bytes = max(peak_rss_bytes, rss)
        cpu_samples.append(cpu)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass

    return {
        "peak_ram_mb": round(peak_rss_bytes / (1024 * 1024), 2),
        "avg_cpu_percent": round(sum(cpu_samples) / len(cpu_samples), 2) if cpu_samples else 0.0,
        "samples": len(cpu_samples),
    }
