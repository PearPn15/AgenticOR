from __future__ import annotations

import time
import os
from collections import deque
from typing import Optional
import psutil

from agentic_or.models import TelemetrySnapshot


class SystemTelemetryMonitor:
    """
    Real-time OS Telemetry Monitor.
    Queries the OS kernel via psutil to extract hardware load and system health metrics.
    """

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.cpu_cores = os.cpu_count() or 4
        # Sliding window for 429 / 403 error tracking: deque of (timestamp, is_error_429)
        self._request_history: deque[tuple[float, bool]] = deque()
        self._last_cpu_sample_time = time.time()
        # Initial prime call to psutil.cpu_percent
        psutil.cpu_percent(interval=None)

    def record_request_result(self, is_rate_limited: bool) -> None:
        """Record an API request outcome to compute rolling 429 error rate."""
        now = time.time()
        self._request_history.append((now, is_rate_limited))
        self._evict_old_records(now)

    def _evict_old_records(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._request_history and self._request_history[0][0] < cutoff:
            self._request_history.popleft()

    def get_current_error_rate_429(self) -> float:
        now = time.time()
        self._evict_old_records(now)
        if not self._request_history:
            return 0.0
        error_count = sum(1 for _, err in self._request_history if err)
        return error_count / len(self._request_history)

    def read_cpu_temperature(self) -> float:
        """Attempt to read CPU temperature via psutil.sensors_temperatures."""
        try:
            sensors = psutil.sensors_temperatures()
            if not sensors:
                return 50.0
            for name, entries in sensors.items():
                for entry in entries:
                    if entry.current is not None and entry.current > 0:
                        return float(entry.current)
        except Exception:
            pass
        return 50.0

    def capture_snapshot(self, queue_backlog: int = 0) -> TelemetrySnapshot:
        """Capture an instantaneous snapshot of system hardware and OS status."""
        now = time.time()
        
        # CPU
        cpu_pct = psutil.cpu_percent(interval=None)
        try:
            load_1m, _, _ = os.getloadavg()
            cpu_load_1m = min(1.0, load_1m / self.cpu_cores)
        except (AttributeError, OSError):
            cpu_load_1m = cpu_pct / 100.0

        # Memory
        mem = psutil.virtual_memory()
        ram_free_mb = mem.available / (1024 * 1024)
        ram_total_mb = mem.total / (1024 * 1024)
        ram_free_ratio = mem.available / mem.total if mem.total > 0 else 1.0

        # Battery
        battery = psutil.sensors_battery()
        if battery is not None:
            battery_pct = float(battery.percent)
            is_charging = bool(battery.power_plugged) if battery.power_plugged is not None else True
        else:
            battery_pct = 100.0
            is_charging = True

        # Thermal
        temp = self.read_cpu_temperature()

        # 429 error rate
        err_rate = self.get_current_error_rate_429()

        return TelemetrySnapshot(
            timestamp=now,
            cpu_load_1m=cpu_load_1m,
            cpu_percent=cpu_pct,
            ram_free_mb=ram_free_mb,
            ram_total_mb=ram_total_mb,
            ram_free_ratio=ram_free_ratio,
            battery_percent=battery_pct,
            is_charging=is_charging,
            cpu_temperature_c=temp,
            error_rate_429=err_rate,
            queue_backlog=queue_backlog,
        )

