from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WorkloadType(str, Enum):
    LOCAL = "LOCAL"        # Light I/O, Python, script, file processing
    API = "API"            # HTTP/Network, Async LLM, token-bound
    BROWSER = "BROWSER"    # Playwright, Headless Chromium, heavy RAM/Context


class ExecutionProfile(str, Enum):
    TURBO_SPEED = "TURBO_SPEED"  # Max concurrency when resources are abundant and AC is connected
    BALANCED = "BALANCED"        # Standard desktop workflow balance
    ECO_SILENT = "ECO_SILENT"    # Low battery, high CPU temperature, strictly throttling


class TaskNode(BaseModel):
    task_id: str
    name: str = ""
    workload_type: WorkloadType = WorkloadType.LOCAL
    predecessors: List[str] = Field(default_factory=list)
    estimated_duration_ms: int = Field(default=1000, ge=1)
    ram_mb: int = Field(default=100, ge=0)
    cpu_percent: float = Field(default=10.0, ge=0.0, le=100.0)
    token_cost: int = Field(default=0, ge=0)
    affinity_key: str = ""  # e.g. "domain:github.com" or "session:user1"
    deadline_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TelemetrySnapshot(BaseModel):
    timestamp: float
    cpu_load_1m: float = 0.0          # Normalized 0..1 (loadavg / cores)
    cpu_percent: float = 0.0          # Instantaneous CPU usage 0..100
    ram_free_mb: float = 0.0          # Available RAM in MB
    ram_total_mb: float = 0.0         # Total system RAM in MB
    ram_free_ratio: float = 1.0       # ram_free / ram_total (0..1)
    battery_percent: float = 100.0    # 0..100
    is_charging: bool = True
    cpu_temperature_c: float = 50.0   # Degree Celsius
    error_rate_429: float = 0.0       # Rate of HTTP 429/403 errors in sliding window (0..1)
    queue_backlog: int = 0            # Number of tasks awaiting execution

    def to_bandit_feature_vector(self) -> List[float]:
        """
        Produce a normalized feature vector x_t in [0, 1]^6 for the Contextual Bandit:
        [ram_free_ratio, cpu_load_norm, battery_ratio, thermal_norm, error_rate_429, backlog_norm]
        """
        cpu_load_norm = min(1.0, max(0.0, self.cpu_percent / 100.0))
        battery_ratio = min(1.0, max(0.0, self.battery_percent / 100.0))
        thermal_norm = min(1.0, max(0.0, (self.cpu_temperature_c - 30.0) / 70.0))
        backlog_norm = min(1.0, self.queue_backlog / 50.0)

        return [
            round(self.ram_free_ratio, 4),
            round(cpu_load_norm, 4),
            round(battery_ratio, 4),
            round(thermal_norm, 4),
            round(self.error_rate_429, 4),
            round(backlog_norm, 4),
        ]


class DispatchActionType(str, Enum):
    EXECUTE = "EXECUTE"
    WAIT = "WAIT"
    GC_CLEAN = "GC_CLEAN"


class DispatchAction(BaseModel):
    task_id: str
    worker_id: str
    action_type: DispatchActionType = DispatchActionType.EXECUTE
    scheduled_start_ms: int = 0
    estimated_duration_ms: int = 0
    setup_cost_ms: int = 0


class DispatchPlan(BaseModel):
    actions: List[DispatchAction] = Field(default_factory=list)
    makespan_ms: int = 0
    total_setup_cost_ms: int = 0
    objective_score: float = 0.0

