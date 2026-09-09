from __future__ import annotations

import time
import logging
from typing import Dict, Optional, Tuple
from agentic_or.models import TelemetrySnapshot, ExecutionProfile

logger = logging.getLogger(__name__)


class OOMGuard:
    """
    Rào cản an toàn bộ nhớ (Hardware OOM-Guard Invariant).
    Bắt buộc duy trì RAM khả dụng >= min_ram_free_mb (Mặc định 1500MB).
    """

    def __init__(self, min_ram_free_mb: float = 1500.0, critical_ram_free_mb: float = 800.0):
        self.min_ram_free_mb = min_ram_free_mb
        self.critical_ram_free_mb = critical_ram_free_mb

    def evaluate(self, snapshot: TelemetrySnapshot) -> Tuple[bool, bool]:
        """
        Trả về (can_dispatch_heavy_tasks, trigger_emergency_gc)
        - can_dispatch_heavy_tasks: True nếu RAM khả dụng trên ngưỡng an toàn.
        - trigger_emergency_gc: True nếu RAM khả dụng rơi xuống mức báo động đỏ.
        """
        if snapshot.ram_free_mb < self.critical_ram_free_mb:
            logger.warning(
                f"[OOMGuard CRITICAL] RAM free={snapshot.ram_free_mb:.1f}MB < {self.critical_ram_free_mb}MB. "
                f"Emergency GC and hard throttle triggered!"
            )
            return False, True
        elif snapshot.ram_free_mb < self.min_ram_free_mb:
            logger.info(
                f"[OOMGuard WARNING] RAM free={snapshot.ram_free_mb:.1f}MB < {self.min_ram_free_mb}MB. "
                f"Heavy tasks paused."
            )
            return False, False
        return True, False


class ThermalBatteryGuard:
    """
    Rào cản an toàn nhiệt độ và pin (Thermal & Battery Invariant).
    Cưỡng chế chuyển sang chế độ ECO_SILENT nếu CPU quá 85°C hoặc pin dưới 25% khi rút sạc.
    """

    def __init__(self, max_cpu_temp_c: float = 85.0, min_battery_percent: float = 25.0):
        self.max_cpu_temp_c = max_cpu_temp_c
        self.min_battery_percent = min_battery_percent

    def determine_profile(self, snapshot: TelemetrySnapshot) -> ExecutionProfile:
        # Check thermal overload
        if snapshot.cpu_temperature_c >= self.max_cpu_temp_c:
            logger.warning(
                f"[ThermalGuard] CPU Temp={snapshot.cpu_temperature_c:.1f}°C >= {self.max_cpu_temp_c}°C. "
                f"Forcing ECO_SILENT profile."
            )
            return ExecutionProfile.ECO_SILENT

        # Check battery level
        if not snapshot.is_charging and snapshot.battery_percent <= self.min_battery_percent:
            logger.warning(
                f"[BatteryGuard] Battery={snapshot.battery_percent:.1f}% (Unplugged) <= {self.min_battery_percent}%. "
                f"Forcing ECO_SILENT profile."
            )
            return ExecutionProfile.ECO_SILENT

        # Turbo mode conditions: AC plugged, Battery > 80%, RAM free > 40%, CPU load < 50%
        if (
            snapshot.is_charging
            and snapshot.ram_free_ratio >= 0.40
            and snapshot.cpu_load_1m <= 0.50
            and snapshot.cpu_temperature_c < 75.0
        ):
            return ExecutionProfile.TURBO_SPEED

        return ExecutionProfile.BALANCED


class DomainCircuitBreaker:
    """
    Rào cản an toàn Rate Limit và Domain (Domain Circuit Breaker Invariant).
    Nếu 1 domain bị dính 3 lỗi 429/403 liên tiếp -> Đóng băng domain và thông báo đổi Proxy.
    """

    def __init__(self, failure_threshold: int = 3, initial_cooldown_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.initial_cooldown_seconds = initial_cooldown_seconds
        
        # domain -> consecutive failure count
        self._consecutive_failures: Dict[str, int] = {}
        # domain -> timestamp when cooldown expires
        self._cooldown_expiry: Dict[str, float] = {}

    def record_success(self, domain: str) -> None:
        """Reset failure counter when request succeeds."""
        if not domain:
            return
        self._consecutive_failures[domain] = 0
        if domain in self._cooldown_expiry:
            del self._cooldown_expiry[domain]

    def record_failure(self, domain: str) -> Tuple[bool, float]:
        """
        Ghi nhận lỗi 429/403.
        Trả về: (is_tripped, cooldown_remaining_seconds)
        """
        if not domain:
            return False, 0.0

        fails = self._consecutive_failures.get(domain, 0) + 1
        self._consecutive_failures[domain] = fails

        if fails >= self.failure_threshold:
            # Exponential backoff based on how many times threshold is exceeded
            multiplier = 2 ** (fails - self.failure_threshold)
            cooldown_duration = min(300.0, self.initial_cooldown_seconds * multiplier)
            expiry = time.time() + cooldown_duration
            self._cooldown_expiry[domain] = expiry
            logger.warning(
                f"[CircuitBreaker TRIPPED] Domain '{domain}' hit {fails} consecutive failures. "
                f"Freezing domain for {cooldown_duration:.1f}s until {expiry:.1f}."
            )
            return True, cooldown_duration

        return False, 0.0

    def is_domain_allowed(self, domain: str) -> Tuple[bool, float]:
        """
        Kiểm tra xem domain có được phép gửi request không.
        Trả về: (is_allowed, cooldown_remaining_seconds)
        """
        if not domain:
            return True, 0.0

        expiry = self._cooldown_expiry.get(domain)
        if expiry is None:
            return True, 0.0

        now = time.time()
        if now >= expiry:
            # Cooldown passed, allow trial request (Half-Open)
            del self._cooldown_expiry[domain]
            return True, 0.0

        return False, max(0.0, expiry - now)

    def get_blocked_domains(self) -> Dict[str, float]:
        """
        Read-only introspection for monitoring/dashboards: domains currently
        cooling down, mapped to remaining seconds. Never mutates state.
        """
        now = time.time()
        return {
            domain: round(expiry - now, 1)
            for domain, expiry in self._cooldown_expiry.items()
            if expiry > now
        }

