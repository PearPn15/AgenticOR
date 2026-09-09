from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ProxyEntry:
    def __init__(self, proxy_url: str):
        self.proxy_url = proxy_url
        self.is_healthy = True
        self.cooldown_until: float = 0.0
        self.failure_count = 0
        self.success_count = 0


class ProxyRotator:
    """
    Autonomous Proxy Rotation and Cooldown Daemon.
    Manages IP leases and health tracking to avoid rate-limit locks.
    """

    def __init__(self, initial_proxies: Optional[List[str]] = None):
        proxies = initial_proxies or [
            "direct://",
            "socks5://proxy1.internal:1080",
            "socks5://proxy2.internal:1080",
            "socks5://proxy3.internal:1080",
        ]
        self._pool: List[ProxyEntry] = [ProxyEntry(p) for p in proxies]
        self._current_idx = 0

    def lease_proxy(self) -> str:
        """Find the next available and healthy proxy not in cooldown."""
        now = time.time()
        for _ in range(len(self._pool)):
            entry = self._pool[self._current_idx]
            self._current_idx = (self._current_idx + 1) % len(self._pool)

            if entry.is_healthy and now >= entry.cooldown_until:
                return entry.proxy_url

        # If all in cooldown, return least penalty proxy
        return self._pool[0].proxy_url

    def report_error(self, proxy_url: str, cooldown_seconds: float = 60.0) -> None:
        """Mark proxy into cooldown after rate-limit or blocking error."""
        now = time.time()
        for entry in self._pool:
            if entry.proxy_url == proxy_url:
                entry.failure_count += 1
                entry.cooldown_until = now + cooldown_seconds
                logger.warning(
                    f"[ProxyRotator] Proxy '{proxy_url}' put in cooldown for {cooldown_seconds}s "
                    f"(Failures: {entry.failure_count})"
                )
                break

    def report_success(self, proxy_url: str) -> None:
        for entry in self._pool:
            if entry.proxy_url == proxy_url:
                entry.success_count += 1
                entry.failure_count = max(0, entry.failure_count - 1)
                break

