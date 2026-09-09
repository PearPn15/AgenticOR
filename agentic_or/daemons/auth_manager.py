from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionCredentials:
    """JWT token / cookie jar for a single session or domain."""
    token: Optional[str] = None
    cookies: Dict[str, str] = field(default_factory=dict)
    expires_at: float = 0.0  # epoch seconds; 0.0 = expiry unknown/not tracked

    def is_expired(self, skew_seconds: float = 30.0) -> bool:
        """True if unknown-expiry or within `skew_seconds` of expiring (safety margin)."""
        if self.expires_at <= 0.0:
            return False
        return time.time() >= (self.expires_at - skew_seconds)


# A refresh callback takes the session_key and returns fresh SessionCredentials
# (e.g. re-login, exchange a refresh_token, re-run a browser login flow).
RefreshCallback = Callable[[str], Awaitable[SessionCredentials]]


class SessionAuthManager:
    """
    Autonomous Token/Cookie Refresh Daemon (Rào cản Phiên Đăng Nhập).
    Giữ JWT Token và Cookie Jar cho từng session/domain luôn "tươi" (fresh),
    tự động làm mới trước khi hết hạn hoặc ngay khi Worker báo lỗi 401/hết phiên,
    tránh việc Worker bị dispatch với credential đã chết.
    """

    def __init__(
        self,
        refresh_margin_seconds: float = 30.0,
        max_consecutive_refresh_failures: int = 3,
    ):
        self.refresh_margin_seconds = refresh_margin_seconds
        self.max_consecutive_refresh_failures = max_consecutive_refresh_failures

        self._sessions: Dict[str, SessionCredentials] = {}
        self._refresh_callbacks: Dict[str, RefreshCallback] = {}
        self._refresh_failures: Dict[str, int] = {}
        # One lock per session so concurrent workers hitting the same expired
        # session don't all fire a duplicate refresh ("thundering herd").
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, session_key: str) -> asyncio.Lock:
        lock = self._locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_key] = lock
        return lock

    def register_session(
        self,
        session_key: str,
        credentials: Optional[SessionCredentials] = None,
        refresh_callback: Optional[RefreshCallback] = None,
    ) -> None:
        """Register initial credentials and/or the async callback used to refresh them."""
        if credentials is not None:
            self._sessions[session_key] = credentials
        if refresh_callback is not None:
            self._refresh_callbacks[session_key] = refresh_callback
        self._refresh_failures.setdefault(session_key, 0)

    def get_credentials(self, session_key: str) -> Optional[SessionCredentials]:
        return self._sessions.get(session_key)

    def is_locked_out(self, session_key: str) -> bool:
        """True once a session has exhausted its consecutive refresh retries."""
        return self._refresh_failures.get(session_key, 0) >= self.max_consecutive_refresh_failures

    def list_locked_out_sessions(self) -> list[str]:
        """Read-only introspection for monitoring/dashboards. Never mutates state."""
        return [key for key in self._refresh_failures if self.is_locked_out(key)]

    async def ensure_fresh(self, session_key: str) -> Optional[SessionCredentials]:
        """
        Return valid credentials for `session_key`, transparently refreshing first
        if they are near/at expiry. Returns None if there is nothing registered,
        or the session is locked out after too many failed refresh attempts.
        """
        if self.is_locked_out(session_key):
            return None

        current = self._sessions.get(session_key)
        if current is not None and not current.is_expired(self.refresh_margin_seconds):
            return current

        return await self.force_refresh(session_key)

    async def force_refresh(self, session_key: str) -> Optional[SessionCredentials]:
        """
        Force an immediate refresh, bypassing expiry tracking. Called by the
        Orchestrator right after a Worker reports a 401 / "session expired" failure.
        """
        if session_key not in self._refresh_callbacks:
            logger.warning(f"[SessionAuthManager] No refresh_callback registered for '{session_key}'.")
            return self._sessions.get(session_key)

        async with self._lock_for(session_key):
            # Another task may have already refreshed while we waited on the lock.
            current = self._sessions.get(session_key)
            if current is not None and not current.is_expired(self.refresh_margin_seconds):
                return current

            try:
                logger.info(f"[SessionAuthManager] Refreshing session '{session_key}'...")
                fresh = await self._refresh_callbacks[session_key](session_key)
                self._sessions[session_key] = fresh
                self._refresh_failures[session_key] = 0
                logger.info(f"[SessionAuthManager] Session '{session_key}' refreshed successfully.")
                return fresh
            except Exception as e:
                fails = self._refresh_failures.get(session_key, 0) + 1
                self._refresh_failures[session_key] = fails
                logger.warning(
                    f"[SessionAuthManager] Refresh failed for '{session_key}' "
                    f"({fails}/{self.max_consecutive_refresh_failures}): {e}"
                )
                if fails >= self.max_consecutive_refresh_failures:
                    logger.error(
                        f"[SessionAuthManager] Session '{session_key}' locked out after "
                        f"{fails} consecutive refresh failures. Manual re-auth required."
                    )
                return None
