from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ChallengeType(str, Enum):
    CAPTCHA = "CAPTCHA"
    TWO_FACTOR = "2FA"


class ChallengeStatus(str, Enum):
    PENDING = "PENDING"
    SOLVED = "SOLVED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


class PendingChallenge:
    """A single Captcha/2FA wall a Worker is currently blocked on."""

    def __init__(
        self,
        challenge_id: str,
        domain: str,
        challenge_type: ChallengeType,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.challenge_id = challenge_id
        self.domain = domain
        self.challenge_type = challenge_type
        self.context = context or {}
        self.status = ChallengeStatus.PENDING
        self.created_at = time.time()
        self.solution: Optional[str] = None
        self._event = asyncio.Event()


# An auto-solve strategy receives the PendingChallenge and returns a solution
# token/string, or None if it can't solve it (e.g. call out to a 2Captcha/
# Anti-Captcha-style API, or an in-house OCR/vision model).
AutoSolveCallback = Callable[[PendingChallenge], Awaitable[Optional[str]]]


class CaptchaSolver:
    """
    Autonomous Captcha/2FA Resolver Daemon (Đột Phá Captcha/2FA).
    Khi một Worker báo bị chặn bởi Captcha hoặc 2FA, daemon này thử giải tự động
    qua `auto_solve_callback` (vd. gọi dịch vụ 2Captcha/Anti-Captcha), và nếu
    không có/thất bại thì rơi xuống hàng đợi **Human-in-the-loop**: chờ một
    người vận hành nộp lời giải qua `submit_human_solution`, có giới hạn thời
    gian chờ (`human_timeout_seconds`) để Worker không bị treo vô thời hạn.
    Một domain liên tục thất bại sẽ bị "làm nguội" (cooldown) để tránh việc
    dội bom captcha vào một trang đã siết phòng thủ.
    """

    def __init__(
        self,
        auto_solve_callback: Optional[AutoSolveCallback] = None,
        human_timeout_seconds: float = 120.0,
        max_consecutive_failures: int = 3,
        cooldown_seconds: float = 60.0,
    ):
        self.auto_solve_callback = auto_solve_callback
        self.human_timeout_seconds = human_timeout_seconds
        self.max_consecutive_failures = max_consecutive_failures
        self.cooldown_seconds = cooldown_seconds

        self._pending: Dict[str, PendingChallenge] = {}
        self._consecutive_failures: Dict[str, int] = {}
        self._domain_cooldown_until: Dict[str, float] = {}
        self._counter = 0

    def is_domain_blocked(self, domain: str) -> bool:
        """True while `domain` is cooling down after repeated captcha failures."""
        return time.time() < self._domain_cooldown_until.get(domain, 0.0)

    def list_pending_challenges(self) -> List[PendingChallenge]:
        """Snapshot of challenges currently awaiting a human solution (for the Control Tower/CLI)."""
        return [c for c in self._pending.values() if c.status == ChallengeStatus.PENDING]

    async def resolve_challenge(
        self,
        domain: str,
        challenge_type: ChallengeType = ChallengeType.CAPTCHA,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Entry point called by the Orchestrator when a Worker reports being blocked.
        Tries auto-solve first, then waits for a human, bounded by a timeout.
        Returns the solution string, or None if unresolved (caller should treat
        the task as failed and let the Domain Circuit Breaker cool the domain).
        """
        if self.is_domain_blocked(domain):
            logger.warning(f"[CaptchaSolver] Domain '{domain}' is in post-captcha cooldown; refusing new challenge.")
            return None

        self._counter += 1
        challenge_id = f"chal_{self._counter}"
        challenge = PendingChallenge(challenge_id, domain, challenge_type, context)
        self._pending[challenge_id] = challenge

        try:
            if self.auto_solve_callback is not None:
                solution = await self._try_auto_solve(challenge)
                if solution:
                    challenge.status = ChallengeStatus.SOLVED
                    challenge.solution = solution
                    self._record_success(domain)
                    logger.info(
                        f"[CaptchaSolver] Auto-solved {challenge_type.value} for '{domain}' ({challenge_id})."
                    )
                    return solution
                logger.info(
                    f"[CaptchaSolver] Auto-solve unavailable/failed for {challenge_id}; "
                    f"escalating to human-in-the-loop."
                )

            return await self._wait_for_human(challenge)
        finally:
            self._pending.pop(challenge_id, None)

    async def _try_auto_solve(self, challenge: PendingChallenge) -> Optional[str]:
        try:
            return await self.auto_solve_callback(challenge)
        except Exception as e:
            logger.warning(f"[CaptchaSolver] Auto-solve callback raised: {e}")
            return None

    async def _wait_for_human(self, challenge: PendingChallenge) -> Optional[str]:
        logger.warning(
            f"[CaptchaSolver] Waiting up to {self.human_timeout_seconds:.0f}s for a human to solve "
            f"{challenge.challenge_type.value} on '{challenge.domain}' (challenge_id={challenge.challenge_id})."
        )
        try:
            await asyncio.wait_for(challenge._event.wait(), timeout=self.human_timeout_seconds)
        except asyncio.TimeoutError:
            challenge.status = ChallengeStatus.TIMED_OUT
            self._record_failure(challenge.domain)
            logger.error(
                f"[CaptchaSolver] Challenge {challenge.challenge_id} on '{challenge.domain}' "
                f"timed out with no human response."
            )
            return None

        if challenge.status == ChallengeStatus.SOLVED:
            self._record_success(challenge.domain)
            return challenge.solution

        self._record_failure(challenge.domain)
        return None

    def submit_human_solution(self, challenge_id: str, solution: str) -> bool:
        """Called by the Control Tower/CLI once a human operator has solved a pending challenge."""
        challenge = self._pending.get(challenge_id)
        if challenge is None or challenge.status != ChallengeStatus.PENDING:
            return False
        challenge.solution = solution
        challenge.status = ChallengeStatus.SOLVED
        challenge._event.set()
        return True

    def dismiss_challenge(self, challenge_id: str) -> bool:
        """Called when a human explicitly gives up on a pending challenge."""
        challenge = self._pending.get(challenge_id)
        if challenge is None or challenge.status != ChallengeStatus.PENDING:
            return False
        challenge.status = ChallengeStatus.FAILED
        challenge._event.set()
        return True

    def _record_failure(self, domain: str) -> None:
        if not domain:
            return
        fails = self._consecutive_failures.get(domain, 0) + 1
        self._consecutive_failures[domain] = fails
        if fails >= self.max_consecutive_failures:
            expiry = time.time() + self.cooldown_seconds
            self._domain_cooldown_until[domain] = expiry
            logger.warning(
                f"[CaptchaSolver] Domain '{domain}' hit {fails} consecutive captcha failures. "
                f"Cooling down for {self.cooldown_seconds:.1f}s."
            )

    def _record_success(self, domain: str) -> None:
        if not domain:
            return
        self._consecutive_failures[domain] = 0
        self._domain_cooldown_until.pop(domain, None)
