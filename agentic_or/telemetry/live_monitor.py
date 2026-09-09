from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TextIO, Tuple

from agentic_or import theme

if TYPE_CHECKING:
    from agentic_or.orchestrator import Orchestrator

_CLEAR_HOME = "\x1b[H\x1b[2J"
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"

# Where LiveMonitor mirrors its state so a DIFFERENT terminal/process can
# watch a run live via `agentic-or watch` - see LiveMonitor.write_status_file
# and cli.py's cmd_watch. Plain file + poll (~1s), not a socket/server: no
# new moving part, and it's the same pattern already used for persisted
# agents (~/.agentic_or/agents.json).
STATUS_FILE_PATH = Path.home() / ".agentic_or" / "status.json"


def write_status_fields(**fields: Any) -> None:
    """
    Immediately merge `fields` into the status file, independent of
    whether any `LiveMonitor` is currently rendering. Plain
    `Orchestrator.set_monitor_extra_status()` only updates in-memory state
    that gets flushed to disk on a monitor's NEXT render cycle - between two
    `spawn_agent()+run()` calls (e.g. while `main_agent.py` is waiting on
    the LLM to decide the next action) no monitor is active, so that update
    would otherwise sit unwritten and a `watch`-ing terminal would keep
    showing the previous action as if still in progress. Use this instead
    for anything that must be visible right away regardless of monitor
    state - see `write_session_status` below and main_agent.py's
    `_dispatch_action`.
    """
    try:
        existing: Dict[str, Any] = {}
        if STATUS_FILE_PATH.exists():
            existing = json.loads(STATUS_FILE_PATH.read_text())
        existing.update(fields)
        STATUS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = STATUS_FILE_PATH.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(existing))
        tmp_path.replace(STATUS_FILE_PATH)
    except OSError as e:
        logging.getLogger(__name__).warning(f"Could not write status fields to {STATUS_FILE_PATH}: {e}")


def write_session_status(active: bool, error: Optional[str] = None) -> None:
    """
    Mark the WHOLE calling process's lifecycle in the status file -
    `session_active`/`session_error` - as opposed to `monitor_active`
    (LiveMonitor's own field), which only reflects one
    `Orchestrator.run()` call's dispatch loop and flips back to `True` on
    the very next call. A long-lived caller like `main_agent.py` (many
    `run()` calls across a chat session, idle between them waiting on the
    LLM/user) should call this once with `active=True` right at startup,
    and once more with `active=False` (and `error=str(e)` on a crash) in a
    `finally` around its whole main loop - see main_agent.py's `main()`.
    Without this, a `watch`-ing terminal cannot tell "briefly idle between
    actions" apart from "the process crashed or was killed" - both leave
    the status file's last `monitor_active` frame looking identical.
    """
    write_status_fields(session_active=active, session_error=error)


# Loggers whose per-task chatter would otherwise scroll the dashboard away.
# Purely cosmetic - only affects log verbosity, never scheduling/execution.
_CHATTY_LOGGERS = (
    "agentic_or.orchestrator",
    "agentic_or.workers",
    "agentic_or.daemons",
    "agentic_or.safety",
)


def format_agents_table(orchestrator: "Orchestrator", enabled: bool = None) -> str:
    """
    Read-only: one row per Agent/Worker (id, type, busy/idle, current task_id).
    Shared by the `agents` CLI/REPL command and the LiveMonitor dashboard, so
    both always show the exact same view of "how many Agents are running".
    """
    if enabled is None:
        enabled = theme.color_enabled()
    workers = list(orchestrator.local_workers) + list(orchestrator.api_workers) + list(orchestrator.browser_workers)
    busy = sum(1 for w in workers if w.is_busy)
    idle = len(workers) - busy

    title = (
        f"AGENTS  ({len(workers)} total · "
        f"{theme.paint(f'{busy} running', theme.GREEN, enabled=enabled)} · "
        f"{theme.paint(f'{idle} idle', theme.GRAY, enabled=enabled)})"
    )
    if not workers:
        return title

    rows = [
        [
            w.worker_id,
            w.workload_type.value,
            theme.paint("● busy", theme.GREEN, enabled=enabled) if w.is_busy
            else theme.paint("○ idle", theme.GRAY, enabled=enabled),
            w.current_task_id or theme.paint("-", theme.GRAY, enabled=enabled),
        ]
        for w in workers
    ]
    lines = [title] + theme.table(["WORKER ID", "TYPE", "STATUS", "CURRENT TASK"], rows, enabled=enabled)
    return "\n".join(lines)


class LiveMonitor:
    """
    Bảng Giám Sát Thời Gian Thực (Live Monitor) - CHỈ ĐỌC, KHÔNG CAN THIỆP.

    Chạy như một asyncio Task hoàn toàn độc lập với vòng lặp Dispatch của
    Orchestrator: tự chụp Telemetry và đọc các thuộc tính công khai của
    Broker/Worker Pools/Daemons theo chu kỳ riêng (`refresh_seconds`).
    Không hề gọi bất kỳ hàm nào làm thay đổi lịch trình, dispatch hay trạng
    thái tác vụ - vì vậy không thể làm chậm hoặc ảnh hưởng tới tiến trình
    thực thi, kể cả khi bật/tắt giữa chừng.

    Nhẹ theo đúng nghĩa: không có dependency mới, chỉ dùng ANSI escape codes
    + `print()` có sẵn trong stdlib.
    """

    def __init__(
        self,
        orchestrator: "Orchestrator",
        refresh_seconds: float = 1.0,
        stream: Optional[TextIO] = None,
        quiet_other_logs: bool = True,
        render_to_terminal: bool = True,
        write_status_file: bool = True,
    ):
        self.orchestrator = orchestrator
        self.refresh_seconds = max(0.2, refresh_seconds)
        self.stream = stream or sys.stdout
        self.quiet_other_logs = quiet_other_logs
        # render_to_terminal=False is for a caller that owns its OWN stdout
        # for something else (e.g. main_agent.py's chat prompt) - it still
        # gets cross-terminal visibility via write_status_file, just without
        # this process's own terminal being cleared/redrawn every cycle.
        self.render_to_terminal = render_to_terminal
        self.write_status_file = write_status_file
        # Free-form extra fields merged into the status file each cycle -
        # e.g. main_agent.py sets extra_status["current_action"] to whatever
        # it's currently deciding, so a `watch`-ing terminal can show it.
        self.extra_status: Dict[str, Any] = {}

        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._suppressed_loggers: List[Tuple[logging.Logger, int]] = []
        # Written into the status file as "monitor_active" - lets a `watch`
        # process tell "still live" apart from "this is the last frame of a
        # run that already finished/stopped" (otherwise a stale file just
        # looks identical to a live-but-idle one).
        self._active = False

    def start(self) -> None:
        """Begin rendering in the background. Safe to call once per run."""
        if self._task is not None:
            return
        self._stop_event = asyncio.Event()
        self._active = True
        if self.quiet_other_logs:
            self._quiet_chatty_loggers()
        if self._is_tty:
            self.stream.write(_HIDE_CURSOR)
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop rendering and restore whatever the dashboard temporarily changed."""
        if self._task is None:
            return
        assert self._stop_event is not None
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=self.refresh_seconds + 2.0)
        except asyncio.TimeoutError:
            self._task.cancel()
        self._task = None

        if self._is_tty:
            self.stream.write(_SHOW_CURSOR)
            self.stream.flush()
        self._restore_chatty_loggers()

    def _quiet_chatty_loggers(self) -> None:
        for name in _CHATTY_LOGGERS:
            logger = logging.getLogger(name)
            self._suppressed_loggers.append((logger, logger.level))
            logger.setLevel(logging.WARNING)

    def _restore_chatty_loggers(self) -> None:
        for logger, level in self._suppressed_loggers:
            logger.setLevel(level)
        self._suppressed_loggers.clear()

    async def _loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._render()
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.refresh_seconds)
                except asyncio.TimeoutError:
                    pass
            self._active = False
            self._render()  # final frame reflects end-of-run state, monitor_active=False
        except asyncio.CancelledError:
            self._active = False

    def _render(self) -> None:
        orch = self.orchestrator
        broker = orch.broker

        # Read-only: capture_snapshot() only queries live OS metrics via psutil,
        # it never mutates orchestrator/broker state.
        snapshot = orch.telemetry.capture_snapshot(queue_backlog=broker.get_backlog_count())
        profile = orch.thermal_guard.determine_profile(snapshot)

        total = len(broker._tasks)
        completed = len(broker._completed_tasks)
        failed = len(broker._failed_tasks)
        running = sum(1 for cp in broker._checkpoints.values() if cp.status == "RUNNING")
        pending = max(0, total - completed - failed - running)

        blocked_domains = orch.circuit_breaker.get_blocked_domains()
        pending_challenges = orch.captcha_solver.list_pending_challenges()
        locked_sessions = orch.auth_manager.list_locked_out_sessions()

        now_str = time.strftime("%H:%M:%S")
        en = theme.color_enabled(self.stream)
        width = 70
        profile_color = {
            "TURBO_SPEED": theme.CYAN, "BALANCED": theme.GRAY, "ECO_SILENT": theme.YELLOW,
        }.get(profile.value, theme.GRAY)

        lines = theme.header("AGENTOR LIVE MONITOR", now_str, width=width, enabled=en)
        lines.append(theme.panel_line(
            f"Profile: {theme.paint(profile.value, theme.BOLD, profile_color, enabled=en)}", width, en))
        lines.append(theme.divider(width, enabled=en))
        lines.append(theme.panel_line(
            f"RAM     [{theme.bar(snapshot.ram_free_ratio, enabled=en)}] {snapshot.ram_free_ratio * 100:5.1f}% free "
            f"({snapshot.ram_free_mb:.0f}/{snapshot.ram_total_mb:.0f} MB)", width, en))
        lines.append(theme.panel_line(
            f"CPU     [{theme.bar(snapshot.cpu_percent / 100.0, invert=True, enabled=en)}] {snapshot.cpu_percent:5.1f}%", width, en))
        lines.append(theme.panel_line(
            f"Battery [{theme.bar(snapshot.battery_percent / 100.0, enabled=en)}] {snapshot.battery_percent:5.1f}% "
            f"({'charging' if snapshot.is_charging else 'on battery'})", width, en))
        lines.append(theme.panel_line(
            f"Temp    {snapshot.cpu_temperature_c:.1f}°C     429/403 rate: {snapshot.error_rate_429 * 100:.1f}%",
            width, en))
        lines.append(theme.divider(width, enabled=en))
        lines.append(theme.panel_line(
            f"Tasks    total={total}  pending={pending}  running={theme.paint(str(running), theme.GREEN, enabled=en)}  "
            f"completed={theme.paint(str(completed), theme.GREEN, enabled=en)}  "
            f"failed={theme.paint(str(failed), theme.RED, enabled=en) if failed else failed}", width, en))
        lines.append(theme.divider(width, enabled=en))
        lines.append(theme.panel_line(
            "Self-Healing   Circuit-broken domains: " + str(len(blocked_domains))
            + (f" {list(blocked_domains.keys())}" if blocked_domains else ""), width, en))
        lines.append(theme.panel_line(
            f"               Captcha/2FA pending: {len(pending_challenges)}   "
            f"Locked-out sessions: {len(locked_sessions)}", width, en))
        lines.append(theme.footer(width, enabled=en))

        if self.render_to_terminal:
            out = "\n".join(lines) + "\n" + format_agents_table(orch, enabled=en) + "\n"
            if self._is_tty:
                self.stream.write(_CLEAR_HOME)
            self.stream.write(out)
            self.stream.flush()

        if self.write_status_file:
            self._write_status_file(
                now_str, profile.value, snapshot,
                total, pending, running, completed, failed,
                blocked_domains, pending_challenges, locked_sessions,
            )

    def _write_status_file(
        self, now_str, profile_value, snapshot,
        total, pending, running, completed, failed,
        blocked_domains, pending_challenges, locked_sessions,
    ) -> None:
        orch = self.orchestrator
        workers = list(orch.local_workers) + list(orch.api_workers) + list(orch.browser_workers)

        data: Dict[str, Any] = {
            "time": now_str,
            "monitor_active": self._active,
            "profile": profile_value,
            "ram_free_ratio": snapshot.ram_free_ratio,
            "ram_free_mb": snapshot.ram_free_mb,
            "ram_total_mb": snapshot.ram_total_mb,
            "cpu_percent": snapshot.cpu_percent,
            "battery_percent": snapshot.battery_percent,
            "is_charging": snapshot.is_charging,
            "cpu_temperature_c": snapshot.cpu_temperature_c,
            "error_rate_429": snapshot.error_rate_429,
            "tasks": {
                "total": total, "pending": pending, "running": running,
                "completed": completed, "failed": failed,
            },
            "self_healing": {
                "circuit_broken_domains": list(blocked_domains.keys()),
                "captcha_pending": len(pending_challenges),
                "locked_sessions": len(locked_sessions),
            },
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "type": w.workload_type.value,
                    "busy": w.is_busy,
                    "current_task": w.current_task_id,
                }
                for w in workers
            ],
            # Parent -> child agent tree (see agentic_or/registry.py) -
            # who registered whom, built dynamically at runtime via
            # Orchestrator.register_sub_agent(), not declared upfront.
            "agent_tree": orch.agent_registry.to_tree_dict()["agents"],
            **self.extra_status,
        }

        try:
            STATUS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a `watch` process reading concurrently
            # never sees a half-written file.
            tmp_path = STATUS_FILE_PATH.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data))
            tmp_path.replace(STATUS_FILE_PATH)
        except OSError as e:
            logging.getLogger(__name__).warning(f"Could not write status file {STATUS_FILE_PATH}: {e}")
