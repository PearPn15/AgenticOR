"""
`agentic-or guard` - a persistent local daemon (unlike Orchestrator, which
only lives for one `run`/`demo`/`llm` call, this is meant to be started
once and left running, ideally auto-started - see hook_client.py's
`sessionstart`) that does three things for an EXTERNAL tool-calling agent
such as Claude Code:

1. Answers "is it safe to run this tool right now?" (POST /decision) with
   one of "allow"/"ask"/"deny", using AgenticOR's own OOMGuard/
   ThermalBatteryGuard against a continuously-sampled real
   TelemetrySnapshot, PLUS a real concurrency cap (how many heavy tool
   calls / subagents are ALREADY running right now, machine-wide) - see
   decision.py for the full policy.
2. Records what that agent is currently doing (POST /event, fed by its
   PreToolUse/PostToolUse/SubagentStart/SubagentStop/SessionEnd hooks) and
   mirrors it into the SAME shared status file `agentic-or watch`/`ui`
   already read (~/.agentic_or/status.json, under "claude_code_agents") -
   so Claude Code's own tool/subagent activity shows up in AgenticOR's
   existing live dashboard, next to its own workers.
3. Keeps a small rolling log of every non-"allow" decision
   (~/.agentic_or/guard_decisions.jsonl) - so you can see later what got
   throttled/blocked and why, not just in the moment it happened.

Pure stdlib (`http.server`), zero new dependency, binds to 127.0.0.1 only -
same posture as agentic_or/webui.py.

SAFETY: this daemon can crash, hang, or simply not be running at all -
none of that may ever block Claude Code. That guarantee lives entirely on
the CALLER's side (hook_client.py fails open on any error/timeout), not
here - this module only has to answer correctly when it IS reachable.
"""
from __future__ import annotations

import http.server
import json
import logging
import os
import socketserver
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_or.guard.decision import classify_tool, decide
from agentic_or.models import ExecutionProfile, TelemetrySnapshot
from agentic_or.safety.guards import OOMGuard, ThermalBatteryGuard
from agentic_or.telemetry.live_monitor import write_status_fields
from agentic_or.telemetry.system_monitor import SystemTelemetryMonitor

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8422
_SAMPLE_INTERVAL_S = 1.0
_STALE_TOOL_AGE_S = 300.0     # drop one tool call entry that's sat "done"/"denied"/"asked" this long
_STALE_SESSION_AGE_S = 3600.0  # drop a WHOLE session that's had no event at all in this long -
                               # a fallback for a process that got killed before SessionEnd could fire

DECISIONS_LOG_PATH = Path.home() / ".agentic_or" / "guard_decisions.jsonl"
_DECISIONS_LOG_MAX_LINES = 500  # bounded - this is a diagnostic tail, not an audit archive


def _append_decision_log(entry: Dict[str, Any]) -> None:
    """Append one non-"allow" decision, then cap the file at
    _DECISIONS_LOG_MAX_LINES so it can never grow unbounded. Best-effort:
    a logging hiccup must never be why a decision response is delayed."""
    try:
        DECISIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DECISIONS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        lines = DECISIONS_LOG_PATH.read_text(encoding="utf-8").splitlines()
        if len(lines) > _DECISIONS_LOG_MAX_LINES:
            DECISIONS_LOG_PATH.write_text(
                "\n".join(lines[-_DECISIONS_LOG_MAX_LINES:]) + "\n", encoding="utf-8"
            )
    except OSError as e:
        logger.warning(f"Could not write {DECISIONS_LOG_PATH}: {e}")


class GuardState:
    """
    All mutable state, behind one lock. One HTTP request is a handful of
    dict reads/writes and the sampler tick is ~1/sec, so a single lock
    keeps this simple without being a real contention point.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.pid = os.getpid()
        self.server: Optional["_ReusableTCPServer"] = None  # set by run_guard() once bound

        self.telemetry = SystemTelemetryMonitor()
        self.oom_guard = OOMGuard()
        self.thermal_guard = ThermalBatteryGuard()
        self.snapshot: TelemetrySnapshot = self.telemetry.capture_snapshot()
        self.profile: ExecutionProfile = ExecutionProfile.BALANCED
        # session_id -> {"cwd", "last_seen", "tools": {key: {tool_name, kind, status, ts}}}
        # status is one of RUNNING/done/denied/asked. Only RUNNING entries
        # of kind heavy/subagent count toward the concurrency caps in
        # decision.py - everything else here is purely for the dashboard.
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def sample(self) -> None:
        snap = self.telemetry.capture_snapshot()
        profile = self.thermal_guard.determine_profile(snap)
        with self.lock:
            self.snapshot = snap
            self.profile = profile

    def _count_running(self, kind: str) -> int:
        """Caller must hold self.lock."""
        return sum(
            1
            for sess in self.sessions.values()
            for t in sess["tools"].values()
            if t["status"] == "RUNNING" and t["kind"] == kind
        )

    def decide(self, tool_name: str) -> tuple[str, str]:
        with self.lock:
            snap, oom, thermal = self.snapshot, self.oom_guard, self.thermal_guard
            running_heavy = self._count_running("heavy")
            running_subagents = self._count_running("subagent")
        return decide(tool_name, snap, oom, thermal, running_heavy, running_subagents)

    def record_decision_result(self, event: Dict[str, Any], decision: str) -> None:
        """
        Called right after decide() for a PreToolUse event - unlike a bare
        /event report, this knows what ACTUALLY happens next per Claude
        Code's hook contract:
        - "allow": marked RUNNING, counts toward the concurrency caps.
        - "ask"/"deny": marked accordingly, NOT counted as running (it may
          never actually execute) - but kept under the SAME key, so if a
          human approves an "ask" and the tool really does run, the
          eventual PostToolUse event still finds this entry and flips it
          to "done" instead of silently doing nothing.
        """
        tool_name = event.get("tool_name", "?")
        status = {"allow": "RUNNING", "ask": "asked", "deny": "denied"}.get(decision, "denied")
        self._upsert_tool(event, tool_name, classify_tool(tool_name), status)

    def record_event(self, event: Dict[str, Any]) -> None:
        session_id = str(event.get("session_id") or "unknown")
        hook = event.get("hook_event_name", "")
        now = time.time()
        with self.lock:
            if hook == "SessionEnd":
                self.sessions.pop(session_id, None)
                return

            sess = self._get_session(session_id, event)

            if hook in ("PostToolUse", "PostToolUseFailure"):
                key = event.get("tool_use_id") or event.get("tool_name", "?")
                if key in sess["tools"]:
                    sess["tools"][key]["status"] = "done" if hook == "PostToolUse" else "failed"
                    sess["tools"][key]["ts"] = now
            elif hook == "SubagentStart":
                key = event.get("agent_id") or "subagent"
                sess["tools"][key] = {
                    "tool_name": f"Task: {event.get('agent_type', '?')}",
                    "kind": "subagent", "status": "RUNNING", "ts": now,
                }
            elif hook == "SubagentStop":
                key = event.get("agent_id") or "subagent"
                sess["tools"].pop(key, None)
            # PreToolUse is intentionally NOT handled here - see
            # record_decision_result(), which is the only place a
            # PreToolUse-driven entry is created, because only there do we
            # know what decision was actually made.

            self._prune_tools(sess, now)

    def _get_session(self, session_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
        """Caller must hold self.lock."""
        sess = self.sessions.setdefault(session_id, {"cwd": event.get("cwd", ""), "tools": {}})
        sess["cwd"] = event.get("cwd", sess["cwd"])
        sess["last_seen"] = time.time()
        return sess

    def _upsert_tool(self, event: Dict[str, Any], tool_name: str, kind: str, status: str) -> None:
        session_id = str(event.get("session_id") or "unknown")
        now = time.time()
        with self.lock:
            sess = self._get_session(session_id, event)
            key = event.get("tool_use_id") or tool_name
            sess["tools"][key] = {"tool_name": tool_name, "kind": kind, "status": status, "ts": now}
            self._prune_tools(sess, now)

    @staticmethod
    def _prune_tools(sess: Dict[str, Any], now: float) -> None:
        stale = [
            k for k, v in sess["tools"].items()
            if v["status"] != "RUNNING" and now - v["ts"] > _STALE_TOOL_AGE_S
        ]
        for k in stale:
            sess["tools"].pop(k, None)

    def prune_stale_sessions(self) -> None:
        """Fallback for a Claude Code process that never got to fire
        SessionEnd (killed, crashed, machine slept) - without this, that
        session's shell would sit in the dashboard forever."""
        now = time.time()
        with self.lock:
            stale = [
                sid for sid, sess in self.sessions.items()
                if now - sess.get("last_seen", now) > _STALE_SESSION_AGE_S
            ]
            for sid in stale:
                self.sessions.pop(sid, None)

    def to_agent_tree(self) -> List[Dict[str, Any]]:
        """Same node shape as AgentRegistry.to_tree_dict()['agents'] (agent_id/
        type/status/current_task/measured_resources/children) so the exact
        same renderers in cli.py/webui.py draw it with no changes there."""
        with self.lock:
            nodes = []
            for session_id, sess in self.sessions.items():
                children = [
                    {
                        "agent_id": key, "type": "CLAUDE_TOOL",
                        "status": "RUNNING" if t["status"] == "RUNNING" else "idle",
                        "current_task": t["tool_name"] if t["status"] == "RUNNING" else f"{t['tool_name']} ({t['status']})",
                        "measured_resources": None, "children": [],
                    }
                    for key, t in sess["tools"].items()
                ]
                running = any(c["status"] == "RUNNING" for c in children)
                nodes.append({
                    "agent_id": f"claude-code:{session_id[:8]}",
                    "type": "CLAUDE_CODE",
                    "status": "RUNNING" if running else "idle",
                    "current_task": sess.get("cwd", ""),
                    "measured_resources": None,
                    "children": children,
                })
            return nodes

    def to_status_dict(self, port: int) -> Dict[str, Any]:
        with self.lock:
            return {
                "ok": True, "pid": self.pid, "port": port,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "session_count": len(self.sessions),
                "profile": self.profile.value,
            }


class _Handler(http.server.BaseHTTPRequestHandler):
    state: GuardState  # set on the class by run_guard() before serving
    port: int = DEFAULT_PORT

    def log_message(self, fmt, *args):  # quiet: no HTTP access-log spam
        pass

    def _read_json_body(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ok": True})
        elif self.path == "/status":
            self._send_json(200, self.state.to_status_dict(self.port))
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        payload = self._read_json_body()
        if self.path == "/decision":
            tool_name = payload.get("tool_name", "")
            decision, reason = self.state.decide(tool_name)
            # Only NOW, knowing the actual decision, do we record it - see
            # record_decision_result()'s docstring for why this can't
            # happen inside decide() itself.
            self.state.record_decision_result(payload, decision)
            if decision != "allow":
                _append_decision_log({
                    "time": time.strftime("%H:%M:%S"), "decision": decision, "reason": reason,
                    "tool_name": tool_name, "session_id": payload.get("session_id"),
                })
            self._send_json(200, {"decision": decision, "reason": reason})
        elif self.path == "/event":
            self.state.record_event(payload)
            self._send_json(200, {"ok": True})
        elif self.path == "/shutdown":
            self._send_json(200, {"ok": True})
            # shutdown() blocks until serve_forever() (running in ITS OWN
            # thread - see run_guard()) actually returns, so it must be
            # called from a different thread than this request handler.
            if self.state.server is not None:
                threading.Thread(target=self.state.server.shutdown, daemon=True).start()
        else:
            self._send_json(404, {"error": "not found"})


class _ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _sampler_loop(state: GuardState, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        state.sample()
        state.prune_stale_sessions()
        with state.lock:
            snap, profile = state.snapshot, state.profile
        # Also write the plain telemetry fields `watch`/`ui` already render
        # (same keys LiveMonitor writes) - so `agentic-or guard` alone,
        # with no pipeline/LiveMonitor running at all, still gives a fully
        # useful `watch`/`ui` view, not just an empty shell around the
        # claude_code_agents tree. Merged via write_status_fields, so if a
        # LiveMonitor IS also running concurrently, whichever wrote most
        # recently simply wins for these particular fields - harmless
        # either way since both read the same real machine.
        write_status_fields(
            time=time.strftime("%H:%M:%S"),
            profile=profile.value,
            ram_free_ratio=snap.ram_free_ratio,
            ram_free_mb=snap.ram_free_mb,
            ram_total_mb=snap.ram_total_mb,
            cpu_percent=snap.cpu_percent,
            battery_percent=snap.battery_percent,
            is_charging=snap.is_charging,
            cpu_temperature_c=snap.cpu_temperature_c,
            claude_code_agents=state.to_agent_tree(),
        )
        stop_event.wait(_SAMPLE_INTERVAL_S)


def run_guard(port: int = DEFAULT_PORT) -> None:
    """Blocking - runs until Ctrl+C or a POST /shutdown (see
    `agentic-or guard stop`). Meant to be started once (its own terminal,
    a service manager, or auto-started by hook_client.py's `sessionstart`)
    and left running, unlike the rest of AgenticOR which only exists for
    one command's duration."""
    state = GuardState()
    _Handler.state = state
    _Handler.port = port
    stop_event = threading.Event()
    sampler = threading.Thread(target=_sampler_loop, args=(state, stop_event), daemon=True)
    sampler.start()

    server: Optional[_ReusableTCPServer] = None
    try:
        server = _ReusableTCPServer(("127.0.0.1", port), _Handler)
    except OSError as e:
        print(f"❌ Could not start the guard daemon on 127.0.0.1:{port}: {e}")
        print(f"   Try a different port: agentic-or guard --port {port + 1}")
        stop_event.set()
        return
    state.server = server

    print(f"🛡️  AgentOR guard listening on http://127.0.0.1:{port}  (pid {state.pid})")
    print("   Point Claude Code's hooks at this daemon - see docs/claude-code-guard.md.")
    print("   Its activity will also show up in `agentic-or watch`/`ui`.")
    print("   Ctrl+C, or `agentic-or guard stop`, to stop.")

    # serve_forever() runs in its OWN thread so a /shutdown request
    # (handled inside one of _ReusableTCPServer's worker threads) can call
    # server.shutdown() without deadlocking on itself.
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()
    try:
        while serve_thread.is_alive():
            serve_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()
