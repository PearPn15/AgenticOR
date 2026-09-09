"""
CLI entry point invoked directly by Claude Code's own hook mechanism (see
docs/claude-code-guard.md for the exact settings.json snippet). Reads the
hook's JSON payload from stdin, asks the `agentic-or guard` daemon for a
decision or just reports the event, then writes back whatever Claude
Code's hook contract for that event expects.

    python3 -m agentic_or.guard.hook_client pretooluse       < PreToolUse payload
    python3 -m agentic_or.guard.hook_client posttooluse      < PostToolUse payload
    python3 -m agentic_or.guard.hook_client posttooluse_fail < PostToolUseFailure payload
    python3 -m agentic_or.guard.hook_client subagentstart    < SubagentStart payload
    python3 -m agentic_or.guard.hook_client subagentstop     < SubagentStop payload
    python3 -m agentic_or.guard.hook_client sessionstart     < SessionStart payload
    python3 -m agentic_or.guard.hook_client sessionend       < SessionEnd payload

SAFETY INVARIANT (the whole reason this file is this defensive): this
script must NEVER be the reason Claude Code hangs or gets wrongly
blocked. Every network call has a short timeout, every exception is
caught, and on ANY problem reaching the daemon (not running, slow,
malformed response) `pretooluse` falls back to allow and every other
event is just dropped. `agentic-or guard` crashing, not being started, or
this script itself blowing up must all be indistinguishable, from Claude
Code's point of view, from this hook never having been installed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

DAEMON_URL = "http://127.0.0.1:8422"
TIMEOUT_S = 1.0


def _post(path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{DAEMON_URL}{path}", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        # Daemon not running / unreachable / slow / malformed reply - this
        # is the fail-open path described in the module docstring above.
        return None


def _get(path: str) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(f"{DAEMON_URL}{path}", timeout=TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _read_hook_input() -> Dict[str, Any]:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def _pretooluse() -> None:
    payload = _read_hook_input()
    result = _post("/decision", payload)
    decision = (result or {}).get("decision", "allow")

    if decision not in ("ask", "deny"):
        sys.exit(0)  # allow - default on any doubt whatsoever (daemon down, malformed reply, ...)

    reason = (result or {}).get("reason") or "AgenticOR guard flagged this tool call."
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }))
    if decision == "deny":
        # Only a hard "deny" uses the legacy exit-code block signal too
        # (belt-and-suspenders in case the JSON contract isn't honored by
        # this Claude Code version) - "ask" must NOT exit 2, that would
        # block outright instead of prompting.
        print(reason, file=sys.stderr)
        sys.exit(2)
    sys.exit(0)


def _report() -> None:
    """PostToolUse/PostToolUseFailure/SubagentStart/SubagentStop/SessionEnd:
    purely informational for the dashboard, never blocks - report and
    exit 0 regardless of whether the daemon was reachable."""
    payload = _read_hook_input()
    _post("/event", payload)
    sys.exit(0)


def _sessionstart() -> None:
    """Auto-start `agentic-or guard` if it isn't already running, so you
    never have to remember to launch it by hand. Still reports the
    SessionStart event itself afterward like any other hook - purely
    informational, same as _report()."""
    payload = _read_hook_input()
    if _get("/health") is None:
        try:
            subprocess.Popen(
                [sys.executable, "-m", "agentic_or.cli", "guard"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,  # detach fully - must outlive this short-lived hook process
            )
        except Exception:
            pass  # best-effort only - a failed auto-start just means guard stays off, never blocks the session
    _post("/event", payload)
    sys.exit(0)


_DISPATCH = {
    "pretooluse": _pretooluse,
    "sessionstart": _sessionstart,
}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(0)  # misconfigured hook call - fail open, not fail loud
    _DISPATCH.get(sys.argv[1], _report)()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        # Absolute last resort: this hook must never itself crash Claude
        # Code's hook runner into an unclear state - always allow.
        sys.exit(0)
