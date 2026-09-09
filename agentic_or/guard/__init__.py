"""
`agentic_or.guard` - AgenticOR acting as a resource governor for an
EXTERNAL tool-calling agent (Claude Code, via its hooks - see
docs/claude-code-guard.md), not just for its own TaskNode workers.

This is the one place in AgenticOR that is explicitly allowed to say no.
Everywhere else in this project (Orchestrator/LiveMonitor/`watch`/`ui`) is
deliberately read-only - see agentic_or/webui.py's docstring, "AgenticOR
observes, it doesn't gate". `guard` is the exception, by design, and is
built around one non-negotiable safety invariant: the hook client
(hook_client.py) that talks to this daemon must ALWAYS fail open. A crash,
a timeout, or `agentic-or guard` simply not running must be indistinguishable
from Claude Code's point of view from not having this integration at all -
never a hang, never a wrongly-blocked tool call.
"""
