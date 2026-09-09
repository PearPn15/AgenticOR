"""
Pure decision logic for `agentic-or guard`: given a tool name, the current
concurrency counts, and a live TelemetrySnapshot, decide whether Claude
Code should be allowed to run that tool right now. Deliberately reuses
OOMGuard/ThermalBatteryGuard as-is (agentic_or/safety/guards.py) rather
than duplicating threshold logic - a tool call and an AgenticOR-scheduled
TaskNode are governed by the exact same real machine thresholds.

No I/O, no state here - plain functions, easy to unit test and to reason
about independently of the HTTP daemon around them.

Three-way decision, matching Claude Code's own PreToolUse hook contract:
- "allow" - runs normally.
- "ask"   - Claude Code prompts the user for a permission decision, same
            as any other permission prompt. Used for POLICY throttles
            (concurrency caps, thermal/battery caution) - not immediately
            dangerous, just something worth a human's call.
- "deny"  - hard block, no prompt. Reserved for actual crash risk (RAM
            genuinely low/critical) where waiting on a human answer isn't
            safe to do.
"""
from __future__ import annotations

from typing import Tuple

from agentic_or.models import ExecutionProfile, TelemetrySnapshot
from agentic_or.safety.guards import OOMGuard, ThermalBatteryGuard

# Tools that are ALWAYS allowed, no matter how bad the machine's resource
# state is. Without this, a machine already under memory/thermal pressure
# could deny even the tools Claude Code needs to read the situation and
# explain itself to the user - bricking the assistant at the exact moment
# it's needed to diagnose the problem. Kept small and conservative: pure
# read/introspection/user-interaction tools that never spawn a process or
# do real work, so denying them was never going to help anyway.
ALWAYS_ALLOW = frozenset({
    "Read", "Grep", "Glob", "TodoWrite", "AskUserQuestion",
    "ExitPlanMode", "EnterPlanMode",
})

# Tools that spawn a real OS process / subprocess/network call heavy
# enough to be worth gating. Anything else (Write, Edit, ...) is a plain
# file write - cheap enough that it's left alone even under pressure,
# consistent with ALWAYS_ALLOW's reasoning: gating it wouldn't free up
# meaningful RAM/CPU, it would just make editing files impossible too.
HEAVY_TOOLS = frozenset({"Bash", "WebFetch", "WebSearch"})

# Subagent spawns are tracked and capped SEPARATELY from other heavy
# tools: a fan-out of many Task calls is a different (and often bigger)
# resource multiplier than one Bash call, and denying/asking about it
# shouldn't be starved or amplified by ordinary Bash traffic sharing the
# same counter.
SUBAGENT_TOOL = "Task"

# Max concurrent heavy tool calls (Bash/WebFetch/WebSearch combined,
# machine-wide across every Claude Code session this daemon knows about)
# allowed per ExecutionProfile - the same profile AgenticOR's own
# ThermalBatteryGuard already computes for its own workers. A deliberately
# simple static table rather than consulting the C++ contextual bandit
# directly: the bandit is tuned for TaskNode/DispatchPlan scheduling
# reward signals, not "should an interactive assistant's shell command
# wait a moment" - this stays a plain, predictable policy instead.
CONCURRENCY_CAP_BY_PROFILE = {
    ExecutionProfile.ECO_SILENT: 1,
    ExecutionProfile.BALANCED: 3,
    ExecutionProfile.TURBO_SPEED: 6,
}
SUBAGENT_CAP_BY_PROFILE = {
    ExecutionProfile.ECO_SILENT: 1,
    ExecutionProfile.BALANCED: 2,
    ExecutionProfile.TURBO_SPEED: 4,
}


def classify_tool(tool_name: str) -> str:
    if tool_name in ALWAYS_ALLOW:
        return "always_allow"
    if tool_name == SUBAGENT_TOOL:
        return "subagent"
    if tool_name in HEAVY_TOOLS:
        return "heavy"
    return "light"


def decide(
    tool_name: str,
    snapshot: TelemetrySnapshot,
    oom_guard: OOMGuard,
    thermal_guard: ThermalBatteryGuard,
    running_heavy_count: int = 0,
    running_subagent_count: int = 0,
) -> Tuple[str, str]:
    """
    Returns (decision, reason) where decision is "allow"/"ask"/"deny".
    Only "heavy"/"subagent" tools are ever gated; ALWAYS_ALLOW and
    everything else ("light") always passes regardless of machine state.
    """
    kind = classify_tool(tool_name)
    if kind not in ("heavy", "subagent"):
        return "allow", ""

    # RAM is the one real crash-risk category here - a hard "deny", no
    # "ask", because waiting on a human's answer isn't safe when memory is
    # already this tight.
    can_dispatch, emergency = oom_guard.evaluate(snapshot)
    if emergency:
        return "deny", (
            f"RAM critical ({snapshot.ram_free_mb:.0f}MB free) - "
            f"AgenticOR guard is blocking '{tool_name}' until memory recovers."
        )
    if not can_dispatch:
        return "deny", (
            f"RAM low ({snapshot.ram_free_mb:.0f}MB free, below safety threshold) - "
            f"AgenticOR guard is blocking '{tool_name}'."
        )

    profile = thermal_guard.determine_profile(snapshot)

    # Thermal/battery and concurrency are POLICY throttles, not imminent
    # crash risk - "ask" instead of a hard block, so you can still say
    # "go ahead anyway" for something you know is fine.
    if profile == ExecutionProfile.ECO_SILENT:
        return "ask", (
            f"Thermal/battery guard forced ECO_SILENT "
            f"(CPU {snapshot.cpu_temperature_c:.1f}°C, battery {snapshot.battery_percent:.0f}%) - "
            f"AgenticOR guard suggests holding off on '{tool_name}'."
        )

    if kind == "subagent":
        cap = SUBAGENT_CAP_BY_PROFILE[profile]
        if running_subagent_count >= cap:
            return "ask", (
                f"{running_subagent_count} subagents already running "
                f"(cap {cap} under {profile.value}) - "
                f"AgenticOR guard suggests waiting before spawning another."
            )
    else:
        cap = CONCURRENCY_CAP_BY_PROFILE[profile]
        if running_heavy_count >= cap:
            return "ask", (
                f"{running_heavy_count} heavy tool calls already running "
                f"(cap {cap} under {profile.value}) - "
                f"AgenticOR guard suggests waiting before running '{tool_name}'."
            )

    return "allow", ""
